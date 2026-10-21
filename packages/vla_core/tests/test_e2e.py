"""Suite E2E: el Learner real conduce un Core real por los tres flujos.

Integra las dos mitades del harness de punta a punta: un ``Learner`` real
(política, algoritmo, optimizador y fuente de datos) hablando por ZeroMQ
(``CoreClient``) con un ``CoreServer`` que envuelve el gateway ``Core`` real y
sus componentes fake (Environment, WorldModel, Reward Model, Dataset reach 2D en
CPU). A diferencia de la suite LEARN —que usa el ``FakeCoreClient`` en memoria—,
aquí todo pasa por el wire y por el gateway real.

La tesis es que **el entrenamiento avanza** en los tres modos (supervisado, RL en
ambiente, RL sobre el world model). Los umbrales van con margen por debajo de los
de LEARN: aquí se prueba la integración, no se recalibra la convergencia.

Marcada ``slow``: corre en CPU, puede tardar.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from vla_core.server import CoreServer
from vla_core.testing.fakes import make_fake_core
from vla_learner.learner import Learner
from vla_learner.testing.evaluation import success_rate

pytestmark = pytest.mark.slow


@contextmanager
def core_zmq(demo_chunk_size: int = 1) -> Iterator[str]:
    """Levanta un Core real (gateway + fakes) tras un ``CoreServer`` ZeroMQ."""
    server = CoreServer(make_fake_core(demo_chunk_size=demo_chunk_size), "tcp://127.0.0.1:0")
    address = server.start()
    try:
        yield address
    finally:
        server.stop()


def _policy(with_value: bool = False, init_log_std: float = -0.5) -> dict:
    cfg: dict = {
        "type": "composed",
        "action_dim": 2,
        "chunk_size": 1,
        "feature_dim": 32,
        "backbone": {"type": "tiny", "seq_len": 1},
        "action_head": {"type": "gaussian", "init_zero": True, "init_log_std": init_log_std},
        "processor": {"type": "simple", "cameras": ["front"]},
        "normalizer": {"mode": "mean_std"},
    }
    if with_value:
        cfg["value_head"] = {"hidden_dim": 32}
    return cfg


def _learner(
    address: str, tmp_path, policy: dict, algorithm: dict, data: dict, lr: float
) -> Learner:
    cfg = {
        "seed": 0,
        "core": {"type": "zmq", "address": address, "timeout_s": 10.0},
        "policy": policy,
        "algorithm": algorithm,
        "data": data,
        "optimizer": {
            "name": "adamw",
            "lr": lr,
            "grad_clip": 1.0,
            "scheduler": None,
            "precision": "fp32",
        },
        "checkpoint": {"directory": str(tmp_path / "ckpt"), "every": 10_000},
    }
    learner = Learner.from_config(cfg)
    learner.logger.is_main = False
    return learner


def test_e2e_supervisado(tmp_path) -> None:
    with core_zmq() as address:
        learner = _learner(
            address,
            tmp_path,
            _policy(init_log_std=-1.0),
            {"type": "bc"},
            {"type": "supervised", "batch_size": 64},
            lr=1e-3,
        )
        learner.train(250)
        assert success_rate(learner.policy) > 0.8


def test_e2e_rl_en_ambiente(tmp_path) -> None:
    with core_zmq() as address:
        learner = _learner(
            address,
            tmp_path,
            _policy(with_value=True, init_log_std=-1.0),
            {
                "type": "ppo",
                "clip_ratio": 0.2,
                "update_epochs": 10,
                "minibatch_size": 64,
                "value_coef": 0.5,
                "entropy_coef": 0.0,
                "advantage": {"type": "gae", "gamma": 0.99, "lam": 0.95},
            },
            {
                "type": "env_rollout",
                "env_ids": [f"s{i}" for i in range(16)],
                "steps_per_batch": 128,
            },
            lr=1e-3,
        )
        antes = success_rate(learner.policy)
        assert antes < 0.15  # la política inicial casi nunca llega
        learner.train(250)
        despues = success_rate(learner.policy)
        assert despues > antes + 0.3 and despues > 0.6


def test_e2e_rl_en_world_model(tmp_path) -> None:
    with core_zmq() as address:
        learner = _learner(
            address,
            tmp_path,
            _policy(init_log_std=-0.5),
            {
                "type": "grpo",
                "clip_ratio": 0.2,
                "update_epochs": 10,
                "minibatch_size": 256,
                "advantage": {"type": "group_relative"},
            },
            {
                "type": "world_model",
                "env_ids": ["s0"],
                "start_selector": "reset",
                "sequences_per_start": 8,
                "horizon": 1,
                "num_starts": 8,
            },
            lr=2e-3,
        )
        antes = success_rate(learner.policy)  # evalúa en el ambiente real
        learner.train(350)  # entrena solo con el WorldModel imaginado
        despues = success_rate(learner.policy)
        assert despues > antes + 0.3 and despues > 0.6
