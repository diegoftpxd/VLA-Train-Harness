"""Suite E2E-ENV: el Learner real entrena por Core contra procesos ``vla-env``.

Es la misma tesis que ``test_e2e.py`` —el entrenamiento avanza—, pero con el
Environment fuera de Core: cada ambiente es un proceso ``vla-env`` (``reach``)
y Core lo consume por la red, armado desde el bloque ``environments``. El
Learner no cambia. ``ReachEnvironment`` reproduce la dinámica del fake
(REACH-02), así que valen los **mismos umbrales** que la E2E actual y la misma
evaluación (``success_rate``).

Marcada ``slow``: levanta 16 procesos y entrena en CPU.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest
from conftest import lanzar_vla_env

from vla_common.types import Action, ActionChunk
from vla_core.builder import build_core
from vla_core.server import CoreServer
from vla_learner.data.core_client import CoreClient
from vla_learner.learner import Learner
from vla_learner.testing.evaluation import success_rate

pytestmark = pytest.mark.slow

N_AMBIENTES = 16
ENV_IDS = [f"reach-{i}" for i in range(N_AMBIENTES)]


@contextmanager
def core_con_procesos(n: int = N_AMBIENTES) -> Iterator[CoreServer]:
    """Core real (resto de componentes fake) conectado a ``n`` procesos ``vla-env``."""
    with lanzar_vla_env(n) as procesos:
        config = {
            "server": {"address": "tcp://127.0.0.1:0"},
            "backend": {"type": "fake"},
            "environments": [
                {"id": env_id, "address": address}
                for env_id, (address, _) in zip(ENV_IDS, procesos, strict=False)
            ],
            "timeouts": {"environment_s": 5, "retries": 1},
        }
        server = build_core(config)
        server.start()
        try:
            yield server
        finally:
            server.shutdown()


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


def _learner(address: str, tmp_path, policy: dict, algorithm: dict, data: dict, lr: float):
    cfg = {
        "seed": 0,
        "core": {"type": "zmq", "address": address, "timeout_s": 30.0},
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


def test_e2e_env01_rl_en_ambiente(tmp_path) -> None:
    with core_con_procesos() as server:
        learner = _learner(
            server.address,
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
            {"type": "env_rollout", "env_ids": ENV_IDS, "steps_per_batch": 128},
            lr=1e-3,
        )
        antes = success_rate(learner.policy)
        assert antes < 0.15
        learner.train(250)
        despues = success_rate(learner.policy)
        assert despues > antes + 0.3 and despues > 0.6


def test_e2e_env02_rl_en_world_model_con_inicios_reales(tmp_path) -> None:
    # Los inicios salen de los procesos vla-env (reset por Core) y el rollout del
    # FakeWorldModel. Se usan los inicios fijos de `env_ids`, no
    # `start_selector: reset`: ese selector inventa env_ids nuevos y depende del
    # autoregistro, que un Core con ambientes declarados no hace.
    with core_con_procesos() as server:
        learner = _learner(
            server.address,
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
                "env_ids": ENV_IDS,
                "sequences_per_start": 8,
                "horizon": 1,
                "num_starts": 8,
            },
            lr=2e-3,
        )
        antes = success_rate(learner.policy)
        learner.train(350)
        despues = success_rate(learner.policy)
        assert despues > antes + 0.3 and despues > 0.6


def _a_la_meta(proprio: np.ndarray) -> ActionChunk:
    return [Action(values=(proprio[2:4] - proprio[:2]).astype(np.float32))]


def test_e2e_env03_episodios_reales_y_reinyeccion() -> None:
    with core_con_procesos(2) as server:
        cliente = CoreClient(server.address, timeout_s=10.0)
        try:
            ((exitoso, obs),) = cliente.reset(["reach-0"], seed=1)
            (resultado,) = cliente.step(["reach-0"], [_a_la_meta(obs.proprio)], with_reward=True)
            assert resultado.terminated and resultado.info["success"]

            ((fallido, _),) = cliente.reset(["reach-1"], seed=2)
            quieto = [Action(values=np.zeros(2, dtype=np.float32))]
            while not (resultado := cliente.step(["reach-1"], [quieto], False)[0]).truncated:
                pass

            transiciones = cliente.get_episode(exitoso)
            assert [t.source for t in transiciones] == ["env"]
            assert transiciones[0].result.info["success"] is True

            core = server.backend
            assert core.ingest_episode(fallido) == 0  # type: ignore[attr-defined]
            assert core.ingest_episode(exitoso) == 1  # type: ignore[attr-defined]
            (obs_demo, accion_demo), *_ = cliente.sample_supervised(1)
            assert np.array_equal(obs_demo.proprio, obs.proprio)  # la demo reinyectada
            assert np.array_equal(accion_demo[0].values, _a_la_meta(obs.proprio)[0].values)
        finally:
            cliente.close()
