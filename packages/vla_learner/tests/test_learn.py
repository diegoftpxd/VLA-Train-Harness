"""Suite LEARN: el sistema completo *aprende* en ToyReachEnv (no solo no falla).

Son tests lentos (marca ``slow``). La evaluación es determinista: una política
aleatoria casi nunca llega al objetivo (tasa de éxito ~0), así que una mejora
clara demuestra aprendizaje. Hiperparámetros e iteraciones calibrados para esta
máquina; si algún algoritmo no aprende, se asume un bug (ver CLAUDE.md).
"""

import pytest

from vla_learner.algorithms.advantages import GAE
from vla_learner.algorithms.ppo import PPO, PPOConfig
from vla_learner.data.collector import RolloutCollector
from vla_learner.data.sources import EnvRolloutSource
from vla_learner.infra.checkpoint import CheckpointManager
from vla_learner.infra.distributed import SingleDevice
from vla_learner.infra.logging import StdoutLogger
from vla_learner.infra.optimizer import OptimizerConfig, OptimizerManager
from vla_learner.learner import Learner, LearnerConfig
from vla_learner.testing.evaluation import success_rate
from vla_learner.testing.fake_core import FakeCoreClient

pytestmark = pytest.mark.slow


def _policy(with_value: bool = False, init_log_std: float = -0.5) -> dict:
    cfg = {
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


def _cfg(tmp_path, policy, algorithm, data, lr) -> dict:
    return {
        "seed": 0,
        "core": {"type": "fake", "seed": 0, "demo_chunk_size": 1},
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


def _learner(cfg: dict) -> Learner:
    learner = Learner.from_config(cfg)
    learner.logger.is_main = False  # sin log por iteración en los tests
    return learner


def test_learn01_bc_aprende(tmp_path):
    learner = _learner(
        _cfg(
            tmp_path,
            _policy(init_log_std=-1.0),
            {"type": "bc"},
            {"type": "supervised", "batch_size": 64},
            lr=1e-3,
        )
    )
    learner.train(250)
    assert success_rate(learner.policy) > 0.9


def test_learn02_ppo_mejora(tmp_path):
    learner = _learner(
        _cfg(
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
    )
    assert success_rate(learner.policy) < 0.1
    learner.train(250)
    assert success_rate(learner.policy) > 0.8


def test_learn03_grpo_mejora(tmp_path):
    learner = _learner(
        _cfg(
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
                "type": "env_rollout",
                "env_ids": [f"s{i}" for i in range(64)],
                "steps_per_batch": 64,
                "group_size": 8,
            },
            lr=2e-3,
        )
    )
    assert success_rate(learner.policy) < 0.1
    learner.train(350)
    assert success_rate(learner.policy) > 0.8


def test_learn04_world_model_transfiere(tmp_path):
    learner = _learner(
        _cfg(
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
    )
    antes = success_rate(learner.policy)
    learner.train(350)  # entrena solo con ToyWorldModel
    despues = success_rate(learner.policy)  # evalúa en ToyReachEnv
    assert despues > antes
    assert despues > 0.8


def test_learn05_bc_mas_rl_no_empeora(tmp_path):
    bc = _learner(
        _cfg(
            tmp_path,
            _policy(with_value=True, init_log_std=-1.0),
            {"type": "bc"},
            {"type": "supervised", "batch_size": 64},
            lr=1e-3,
        )
    )
    bc.train(250)
    exito_bc = success_rate(bc.policy)
    assert exito_bc > 0.9

    # continúa con PPO sobre la MISMA política entrenada con BC
    policy = bc.policy
    client = FakeCoreClient()
    source = EnvRolloutSource(RolloutCollector(client), [f"s{i}" for i in range(16)], 128)
    optimizer = OptimizerManager(
        policy.trainable_parameters(),
        OptimizerConfig(name="adamw", lr=3e-4, grad_clip=1.0, scheduler=None, precision="fp32"),
    )
    rl = Learner(
        policy,
        PPO(PPOConfig(update_epochs=10, minibatch_size=64), GAE()),
        source,
        optimizer,
        CheckpointManager(tmp_path / "rl", keep_last=1),
        StdoutLogger(is_main=False),
        SingleDevice(),
        LearnerConfig(checkpoint_every=10_000),
    )
    rl.train(60)
    exito_rl = success_rate(policy)
    assert exito_rl >= exito_bc - 0.1  # el RL no empeora
    assert exito_rl > 0.8
