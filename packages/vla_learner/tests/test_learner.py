"""Suite LRN: el Learner completo (vía from_config) contra FakeCoreClient."""

import pytest
import torch

from vla_learner.learner import Learner
from vla_learner.policy.composed import ComposedPolicy
from vla_learner.policy.heads import GaussianActionHead

pytestmark = pytest.mark.fast


def _policy_cfg(with_value: bool = False, cameras: list[str] | None = None) -> dict:
    cfg = {
        "type": "composed",
        "action_dim": 2,
        "chunk_size": 1,
        "feature_dim": 16,
        "backbone": {"type": "tiny", "seq_len": 1},
        "action_head": {"type": "gaussian"},
        "processor": {"type": "simple", "cameras": cameras or ["front"]},
        "normalizer": {"mode": "mean_std"},
    }
    if with_value:
        cfg["value_head"] = {"hidden_dim": 16}
    return cfg


def _base(tmp_path, **over) -> dict:
    cfg = {
        "seed": 0,
        "core": {"type": "fake", "seed": 0, "demo_chunk_size": 1},
        "optimizer": {
            "name": "adamw",
            "lr": 0.001,
            "grad_clip": 1.0,
            "scheduler": None,
            "precision": "fp32",
        },
        "checkpoint": {"directory": str(tmp_path / "ckpt"), "every": 1, "keep_last": 10},
    }
    cfg.update(over)
    return cfg


def _mean_weight(learner: Learner) -> torch.Tensor:
    assert isinstance(learner.policy, ComposedPolicy)
    head = learner.policy.action_head
    assert isinstance(head, GaussianActionHead)
    return head.mean_proj.weight.detach().clone()


# --- LRN-01: incompatibilidades fallan al construir --------------------


def test_lrn01_ppo_sin_value_falla(tmp_path):
    cfg = _base(
        tmp_path,
        policy=_policy_cfg(with_value=False),
        algorithm={"type": "ppo"},
        data={"type": "env_rollout", "env_ids": ["sim-0", "sim-1"], "steps_per_batch": 4},
    )
    with pytest.raises(ValueError):
        Learner.from_config(cfg)


def test_lrn01_tipos_de_batch_distintos_falla(tmp_path):
    cfg = _base(
        tmp_path,
        policy=_policy_cfg(),
        algorithm={"type": "bc"},
        data={"type": "env_rollout", "env_ids": ["sim-0"], "steps_per_batch": 2},
    )
    with pytest.raises(ValueError):
        Learner.from_config(cfg)


def test_lrn01_camaras_faltantes_falla(tmp_path):
    cfg = _base(
        tmp_path,
        policy=_policy_cfg(cameras=["front", "wrist"]),
        algorithm={"type": "bc"},
        data={"type": "supervised", "batch_size": 4},
        available_cameras=["front"],
    )
    with pytest.raises(ValueError):
        Learner.from_config(cfg)


# --- LRN-02: una iteración de cada modo --------------------------------


@pytest.mark.parametrize(
    "algorithm, data, with_value",
    [
        ({"type": "bc"}, {"type": "supervised", "batch_size": 8}, False),
        (
            {"type": "ppo", "update_epochs": 2, "minibatch_size": 32},
            {"type": "env_rollout", "env_ids": ["sim-0", "sim-1"], "steps_per_batch": 8},
            True,
        ),
        (
            {"type": "grpo", "update_epochs": 1, "minibatch_size": 32},
            {
                "type": "world_model",
                "env_ids": ["sim-0"],
                "sequences_per_start": 4,
                "horizon": 2,
                "num_starts": 1,
            },
            False,
        ),
    ],
)
def test_lrn02_una_iteracion(tmp_path, capsys, algorithm, data, with_value):
    cfg = _base(tmp_path, policy=_policy_cfg(with_value=with_value), algorithm=algorithm, data=data)
    learner = Learner.from_config(cfg)
    antes = _mean_weight(learner)
    learner.train(1)
    assert not torch.equal(antes, _mean_weight(learner))  # cambian los pesos
    assert capsys.readouterr().out.strip() != ""  # publica métricas


# --- LRN-03: cantidad de pasos del optimizador -------------------------


def test_lrn03_cantidad_de_pasos(tmp_path):
    cfg = _base(
        tmp_path,
        policy=_policy_cfg(with_value=True),
        algorithm={"type": "ppo", "update_epochs": 4, "minibatch_size": 8},
        data={"type": "env_rollout", "env_ids": ["sim-0", "sim-1"], "steps_per_batch": 16},
    )
    learner = Learner.from_config(cfg)
    llamadas = []
    original = learner.optimizer.step

    def contando(loss):
        llamadas.append(1)
        return original(loss)

    learner.optimizer.step = contando  # type: ignore[method-assign]
    learner.train(1)
    # update_epochs(4) × ceil(N=16 / minibatch=8)=2 → 8
    assert len(llamadas) == 8


# --- LRN-04: checkpoints periódicos ------------------------------------


def test_lrn04_checkpoints_periodicos(tmp_path):
    cfg = _base(
        tmp_path,
        policy=_policy_cfg(),
        algorithm={"type": "bc"},
        data={"type": "supervised", "batch_size": 4},
    )
    cfg["checkpoint"]["every"] = 2
    learner = Learner.from_config(cfg)
    learner.train(5)  # iteraciones 0..4 → guarda en 0, 2, 4
    for paso in (0, 2, 4):
        assert (tmp_path / "ckpt" / f"step_{paso:06d}").is_dir()
    assert not (tmp_path / "ckpt" / "step_000001").exists()


# --- LRN-05: resume continúa donde quedó -------------------------------


def test_lrn05_resume(tmp_path):
    cfg = _base(
        tmp_path,
        policy=_policy_cfg(),
        algorithm={"type": "bc"},
        data={"type": "supervised", "batch_size": 4},
    )
    cfg["checkpoint"]["every"] = 2
    primero = Learner.from_config(cfg)
    primero.train(3)  # guarda en 0 y 2

    segundo = Learner.from_config(cfg)
    segundo.resume()
    assert segundo.start_iteration == 3  # la siguiente a la guardada (2)
