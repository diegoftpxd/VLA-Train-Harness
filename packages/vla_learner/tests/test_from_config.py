"""Smoke de ``from_config`` sobre los configs tiny de ejemplo (el camino de la CLI)."""

from pathlib import Path

import pytest
import torch
import yaml

from vla_learner.learner import Learner
from vla_learner.policy.composed import ComposedPolicy
from vla_learner.policy.heads import GaussianActionHead

pytestmark = pytest.mark.fast

_CONFIGS = Path(__file__).resolve().parents[3] / "configs"


@pytest.mark.parametrize("nombre", ["bc_tiny.yaml", "ppo_tiny.yaml"])
def test_from_config_ejemplos(tmp_path, nombre):
    config = yaml.safe_load((_CONFIGS / nombre).read_text(encoding="utf-8"))
    config["checkpoint"]["directory"] = str(tmp_path / "ckpt")
    learner = Learner.from_config(config)
    assert isinstance(learner.policy, ComposedPolicy)
    head = learner.policy.action_head
    assert isinstance(head, GaussianActionHead)
    antes = head.mean_proj.weight.detach().clone()
    learner.train(1)
    assert not torch.equal(antes, head.mean_proj.weight.detach())


def test_from_config_flow_matching(tmp_path):
    # Flow matching (mecánica de π0) de punta a punta: se arma desde el YAML y
    # entrena por BC. El π0 pesado real (policy type `pi0`) va en el cluster.
    from vla_learner.policy.heads import FlowMatchingActionHead

    config = yaml.safe_load((_CONFIGS / "flow_matching_bc.yaml").read_text(encoding="utf-8"))
    config["checkpoint"]["directory"] = str(tmp_path / "ckpt")
    learner = Learner.from_config(config)
    assert isinstance(learner.policy, ComposedPolicy)
    head = learner.policy.action_head
    assert isinstance(head, FlowMatchingActionHead)
    assert head.capabilities == set()  # sin log-prob: solo BC
    antes = [p.detach().clone() for p in head.parameters()]
    learner.train(1)
    assert any(not torch.equal(a, p) for a, p in zip(antes, head.parameters(), strict=True))


def test_normalizer_stats_fit_ajusta_con_muestras_de_core():
    # ``normalizer.stats: fit`` (Learner Design.md): el builder pide una muestra a
    # Core y arma los normalizadores de acción y de propiocepción.
    import numpy as np

    from vla_learner.builder import _resolve_normalizer
    from vla_learner.testing.fake_core import FakeCoreClient

    resuelto = _resolve_normalizer(
        {"mode": "mean_std", "stats": "fit", "fit_samples": 32},
        FakeCoreClient(seed=0, demo_chunk_size=3),
    )
    assert resuelto is not None and "stats" not in resuelto
    muestras = FakeCoreClient(seed=0, demo_chunk_size=3).sample_supervised(32)
    acciones = np.concatenate([np.stack([a.values for a in ch]) for _, ch in muestras])
    proprios = np.stack([obs.proprio for obs, _ in muestras])
    assert resuelto["action"].dim() == acciones.shape[1]
    assert resuelto["state"].dim() == proprios.shape[1]
    centro = resuelto["action"].normalize(torch.as_tensor(acciones.mean(0)), "default")
    assert torch.allclose(centro, torch.zeros_like(centro), atol=1e-5)


def test_composed_usa_las_estadisticas_ajustadas(tmp_path):
    # Antes ``stats`` se ignoraba en silencio en ``composed`` (siempre unitario).
    config = yaml.safe_load((_CONFIGS / "bc_tiny.yaml").read_text(encoding="utf-8"))
    config["checkpoint"]["directory"] = str(tmp_path / "ckpt")
    config["policy"]["normalizer"] = {"mode": "mean_std", "stats": "fit", "fit_samples": 64}
    learner = Learner.from_config(config)
    policy = learner.policy
    assert isinstance(policy, ComposedPolicy)
    x = torch.ones(1, 1, 2)
    # Un normalizador unitario (mean 0, std 1) devolvería x tal cual.
    assert not torch.allclose(policy.normalizer.normalize(x, "default"), x)
    assert config["policy"]["normalizer"]["stats"] == "fit"  # el config no se muta
    learner.train(1)
