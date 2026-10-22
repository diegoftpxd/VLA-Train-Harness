"""Suite HEAD/POL para ``FlowMatchingActionHead`` montada en ``ComposedPolicy``.

Es la integración de π0 "siguiendo las interfaces ya existentes": un VLA de flow
matching entra como backbone + cabeza de flow matching, y pasa el contrato POL
igual que cualquier otra política (la rama sin ``LOG_PROB`` de ``VLAPolicyContract``).
Todo en CPU, sin pesos reales.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vla_learner.policy import (
    ActionNormalizer,
    ActionStats,
    Capability,
    ComposedPolicy,
    FlowMatchingActionHead,
    SimpleProcessor,
)
from vla_learner.testing.contracts.head_contract import HeadContract
from vla_learner.testing.contracts.policy_contract import VLAPolicyContract
from vla_learner.testing.data import make_observations
from vla_learner.testing.tiny_backbone import TinyBackbone

pytestmark = pytest.mark.fast

FEATURE_DIM, ACTION_DIM, CHUNK = 16, 2, 4


def _wide_normalizer() -> ActionNormalizer:
    # escalas muy distintas por dimensión → acciones fuera de [-1, 1] (POL-07)
    stats = {
        "default": ActionStats(
            mean=np.zeros(ACTION_DIM, np.float32),
            std=np.array([0.05, 50.0], np.float32),
            min=np.array([-0.1, -100.0], np.float32),
            max=np.array([0.1, 100.0], np.float32),
            q01=np.array([-0.09, -90.0], np.float32),
            q99=np.array([0.09, 90.0], np.float32),
        )
    }
    return ActionNormalizer(stats, mode="mean_std")


def _build_policy() -> ComposedPolicy:
    policy = ComposedPolicy(
        backbone=TinyBackbone(FEATURE_DIM, seq_len=1),
        action_head=FlowMatchingActionHead(FEATURE_DIM, ACTION_DIM, CHUNK, hidden_dim=64),
        processor=SimpleProcessor(["front"]),
        normalizer=_wide_normalizer(),
    )
    # materializa las capas perezosas antes de usarla en los contratos
    policy.act(
        policy.processor(make_observations(4, cameras=["front"], seed=0)), deterministic=True
    )
    return policy


class TestFlowMatchingPolicy(VLAPolicyContract):
    @pytest.fixture
    def policy(self) -> ComposedPolicy:
        return _build_policy()


class TestFlowMatchingHead(HeadContract):
    FEATURE_DIM = FEATURE_DIM

    @pytest.fixture
    def head(self) -> FlowMatchingActionHead:
        torch.manual_seed(0)
        return FlowMatchingActionHead(FEATURE_DIM, ACTION_DIM, CHUNK, hidden_dim=64)


def test_capacidades_vacias() -> None:
    head = FlowMatchingActionHead(FEATURE_DIM, ACTION_DIM, CHUNK)
    assert head.capabilities == set()


def test_log_prob_y_entropy_no_implementados() -> None:
    head = FlowMatchingActionHead(FEATURE_DIM, ACTION_DIM, CHUNK)
    features = torch.randn(2, 1, FEATURE_DIM)
    actions = torch.zeros(2, CHUNK, ACTION_DIM)
    with pytest.raises(NotImplementedError):
        head.log_prob(features, actions, None)
    with pytest.raises(NotImplementedError):
        head.entropy(features)


def test_determinista_reproducible_y_estocastico_varia() -> None:
    torch.manual_seed(0)
    head = FlowMatchingActionHead(FEATURE_DIM, ACTION_DIM, CHUNK, hidden_dim=64)
    features = torch.randn(3, 1, FEATURE_DIM)
    a, _, _ = head.sample(features, deterministic=True)
    b, _, _ = head.sample(features, deterministic=True)
    assert torch.equal(a, b)  # el arranque desde el origen es determinista
    torch.manual_seed(1)
    c, _, _ = head.sample(features, deterministic=False)
    torch.manual_seed(2)
    d, _, _ = head.sample(features, deterministic=False)
    assert not torch.allclose(c, d)  # el ruido inicial cambia la muestra


def test_muestreo_recupera_el_objetivo() -> None:
    # Entrenar el campo de velocidad sobre un objetivo fijo debe hacer que el
    # muestreo (integración del ODE) converja hacia ese objetivo.
    torch.manual_seed(0)
    head = FlowMatchingActionHead(FEATURE_DIM, ACTION_DIM, CHUNK, hidden_dim=64)
    features = torch.randn(8, 1, FEATURE_DIM)
    objetivo = torch.rand(8, CHUNK, ACTION_DIM) * 1.6 - 0.8
    opt = torch.optim.Adam(head.parameters(), lr=0.02)
    for _ in range(400):
        opt.zero_grad()
        head.bc_loss(features, objetivo).loss.backward()
        opt.step()
    with torch.no_grad():
        muestra, _, _ = head.sample(features, deterministic=False)
    error = (muestra - objetivo).abs().mean().item()
    assert error < 0.2, f"el muestreo no recuperó el objetivo (error={error:.3f})"


def test_registrado_en_builder() -> None:
    import vla_learner.builder  # noqa: F401
    from vla_learner.registry import Registry

    assert "flow_matching" in Registry.available("action_head")
    head = Registry.build(
        "action_head",
        {"type": "flow_matching", "feature_dim": FEATURE_DIM, "action_dim": 2, "chunk_size": 4},
    )
    assert isinstance(head, FlowMatchingActionHead)
    assert Capability.LOG_PROB not in head.capabilities
