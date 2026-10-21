"""Suite HEAD (específica de Gaussiana) y POL montada en ``ComposedPolicy``."""

import numpy as np
import pytest
import torch

from vla_learner.policy import (
    ActionNormalizer,
    ActionStats,
    ComposedPolicy,
    GaussianActionHead,
    SimpleProcessor,
    ValueHead,
)
from vla_learner.testing.contracts.head_contract import HeadContract
from vla_learner.testing.contracts.policy_contract import VLAPolicyContract
from vla_learner.testing.data import make_observations
from vla_learner.testing.tiny_backbone import TinyBackbone

pytestmark = pytest.mark.fast

FEATURE_DIM, ACTION_DIM, CHUNK = 16, 2, 2


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
        action_head=GaussianActionHead(FEATURE_DIM, ACTION_DIM, CHUNK),
        processor=SimpleProcessor(["front"]),
        normalizer=_wide_normalizer(),
        value_head=ValueHead(FEATURE_DIM, 8),
    )
    # materializa las capas perezosas antes de usarla en los contratos
    policy.act(
        policy.processor(make_observations(4, cameras=["front"], seed=0)), deterministic=True
    )
    return policy


class TestGaussianPolicy(VLAPolicyContract):
    @pytest.fixture
    def policy(self) -> ComposedPolicy:
        return _build_policy()


class TestGaussianHead(HeadContract):
    @pytest.fixture
    def head(self) -> GaussianActionHead:
        return GaussianActionHead(FEATURE_DIM, ACTION_DIM, CHUNK)


def test_head02_gaussian_coincide_con_normal():
    torch.manual_seed(0)
    head = GaussianActionHead(FEATURE_DIM, ACTION_DIM, CHUNK)
    features = torch.randn(5, 1, FEATURE_DIM)
    actions = torch.randn(5, CHUNK, ACTION_DIM) * 0.3
    mean = head.mean_proj(features.mean(dim=1)).reshape(5, CHUNK, ACTION_DIM)
    referencia = (
        torch.distributions.Normal(mean, head.log_std.exp()).log_prob(actions).sum(dim=(1, 2))
    )
    assert torch.allclose(head.log_prob(features, actions, None), referencia, atol=1e-5)
