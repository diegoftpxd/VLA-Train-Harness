"""Suite HEAD (específica de tokens) y POL montada en ``ComposedPolicy``."""

import numpy as np
import pytest
import torch

from vla_learner.policy import (
    ActionNormalizer,
    ActionStats,
    ComposedPolicy,
    SimpleProcessor,
    TokenActionHead,
    ValueHead,
)
from vla_learner.testing.contracts.head_contract import HeadContract
from vla_learner.testing.contracts.policy_contract import VLAPolicyContract
from vla_learner.testing.data import make_observations
from vla_learner.testing.tiny_backbone import TinyBackbone

pytestmark = pytest.mark.fast

FEATURE_DIM, ACTION_DIM, CHUNK, BINS = 16, 2, 2, 51


def _wide_normalizer() -> ActionNormalizer:
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
        action_head=TokenActionHead(FEATURE_DIM, ACTION_DIM, CHUNK, bins=BINS),
        processor=SimpleProcessor(["front"]),
        normalizer=_wide_normalizer(),
        value_head=ValueHead(FEATURE_DIM, 8),
    )
    policy.act(
        policy.processor(make_observations(4, cameras=["front"], seed=0)), deterministic=True
    )
    return policy


class TestTokenPolicy(VLAPolicyContract):
    @pytest.fixture
    def policy(self) -> ComposedPolicy:
        return _build_policy()


class TestTokenHead(HeadContract):
    @pytest.fixture
    def head(self) -> TokenActionHead:
        return TokenActionHead(FEATURE_DIM, ACTION_DIM, CHUNK, bins=BINS)


def test_head02_token_coincide_con_logsoftmax():
    torch.manual_seed(0)
    head = TokenActionHead(FEATURE_DIM, ACTION_DIM, CHUNK, bins=BINS)
    features = torch.randn(5, 1, FEATURE_DIM)
    actions = torch.rand(5, CHUNK, ACTION_DIM) * 1.8 - 0.9
    tokens = head.tokenize(actions)
    log_probs = torch.log_softmax(head._logits(features), dim=-1)
    referencia = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).sum(dim=(1, 2))
    assert torch.allclose(head.log_prob(features, actions, None), referencia, atol=1e-5)


def test_head04_token_discretizacion_estable():
    head = TokenActionHead(FEATURE_DIM, ACTION_DIM, CHUNK, bins=BINS)
    actions = torch.rand(5, CHUNK, ACTION_DIM) * 1.8 - 0.9
    tokens = head.tokenize(actions)
    reconstruidas = head.detokenize(tokens)
    assert torch.equal(head.tokenize(reconstruidas), tokens)
