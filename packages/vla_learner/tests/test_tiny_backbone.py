"""Tests de ``TinyBackbone``."""

import pytest
import torch

from vla_learner.testing.tiny_backbone import TinyBackbone
from vla_learner.types import ModelInput

pytestmark = pytest.mark.fast


def test_encode_forma_y_feature_dim():
    backbone = TinyBackbone(feature_dim=16, seq_len=1)
    inputs = ModelInput(
        tensors={"image": torch.randn(4, 3, 8, 8), "proprio": torch.randn(4, 7)},
        batch_size=4,
    )
    features = backbone.encode(inputs)
    assert features.shape == (4, 1, 16)
    assert backbone.feature_dim == 16


def test_encode_diferenciable():
    backbone = TinyBackbone(feature_dim=8)
    inputs = ModelInput(tensors={"x": torch.randn(3, 5)}, batch_size=3)
    backbone.encode(inputs).sum().backward()
    assert any(p.grad is not None for p in backbone.parameters())


def test_seq_len():
    backbone = TinyBackbone(feature_dim=4, seq_len=3)
    inputs = ModelInput(tensors={"x": torch.randn(2, 6)}, batch_size=2)
    assert backbone.encode(inputs).shape == (2, 3, 4)
