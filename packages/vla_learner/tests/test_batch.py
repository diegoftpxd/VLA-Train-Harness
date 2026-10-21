"""Suite BAT sobre ``SupervisedBatch`` y ``RolloutBatch``."""

import pytest
import torch

from vla_learner.testing.contracts.batch_contract import BatchContract
from vla_learner.types import ModelInput, RolloutBatch, SupervisedBatch

pytestmark = pytest.mark.fast


def _make_supervised(n: int = 6, h: int = 2, d: int = 3) -> SupervisedBatch:
    inputs = ModelInput(
        tensors={"img": torch.randn(n, 4), "proprio": torch.randn(n, 2)},
        batch_size=n,
    )
    return SupervisedBatch(inputs=inputs, actions=torch.randn(n, h, d))


def _make_rollout(n: int = 6, h: int = 2, d: int = 3) -> RolloutBatch:
    inputs = ModelInput(tensors={"img": torch.randn(n, 4)}, batch_size=n)
    return RolloutBatch(
        inputs=inputs,
        actions=torch.randn(n, h, d),
        old_log_probs=torch.randn(n),
        rewards=torch.randn(n),
        terminated=torch.zeros(n, dtype=torch.bool),
        truncated=torch.zeros(n, dtype=torch.bool),
        episode_ids=[f"ep{i}" for i in range(n)],
        step_ids=torch.arange(n),
        source="env",
        old_values=torch.randn(n),
        group_ids=torch.arange(n),
        extras={"token_ids": torch.randint(0, 10, (n, h))},
    )


class TestSupervisedBatch(BatchContract):
    @pytest.fixture
    def batch(self) -> SupervisedBatch:
        return _make_supervised()


class TestRolloutBatch(BatchContract):
    @pytest.fixture
    def batch(self) -> RolloutBatch:
        return _make_rollout()
