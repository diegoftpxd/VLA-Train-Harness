"""Suite ALG y BC-01 sobre ``BehaviorCloning``."""

import pytest
import torch

from vla_learner.algorithms.base import Algorithm
from vla_learner.algorithms.bc import BehaviorCloning
from vla_learner.testing.contracts.algorithm_contract import AlgorithmContract
from vla_learner.testing.stub_policy import StubPolicy
from vla_learner.types import ModelInput, SupervisedBatch

pytestmark = pytest.mark.fast


def _supervised() -> SupervisedBatch:
    return SupervisedBatch(
        inputs=ModelInput(tensors={"proprio": torch.randn(4, 4)}, batch_size=4),
        actions=torch.randn(4, 1, 2),
    )


class TestBehaviorCloning(AlgorithmContract):
    @pytest.fixture
    def algorithm(self) -> Algorithm:
        return BehaviorCloning()

    @pytest.fixture
    def policy(self) -> StubPolicy:
        return StubPolicy(action_dim=2, chunk_size=1)

    @pytest.fixture
    def batch(self) -> SupervisedBatch:
        return _supervised()


def test_bc01_delega_en_el_modelo():
    policy = StubPolicy(action_dim=2, chunk_size=1)
    batch = _supervised()
    esperado = policy.imitation_loss(batch.inputs, batch.actions).loss
    obtenido = BehaviorCloning().loss(policy, batch).loss
    assert torch.equal(obtenido, esperado)
