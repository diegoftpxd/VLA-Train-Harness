"""Suite ALG, GRPO-01..03 y ALG-04 sobre ``GRPO``."""

import pytest
import torch

from vla_learner.algorithms.advantages import GroupRelativeAdvantage
from vla_learner.algorithms.base import Algorithm
from vla_learner.algorithms.grpo import GRPO, GRPOConfig
from vla_learner.algorithms.reference import ReferencePolicy
from vla_learner.testing.contracts.algorithm_contract import AlgorithmContract
from vla_learner.testing.stub_policy import StubPolicy
from vla_learner.types import ModelInput, RolloutBatch

pytestmark = pytest.mark.fast


def _rollout(policy: StubPolicy, n: int = 6, groups: bool = True) -> RolloutBatch:
    inputs = ModelInput(tensors={"proprio": torch.randn(n, 4)}, batch_size=n)
    actions = torch.randn(n, policy.chunk_size, policy.action_dim)
    with torch.no_grad():
        ev = policy.evaluate_actions(inputs, actions)
    return RolloutBatch(
        inputs=inputs,
        actions=actions,
        old_log_probs=ev.log_probs.detach(),
        rewards=torch.randn(n),
        terminated=torch.tensor([False, False, True, False, False, True][:n]),
        truncated=torch.zeros(n, dtype=torch.bool),
        episode_ids=["e0", "e0", "e0", "e1", "e1", "e1"][:n],
        step_ids=torch.tensor([0, 1, 2, 0, 1, 2][:n]),
        source="env",
        old_values=ev.values.detach() if ev.values is not None else None,
        group_ids=torch.tensor([0, 0, 0, 1, 1, 1][:n]) if groups else None,
        extras={},
    )


class TestGRPO(AlgorithmContract):
    @pytest.fixture
    def algorithm(self) -> Algorithm:
        return GRPO(GRPOConfig(), GroupRelativeAdvantage())

    @pytest.fixture
    def policy(self) -> StubPolicy:
        return StubPolicy(action_dim=2, chunk_size=1)

    @pytest.fixture
    def batch(self, policy: StubPolicy) -> RolloutBatch:
        return _rollout(policy, groups=True)


def test_grpo01_exige_grupos():
    policy = StubPolicy(action_dim=2, chunk_size=1)
    grpo = GRPO(GRPOConfig(), GroupRelativeAdvantage())
    with pytest.raises(ValueError):
        grpo.prepare(_rollout(policy, groups=False), policy)


def test_grpo02_referencia_coherente_con_kl_coef():
    with pytest.raises(ValueError):
        GRPO(GRPOConfig(kl_coef=0.5), GroupRelativeAdvantage(), reference=None)


def test_grpo03_kl_nulo_en_el_origen():
    policy = StubPolicy(action_dim=2, chunk_size=1)
    grpo = GRPO(GRPOConfig(kl_coef=1.0), GroupRelativeAdvantage(), ReferencePolicy(policy))
    batch = grpo.prepare(_rollout(policy, groups=True), policy)
    salida = grpo.loss(policy, batch)
    assert abs(salida.metrics["kl"]) < 1e-6  # política == referencia → KL 0


def test_alg04_direccion_grpo():
    torch.manual_seed(0)
    policy = StubPolicy(action_dim=2, chunk_size=1)
    inputs = ModelInput(tensors={"proprio": torch.zeros(2, 4)}, batch_size=2)
    actions = torch.tensor([[[1.0, 1.0]], [[-1.0, -1.0]]])
    with torch.no_grad():
        ev = policy.evaluate_actions(inputs, actions)
    batch = RolloutBatch(
        inputs=inputs,
        actions=actions,
        old_log_probs=ev.log_probs.detach(),
        rewards=torch.zeros(2),
        terminated=torch.tensor([True, True]),
        truncated=torch.tensor([False, False]),
        episode_ids=["a", "b"],
        step_ids=torch.tensor([0, 0]),
        source="env",
        old_values=None,
        group_ids=torch.tensor([0, 0]),
        extras={},
        advantages=torch.tensor([1.0, -1.0]),
        returns=torch.zeros(2),
    )
    grpo = GRPO(GRPOConfig(), GroupRelativeAdvantage())
    antes = policy.evaluate_actions(inputs, actions).log_probs.detach().clone()
    policy.zero_grad(set_to_none=True)
    grpo.loss(policy, batch).loss.backward()
    with torch.no_grad():
        for parametro in policy.parameters():
            if parametro.grad is not None:
                parametro -= 0.5 * parametro.grad
    despues = policy.evaluate_actions(inputs, actions).log_probs.detach()
    assert despues[0] > antes[0]
    assert despues[1] < antes[1]
