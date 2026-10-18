"""Suite ALG, PPO-01..04 y ALG-04 sobre ``PPO``."""

import pytest
import torch

from vla_learner.algorithms.advantages import GAE
from vla_learner.algorithms.base import Algorithm
from vla_learner.algorithms.ppo import PPO, PPOConfig, clipped_surrogate, value_loss
from vla_learner.testing.contracts.algorithm_contract import AlgorithmContract
from vla_learner.testing.stub_policy import StubPolicy
from vla_learner.types import ModelInput, RolloutBatch

pytestmark = pytest.mark.fast


def _rollout(policy: StubPolicy, n: int = 6) -> RolloutBatch:
    inputs = ModelInput(tensors={"proprio": torch.randn(n, 4)}, batch_size=n)
    actions = torch.randn(n, policy.chunk_size, policy.action_dim)
    with torch.no_grad():
        ev = policy.evaluate_actions(inputs, actions)
    assert ev.values is not None
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
        old_values=ev.values.detach(),
        extras={},
    )


class TestPPO(AlgorithmContract):
    @pytest.fixture
    def algorithm(self) -> Algorithm:
        return PPO(PPOConfig(), GAE())

    @pytest.fixture
    def policy(self) -> StubPolicy:
        return StubPolicy(action_dim=2, chunk_size=1)

    @pytest.fixture
    def batch(self, policy: StubPolicy) -> RolloutBatch:
        return _rollout(policy)


def test_ppo01_sin_cambio_de_pesos_no_hay_recorte():
    policy = StubPolicy(action_dim=2, chunk_size=1)
    ppo = PPO(PPOConfig(), GAE())
    batch = ppo.prepare(_rollout(policy), policy)
    salida = ppo.loss(policy, batch)
    assert salida.metrics["clip_fraction"] == 0.0
    assert abs(salida.metrics["approx_kl"]) < 1e-6


def test_ppo02_el_recorte_corta_el_gradiente():
    log_probs = torch.tensor([0.0, 0.0], requires_grad=True)
    old_log_probs = torch.tensor([-1.0, -1.0])  # ratio ≈ 2.72 > 1 + 0.2
    advantages = torch.tensor([1.0, -1.0])  # fila 0 recortada (A>0); fila 1 no
    perdida, _, clip_fraction = clipped_surrogate(log_probs, old_log_probs, advantages, 0.2)
    perdida.backward()
    assert clip_fraction > 0.0
    grad = log_probs.grad
    assert grad is not None
    assert grad[0].abs() < 1e-8  # fila recortada no aporta gradiente
    assert grad[1].abs() > 1e-8  # fila no recortada sí


def test_ppo03_value_loss_con_y_sin_clip():
    values = torch.tensor([1.0, 2.0])
    returns = torch.tensor([1.5, 1.0])
    old_values = torch.tensor([0.9, 2.1])

    esperado = 0.5 * ((values - returns) ** 2).mean()
    assert torch.allclose(value_loss(values, returns, old_values, None), esperado)

    clip = 0.2
    recortado = old_values + (values - old_values).clamp(-clip, clip)
    esperado_clip = 0.5 * torch.max((values - returns) ** 2, (recortado - returns) ** 2).mean()
    assert torch.allclose(value_loss(values, returns, old_values, clip), esperado_clip)


def test_ppo04_target_kl_corta_las_epocas():
    assert PPO(PPOConfig(target_kl=0.05), GAE()).should_stop_epochs({"approx_kl": 0.06}) is True
    assert PPO(PPOConfig(target_kl=0.05), GAE()).should_stop_epochs({"approx_kl": 0.04}) is False
    assert PPO(PPOConfig(target_kl=None), GAE()).should_stop_epochs({"approx_kl": 9.9}) is False


def test_alg04_direccion_ppo():
    torch.manual_seed(0)
    policy = StubPolicy(action_dim=2, chunk_size=1)
    inputs = ModelInput(tensors={"proprio": torch.zeros(2, 4)}, batch_size=2)
    actions = torch.tensor([[[1.0, 1.0]], [[-1.0, -1.0]]])
    with torch.no_grad():
        ev = policy.evaluate_actions(inputs, actions)
    assert ev.values is not None
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
        old_values=ev.values.detach(),
        extras={},
        advantages=torch.tensor([1.0, -1.0]),
        returns=torch.zeros(2),
    )
    # value_coef=0 aísla el término de política
    ppo = PPO(PPOConfig(value_coef=0.0, entropy_coef=0.0), GAE())
    antes = policy.evaluate_actions(inputs, actions).log_probs.detach().clone()
    policy.zero_grad(set_to_none=True)
    ppo.loss(policy, batch).loss.backward()
    with torch.no_grad():
        for parametro in policy.parameters():
            if parametro.grad is not None:
                parametro -= 0.5 * parametro.grad
    despues = policy.evaluate_actions(inputs, actions).log_probs.detach()
    assert despues[0] > antes[0]  # ventaja positiva → sube la probabilidad
    assert despues[1] < antes[1]  # ventaja negativa → baja
