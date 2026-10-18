"""Tests de ``StubPolicy`` (y ``StubProcessor``)."""

import pytest
import torch

from vla_learner.policy.interfaces import Capability
from vla_learner.testing.stub_policy import StubPolicy
from vla_learner.types import ModelInput

pytestmark = pytest.mark.fast


def _inputs(batch_size: int) -> ModelInput:
    return ModelInput(tensors={"proprio": torch.randn(batch_size, 4)}, batch_size=batch_size)


def test_act_formas_y_capacidades():
    policy = StubPolicy(action_dim=3, chunk_size=2)
    out = policy.act(_inputs(5))
    assert out.actions.shape == (5, 2, 3)
    assert out.log_probs is not None and out.log_probs.shape == (5,)
    assert out.values is not None and out.values.shape == (5,)


def test_act_sin_value_ni_entropy():
    policy = StubPolicy(capabilities={Capability.LOG_PROB})
    out = policy.act(_inputs(4))
    assert out.values is None
    ev = policy.evaluate_actions(_inputs(4), out.actions)
    assert ev.values is None
    assert ev.entropy is None


def test_evaluate_sin_log_prob_falla():
    policy = StubPolicy(capabilities=set())
    actions = torch.zeros(2, policy.chunk_size, policy.action_dim)
    with pytest.raises(NotImplementedError):
        policy.evaluate_actions(_inputs(2), actions)


def test_consistencia_act_evaluate():
    policy = StubPolicy(action_dim=3, chunk_size=2)
    inputs = _inputs(5)
    out = policy.act(inputs)
    ev = policy.evaluate_actions(inputs, out.actions)
    assert out.log_probs is not None
    assert torch.allclose(out.log_probs, ev.log_probs, atol=1e-5)


def test_evaluate_actions_produce_gradientes():
    policy = StubPolicy()
    actions = torch.randn(4, policy.chunk_size, policy.action_dim)
    policy.evaluate_actions(_inputs(4), actions).log_probs.sum().backward()
    assert policy.mean.grad is not None
    assert torch.isfinite(policy.mean.grad).all()


def test_imitation_loss_escalar_y_gradiente():
    policy = StubPolicy()
    actions = torch.randn(4, policy.chunk_size, policy.action_dim)
    out = policy.imitation_loss(_inputs(4), actions)
    assert out.loss.ndim == 0
    assert isinstance(out.metrics["nll"], float)
    out.loss.backward()
    assert policy.mean.grad is not None


def test_act_determinista_reproducible():
    policy = StubPolicy()
    a = policy.act(_inputs(3), deterministic=True)
    b = policy.act(_inputs(3), deterministic=True)
    assert torch.equal(a.actions, b.actions)


def test_clone_frozen_independiente():
    policy = StubPolicy()
    clon = policy.clone_frozen()
    assert all(not p.requires_grad for p in clon.parameters())

    actions = torch.randn(4, policy.chunk_size, policy.action_dim)
    inputs = _inputs(4)
    antes = clon.evaluate_actions(inputs, actions).log_probs.detach().clone()
    with torch.no_grad():
        policy.mean.add_(1.0)
    despues = clon.evaluate_actions(inputs, actions).log_probs.detach()
    assert torch.equal(antes, despues)
