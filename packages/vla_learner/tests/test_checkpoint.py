"""Suite CKPT sobre ``CheckpointManager``."""

import pytest
import torch

from vla_learner.infra.checkpoint import CheckpointManager
from vla_learner.infra.optimizer import OptimizerConfig, OptimizerManager
from vla_learner.testing.stub_policy import StubPolicy
from vla_learner.types import ModelInput

pytestmark = pytest.mark.fast


def _manager(policy: StubPolicy) -> OptimizerManager:
    return OptimizerManager(
        policy.parameters(),
        OptimizerConfig(name="adamw", lr=0.1, grad_clip=None, scheduler=None, precision="fp32"),
    )


def _entrenar(policy: StubPolicy, optimizer: OptimizerManager, pasos: int, seed: int | None):
    if seed is not None:
        torch.manual_seed(seed)
    for _ in range(pasos):
        actions = torch.randn(4, 1, 2)
        inputs = ModelInput(tensors={"proprio": torch.randn(4, 4)}, batch_size=4)
        optimizer.step(policy.imitation_loss(inputs, actions).loss)


def test_ckpt01_estructura(tmp_path):
    policy = StubPolicy(action_dim=2, chunk_size=1)
    manager = CheckpointManager(tmp_path, keep_last=2)
    primero = manager.save(1, policy, _manager(policy), None, {})
    assert (primero / "policy").is_dir()
    assert (primero / "training_state.pt").is_file()
    manager.save(2, policy, _manager(policy), None, {})
    manager.save(3, policy, _manager(policy), None, {})
    assert not (tmp_path / "step_000001").exists()  # keep_last=2 borra el más viejo
    assert manager.latest() == tmp_path / "step_000003"


def test_ckpt02_retomar_es_exacto(tmp_path):
    seguido = StubPolicy(action_dim=2, chunk_size=1)
    opt_seguido = _manager(seguido)
    _entrenar(seguido, opt_seguido, 6, seed=123)

    parcial = StubPolicy(action_dim=2, chunk_size=1)
    opt_parcial = _manager(parcial)
    _entrenar(parcial, opt_parcial, 3, seed=123)
    manager = CheckpointManager(tmp_path)
    manager.save(3, parcial, opt_parcial, None, {})

    retomado = StubPolicy(action_dim=2, chunk_size=1)
    opt_retomado = _manager(retomado)
    paso = manager.load_training_state(tmp_path / "step_000003", retomado, opt_retomado)
    assert paso == 3
    _entrenar(retomado, opt_retomado, 3, seed=None)  # continúa el RNG restaurado

    assert torch.allclose(seguido.mean.detach(), retomado.mean.detach(), atol=1e-6)
    assert torch.allclose(seguido.log_std.detach(), retomado.log_std.detach(), atol=1e-6)
