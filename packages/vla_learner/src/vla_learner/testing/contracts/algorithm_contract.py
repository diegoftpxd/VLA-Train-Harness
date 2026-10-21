"""Suite de contrato ALG (genérica) para ``Algorithm``.

Cubre ALG-01 (declaraciones), ALG-02 (loss válida) y ALG-03 (prepare no entrena).
ALG-04 (dirección del gradiente) es específica de los algoritmos de RL y vive en
sus módulos de test. La implementación entrega ``algorithm``, ``policy`` (un
``StubPolicy``) y ``batch`` del tipo que declare el algoritmo.
"""

from __future__ import annotations

import pytest
import torch

from vla_learner.algorithms.base import Algorithm
from vla_learner.policy.interfaces import Capability, VLAPolicy
from vla_learner.types import Batch, RolloutBatch, SupervisedBatch


class AlgorithmContract:
    @pytest.fixture
    def algorithm(self) -> Algorithm:
        raise NotImplementedError

    @pytest.fixture
    def policy(self) -> VLAPolicy:
        raise NotImplementedError

    @pytest.fixture
    def batch(self) -> Batch:
        raise NotImplementedError

    def test_alg01_declaraciones(self, algorithm):
        caps = algorithm.required_capabilities
        assert isinstance(caps, set)
        assert all(isinstance(c, Capability) for c in caps)
        assert algorithm.batch_type in (SupervisedBatch, RolloutBatch)

    def test_alg02_loss_valida(self, algorithm, policy, batch):
        preparado = algorithm.prepare(batch, policy)
        policy.zero_grad(set_to_none=True)
        salida = algorithm.loss(policy, preparado)
        assert salida.loss.ndim == 0
        assert torch.isfinite(salida.loss)
        assert all(isinstance(v, float) for v in salida.metrics.values())
        salida.loss.backward()
        assert any(p.grad is not None for p in policy.parameters())

    def test_alg03_prepare_no_entrena(self, algorithm, policy, batch):
        modo = policy.training
        snapshot = [p.detach().clone() for p in policy.parameters()]

        primero = algorithm.prepare(batch, policy)
        for parametro, previo in zip(policy.parameters(), snapshot, strict=True):
            assert torch.equal(parametro, previo)
        assert policy.training == modo
        if isinstance(primero, RolloutBatch) and primero.advantages is not None:
            assert not primero.advantages.requires_grad

        segundo = algorithm.prepare(batch, policy)
        if isinstance(primero, RolloutBatch) and isinstance(segundo, RolloutBatch):
            if primero.advantages is not None and segundo.advantages is not None:
                assert torch.allclose(primero.advantages, segundo.advantages)
