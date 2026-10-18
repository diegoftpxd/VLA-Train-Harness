"""Suite de contrato HEAD (parte genérica) para ``ActionHead`` con ``LOG_PROB``.

Cubre HEAD-01 (formas) y HEAD-05 (``bc_loss`` mínima en el objetivo). Los tests
específicos de cada cabeza (HEAD-02 log-prob correcta, HEAD-04 discretización
estable) viven en los módulos de cada cabeza. HEAD-03 (log-prob del chunk = suma)
queda cubierto por la coincidencia exacta de HEAD-02.

La implementación entrega el fixture ``head``; ``FEATURE_DIM`` debe coincidir con
el de la cabeza.
"""

from __future__ import annotations

import pytest
import torch

from vla_learner.policy.interfaces import ActionHead, Capability


class HeadContract:
    FEATURE_DIM = 16
    BATCH = 5

    @pytest.fixture
    def head(self) -> ActionHead:
        raise NotImplementedError

    @pytest.fixture
    def features(self) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randn(self.BATCH, 1, self.FEATURE_DIM)

    @pytest.fixture
    def actions(self, head: ActionHead) -> torch.Tensor:
        torch.manual_seed(1)
        return torch.rand(self.BATCH, head.chunk_size, head.action_dim) * 1.8 - 0.9

    def test_head01_formas(self, head, features):
        acciones, log_probs, extras = head.sample(features, deterministic=False)
        assert acciones.shape == (self.BATCH, head.chunk_size, head.action_dim)
        if Capability.LOG_PROB in head.capabilities:
            assert log_probs is not None
            assert log_probs.shape == (self.BATCH,)
        else:
            assert log_probs is None
        for tensor in extras.values():
            assert tensor.shape[0] == self.BATCH  # extras seleccionable por fila

    def test_head05_bc_loss_minima_en_el_objetivo(self, head, features, actions):
        optimizador = torch.optim.Adam(head.parameters(), lr=0.05)
        for _ in range(200):
            optimizador.zero_grad()
            perdida = head.bc_loss(features, actions).loss
            perdida.backward()
            optimizador.step()
        with torch.no_grad():
            en_objetivo = head.bc_loss(features, actions).loss
            aleatorio = head.bc_loss(features, torch.rand_like(actions) * 1.8 - 0.9).loss
        assert en_objetivo < aleatorio
