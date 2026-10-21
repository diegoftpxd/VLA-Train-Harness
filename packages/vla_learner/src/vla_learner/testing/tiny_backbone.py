"""``TinyBackbone``: backbone de pocos parámetros para probar cabezas de acción.

Definido en ``docs/learner/Test Learners.md``. Aplana y concatena todos los tensores de
la entrada y los proyecta a ``feature_dim`` con una capa lineal perezosa (así no
depende del esquema exacto del procesador). Devuelve features ``[B, T, F]``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vla_learner.policy.interfaces import Backbone
from vla_learner.types import ModelInput


class TinyBackbone(Backbone):
    def __init__(self, feature_dim: int = 16, seq_len: int = 1) -> None:
        super().__init__()
        self._feature_dim = feature_dim
        self._seq_len = seq_len
        self.proj = nn.LazyLinear(feature_dim * seq_len)

    def encode(self, inputs: ModelInput) -> Tensor:
        batch_size = inputs.batch_size
        aplanados = [t.reshape(batch_size, -1).float() for t in inputs.tensors.values()]
        concatenado = torch.cat(aplanados, dim=1)
        proyectado = self.proj(concatenado)
        return proyectado.reshape(batch_size, self._seq_len, self._feature_dim)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim
