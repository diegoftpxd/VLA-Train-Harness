"""Interfaz ``Algorithm``.

Definida en ``docs/learner/Learner Design.md``. Cada algoritmo declara qué datos
necesita, qué capacidades exige del modelo, cómo prepara el batch y cómo calcula
la loss. Nunca toca el optimizador.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vla_learner.policy.interfaces import Capability, VLAPolicy
from vla_learner.types import Batch, LossOutput


class Algorithm(ABC):
    @property
    @abstractmethod
    def required_capabilities(self) -> set[Capability]: ...

    @property
    @abstractmethod
    def batch_type(self) -> type[Batch]:
        """``SupervisedBatch`` o ``RolloutBatch``."""

    def prepare(self, batch: Batch, policy: VLAPolicy) -> Batch:
        """Trabajo previo a las actualizaciones, sin gradientes. Por defecto, nada."""
        return batch

    @abstractmethod
    def loss(self, policy: VLAPolicy, batch: Batch) -> LossOutput: ...

    @property
    def update_epochs(self) -> int:
        """Cuántas pasadas sobre el mismo batch. 1 para BC."""
        return 1

    @property
    def minibatch_size(self) -> int | None:
        """``None`` = el batch completo."""
        return None

    def should_stop_epochs(self, metrics: dict[str, float]) -> bool:
        """El Learner lo consulta tras cada época; si devuelve True, corta las
        épocas restantes de esta iteración. Por defecto no corta nunca."""
        return False
