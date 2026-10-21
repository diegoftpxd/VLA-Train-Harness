"""Interfaz ``Dataset``: demos supervisadas servidas desde dentro de Core.

Definido en ``docs/core/Core Design.md`` y ``docs/Architecture.md``. En v1 el
Dataset no es un proceso propio: vive dentro de Core y se accede solo por él
(``sample_supervised``). Entrega las acciones en **unidades reales**, sin
normalizar: cada política del Learner normaliza a su manera.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vla_common.types import ActionChunk, Observation


class Dataset(ABC):
    @abstractmethod
    def sample(self, n: int, filters: dict | None = None) -> list[tuple[Observation, ActionChunk]]:
        """Devuelve ``n`` pares ``(Observation, ActionChunk)`` en unidades reales."""
