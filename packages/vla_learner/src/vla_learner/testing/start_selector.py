"""``ListStartSelector``: un ``StartSelector`` mínimo para los tests.

Devuelve, en orden y de forma cíclica, puntos de partida de una lista fija
(observaciones reales o referencias ``(episode_id, paso)``).
"""

from __future__ import annotations

from vla_common import Observation
from vla_learner.data.sources import StartSelector


class ListStartSelector(StartSelector):
    def __init__(self, starts: list[Observation | tuple[str, int]]) -> None:
        self._starts = list(starts)

    def select(self, n: int) -> list[Observation | tuple[str, int]]:
        return [self._starts[i % len(self._starts)] for i in range(n)]
