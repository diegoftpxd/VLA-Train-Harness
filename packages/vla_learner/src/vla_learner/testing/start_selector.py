"""``ListStartSelector``: un ``StartSelector`` mínimo para los tests.

Devuelve, en orden y de forma cíclica, puntos de partida de una lista fija
(observaciones reales o referencias ``(episode_id, paso)``).
"""

from __future__ import annotations

from vla_common import Observation
from vla_learner.data.core_client import CoreClientProtocol
from vla_learner.data.sources import StartSelector


class ListStartSelector(StartSelector):
    def __init__(self, starts: list[Observation | tuple[str, int]]) -> None:
        self._starts = list(starts)

    def select(self, n: int) -> list[Observation | tuple[str, int]]:
        return [self._starts[i % len(self._starts)] for i in range(n)]


class ResetStartSelector(StartSelector):
    """Reinicia ``n`` ambientes frescos en cada llamada y devuelve sus observaciones.

    Da puntos de partida diversos para entrenar en el WorldModel (así la política
    generaliza en vez de sobreajustar a unos pocos estados fijos).
    """

    def __init__(self, client: CoreClientProtocol, prefix: str = "wm-start") -> None:
        self._client = client
        self._prefix = prefix
        self._contador = 0

    def select(self, n: int) -> list[Observation | tuple[str, int]]:
        inicios: list[Observation | tuple[str, int]] = []
        for _ in range(n):
            self._contador += 1
            ((_, obs),) = self._client.reset([f"{self._prefix}-{self._contador}"])
            inicios.append(obs)
        return inicios
