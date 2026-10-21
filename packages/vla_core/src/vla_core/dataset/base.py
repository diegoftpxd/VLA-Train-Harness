"""Interfaz ``Dataset``: demos supervisadas servidas desde dentro de Core.

Definido en ``docs/core/Core Design.md``, ``docs/Architecture.md`` y
``docs/dataset/Dataset Design.md``. En v1 el Dataset no es un proceso propio:
vive dentro de Core y se accede solo por él (``sample_supervised``). Entrega las
acciones en **unidades reales**, sin normalizar: cada política del Learner
normaliza a su manera.

``filters`` es un diccionario opcional de criterios (``task``, ``embodiment``,
...); cada implementación respeta los que entiende e ignora el resto. Un corpus
de solo-lectura (Open-X, LeRobot) implementa solo ``Dataset``; un dataset que
admite reinyectar demos nuevas —el gancho de episodios exitosos (ver
``AppendableDataset``)— implementa ``AppendableDataset``.

Esta es la **única frontera** que Core ve: los adaptadores concretos de cada
formato viven detrás de estas ABCs y se construyen con ``build_dataset`` (ver
``factory.py``), de modo que Core queda agnóstico del formato y de cómo se
consulta cada corpus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vla_common.types import ActionChunk, Observation


class Dataset(ABC):
    @abstractmethod
    def sample(self, n: int, filters: dict | None = None) -> list[tuple[Observation, ActionChunk]]:
        """Devuelve ``n`` pares ``(Observation, ActionChunk)`` en unidades reales.

        Si ``filters`` trae criterios que la implementación entiende (p. ej.
        ``{"task": "..."}``), los pares devueltos los cumplen.
        """


class AppendableDataset(Dataset):
    """Dataset que además acepta reinyectar demos nuevas.

    Es el soporte del **gancho de episodios exitosos**: Core puede tomar un
    episodio que registró, filtrarlo por éxito y agregar sus pares
    ``(Observation, ActionChunk)`` para que ``sample`` los sirva después
    (ver ``docs/Architecture.md``: "Los episodios exitosos registrados por Core
    pueden agregarse aquí después de un filtro"). Un corpus de solo-lectura no
    implementa esta interfaz.
    """

    @abstractmethod
    def add(self, pairs: list[tuple[Observation, ActionChunk]]) -> int:
        """Agrega ``pairs`` al dataset (en unidades reales). Devuelve cuántos se agregaron."""
