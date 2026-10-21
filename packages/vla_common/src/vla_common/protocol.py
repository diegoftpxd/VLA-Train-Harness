"""Contrato de Core: la interfaz que el Learner ve y que Core implementa.

Definido en ``docs/Architecture.md`` y ``docs/core/Core Design.md``. Vive en
``vla_common`` para que haya una única definición que compartan el cliente
(``CoreClient``), el servidor (``CoreServer``) y los backends (``FakeCoreClient``
en memoria y el ``Core`` real). Así cliente y servidor no pueden divergir.
"""

from __future__ import annotations

from typing import Protocol

from vla_common.types import ActionChunk, Observation, StepResult, Transition


class CoreInterface(Protocol):
    """Interfaz de Core. Todas las llamadas aceptan listas para atender varios
    ambientes en paralelo con un solo mensaje (ver ``docs/Architecture.md``).
    """

    def reset(
        self, env_ids: list[str], task: str | None = ..., seed: int | None = ...
    ) -> list[tuple[str, Observation]]: ...

    def step(
        self, env_ids: list[str], actions: list[ActionChunk], with_reward: bool
    ) -> list[StepResult]: ...

    def simulate(
        self,
        start: Observation | tuple[str, int],
        action_sequences: list[list[ActionChunk]],
        with_reward: bool,
    ) -> list[list[StepResult]]: ...

    def sample_supervised(
        self, n: int, filters: dict | None = ...
    ) -> list[tuple[Observation, ActionChunk]]: ...

    def get_episode(self, episode_id: str) -> list[Transition]: ...

    def close(self) -> None: ...
