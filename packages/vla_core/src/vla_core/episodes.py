"""Estado de episodios: ``EpisodeManager`` (en vuelo) y ``EpisodeStore`` (almacén).

Definido en ``docs/core/Core Design.md``. Dos responsabilidades distintas:

- ``EpisodeManager`` lleva el estado vivo de cada ambiente: su ``episode_id``, el
  paso actual, la observación anterior (la necesita el Reward Model) y si el
  episodio ya terminó (para prohibir un ``step`` sin ``reset``).
- ``EpisodeStore`` guarda y recupera las transiciones ya cerradas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from vla_common.types import Observation, Transition


@dataclass
class _EpisodeState:
    episode_id: str
    step: int
    last_observation: Observation
    done: bool


class EpisodeManager:
    def __init__(self) -> None:
        self._by_env: dict[str, _EpisodeState] = {}
        self._counter = 0

    def open(self, env_id: str, observation: Observation) -> str:
        """Abre un episodio nuevo para ``env_id`` y devuelve su ``episode_id`` único."""
        self._counter += 1
        episode_id = f"{env_id}-ep{self._counter}"
        self._by_env[env_id] = _EpisodeState(episode_id, 0, observation, done=False)
        return episode_id

    def episode_of(self, env_id: str) -> str:
        return self._by_env[env_id].episode_id

    def last_observation(self, env_id: str) -> Observation:
        return self._by_env[env_id].last_observation

    def is_done(self, env_id: str) -> bool:
        return self._by_env[env_id].done

    def advance(
        self, env_id: str, next_observation: Observation, terminated: bool, truncated: bool
    ) -> int:
        """Registra un paso y devuelve el índice de la transición recién cerrada."""
        estado = self._by_env[env_id]
        indice = estado.step
        estado.step += 1
        estado.last_observation = next_observation
        estado.done = terminated or truncated
        return indice


class EpisodeStore(ABC):
    @abstractmethod
    def append(self, transition: Transition) -> None: ...

    @abstractmethod
    def get(self, episode_id: str) -> list[Transition]: ...


class InMemoryEpisodeStore(EpisodeStore):
    """Almacén de episodios en memoria (v1). Una variante en disco se agrega
    sin cambiar la interfaz."""

    def __init__(self) -> None:
        self._episodes: dict[str, list[Transition]] = {}

    def append(self, transition: Transition) -> None:
        self._episodes.setdefault(transition.episode_id, []).append(transition)

    def get(self, episode_id: str) -> list[Transition]:
        if episode_id not in self._episodes:
            raise KeyError(f"episodio {episode_id!r} no registrado")
        return list(self._episodes[episode_id])
