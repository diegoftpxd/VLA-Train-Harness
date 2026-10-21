"""``Core``: el gateway. Coordina cada llamada entre sus colaboradores.

Definido en ``docs/core/Core Design.md``. Implementa ``CoreInterface`` (el mismo
contrato que ``FakeCoreClient`` y que el ``CoreClient`` consume), pero delgado:
cada método es una secuencia corta de llamadas a colaboradores. No abre sockets,
no simula física, no puntúa ni guarda transiciones a mano: lo pide.

Estado de avance: ``reset``, ``step`` (con recompensa), ``get_episode`` (Fase 3)
y ``simulate`` (Fase 4) están implementados. ``sample_supervised`` (Fase 5)
todavía levanta ``NotImplementedError``.
"""

from __future__ import annotations

import dataclasses

from vla_common.types import ActionChunk, Observation, StepResult, Transition
from vla_core.dataset import Dataset
from vla_core.environments import EnvironmentRegistry
from vla_core.episodes import EpisodeManager, EpisodeStore
from vla_core.errors import EpisodeAlreadyDone
from vla_core.reward import RewardScorer, ScoreItem
from vla_core.world_model import WorldModel


class Core:
    def __init__(
        self,
        environments: EnvironmentRegistry,
        episodes: EpisodeManager,
        store: EpisodeStore,
        scorer: RewardScorer,
        world_model: WorldModel,
        dataset: Dataset,
    ) -> None:
        self._environments = environments
        self._episodes = episodes
        self._store = store
        self._scorer = scorer
        self._world_model = world_model
        self._dataset = dataset

    def reset(
        self, env_ids: list[str], task: str | None = None, seed: int | None = None
    ) -> list[tuple[str, Observation]]:
        salida: list[tuple[str, Observation]] = []
        for env_id in env_ids:
            obs = self._environments.get(env_id).reset(task, seed)
            episode_id = self._episodes.open(env_id, obs)
            salida.append((episode_id, obs))
        return salida

    def step(
        self, env_ids: list[str], actions: list[ActionChunk], with_reward: bool
    ) -> list[StepResult]:
        # Regla de episodio: no se puede pisar un episodio ya terminado.
        for env_id in env_ids:
            if self._episodes.is_done(env_id):
                raise EpisodeAlreadyDone(
                    f"el episodio de {env_id!r} ya terminó; hace falta reset()"
                )

        # Ejecuta cada chunk en su ambiente (el StepResult viene sin recompensa).
        pendientes: list[tuple[str, ActionChunk, Observation, StepResult]] = []
        for env_id, chunk in zip(env_ids, actions, strict=True):
            previa = self._episodes.last_observation(env_id)
            resultado = self._environments.get(env_id).step(chunk)
            pendientes.append((env_id, chunk, previa, resultado))

        # Una sola llamada a score para todo el batch (regla "un score por paso").
        if with_reward:
            items = [
                ScoreItem(
                    instruction=previa.instruction,
                    observation=previa,
                    action=chunk,
                    next_observation=resultado.observation,
                )
                for _, chunk, previa, resultado in pendientes
            ]
            recompensas: list[float | None] = list(self._scorer.score(items))
        else:
            recompensas = [None] * len(pendientes)

        # Adjunta la recompensa, registra la transición y avanza el episodio.
        salida: list[StepResult] = []
        for (env_id, chunk, previa, resultado), recompensa in zip(
            pendientes, recompensas, strict=True
        ):
            final = dataclasses.replace(resultado, reward=recompensa)
            indice = self._episodes.advance(
                env_id, resultado.observation, resultado.terminated, resultado.truncated
            )
            self._store.append(
                Transition(
                    episode_id=self._episodes.episode_of(env_id),
                    step=indice,
                    source="env",
                    observation=previa,
                    action=chunk,
                    result=final,
                )
            )
            salida.append(final)
        return salida

    def simulate(
        self,
        start: Observation | tuple[str, int],
        action_sequences: list[list[ActionChunk]],
        with_reward: bool,
    ) -> list[list[StepResult]]:
        start_obs = self._resolve_start(start)
        latent = self._world_model.encode(start_obs)
        # El WorldModel devuelve StepResult sin recompensa (puede cortar una
        # secuencia si el paso predicho termina el episodio).
        predichos = self._world_model.rollout(latent, action_sequences)
        if not with_reward:
            return predichos

        # Una sola llamada a score para todas las transiciones imaginadas. La
        # observación anterior del paso j es el inicio (j=0) o el paso j-1.
        items: list[ScoreItem] = []
        indices: list[tuple[int, int]] = []
        for i, secuencia in enumerate(predichos):
            previa = start_obs
            for j, paso in enumerate(secuencia):
                items.append(
                    ScoreItem(
                        instruction=previa.instruction,
                        observation=previa,
                        action=action_sequences[i][j],
                        next_observation=paso.observation,
                    )
                )
                indices.append((i, j))
                previa = paso.observation

        recompensas = self._scorer.score(items)
        salida = [list(secuencia) for secuencia in predichos]
        for (i, j), recompensa in zip(indices, recompensas, strict=True):
            salida[i][j] = dataclasses.replace(salida[i][j], reward=recompensa)
        return salida

    def _resolve_start(self, start: Observation | tuple[str, int]) -> Observation:
        if isinstance(start, Observation):
            return start
        episode_id, step = start
        return self._store.get(episode_id)[step].observation

    def sample_supervised(
        self, n: int, filters: dict | None = None
    ) -> list[tuple[Observation, ActionChunk]]:
        raise NotImplementedError("Core.sample_supervised se implementa en la Fase 5.")

    def get_episode(self, episode_id: str) -> list[Transition]:
        return self._store.get(episode_id)

    def close(self) -> None:
        for env_id in self._environments.available():
            self._environments.get(env_id).close()
