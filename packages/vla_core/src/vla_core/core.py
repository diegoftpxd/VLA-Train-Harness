"""``Core``: el gateway. Coordina cada llamada entre sus colaboradores.

Definido en ``docs/core/Core Design.md``. Implementa ``CoreInterface`` (el mismo
contrato que ``FakeCoreClient`` y que el ``CoreClient`` consume), pero delgado:
cada método es una secuencia corta de llamadas a colaboradores. No abre sockets,
no simula física, no puntúa ni guarda transiciones a mano: lo pide.

Estado de avance: ``reset``, ``step`` (con recompensa), ``get_episode`` (Fase 3),
``simulate`` (Fase 4), ``sample_supervised`` + el gancho ``ingest_episode``
(Fase 5) y el manejo de caídas de ambiente (Fase 6: timeout → no disponible +
parada de seguridad) están implementados.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from vla_common.environment import Environment, check_action_chunk
from vla_common.types import ActionChunk, Observation, StepResult, Transition
from vla_core.dataset import AppendableDataset, Dataset
from vla_core.environments import EnvironmentRegistry
from vla_core.episodes import EpisodeManager, EpisodeStore
from vla_core.errors import (
    DatasetNotAppendable,
    DownstreamTimeout,
    DownstreamUnavailable,
    EpisodeAlreadyDone,
)
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
            obs = self._usar_ambiente(env_id, lambda env: env.reset(task, seed))
            episode_id = self._episodes.open(env_id, obs)
            salida.append((episode_id, obs))
        return salida

    def step(
        self, env_ids: list[str], actions: list[ActionChunk], with_reward: bool
    ) -> list[StepResult]:
        # Pre-chequeo antes de tocar ningún ambiente: disponibilidad, regla de
        # episodio (no se puede pisar un episodio ya terminado) y forma de cada
        # acción contra el spec (un chunk mal formado nunca llega a un robot).
        for env_id, chunk in zip(env_ids, actions, strict=True):
            if not self._environments.is_available(env_id):
                raise DownstreamUnavailable(
                    f"ambiente {env_id!r} no disponible; hace falta recuperarlo"
                )
            if self._episodes.is_done(env_id):
                raise EpisodeAlreadyDone(
                    f"el episodio de {env_id!r} ya terminó; hace falta reset()"
                )
            # El spec puede venir de un proceso remoto: se pide por _usar_ambiente
            # para que su caída se escale igual que la de reset/step.
            action_dim = self._usar_ambiente(env_id, lambda env: env.spec.action_dim)
            try:
                check_action_chunk(chunk, action_dim)
            except ValueError as error:
                raise ValueError(f"ambiente {env_id!r}: {error}") from error

        # Ejecuta cada chunk en su ambiente (el StepResult viene sin recompensa).
        pendientes: list[tuple[str, ActionChunk, Observation, StepResult]] = []
        for env_id, chunk in zip(env_ids, actions, strict=True):
            previa = self._episodes.last_observation(env_id)
            resultado = self._usar_ambiente(env_id, lambda env, c=chunk: env.step(c))
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

    def _usar_ambiente(self, env_id: str, operacion: Callable[[Environment], Any]) -> Any:
        """Ejecuta ``operacion(env)`` manejando la caída del ambiente.

        Si el ambiente ya está marcado no disponible, levanta
        ``DownstreamUnavailable`` sin tocarlo. Si no responde (su cliente
        ``DEALER`` agotó timeout y reintentos → ``DownstreamTimeout``), Core lo
        escala: marca el ambiente no disponible, invoca su parada de seguridad
        (best-effort; no-op en un simulador, frenado en el robot real) y levanta
        ``DownstreamUnavailable``. El resto de los ambientes sigue intacto.
        """
        # ``get`` resuelve el ambiente (o lo autoregistra, en el registro fake, o
        # levanta ``UnknownEnvironment`` si no existe). La disponibilidad se
        # chequea después: solo un ambiente ya resuelto puede estar "caído".
        env = self._environments.get(env_id)
        if not self._environments.is_available(env_id):
            raise DownstreamUnavailable(f"ambiente {env_id!r} no disponible")
        try:
            return operacion(env)
        except DownstreamTimeout as error:
            self._environments.mark_unavailable(env_id)
            self._parada_de_seguridad(env)
            raise DownstreamUnavailable(
                f"ambiente {env_id!r} no respondió; marcado no disponible"
            ) from error

    @staticmethod
    def _parada_de_seguridad(env: Environment) -> None:
        # La parada es best-effort: si el ambiente caído tampoco atiende la
        # parada, no se enmascara la falla original.
        try:
            env.emergency_stop()
        except Exception:  # noqa: BLE001 - best-effort
            pass

    def _resolve_start(self, start: Observation | tuple[str, int]) -> Observation:
        if isinstance(start, Observation):
            return start
        episode_id, step = start
        return self._store.get(episode_id)[step].observation

    def sample_supervised(
        self, n: int, filters: dict | None = None
    ) -> list[tuple[Observation, ActionChunk]]:
        # Delegación directa en el Dataset (ver flujo "Supervisado" en
        # Architecture.md). Las acciones salen en unidades reales; el Learner las
        # marca con source="dataset" al armar su batch.
        return self._dataset.sample(n, filters)

    def ingest_episode(self, episode_id: str) -> int:
        """Gancho de episodios exitosos: reinyecta un episodio en el Dataset.

        Toma las transiciones que Core registró para ``episode_id``, aplica el
        filtro de éxito (la última transición tiene ``info["success"]``; un
        ``terminated`` solo no alcanza, porque un episodio también termina por
        fallo) y agrega sus pares ``(observación,
        acción)`` al Dataset como demos supervisadas. Devuelve cuántos pares se
        agregaron (``0`` si el episodio no fue exitoso).

        El Dataset debe ser ``AppendableDataset`` (``DatasetNotAppendable`` si no).
        Las transiciones imaginadas no llegan acá: solo lo que Core guardó en el
        almacén, que es siempre real (``source="env"``).
        """
        if not isinstance(self._dataset, AppendableDataset):
            raise DatasetNotAppendable("el Dataset configurado no admite reinyectar episodios")
        transiciones = self._store.get(episode_id)
        if not self._episode_succeeded(transiciones):
            return 0
        pares = [(t.observation, t.action) for t in transiciones]
        return self._dataset.add(pares)

    @staticmethod
    def _episode_succeeded(transitions: list[Transition]) -> bool:
        """Éxito = la última transición lo marca en ``info["success"]``."""
        return bool(transitions) and bool(transitions[-1].result.info.get("success", False))

    def get_episode(self, episode_id: str) -> list[Transition]:
        return self._store.get(episode_id)

    def close(self) -> None:
        # Cierra todos los ambientes registrados, disponibles o no.
        for env_id in self._environments.registered():
            self._environments.get(env_id).close()
