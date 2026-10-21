"""``FakeCoreClient``: implementación en memoria de la interfaz de Core.

Definido en ``docs/learner/Learner Design.md`` y ``docs/learner/Test Learners.md``. Tiene la
misma interfaz que el ``CoreClient`` real (Fase 5), pero resuelve todo en
memoria con ``ToyReachEnv`` y ``ToyWorldModel``, sin red ni ZeroMQ. Registra
cada llamada en ``self.calls`` para que los tests la inspeccionen.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_common import Action, ActionChunk, Observation, StepResult, Transition
from vla_learner.testing.toy_env import BOUND, ToyReachEnv, ToyWorldModel


def _strip_reward(result: StepResult) -> StepResult:
    """Copia el resultado con ``reward=None`` (cuando ``with_reward=False``)."""
    return StepResult(
        observation=result.observation,
        reward=None,
        terminated=result.terminated,
        truncated=result.truncated,
        info=result.info,
    )


class FakeCoreClient:
    def __init__(
        self,
        seed: int = 0,
        demo_chunk_size: int = 1,
        world_model: ToyWorldModel | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._demo_chunk_size = demo_chunk_size
        self._world_model = world_model or ToyWorldModel()
        self._envs: dict[str, ToyReachEnv] = {}
        self._episodes: dict[str, list[Transition]] = {}
        self._episode_of: dict[str, str] = {}
        self._last_obs: dict[str, Observation] = {}
        self._episode_counter = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    # -- Interfaz de Core -------------------------------------------------

    def reset(
        self,
        env_ids: list[str],
        task: str | None = None,
        seed: int | None = None,
    ) -> list[tuple[str, Observation]]:
        self.calls.append(("reset", {"env_ids": list(env_ids), "task": task, "seed": seed}))
        salida: list[tuple[str, Observation]] = []
        for env_id in env_ids:
            env = self._envs.setdefault(env_id, ToyReachEnv())
            obs = env.reset(task=task, seed=seed)
            self._episode_counter += 1
            episode_id = f"{env_id}-ep{self._episode_counter}"
            self._episode_of[env_id] = episode_id
            self._episodes[episode_id] = []
            self._last_obs[env_id] = obs
            salida.append((episode_id, obs))
        return salida

    def step(
        self,
        env_ids: list[str],
        actions: list[ActionChunk],
        with_reward: bool,
    ) -> list[StepResult]:
        self.calls.append(
            ("step", {"env_ids": list(env_ids), "actions": actions, "with_reward": with_reward})
        )
        resultados: list[StepResult] = []
        for env_id, chunk in zip(env_ids, actions, strict=True):
            env = self._envs[env_id]
            previa = self._last_obs[env_id]
            result = env.step(chunk)
            if not with_reward:
                result = _strip_reward(result)
            episode_id = self._episode_of[env_id]
            self._episodes[episode_id].append(
                Transition(
                    episode_id=episode_id,
                    step=len(self._episodes[episode_id]),
                    source="env",
                    observation=previa,
                    action=chunk,
                    result=result,
                )
            )
            self._last_obs[env_id] = result.observation
            resultados.append(result)
        return resultados

    def simulate(
        self,
        start: Observation | tuple[str, int],
        action_sequences: list[list[ActionChunk]],
        with_reward: bool,
    ) -> list[list[StepResult]]:
        self.calls.append(
            (
                "simulate",
                {"start": start, "action_sequences": action_sequences, "with_reward": with_reward},
            )
        )
        start_obs = self._resolve_start(start)
        resultados = self._world_model.rollout(start_obs, action_sequences)
        if not with_reward:
            resultados = [[_strip_reward(r) for r in seq] for seq in resultados]
        return resultados

    def sample_supervised(
        self,
        n: int,
        filters: dict | None = None,
    ) -> list[tuple[Observation, ActionChunk]]:
        self.calls.append(("sample_supervised", {"n": n, "filters": filters}))
        pares: list[tuple[Observation, ActionChunk]] = []
        for _ in range(n):
            pos = self._rng.uniform(-BOUND, BOUND, size=2).astype(np.float32)
            goal = self._rng.uniform(-BOUND, BOUND, size=2).astype(np.float32)
            obs = Observation(
                images={"front": np.zeros((3, 8, 8), dtype=np.float32)},
                proprio=np.concatenate([pos, goal]).astype(np.float32),
                instruction="reach the goal",
            )
            # Controlador perfecto: repartir el desplazamiento (goal - pos) en el chunk.
            delta = ((goal - pos) / self._demo_chunk_size).astype(np.float32)
            chunk: ActionChunk = [Action(values=delta.copy()) for _ in range(self._demo_chunk_size)]
            pares.append((obs, chunk))
        return pares

    def get_episode(self, episode_id: str) -> list[Transition]:
        self.calls.append(("get_episode", {"episode_id": episode_id}))
        return list(self._episodes[episode_id])

    def close(self) -> None:
        self.calls.append(("close", {}))
        self._envs.clear()

    # -- Interno ----------------------------------------------------------

    def _resolve_start(self, start: Observation | tuple[str, int]) -> Observation:
        if isinstance(start, Observation):
            return start
        episode_id, step = start
        return self._episodes[episode_id][step].observation
