"""``StaticCoreBackend``: backend mínimo que cumple ``CoreInterface``.

Es un *andamiaje* de la Fase 1, no el Core real: no simula física, no termina
episodios ni calcula recompensas de verdad. Su único propósito es ejercitar la
capa de transporte (``CoreServer``) y comprobar que cada llamada hace
*round-trip* correcto contra el ``CoreClient`` real: que los arrays sobreviven
el wire bit a bit, que el sobre ``ok``/``error`` funciona y que el campo
``reward`` opcional se respeta.

El ``Core`` real y sus colaboradores (Fases 2 a 6) reemplazan este backend.
"""

from __future__ import annotations

import numpy as np

from vla_common import Action, ActionChunk, Observation, StepResult, Transition

# Un ``env_id`` reservado para forzar un error y probar el sobre ``{ok: False}``.
BOOM = "boom"


def _observation(instruction: str = "reach the goal") -> Observation:
    """Observación canónica con arrays ``float32`` (para verificar el wire)."""
    return Observation(
        images={"front": np.zeros((3, 8, 8), dtype=np.float32)},
        proprio=np.zeros(4, dtype=np.float32),
        instruction=instruction,
    )


class StaticCoreBackend:
    def __init__(self) -> None:
        self._episodes: dict[str, list[Transition]] = {}
        self._episode_of: dict[str, str] = {}
        self._last_obs: dict[str, Observation] = {}
        self._counter = 0

    def reset(
        self, env_ids: list[str], task: str | None = None, seed: int | None = None
    ) -> list[tuple[str, Observation]]:
        salida: list[tuple[str, Observation]] = []
        for env_id in env_ids:
            if env_id == BOOM:
                raise ValueError(f"ambiente {env_id!r} reservado para probar errores")
            self._counter += 1
            episode_id = f"{env_id}-ep{self._counter}"
            obs = _observation()
            self._episodes[episode_id] = []
            self._episode_of[env_id] = episode_id
            self._last_obs[env_id] = obs
            salida.append((episode_id, obs))
        return salida

    def step(
        self, env_ids: list[str], actions: list[ActionChunk], with_reward: bool
    ) -> list[StepResult]:
        resultados: list[StepResult] = []
        for env_id, chunk in zip(env_ids, actions, strict=True):
            if env_id == BOOM:
                raise ValueError(f"ambiente {env_id!r} reservado para probar errores")
            previa = self._last_obs[env_id]
            siguiente = _observation()
            resultado = StepResult(
                observation=siguiente,
                reward=1.0 if with_reward else None,
                terminated=False,
                truncated=False,
                info={},
            )
            episode_id = self._episode_of[env_id]
            self._episodes[episode_id].append(
                Transition(
                    episode_id=episode_id,
                    step=len(self._episodes[episode_id]),
                    source="env",
                    observation=previa,
                    action=chunk,
                    result=resultado,
                )
            )
            self._last_obs[env_id] = siguiente
            resultados.append(resultado)
        return resultados

    def simulate(
        self,
        start: Observation | tuple[str, int],
        action_sequences: list[list[ActionChunk]],
        with_reward: bool,
    ) -> list[list[StepResult]]:
        salida: list[list[StepResult]] = []
        for secuencia in action_sequences:
            pasos = [
                StepResult(
                    observation=_observation(),
                    reward=0.5 if with_reward else None,
                    terminated=False,
                    truncated=False,
                    info={"source": "world_model"},
                )
                for _ in secuencia
            ]
            salida.append(pasos)
        return salida

    def sample_supervised(
        self, n: int, filters: dict | None = None
    ) -> list[tuple[Observation, ActionChunk]]:
        chunk: ActionChunk = [Action(values=np.zeros(2, dtype=np.float32))]
        return [(_observation(), chunk) for _ in range(n)]

    def get_episode(self, episode_id: str) -> list[Transition]:
        return list(self._episodes[episode_id])

    def close(self) -> None:
        self._episodes.clear()
        self._episode_of.clear()
        self._last_obs.clear()
