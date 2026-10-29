"""``GymEnvironment``: base para envolver un simulador con API tipo Gymnasium.

Definida en ``docs/environment/Environment Design.md``. Aplica **una sola vez**
las reglas del harness, para que ningún adaptador las repita:

- el chunk cuenta como un paso y se corta en la primera acción que termina el
  episodio (éxito, fallo o límite);
- ``max_steps`` cuenta acciones de bajo nivel, no chunks;
- el éxito sale de ``info["is_success"]`` del simulador (convención de
  Gymnasium) y viaja en ``info["success"]``;
- la recompensa del simulador se ignora (la pone Core);
- la acción se valida antes de mover nada, y un ``step`` tras un final es error.

Un adaptador solo implementa ``_make`` (el simulador para una tarea) y
``_observation`` (traducir la observación cruda al tipo canónico), y
``_success`` si su simulador marca el éxito de otra forma. No importa
``gymnasium``: alcanza con que el simulador tenga ``reset(seed=...) -> (obs,
info)`` y ``step(action) -> (obs, reward, terminated, truncated, info)``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from vla_common.environment import Environment, EnvSpec, check_action_chunk, resolve_task
from vla_common.types import ActionChunk, Observation, StepResult


def to_canonical_image(image: np.ndarray) -> np.ndarray:
    """Imagen RGB → ``float32`` CHW en ``[0, 1]`` (el formato canónico).

    Acepta HWC o CHW (3 canales), en ``uint8`` (0–255) o ``float`` (ya en
    ``[0, 1]``). Cualquier otra forma es ``ValueError``.
    """
    array = np.asarray(image)
    if array.ndim != 3 or 3 not in (array.shape[0], array.shape[-1]):
        raise ValueError(f"se esperaba una imagen RGB HWC o CHW; forma {array.shape}")
    if array.shape[0] != 3:  # HWC → CHW
        array = np.transpose(array, (2, 0, 1))
    if array.dtype == np.uint8:
        return array.astype(np.float32) / np.float32(255.0)
    return np.clip(array.astype(np.float32), 0.0, 1.0)


class GymEnvironment(Environment):
    """Envuelve un simulador tipo Gymnasium y aplica las reglas del harness."""

    def __init__(self, spec: EnvSpec, max_steps: int) -> None:
        self._spec = spec
        self.max_steps = max_steps
        self._sim: Any = None
        self._task = spec.tasks[0]
        self._actions_done = 0
        self._done = True  # sin reset todavía

    # --- Lo que implementa cada adaptador ----------------------------------

    @abstractmethod
    def _make(self, task: str) -> Any:
        """Devuelve el simulador para ``task`` (puede reusar el actual)."""

    @abstractmethod
    def _observation(self, raw: Any, task: str) -> Observation:
        """Traduce la observación cruda del simulador al tipo canónico."""

    def _success(self, raw_info: dict[str, Any]) -> bool:
        return bool(raw_info.get("is_success", False))

    # --- Lo que resuelve la base -------------------------------------------

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    def reset(self, task: str | None, seed: int | None) -> Observation:
        self._task = resolve_task(task, self._spec)
        self._sim = self._make(self._task)
        raw, _ = self._sim.reset(seed=seed)
        self._actions_done = 0
        self._done = False
        return self._observation(raw, self._task)

    def step(self, actions: ActionChunk) -> StepResult:
        if self._done:
            raise RuntimeError("el episodio ya terminó (o no empezó); hace falta reset()")
        check_action_chunk(actions, self._spec.action_dim)
        raw: Any = None
        terminated = truncated = success = False
        for action in actions:
            raw, _, terminated, truncated, info = self._sim.step(
                np.asarray(action.values, dtype=np.float32)
            )
            self._actions_done += 1
            success = self._success(info)
            if terminated or truncated or success or self._actions_done >= self.max_steps:
                break
        terminated = bool(terminated or success)
        truncated = not terminated and bool(truncated or self._actions_done >= self.max_steps)
        self._done = terminated or truncated
        return StepResult(
            observation=self._observation(raw, self._task),
            reward=None,  # la agrega Core
            terminated=terminated,
            truncated=truncated,
            info={"success": success},
        )

    def close(self) -> None:
        if self._sim is not None and hasattr(self._sim, "close"):
            self._sim.close()
        self._sim = None

    def emergency_stop(self) -> None:
        pass  # un simulador no tiene nada que frenar
