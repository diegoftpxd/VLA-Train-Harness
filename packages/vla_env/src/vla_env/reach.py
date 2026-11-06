"""``ReachEnvironment``: adaptador de referencia (``type: reach``).

Definido en ``docs/environment/Environment Design.md``. Un punto en 2D debe
llegar a una meta. Sirve de **plantilla** para agregar un simulador: se copia
este archivo, se reemplaza ``_ReachSim`` por el simulador real y se reescriben
``_make`` y ``_observation``.

Mapeo simulador → canónico (lo que cada adaptador documenta aquí):

- **Cámaras:** ``front``. El simulador entrega una imagen HWC ``uint8`` negra
  (8×8): la tarea se resuelve con el estado, y así la observación es idéntica a
  la del ``FakeEnvironment`` de Core y la integración con el Learner mide lo mismo.
- **Estado:** ``proprio = [pos_x, pos_y, goal_x, goal_y]`` (``float32``).
- **Acción:** desplazamiento 2D en unidades reales; la posición se recorta a
  ``[-1, 1]``.
- **Éxito:** distancia a la meta menor que 0,1 (``info["is_success"]``).
- **Límite:** ``max_steps`` acciones (por defecto 20).

Con ``chunk_size = 1`` reproduce exactamente la dinámica del ``FakeEnvironment``
(misma semilla → mismas posiciones), que a su vez replica el ``ToyReachEnv`` del
Learner.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_common.environment import EnvSpec
from vla_common.types import Observation
from vla_env.gym import GymEnvironment, to_canonical_image

BOUND = 1.0
THRESHOLD = 0.1
TASK = "reach the goal"
IMAGE_SIZE = 8


class _ReachSim:
    """Mini-simulador con API tipo Gymnasium. Ignora la tarea: hay una sola."""

    def __init__(self) -> None:
        self.pos = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)

    def _raw(self) -> dict[str, np.ndarray]:
        return {
            "image": np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            "pos": self.pos.copy(),
            "goal": self.goal.copy(),
        }

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        rng = np.random.default_rng(seed)
        self.pos = rng.uniform(-BOUND, BOUND, size=2).astype(np.float32)
        self.goal = rng.uniform(-BOUND, BOUND, size=2).astype(np.float32)
        return self._raw(), {}

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        self.pos = np.clip(self.pos + action, -BOUND, BOUND).astype(np.float32)
        distancia = float(np.linalg.norm(self.pos - self.goal))
        exito = distancia < THRESHOLD
        return self._raw(), -distancia, False, False, {"is_success": exito}


class ReachEnvironment(GymEnvironment):
    def __init__(self, max_steps: int = 20) -> None:
        super().__init__(EnvSpec(action_dim=2, cameras=["front"], tasks=[TASK]), max_steps)

    def _make(self, task: str) -> _ReachSim:
        # Un solo simulador para la única tarea: se reusa entre episodios.
        return self._sim if self._sim is not None else _ReachSim()

    def _observation(self, raw: dict[str, np.ndarray], task: str) -> Observation:
        return Observation(
            images={"front": to_canonical_image(raw["image"])},
            proprio=np.concatenate([raw["pos"], raw["goal"]]).astype(np.float32),
            instruction=task,
        )
