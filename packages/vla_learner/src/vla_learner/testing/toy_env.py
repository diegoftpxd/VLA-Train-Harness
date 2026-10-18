"""``ToyReachEnv`` y ``ToyWorldModel``: un problema mínimo de control 2D.

Definidos en ``docs/Test Learners.md``. Un punto en 2D debe llegar a un
objetivo. La observación lleva en la propiocepción la posición y el objetivo,
``[pos_x, pos_y, goal_x, goal_y]``, para que una política tiny pueda aprender a
moverse hacia el objetivo. La recompensa es ``-distancia``; ``terminated`` al
llegar; ``truncated`` a los ``MAX_STEPS`` pasos (un chunk = un paso).
"""

from __future__ import annotations

import numpy as np

from vla_common import ActionChunk, Observation, StepResult

GOAL_RADIUS = 0.1  # distancia a la que se considera alcanzado el objetivo
MAX_STEPS = 20  # pasos (chunks) antes de truncar
BOUND = 1.0  # el punto vive en [-BOUND, BOUND]^2
IMAGE_SHAPE = (3, 8, 8)  # imagen mínima para que la observación esté bien formada


def _image() -> dict[str, np.ndarray]:
    return {"front": np.zeros(IMAGE_SHAPE, dtype=np.float32)}


class ToyReachEnv:
    """Ambiente mínimo de alcance en 2D. Opera con numpy, sin torch."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self.pos = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.done = False

    def reset(self, task: str | None = None, seed: int | None = None) -> Observation:
        """Reinicia a una posición y un objetivo. Con la misma ``seed`` da el mismo episodio."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.pos = self._rng.uniform(-BOUND, BOUND, size=2).astype(np.float32)
        self.goal = self._rng.uniform(-BOUND, BOUND, size=2).astype(np.float32)
        self.steps = 0
        self.done = False
        return self.observation()

    def step(self, chunk: ActionChunk) -> StepResult:
        """Aplica el chunk en secuencia. Si llega a mitad del chunk, se detiene ahí."""
        if self.done:
            raise RuntimeError("step sobre un episodio terminado; falta reset.")
        reached = False
        for action in chunk:
            delta = np.asarray(action.values, dtype=np.float32).reshape(2)
            self.pos = np.clip(self.pos + delta, -BOUND, BOUND).astype(np.float32)
            if self._distance() <= GOAL_RADIUS:
                reached = True
                break
        self.steps += 1
        terminated = reached
        truncated = (not terminated) and self.steps >= MAX_STEPS
        self.done = terminated or truncated
        return StepResult(
            observation=self.observation(),
            reward=-self._distance(),
            terminated=terminated,
            truncated=truncated,
            info={"distance": self._distance()},
        )

    def observation(self) -> Observation:
        proprio = np.concatenate([self.pos, self.goal]).astype(np.float32)
        return Observation(images=_image(), proprio=proprio, instruction="reach the goal")

    def _distance(self) -> float:
        return float(np.linalg.norm(self.pos - self.goal))


class ToyWorldModel:
    """Simulador perfecto de ``ToyReachEnv``, para el camino de ``simulate``.

    No tiene estado propio: parte de una observación dada y reproduce la misma
    dinámica que el ambiente real.
    """

    def rollout(
        self,
        start: Observation,
        action_sequences: list[list[ActionChunk]],
    ) -> list[list[StepResult]]:
        """Simula cada secuencia de chunks desde ``start`` y devuelve sus pasos."""
        resultados: list[list[StepResult]] = []
        for secuencia in action_sequences:
            env = ToyReachEnv()
            env.pos = np.asarray(start.proprio[:2], dtype=np.float32).copy()
            env.goal = np.asarray(start.proprio[2:4], dtype=np.float32).copy()
            pasos: list[StepResult] = []
            for chunk in secuencia:
                pasos.append(env.step(chunk))
                if env.done:
                    break
            resultados.append(pasos)
        return resultados
