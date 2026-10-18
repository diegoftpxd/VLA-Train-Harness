"""Tipos de datos compartidos entre todos los procesos del harness.

Definidos en ``docs/Architecture.md``. Todos los procesos usan estos mismos
tipos para no inventar cada uno su propio formato. Las imágenes y la
propiocepción son arrays de numpy (``Array``); nunca tensores de torch, para
que este paquete sea independiente del framework de modelado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

# Alias del tipo de array usado en todo el protocolo.
Array = np.ndarray


@dataclass
class Observation:
    """Entrada del modelo en un instante: imágenes, estado del robot e instrucción."""

    images: dict[str, Array]  # una entrada por cámara
    proprio: Array  # estado del robot (articulaciones, gripper)
    instruction: str  # tarea en lenguaje natural


@dataclass
class Action:
    """Una acción de bajo nivel, en unidades reales (sin normalizar)."""

    values: Array


# Acciones ejecutadas en secuencia; para RL, un chunk cuenta como un solo paso.
ActionChunk = list[Action]


@dataclass
class StepResult:
    """Resultado de aplicar un ``ActionChunk`` en el ambiente o el world model."""

    observation: Observation
    reward: float | None  # None si no se pidió recompensa
    terminated: bool  # la tarea terminó (éxito o fallo)
    truncated: bool  # se cortó por límite de pasos
    info: dict[str, Any]


@dataclass
class Transition:
    """Una transición registrada por Core; la unidad del almacén de episodios."""

    episode_id: str
    step: int
    source: Literal["env", "world_model", "dataset"]
    observation: Observation
    action: ActionChunk
    result: StepResult
