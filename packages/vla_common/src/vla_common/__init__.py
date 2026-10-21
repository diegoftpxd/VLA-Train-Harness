"""Tipos compartidos y serialización del harness VLA.

Este paquete lo usan todos los procesos (Learner, Core y los demás),
por eso no depende de ``torch``: solo de ``numpy`` y ``msgpack``.
"""

from vla_common.types import (
    Action,
    ActionChunk,
    Array,
    Observation,
    StepResult,
    Transition,
)

__all__ = [
    "Action",
    "ActionChunk",
    "Array",
    "Observation",
    "StepResult",
    "Transition",
]
