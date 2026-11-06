"""Contrato y protocolo del Environment, compartidos por Core y ``vla_env``.

Definido en ``docs/environment/Environment Design.md``. Un Environment es un
ambiente real o simulado con API tipo Gymnasium. Core lo consume (por un cliente
``DEALER``) y ``vla_env`` lo implementa y sirve (con un ``ROUTER``); los dos se
encuentran solo aquí, así que ninguno depende del otro. Como el resto de
``vla_common``, este módulo no hace I/O: nada de ``zmq`` ni ``torch``.

Convenciones que todo ambiente cumple (las verifica la suite ENV):

- ``step`` devuelve el ``StepResult`` con ``reward=None`` (la recompensa la
  agrega Core) e ``info["success"]: bool`` (``terminated`` no distingue éxito
  de fallo). Los valores de ``info`` son escalares, para viajar en msgpack.
- Si el episodio termina a mitad de un chunk, se corta ahí y se devuelve la
  observación de ese punto: el chunk cuenta como un solo paso.
- Las imágenes son ``float32``, CHW, de 3 canales, con valores en ``[0, 1]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from vla_common import wire
from vla_common.types import ActionChunk, Observation, StepResult


@dataclass
class EnvSpec:
    """Lo que un ambiente declara: dimensión de acción, cámaras y tareas."""

    action_dim: int
    cameras: list[str]
    tasks: list[str]  # las que acepta reset(task=...); task=None usa tasks[0]


class Environment(ABC):
    @abstractmethod
    def reset(self, task: str | None, seed: int | None) -> Observation:
        """Inicia un episodio. ``task=None`` usa ``spec.tasks[0]``; una tarea
        fuera de ``spec.tasks`` es error. La misma semilla da la misma
        observación."""

    @abstractmethod
    def step(self, actions: ActionChunk) -> StepResult:
        """Ejecuta el chunk y devuelve el ``StepResult`` con ``reward=None``."""

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def emergency_stop(self) -> None:
        """Parada de seguridad del robot real; no-op en simuladores."""

    @property
    @abstractmethod
    def spec(self) -> EnvSpec: ...


def check_action_chunk(actions: ActionChunk, action_dim: int) -> None:
    """Levanta ``ValueError`` si el chunk está vacío o alguna acción no es un
    vector de largo ``action_dim``. Es la única definición de esa regla: la usan
    Core (antes de tocar ningún ambiente) y los ambientes (antes de mover nada).
    """
    if not actions:
        raise ValueError("el chunk de acciones está vacío")
    for i, action in enumerate(actions):
        forma = np.shape(action.values)
        if forma != (action_dim,):
            raise ValueError(
                f"la acción {i} del chunk tiene forma {forma}; se esperaba ({action_dim},)"
            )


def resolve_task(task: str | None, spec: EnvSpec) -> str:
    """Tarea efectiva de un ``reset``: ``task`` si está en ``spec.tasks``, o
    ``spec.tasks[0]`` si es ``None``. Una tarea desconocida es ``ValueError``."""
    if task is None:
        return spec.tasks[0]
    if task not in spec.tasks:
        raise ValueError(f"tarea desconocida {task!r}. Disponibles: {spec.tasks}")
    return task


# -- Protocolo ZeroMQ del Environment ---------------------------------------


def spec_to_wire(spec: EnvSpec) -> dict[str, Any]:
    return {"action_dim": spec.action_dim, "cameras": list(spec.cameras), "tasks": list(spec.tasks)}


def spec_from_wire(data: dict[str, Any]) -> EnvSpec:
    return EnvSpec(
        action_dim=data["action_dim"], cameras=list(data["cameras"]), tasks=list(data["tasks"])
    )


def handle_environment(request: dict[str, Any], env: Environment) -> dict[str, Any]:
    """Despacha un request del protocolo de Environment contra ``env``.

    Es la única definición del protocolo del lado servidor: la usan el
    ``EnvironmentServer`` de ``vla_env`` y los servidores de prueba de Core.
    """
    call = request["call"]
    if call == "reset":
        obs = env.reset(request["task"], request["seed"])
        return {"observation": wire.observation_to_wire(obs)}
    if call == "step":
        result = env.step(wire.action_chunk_from_wire(request["actions"]))
        return {"result": wire.step_result_to_wire(result)}
    if call == "close":
        env.close()
        return {}
    if call == "emergency_stop":
        env.emergency_stop()
        return {}
    if call == "spec":
        return {"spec": spec_to_wire(env.spec)}
    raise ValueError(f"Llamada desconocida de Environment: {call!r}")
