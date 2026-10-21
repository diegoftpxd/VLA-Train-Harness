"""Interfaz ``Environment`` y su alrededor: registro y cliente ZeroMQ.

Definido en ``docs/core/Core Design.md`` y ``docs/Architecture.md``. Un
Environment es un ambiente real o simulado con API tipo Gymnasium. Core lo
consume: nunca ejecuta física, solo enruta. La recompensa la agrega Core, así
que ``step`` devuelve el ``StepResult`` con ``reward=None``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from vla_common import wire
from vla_common.types import ActionChunk, Observation, StepResult
from vla_core.errors import UnknownEnvironment
from vla_core.rpc import RpcChannel


@dataclass
class EnvSpec:
    """Lo que un ambiente declara al registrarse: dimensión de acción y cámaras."""

    action_dim: int
    cameras: list[str]


class Environment(ABC):
    @abstractmethod
    def reset(self, task: str | None, seed: int | None) -> Observation: ...

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


class EnvironmentRegistry:
    """Mapa ``env_id`` → ``Environment``. Resuelve; no enruta lógica de Core."""

    def __init__(self) -> None:
        self._envs: dict[str, Environment] = {}

    def register(self, env_id: str, environment: Environment) -> None:
        self._envs[env_id] = environment

    def get(self, env_id: str) -> Environment:
        if env_id not in self._envs:
            raise UnknownEnvironment(
                f"ambiente {env_id!r} no registrado. Disponibles: {sorted(self._envs)}"
            )
        return self._envs[env_id]

    def available(self) -> list[str]:
        return sorted(self._envs)


# -- Protocolo ZeroMQ del Environment ---------------------------------------


def handle_environment(request: dict[str, Any], env: Environment) -> dict[str, Any]:
    """Despacha un request del protocolo de Environment contra ``env``."""
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
        return {"spec": {"action_dim": env.spec.action_dim, "cameras": list(env.spec.cameras)}}
    raise ValueError(f"Llamada desconocida de Environment: {call!r}")


class EnvironmentClient(Environment):
    """Habla el protocolo de un Environment por un ``RpcChannel`` (``DEALER``)."""

    def __init__(self, address: str, timeout_s: float = 5.0, retries: int = 1) -> None:
        self._channel = RpcChannel(address, timeout_s, retries)
        self._spec: EnvSpec | None = None
        self._closed = False

    def reset(self, task: str | None, seed: int | None) -> Observation:
        response = self._channel.call({"call": "reset", "task": task, "seed": seed})
        return wire.observation_from_wire(response["observation"])

    def step(self, actions: ActionChunk) -> StepResult:
        response = self._channel.call(
            {"call": "step", "actions": wire.action_chunk_to_wire(actions)}
        )
        return wire.step_result_from_wire(response["result"])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._channel.call({"call": "close"})
        finally:
            self._channel.close()

    def emergency_stop(self) -> None:
        self._channel.call({"call": "emergency_stop"})

    @property
    def spec(self) -> EnvSpec:
        if self._spec is None:
            data = self._channel.call({"call": "spec"})["spec"]
            self._spec = EnvSpec(action_dim=data["action_dim"], cameras=list(data["cameras"]))
        return self._spec
