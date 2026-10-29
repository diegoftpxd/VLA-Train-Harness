"""Lado de Core del Environment: registro de ambientes y cliente ZeroMQ.

Definido en ``docs/core/Core Design.md``. El contrato (``Environment``,
``EnvSpec``) y el protocolo (``handle_environment``) viven en
``vla_common.environment``, compartidos con ``vla_env``, que implementa y sirve
los ambientes. Core nunca ejecuta física, solo enruta. La recompensa la agrega
Core, así que ``step`` devuelve el ``StepResult`` con ``reward=None``.
"""

from __future__ import annotations

from vla_common import wire
from vla_common.environment import Environment, EnvSpec, spec_from_wire
from vla_common.types import ActionChunk, Observation, StepResult
from vla_core.errors import DownstreamTimeout, UnknownEnvironment
from vla_core.rpc import RpcChannel


class EnvironmentRegistry:
    """Mapa ``env_id`` → ``Environment`` con su disponibilidad.

    Resuelve y lleva qué ambientes están disponibles; no enruta lógica de Core.
    Core marca un ambiente **no disponible** cuando su proceso deja de responder
    (ver el manejo de fallas en ``Core``), y vuelve a marcarlo disponible si se
    recupera. ``available`` lista solo los disponibles; ``registered`` los lista
    todos (para cerrarlos, por ejemplo).
    """

    def __init__(self) -> None:
        self._envs: dict[str, Environment] = {}
        self._unavailable: set[str] = set()

    def register(self, env_id: str, environment: Environment) -> None:
        self._envs[env_id] = environment
        self._unavailable.discard(env_id)

    def get(self, env_id: str) -> Environment:
        if env_id not in self._envs:
            raise UnknownEnvironment(
                f"ambiente {env_id!r} no registrado. Disponibles: {sorted(self._envs)}"
            )
        return self._envs[env_id]

    def is_available(self, env_id: str) -> bool:
        return env_id in self._envs and env_id not in self._unavailable

    def mark_unavailable(self, env_id: str) -> None:
        self._unavailable.add(env_id)

    def mark_available(self, env_id: str) -> None:
        self._unavailable.discard(env_id)

    def available(self) -> list[str]:
        return sorted(env_id for env_id in self._envs if env_id not in self._unavailable)

    def registered(self) -> list[str]:
        return sorted(self._envs)


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
        # El ``close`` es best-effort: si el proceso ya no responde, igual se
        # libera el socket local sin propagar el timeout (es teardown).
        try:
            self._channel.call({"call": "close"})
        except (DownstreamTimeout, RuntimeError):
            pass
        finally:
            self._channel.close()

    def emergency_stop(self) -> None:
        self._channel.call({"call": "emergency_stop"})

    @property
    def spec(self) -> EnvSpec:
        if self._spec is None:
            self._spec = spec_from_wire(self._channel.call({"call": "spec"})["spec"])
        return self._spec
