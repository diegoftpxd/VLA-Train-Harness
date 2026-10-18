"""``CoreClient``: la única clase que habla por la red (ZeroMQ).

Definido en ``docs/Architecture.md`` y ``docs/Learner Design.md``. Expone la
interfaz de Core y es el único módulo que importa ``zmq`` (lo verifica ARCH-01).
Arma los mensajes con ``vla_common.wire`` + ``vla_common.serialization`` (sin
pickle) y usa un socket ``DEALER`` con timeout y sin bloqueo.
"""

from __future__ import annotations

from typing import Any, Protocol

import zmq

from vla_common import wire
from vla_common.serialization import decode, encode
from vla_common.types import ActionChunk, Observation, StepResult, Transition


class CoreClientProtocol(Protocol):
    """Interfaz de Core. La cumplen ``CoreClient`` (red) y ``FakeCoreClient`` (memoria)."""

    def reset(
        self, env_ids: list[str], task: str | None = ..., seed: int | None = ...
    ) -> list[tuple[str, Observation]]: ...

    def step(
        self, env_ids: list[str], actions: list[ActionChunk], with_reward: bool
    ) -> list[StepResult]: ...

    def simulate(
        self,
        start: Observation | tuple[str, int],
        action_sequences: list[list[ActionChunk]],
        with_reward: bool,
    ) -> list[list[StepResult]]: ...

    def sample_supervised(
        self, n: int, filters: dict | None = ...
    ) -> list[tuple[Observation, ActionChunk]]: ...

    def get_episode(self, episode_id: str) -> list[Transition]: ...

    def close(self) -> None: ...


class CoreClient:
    def __init__(self, address: str, timeout_s: float) -> None:
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(address)
        self._timeout_ms = int(timeout_s * 1000)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

    def _rpc(self, request: dict[str, Any]) -> dict[str, Any]:
        self._socket.send_multipart(encode(request))
        listos = dict(self._poller.poll(self._timeout_ms))
        if self._socket not in listos:
            raise TimeoutError(f"Core no respondió en {self._timeout_ms} ms.")
        response = decode(self._socket.recv_multipart())
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "Error desconocido de Core."))
        return response

    def reset(
        self,
        env_ids: list[str],
        task: str | None = None,
        seed: int | None = None,
    ) -> list[tuple[str, Observation]]:
        response = self._rpc(
            {"call": "reset", "env_ids": list(env_ids), "task": task, "seed": seed}
        )
        return [
            (episode_id, wire.observation_from_wire(obs))
            for episode_id, obs in response["episodes"]
        ]

    def step(
        self,
        env_ids: list[str],
        actions: list[ActionChunk],
        with_reward: bool,
    ) -> list[StepResult]:
        response = self._rpc(
            {
                "call": "step",
                "env_ids": list(env_ids),
                "actions": [wire.action_chunk_to_wire(c) for c in actions],
                "with_reward": with_reward,
            }
        )
        return [wire.step_result_from_wire(r) for r in response["results"]]

    def simulate(
        self,
        start: Observation | tuple[str, int],
        action_sequences: list[list[ActionChunk]],
        with_reward: bool,
    ) -> list[list[StepResult]]:
        if isinstance(start, Observation):
            start_payload: dict[str, Any] = {"start_obs": wire.observation_to_wire(start)}
        else:
            start_payload = {"start_ref": [start[0], start[1]]}
        response = self._rpc(
            {
                "call": "simulate",
                **start_payload,
                "action_sequences": [
                    [wire.action_chunk_to_wire(c) for c in seq] for seq in action_sequences
                ],
                "with_reward": with_reward,
            }
        )
        return [[wire.step_result_from_wire(r) for r in seq] for seq in response["sequences"]]

    def sample_supervised(
        self,
        n: int,
        filters: dict | None = None,
    ) -> list[tuple[Observation, ActionChunk]]:
        response = self._rpc({"call": "sample_supervised", "n": n, "filters": filters})
        return [
            (wire.observation_from_wire(obs), wire.action_chunk_from_wire(chunk))
            for obs, chunk in response["pairs"]
        ]

    def get_episode(self, episode_id: str) -> list[Transition]:
        response = self._rpc({"call": "get_episode", "episode_id": episode_id})
        return [wire.transition_from_wire(t) for t in response["transitions"]]

    def close(self) -> None:
        try:
            self._rpc({"call": "close"})
        except (TimeoutError, RuntimeError):
            pass
        self._socket.close(0)
