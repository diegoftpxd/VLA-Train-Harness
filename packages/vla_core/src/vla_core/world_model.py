"""Interfaz ``WorldModel`` y su cliente ZeroMQ.

Definido en ``docs/core/Core Design.md`` y ``docs/Architecture.md``. El
WorldModel predice cómo cambian las observaciones ante secuencias de acciones.
``encode`` lleva una observación real a estado latente y ``rollout`` simula
varias secuencias desde ahí, devolviendo **observaciones ya decodificadas** (una
por chunk de cada secuencia). Así Core arma las transiciones imaginadas sin tocar
el espacio latente: sigue siendo *model-free*.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from vla_common import Array, wire
from vla_common.types import ActionChunk, Observation
from vla_core.rpc import RpcChannel


@dataclass
class LatentState:
    """Estado latente del world model. Su contenido es propio del modelo."""

    values: Array


def latent_to_wire(latent: LatentState) -> dict[str, Any]:
    return {"values": latent.values}


def latent_from_wire(data: dict[str, Any]) -> LatentState:
    return LatentState(values=np.asarray(data["values"]))


class WorldModel(ABC):
    @abstractmethod
    def encode(self, observation: Observation) -> LatentState: ...

    @abstractmethod
    def rollout(
        self, latent: LatentState, action_sequences: list[list[ActionChunk]]
    ) -> list[list[Observation]]:
        """Observaciones predichas: una lista por secuencia, una por chunk."""


# -- Protocolo ZeroMQ del WorldModel ----------------------------------------


def handle_world_model(request: dict[str, Any], model: WorldModel) -> dict[str, Any]:
    call = request["call"]
    if call == "encode":
        latent = model.encode(wire.observation_from_wire(request["observation"]))
        return {"latent": latent_to_wire(latent)}
    if call == "rollout":
        secuencias = [
            [wire.action_chunk_from_wire(c) for c in seq] for seq in request["action_sequences"]
        ]
        predichas = model.rollout(latent_from_wire(request["latent"]), secuencias)
        return {"observations": [[wire.observation_to_wire(o) for o in seq] for seq in predichas]}
    raise ValueError(f"Llamada desconocida de WorldModel: {call!r}")


class WorldModelClient(WorldModel):
    def __init__(self, address: str, timeout_s: float = 10.0, retries: int = 1) -> None:
        self._channel = RpcChannel(address, timeout_s, retries)

    def encode(self, observation: Observation) -> LatentState:
        response = self._channel.call(
            {"call": "encode", "observation": wire.observation_to_wire(observation)}
        )
        return latent_from_wire(response["latent"])

    def rollout(
        self, latent: LatentState, action_sequences: list[list[ActionChunk]]
    ) -> list[list[Observation]]:
        response = self._channel.call(
            {
                "call": "rollout",
                "latent": latent_to_wire(latent),
                "action_sequences": [
                    [wire.action_chunk_to_wire(c) for c in seq] for seq in action_sequences
                ],
            }
        )
        return [[wire.observation_from_wire(o) for o in seq] for seq in response["observations"]]

    def close(self) -> None:
        self._channel.close()
