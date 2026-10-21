"""Interfaz ``RewardModel``, su cliente ZeroMQ y el ``RewardScorer``.

Definido en ``docs/core/Core Design.md`` y ``docs/Architecture.md``. El Reward
Model recibe la instrucción y la transición (observación, acción, observación
siguiente) y devuelve una recompensa; acepta batches. El ``RewardScorer`` aísla
la regla "una llamada a ``score`` por paso del loop": junta los items de todos
los ambientes (o secuencias) en un solo batch y reparte las recompensas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from vla_common import wire
from vla_common.types import ActionChunk, Observation
from vla_core.rpc import RpcChannel


@dataclass
class ScoreItem:
    """Una transición a puntuar; la instrucción viaja en la observación, pero se
    pasa explícita para que el Reward Model no dependa de dónde la lee."""

    instruction: str
    observation: Observation
    action: ActionChunk
    next_observation: Observation


class RewardModel(ABC):
    @abstractmethod
    def score(self, items: list[ScoreItem]) -> list[float]:
        """Una recompensa por item, en el mismo orden. Acepta batch vacío."""


class RewardScorer:
    """Puntúa un batch de items en una sola llamada al Reward Model."""

    def __init__(self, reward_model: RewardModel) -> None:
        self._reward_model = reward_model

    def score(self, items: list[ScoreItem]) -> list[float]:
        if not items:
            return []
        return self._reward_model.score(items)


# -- Protocolo ZeroMQ del Reward Model --------------------------------------


def _item_to_wire(item: ScoreItem) -> dict[str, Any]:
    return {
        "instruction": item.instruction,
        "observation": wire.observation_to_wire(item.observation),
        "action": wire.action_chunk_to_wire(item.action),
        "next_observation": wire.observation_to_wire(item.next_observation),
    }


def _item_from_wire(data: dict[str, Any]) -> ScoreItem:
    return ScoreItem(
        instruction=data["instruction"],
        observation=wire.observation_from_wire(data["observation"]),
        action=wire.action_chunk_from_wire(data["action"]),
        next_observation=wire.observation_from_wire(data["next_observation"]),
    )


def handle_reward_model(request: dict[str, Any], model: RewardModel) -> dict[str, Any]:
    call = request["call"]
    if call == "score":
        items = [_item_from_wire(d) for d in request["items"]]
        return {"rewards": [float(r) for r in model.score(items)]}
    raise ValueError(f"Llamada desconocida de RewardModel: {call!r}")


class RewardModelClient(RewardModel):
    def __init__(self, address: str, timeout_s: float = 10.0, retries: int = 1) -> None:
        self._channel = RpcChannel(address, timeout_s, retries)

    def score(self, items: list[ScoreItem]) -> list[float]:
        if not items:
            return []
        response = self._channel.call({"call": "score", "items": [_item_to_wire(i) for i in items]})
        return [float(r) for r in response["rewards"]]

    def close(self) -> None:
        self._channel.close()
