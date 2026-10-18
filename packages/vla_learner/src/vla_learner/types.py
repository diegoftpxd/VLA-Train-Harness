"""Tipos internos del Learner, agrupados en batch con tensores de torch.

Definidos en ``docs/Learner Design.md`` (sección "Tipos internos"). Los tipos
del protocolo (``Observation``, ``Action``, ``ActionChunk``, ``Transition``)
viven en ``vla_common``; aquí se agregan los que trabajan con tensores.

Convenciones que atraviesan todo el Learner:

- **Un chunk es un paso.** La log-prob de un chunk es la suma de las de sus
  acciones; PPO y GRPO operan sobre esa suma.
- **Fuera de ``VLAPolicy``, las acciones están en unidades reales.**
- **``extras`` es el espacio propio del modelo**, que el resto no interpreta.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor


@dataclass
class ModelInput:
    """Salida de ``ObservationProcessor``. Su contenido depende del modelo."""

    tensors: dict[str, Tensor]
    batch_size: int


@dataclass
class PolicyOutput:
    actions: Tensor  # [B, H, D] en unidades reales; H = largo del chunk
    log_probs: Tensor | None  # [B], log-prob del chunk completo
    values: Tensor | None  # [B], si el modelo tiene cabeza de valor
    extras: dict[str, Tensor] = field(default_factory=dict)  # datos propios del modelo


@dataclass
class ActionEvaluation:
    log_probs: Tensor  # [B]
    entropy: Tensor | None  # [B]
    values: Tensor | None  # [B]


@dataclass
class LossOutput:
    loss: Tensor
    metrics: dict[str, float]


def _select_tensor(tensor: Tensor, indices: Tensor) -> Tensor:
    """Indexa la dimensión 0 de ``tensor`` con ``indices`` (los mueve a su device)."""
    return tensor.index_select(0, indices.to(device=tensor.device, dtype=torch.long))


def _select_tensor_opt(tensor: Tensor | None, indices: Tensor) -> Tensor | None:
    return None if tensor is None else _select_tensor(tensor, indices)


def _select_model_input(inputs: ModelInput, indices: Tensor) -> ModelInput:
    return ModelInput(
        tensors={name: _select_tensor(t, indices) for name, t in inputs.tensors.items()},
        batch_size=int(indices.shape[0]),
    )


class Batch(ABC):
    """Todo batch sabe su largo y cómo seleccionar un subconjunto de filas."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def select(self, indices: Tensor) -> Batch: ...


@dataclass
class SupervisedBatch(Batch):
    inputs: ModelInput
    actions: Tensor  # [B, H, D] en unidades reales

    def __len__(self) -> int:
        return self.inputs.batch_size

    def select(self, indices: Tensor) -> SupervisedBatch:
        return SupervisedBatch(
            inputs=_select_model_input(self.inputs, indices),
            actions=_select_tensor(self.actions, indices),
        )


@dataclass
class RolloutBatch(Batch):
    inputs: ModelInput
    actions: Tensor  # [B, H, D] en unidades reales
    old_log_probs: Tensor  # [B], al momento de actuar
    rewards: Tensor  # [B]
    terminated: Tensor  # [B] bool
    truncated: Tensor  # [B] bool
    episode_ids: list[str]
    step_ids: Tensor  # [B], para ordenar dentro del episodio
    source: Literal["env", "world_model"]
    old_values: Tensor | None = None
    group_ids: Tensor | None = None  # para GRPO
    extras: dict[str, Tensor] = field(default_factory=dict)  # copiados de PolicyOutput
    advantages: Tensor | None = None  # lo llena Algorithm.prepare
    returns: Tensor | None = None  # lo llena Algorithm.prepare

    def __len__(self) -> int:
        return self.inputs.batch_size

    def select(self, indices: Tensor) -> RolloutBatch:
        return RolloutBatch(
            inputs=_select_model_input(self.inputs, indices),
            actions=_select_tensor(self.actions, indices),
            old_log_probs=_select_tensor(self.old_log_probs, indices),
            rewards=_select_tensor(self.rewards, indices),
            terminated=_select_tensor(self.terminated, indices),
            truncated=_select_tensor(self.truncated, indices),
            episode_ids=[self.episode_ids[i] for i in indices.to(torch.long).tolist()],
            step_ids=_select_tensor(self.step_ids, indices),
            source=self.source,
            old_values=_select_tensor_opt(self.old_values, indices),
            group_ids=_select_tensor_opt(self.group_ids, indices),
            extras={name: _select_tensor(t, indices) for name, t in self.extras.items()},
            advantages=_select_tensor_opt(self.advantages, indices),
            returns=_select_tensor_opt(self.returns, indices),
        )
