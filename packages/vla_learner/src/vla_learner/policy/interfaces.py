"""Interfaces del modelo del Learner.

Definidas en ``docs/learner/Learner Design.md`` (sección "Modelo"). Son las únicas que
el resto del Learner usa: todo modelo concreto (``ComposedPolicy`` o un
adaptador) las implementa. En la Fase 2 se adelantan porque ``StubPolicy`` y
``TinyBackbone`` (herramientas de prueba) las implementan.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

import torch
from torch import Tensor, nn

from vla_common import Observation
from vla_learner.types import ActionEvaluation, LossOutput, ModelInput, PolicyOutput


class Capability(Enum):
    """Qué sabe hacer una política; el Learner las revisa al arrancar."""

    LOG_PROB = "log_prob"  # puede calcular log π(a|s)
    VALUE = "value"  # tiene cabeza de valor
    ENTROPY = "entropy"  # puede calcular entropía


class ObservationProcessor(ABC):
    """Convierte observaciones del formato común al formato del modelo."""

    @abstractmethod
    def __call__(self, observations: list[Observation]) -> ModelInput: ...

    @property
    @abstractmethod
    def required_cameras(self) -> list[str]: ...

    def to(self, device: torch.device) -> ObservationProcessor:
        return self


class Backbone(nn.Module, ABC):
    """Visión + lenguaje + propiocepción → features."""

    @abstractmethod
    def encode(self, inputs: ModelInput) -> Tensor:
        """Features ``[B, T, F]`` a partir de imágenes, propiocepción e instrucción."""

    @property
    @abstractmethod
    def feature_dim(self) -> int: ...


class ActionHead(nn.Module, ABC):
    """Features → acciones. Trabaja siempre con acciones normalizadas;
    ``ComposedPolicy`` normaliza antes y desnormaliza después.
    """

    @property
    @abstractmethod
    def capabilities(self) -> set[Capability]: ...

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Largo del chunk (H) que genera esta cabeza."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensión de cada acción (D)."""

    @abstractmethod
    def sample(self, features: Tensor, deterministic: bool) -> tuple[Tensor, Tensor | None, dict]:
        """Devuelve (acciones normalizadas ``[B, H, D]``, log_probs ``[B]`` o None, extras)."""

    @abstractmethod
    def log_prob(self, features: Tensor, actions: Tensor, extras: dict | None) -> Tensor: ...

    @abstractmethod
    def entropy(self, features: Tensor) -> Tensor: ...

    @abstractmethod
    def bc_loss(self, features: Tensor, actions: Tensor) -> LossOutput: ...


class VLAPolicy(nn.Module, ABC):
    """Interfaz única del modelo: actuar, evaluar acciones y loss de imitación."""

    processor: ObservationProcessor

    @property
    @abstractmethod
    def capabilities(self) -> set[Capability]: ...

    @property
    @abstractmethod
    def chunk_size(self) -> int: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @abstractmethod
    def act(self, inputs: ModelInput, deterministic: bool = False) -> PolicyOutput:
        """Genera acciones en unidades reales. Se llama sin gradientes en los rollouts."""

    @abstractmethod
    def evaluate_actions(
        self,
        inputs: ModelInput,
        actions: Tensor,
        extras: dict[str, Tensor] | None = None,
    ) -> ActionEvaluation:
        """Log-prob, entropía y valor de acciones ya tomadas (unidades reales). Con gradientes."""

    @abstractmethod
    def imitation_loss(self, inputs: ModelInput, actions: Tensor) -> LossOutput:
        """Loss de BC propia del modelo (cross-entropy, denoising, flow matching...)."""

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def clone_frozen(self) -> VLAPolicy:
        """Copia sin gradientes, para ``ReferencePolicy``."""
        clone = copy.deepcopy(self)
        clone.eval()
        for parameter in clone.parameters():
            parameter.requires_grad_(False)
        return clone

    @abstractmethod
    def save_pretrained(self, directory: Path) -> None:
        """Guarda pesos, configuración y estadísticas de normalización."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, directory: Path) -> VLAPolicy: ...
