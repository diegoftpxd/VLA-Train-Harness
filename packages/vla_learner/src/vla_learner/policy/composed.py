"""``ComposedPolicy``: implementa ``VLAPolicy`` con backbone + cabezas.

Definido en ``docs/learner/Learner Design.md``. La mayoría de los VLAs se ven como
backbone + cabeza de acción (+ cabeza de valor opcional). Esta clase arma una
``VLAPolicy`` a partir de esas piezas, normalizando las acciones con
``ActionNormalizer``: ``act`` desnormaliza la salida de la cabeza, y
``evaluate_actions``/``imitation_loss`` normalizan las acciones de entrada.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from vla_learner.policy.heads.value import ValueHead
from vla_learner.policy.interfaces import (
    ActionHead,
    Backbone,
    Capability,
    ObservationProcessor,
    VLAPolicy,
)
from vla_learner.policy.normalizer import ActionNormalizer
from vla_learner.types import ActionEvaluation, LossOutput, ModelInput, PolicyOutput


class ComposedPolicy(VLAPolicy):
    def __init__(
        self,
        backbone: Backbone,
        action_head: ActionHead,
        processor: ObservationProcessor,
        normalizer: ActionNormalizer,
        value_head: ValueHead | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.action_head = action_head
        self.processor = processor
        self.normalizer = normalizer
        self.value_head = value_head
        self._embodiment = normalizer.default_embodiment()

    @property
    def capabilities(self) -> set[Capability]:
        caps = set(self.action_head.capabilities)
        if self.value_head is not None:
            caps.add(Capability.VALUE)
        return caps

    @property
    def chunk_size(self) -> int:
        return self.action_head.chunk_size

    @property
    def action_dim(self) -> int:
        return self.action_head.action_dim

    def act(self, inputs: ModelInput, deterministic: bool = False) -> PolicyOutput:
        with torch.no_grad():
            features = self.backbone.encode(inputs)
            actions_norm, log_probs, extras = self.action_head.sample(features, deterministic)
            actions = self.normalizer.denormalize(actions_norm, self._embodiment)
            values = self.value_head(features) if self.value_head is not None else None
        return PolicyOutput(actions=actions, log_probs=log_probs, values=values, extras=extras)

    def evaluate_actions(
        self,
        inputs: ModelInput,
        actions: Tensor,
        extras: dict[str, Tensor] | None = None,
    ) -> ActionEvaluation:
        actions_norm = self.normalizer.normalize(actions, self._embodiment)
        features = self.backbone.encode(inputs)
        log_probs = self.action_head.log_prob(features, actions_norm, extras)
        entropy = (
            self.action_head.entropy(features)
            if Capability.ENTROPY in self.action_head.capabilities
            else None
        )
        values = self.value_head(features) if self.value_head is not None else None
        return ActionEvaluation(log_probs=log_probs, entropy=entropy, values=values)

    def imitation_loss(self, inputs: ModelInput, actions: Tensor) -> LossOutput:
        actions_norm = self.normalizer.normalize(actions, self._embodiment)
        features = self.backbone.encode(inputs)
        return self.action_head.bc_loss(features, actions_norm)

    def save_pretrained(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self, directory / "policy.pt")

    @classmethod
    def from_pretrained(cls, directory: Path) -> ComposedPolicy:
        policy = torch.load(Path(directory) / "policy.pt", weights_only=False)
        assert isinstance(policy, cls)
        return policy
