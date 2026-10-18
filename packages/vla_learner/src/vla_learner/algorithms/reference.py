"""``ReferencePolicy``: copia congelada del modelo para el término KL de GRPO.

Duplica la memoria del modelo, así que solo se crea si ``kl_coef > 0``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from vla_learner.policy.interfaces import VLAPolicy
from vla_learner.types import ModelInput


class ReferencePolicy:
    def __init__(self, policy: VLAPolicy) -> None:
        self.model = policy.clone_frozen()

    @torch.no_grad()
    def log_probs(
        self,
        inputs: ModelInput,
        actions: Tensor,
        extras: dict[str, Tensor] | None = None,
    ) -> Tensor:
        return self.model.evaluate_actions(inputs, actions, extras).log_probs
