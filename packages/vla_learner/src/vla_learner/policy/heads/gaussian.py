"""``GaussianActionHead``: media y desviación por dimensión.

Capacidades: ``LOG_PROB`` y ``ENTROPY``. Loss de BC = log-verosimilitud
negativa. Trabaja con acciones normalizadas (``ComposedPolicy`` normaliza antes
y desnormaliza después). La desviación es un parámetro aprendible independiente
del estado (habitual en políticas gaussianas).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vla_learner.policy.interfaces import ActionHead, Capability
from vla_learner.types import LossOutput


class GaussianActionHead(ActionHead):
    def __init__(self, feature_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        self.mean_proj = nn.Linear(feature_dim, chunk_size * action_dim)
        self.log_std = nn.Parameter(torch.zeros(chunk_size, action_dim))

    @property
    def capabilities(self) -> set[Capability]:
        return {Capability.LOG_PROB, Capability.ENTROPY}

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _distribution(self, features: Tensor) -> torch.distributions.Normal:
        pooled = features.mean(dim=1)  # [B, F]
        mean = self.mean_proj(pooled).reshape(-1, self._chunk_size, self._action_dim)
        return torch.distributions.Normal(mean, self.log_std.exp())

    def sample(self, features: Tensor, deterministic: bool) -> tuple[Tensor, Tensor | None, dict]:
        dist = self._distribution(features)
        actions = dist.mean if deterministic else dist.rsample()
        log_probs = dist.log_prob(actions).sum(dim=(1, 2))
        return actions, log_probs, {}

    def log_prob(self, features: Tensor, actions: Tensor, extras: dict | None) -> Tensor:
        return self._distribution(features).log_prob(actions).sum(dim=(1, 2))

    def entropy(self, features: Tensor) -> Tensor:
        return self._distribution(features).entropy().sum(dim=(1, 2))

    def bc_loss(self, features: Tensor, actions: Tensor) -> LossOutput:
        nll = -self.log_prob(features, actions, None).mean()
        return LossOutput(loss=nll, metrics={"nll": float(nll.detach())})
