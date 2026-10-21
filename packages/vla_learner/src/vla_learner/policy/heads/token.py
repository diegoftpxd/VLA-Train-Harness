"""``TokenActionHead``: acciones discretizadas como tokens.

Capacidades: ``LOG_PROB`` y ``ENTROPY``. Loss de BC = cross-entropy. Cada
dimensión de la acción (normalizada en ``[low, high]``) se discretiza en
``bins`` cubetas. La discretización es estable: token → acción → token devuelve
el mismo token (HEAD-04).

Nota de diseño: el modelo real (p. ej. OpenVLA) genera los tokens de forma
autoregresiva. Esta cabeza de referencia los predice de forma independiente por
posición, lo cual basta para los contratos; un modelo autoregresivo entra como
adaptador (ver ``docs/Learner Design.md``). Para evitar errores de redondeo al
re-tokenizar en ``evaluate_actions``, los ids elegidos viajan en ``extras``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vla_learner.policy.interfaces import ActionHead, Capability
from vla_learner.types import LossOutput


class TokenActionHead(ActionHead):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        chunk_size: int,
        bins: int = 51,
        low: float = -1.0,
        high: float = 1.0,
    ) -> None:
        super().__init__()
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        self.bins = bins
        self.low = low
        self.high = high
        self.logits_proj = nn.Linear(feature_dim, chunk_size * action_dim * bins)

    @property
    def capabilities(self) -> set[Capability]:
        return {Capability.LOG_PROB, Capability.ENTROPY}

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def tokenize(self, actions: Tensor) -> Tensor:
        """Acciones normalizadas ``[B, H, D]`` → tokens enteros ``[B, H, D]``."""
        escala = (actions - self.low) / (self.high - self.low) * (self.bins - 1)
        return escala.round().clamp(0, self.bins - 1).long()

    def detokenize(self, tokens: Tensor) -> Tensor:
        """Tokens ``[B, H, D]`` → acciones normalizadas en el centro de la cubeta."""
        return self.low + tokens.float() / (self.bins - 1) * (self.high - self.low)

    def _logits(self, features: Tensor) -> Tensor:
        pooled = features.mean(dim=1)  # [B, F]
        logits = self.logits_proj(pooled)
        return logits.reshape(-1, self._chunk_size, self._action_dim, self.bins)

    def _gather_log_prob(self, logits: Tensor, tokens: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        elegido = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return elegido.sum(dim=(1, 2))

    def sample(self, features: Tensor, deterministic: bool) -> tuple[Tensor, Tensor | None, dict]:
        logits = self._logits(features)
        if deterministic:
            tokens = logits.argmax(dim=-1)
        else:
            tokens = torch.distributions.Categorical(logits=logits).sample()
        actions = self.detokenize(tokens)
        log_probs = self._gather_log_prob(logits, tokens)
        return actions, log_probs, {"token_ids": tokens}

    def log_prob(self, features: Tensor, actions: Tensor, extras: dict | None) -> Tensor:
        logits = self._logits(features)
        if extras is not None and "token_ids" in extras:
            tokens = extras["token_ids"]
        else:
            tokens = self.tokenize(actions)
        return self._gather_log_prob(logits, tokens)

    def entropy(self, features: Tensor) -> Tensor:
        logits = self._logits(features)
        return torch.distributions.Categorical(logits=logits).entropy().sum(dim=(1, 2))

    def bc_loss(self, features: Tensor, actions: Tensor) -> LossOutput:
        logits = self._logits(features)
        tokens = self.tokenize(actions)
        ce = -self._gather_log_prob(logits, tokens).mean()
        return LossOutput(loss=ce, metrics={"cross_entropy": float(ce.detach())})
