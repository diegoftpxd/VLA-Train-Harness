"""``ValueHead``: features → V(s)."""

from __future__ import annotations

from torch import Tensor, nn


class ValueHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        """``V(s)`` ``[B]`` a partir de features ``[B, T, F]`` (promediadas sobre T)."""
        pooled = features.mean(dim=1)
        return self.net(pooled).squeeze(-1)
