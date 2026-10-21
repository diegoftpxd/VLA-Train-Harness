"""``PPO``: Proximal Policy Optimization con ventajas de un ``AdvantageEstimator``.

Definido en ``docs/learner/Learner Design.md``. La pérdida de política usa el objetivo
recortado; la de valor, el error cuadrático (con ``value_clip`` opcional). Corta
las épocas cuando el KL aproximado supera ``target_kl``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from torch import Tensor

from vla_learner.algorithms.advantages import AdvantageEstimator
from vla_learner.algorithms.base import Algorithm
from vla_learner.policy.interfaces import Capability, VLAPolicy
from vla_learner.types import Batch, LossOutput, RolloutBatch

_EPS = 1e-8


def clipped_surrogate(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    clip_ratio: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Objetivo de política recortado de PPO.

    Devuelve (policy_loss, approx_kl, clip_fraction). Para las filas recortadas
    (cociente fuera de ``[1-ε, 1+ε]`` en la dirección que no mejora) el término
    ``clip`` es constante en ``log_probs``, así que no aporta gradiente.
    """
    log_ratio = log_probs - old_log_probs
    ratio = log_ratio.exp()
    surrogate_1 = ratio * advantages
    surrogate_2 = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
    with torch.no_grad():
        approx_kl = (ratio - 1.0 - log_ratio).mean()  # estimador k3 de Schulman
        clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
    return policy_loss, approx_kl, clip_fraction


def value_loss(
    values: Tensor,
    returns: Tensor,
    old_values: Tensor | None,
    value_clip: float | None,
) -> Tensor:
    """Error cuadrático de la función de valor, con recorte opcional."""
    sin_recorte = (values - returns) ** 2
    if value_clip is None:
        return 0.5 * sin_recorte.mean()
    if old_values is None:
        raise ValueError("value_clip requiere old_values en el batch.")
    recortado = old_values + (values - old_values).clamp(-value_clip, value_clip)
    con_recorte = (recortado - returns) ** 2
    return 0.5 * torch.max(sin_recorte, con_recorte).mean()


@dataclass
class PPOConfig:
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    update_epochs: int = 4
    minibatch_size: int = 64
    value_clip: float | None = None
    target_kl: float | None = None  # corta las épocas si el KL se pasa


class PPO(Algorithm):
    def __init__(self, config: PPOConfig, advantage_estimator: AdvantageEstimator) -> None:
        self.config = config
        self.advantage_estimator = advantage_estimator

    @property
    def required_capabilities(self) -> set[Capability]:
        return {Capability.LOG_PROB, Capability.VALUE}

    @property
    def batch_type(self) -> type[Batch]:
        return RolloutBatch

    @property
    def update_epochs(self) -> int:
        return self.config.update_epochs

    @property
    def minibatch_size(self) -> int | None:
        return self.config.minibatch_size

    def prepare(self, batch: Batch, policy: VLAPolicy) -> Batch:
        assert isinstance(batch, RolloutBatch)
        with torch.no_grad():
            advantages, returns = self.advantage_estimator.compute(batch)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + _EPS)
        return dataclasses.replace(batch, advantages=advantages, returns=returns)

    def loss(self, policy: VLAPolicy, batch: Batch) -> LossOutput:
        assert isinstance(batch, RolloutBatch)
        assert batch.advantages is not None and batch.returns is not None, "falta prepare()"
        evaluation = policy.evaluate_actions(batch.inputs, batch.actions, batch.extras)
        assert evaluation.values is not None, "PPO exige una cabeza de valor"

        policy_loss, approx_kl, clip_fraction = clipped_surrogate(
            evaluation.log_probs, batch.old_log_probs, batch.advantages, self.config.clip_ratio
        )
        v_loss = value_loss(
            evaluation.values, batch.returns, batch.old_values, self.config.value_clip
        )
        if evaluation.entropy is not None:
            entropy = evaluation.entropy.mean()
        else:
            entropy = torch.zeros((), dtype=policy_loss.dtype)

        total = policy_loss + self.config.value_coef * v_loss - self.config.entropy_coef * entropy
        metrics = {
            "loss": float(total.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(v_loss.detach()),
            "entropy": float(entropy.detach()),
            "approx_kl": float(approx_kl),
            "clip_fraction": float(clip_fraction),
        }
        return LossOutput(loss=total, metrics=metrics)

    def should_stop_epochs(self, metrics: dict[str, float]) -> bool:
        return self.config.target_kl is not None and metrics["approx_kl"] > self.config.target_kl
