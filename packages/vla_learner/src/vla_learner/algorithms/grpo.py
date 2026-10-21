"""``GRPO``: Group Relative Policy Optimization.

Definido en ``docs/learner/Learner Design.md``. No usa cabeza de valor: compara cada
rollout con otros del mismo grupo (mismo punto de partida). Reusa el objetivo
recortado de PPO y, opcionalmente, agrega un término KL contra una referencia
congelada.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from torch import Tensor

from vla_learner.algorithms.advantages import GroupRelativeAdvantage
from vla_learner.algorithms.base import Algorithm
from vla_learner.algorithms.ppo import clipped_surrogate
from vla_learner.algorithms.reference import ReferencePolicy
from vla_learner.policy.interfaces import Capability, VLAPolicy
from vla_learner.types import Batch, LossOutput, RolloutBatch


def kl_estimate(log_probs: Tensor, reference_log_probs: Tensor) -> Tensor:
    """Estimador k3 de KL(policy || reference), no negativo y 0 cuando coinciden."""
    log_ratio = reference_log_probs - log_probs
    return (log_ratio.exp() - log_ratio - 1.0).mean()


@dataclass
class GRPOConfig:
    group_size: int = 8
    clip_ratio: float = 0.2
    kl_coef: float = 0.0  # 0 = sin ReferencePolicy
    update_epochs: int = 1
    minibatch_size: int = 64


class GRPO(Algorithm):
    def __init__(
        self,
        config: GRPOConfig,
        advantage_estimator: GroupRelativeAdvantage,
        reference: ReferencePolicy | None = None,
    ) -> None:
        if config.kl_coef > 0 and reference is None:
            raise ValueError("GRPO con kl_coef > 0 necesita una ReferencePolicy.")
        self.config = config
        self.advantage_estimator = advantage_estimator
        self.reference = reference

    @property
    def required_capabilities(self) -> set[Capability]:
        return {Capability.LOG_PROB}

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
        if batch.group_ids is None:
            raise ValueError("GRPO necesita group_ids en el batch.")
        with torch.no_grad():
            advantages, returns = self.advantage_estimator.compute(batch)
        return dataclasses.replace(batch, advantages=advantages, returns=returns)

    def loss(self, policy: VLAPolicy, batch: Batch) -> LossOutput:
        assert isinstance(batch, RolloutBatch)
        assert batch.advantages is not None, "falta prepare()"
        evaluation = policy.evaluate_actions(batch.inputs, batch.actions, batch.extras)
        policy_loss, approx_kl, clip_fraction = clipped_surrogate(
            evaluation.log_probs, batch.old_log_probs, batch.advantages, self.config.clip_ratio
        )
        total = policy_loss
        metrics = {
            "policy_loss": float(policy_loss.detach()),
            "approx_kl": float(approx_kl),
            "clip_fraction": float(clip_fraction),
        }
        if self.config.kl_coef > 0:
            assert self.reference is not None
            reference_log_probs = self.reference.log_probs(
                batch.inputs, batch.actions, batch.extras
            )
            kl = kl_estimate(evaluation.log_probs, reference_log_probs)
            total = total + self.config.kl_coef * kl
            metrics["kl"] = float(kl.detach())
        metrics["loss"] = float(total.detach())
        return LossOutput(loss=total, metrics=metrics)
