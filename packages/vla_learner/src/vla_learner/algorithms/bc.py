"""``BehaviorCloning``: imitación. La loss la pone el modelo.

Funciona con cualquier modelo porque delega en ``policy.imitation_loss``.
"""

from __future__ import annotations

from vla_learner.algorithms.base import Algorithm
from vla_learner.policy.interfaces import Capability, VLAPolicy
from vla_learner.types import Batch, LossOutput, SupervisedBatch


class BehaviorCloning(Algorithm):
    @property
    def required_capabilities(self) -> set[Capability]:
        return set()

    @property
    def batch_type(self) -> type[Batch]:
        return SupervisedBatch

    def loss(self, policy: VLAPolicy, batch: Batch) -> LossOutput:
        assert isinstance(batch, SupervisedBatch)
        return policy.imitation_loss(batch.inputs, batch.actions)
