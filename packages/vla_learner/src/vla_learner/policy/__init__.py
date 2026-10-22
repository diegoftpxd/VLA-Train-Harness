"""Modelo del Learner: interfaces e implementaciones.

Fase 2: interfaces (``interfaces.py``), adelantadas porque ``StubPolicy`` y
``TinyBackbone`` las implementan. Fase 3: piezas concretas (``ComposedPolicy``,
``GaussianActionHead``, ``TokenActionHead``, ``ValueHead``, ``ActionNormalizer``
y ``SimpleProcessor``).
"""

from vla_learner.policy.composed import ComposedPolicy
from vla_learner.policy.heads import (
    FlowMatchingActionHead,
    GaussianActionHead,
    TokenActionHead,
    ValueHead,
)
from vla_learner.policy.interfaces import (
    ActionHead,
    Backbone,
    Capability,
    ObservationProcessor,
    VLAPolicy,
)
from vla_learner.policy.normalizer import ActionNormalizer, ActionStats
from vla_learner.policy.processors import SimpleProcessor

__all__ = [
    "ActionHead",
    "ActionNormalizer",
    "ActionStats",
    "Backbone",
    "Capability",
    "ComposedPolicy",
    "FlowMatchingActionHead",
    "GaussianActionHead",
    "ObservationProcessor",
    "SimpleProcessor",
    "TokenActionHead",
    "VLAPolicy",
    "ValueHead",
]
