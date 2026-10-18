"""Modelo del Learner: interfaces y, en fases posteriores, implementaciones.

En la Fase 2 solo están las interfaces (``interfaces.py``), adelantadas porque
las herramientas de prueba ``StubPolicy`` y ``TinyBackbone`` las implementan.
Las piezas concretas (``ComposedPolicy``, cabezas, ``ActionNormalizer`` y un
procesador real) llegan en la Fase 3.
"""

from vla_learner.policy.interfaces import (
    ActionHead,
    Backbone,
    Capability,
    ObservationProcessor,
    VLAPolicy,
)

__all__ = [
    "ActionHead",
    "Backbone",
    "Capability",
    "ObservationProcessor",
    "VLAPolicy",
]
