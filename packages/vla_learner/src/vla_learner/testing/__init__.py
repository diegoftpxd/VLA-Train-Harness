"""Herramientas de prueba compartidas del Learner.

Las suites de contrato viven en ``vla_learner.testing.contracts`` para que las
implementaciones futuras las importen y hereden (ver CLAUDE.md).
"""

from vla_learner.testing.data import make_action_chunks, make_observations
from vla_learner.testing.fake_core import FakeCoreClient
from vla_learner.testing.stub_policy import StubPolicy, StubProcessor
from vla_learner.testing.tiny_backbone import TinyBackbone
from vla_learner.testing.toy_env import ToyReachEnv, ToyWorldModel

__all__ = [
    "FakeCoreClient",
    "StubPolicy",
    "StubProcessor",
    "TinyBackbone",
    "ToyReachEnv",
    "ToyWorldModel",
    "make_action_chunks",
    "make_observations",
]
