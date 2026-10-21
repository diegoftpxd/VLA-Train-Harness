"""Herramientas de prueba de Core: fakes de componentes y el backend estático."""

from vla_core.testing.fakes import (
    FakeDataset,
    FakeEnvironment,
    FakeEnvironmentRegistry,
    FakeRewardModel,
    FakeWorldModel,
    make_fake_core,
    serve_environment,
    serve_reward_model,
    serve_world_model,
)
from vla_core.testing.static_backend import StaticCoreBackend

__all__ = [
    "FakeDataset",
    "FakeEnvironment",
    "FakeEnvironmentRegistry",
    "FakeRewardModel",
    "FakeWorldModel",
    "StaticCoreBackend",
    "make_fake_core",
    "serve_environment",
    "serve_reward_model",
    "serve_world_model",
]
