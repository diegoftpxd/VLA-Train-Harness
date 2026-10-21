"""Registro de ambientes: resolución por ``env_id`` y error si falta."""

from __future__ import annotations

import pytest

from vla_core.environments import EnvironmentRegistry
from vla_core.errors import UnknownEnvironment
from vla_core.testing.fakes import FakeEnvironment

pytestmark = pytest.mark.fast


def test_registro_resuelve_y_lista() -> None:
    registry = EnvironmentRegistry()
    env0 = FakeEnvironment()
    env1 = FakeEnvironment()
    registry.register("sim-0", env0)
    registry.register("sim-1", env1)
    assert registry.get("sim-0") is env0
    assert registry.available() == ["sim-0", "sim-1"]


def test_registro_ambiente_desconocido_es_error() -> None:
    registry = EnvironmentRegistry()
    with pytest.raises(UnknownEnvironment):
        registry.get("no-existe")
