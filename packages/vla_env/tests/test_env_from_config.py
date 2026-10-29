"""Suites ENV y SRVE contra cualquier ambiente construible desde un YAML (``heavy``).

    VLA_ENV_CONFIG=configs/env/<nombre>.yaml uv run pytest -m heavy \\
        packages/vla_env/tests/test_env_from_config.py

Así un ambiente nuevo (LIBERO, un robot) se prueba sin escribir tests: basta su
YAML. La acción neutra sale de ``environment.smoke_action``. Ver
``docs/environment/Test Environment.md``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from vla_common.environment import Environment
from vla_common.testing.contracts import EnvironmentContract
from vla_core.environments import EnvironmentClient
from vla_env.cli import load_config
from vla_env.factory import build_environment
from vla_env.server import EnvironmentServer

pytestmark = pytest.mark.heavy

RUTA = os.environ.get("VLA_ENV_CONFIG")
if RUTA is None:
    pytest.skip("definí VLA_ENV_CONFIG con el YAML del ambiente", allow_module_level=True)
BLOQUE = load_config(Path(RUTA))["environment"]


@pytest.fixture(scope="module")
def ambiente() -> Iterator[Environment]:
    # Uno por módulo: crear un simulador real (MuJoCo, un robot) es caro.
    env = build_environment(BLOQUE)
    try:
        yield env
    finally:
        env.close()


def _neutra(env: Environment) -> np.ndarray:
    accion = BLOQUE.get("smoke_action")
    if accion is None:
        return np.zeros(env.spec.action_dim, dtype=np.float32)
    return np.asarray(accion, dtype=np.float32)


class TestEnvironmentDesdeConfig(EnvironmentContract):
    @pytest.fixture
    def environment(self, ambiente: Environment) -> Environment:
        return ambiente

    @pytest.fixture
    def neutral_action(self, environment: Environment) -> np.ndarray:
        return _neutra(environment)


class TestEnvironmentDesdeConfigPorLaRed(EnvironmentContract):
    """SRVE-01: el mismo ambiente, servido por ``EnvironmentServer``."""

    @pytest.fixture
    def environment(self, ambiente: Environment) -> Iterator[Environment]:
        with EnvironmentServer(ambiente) as server:
            cliente = EnvironmentClient(server.address, timeout_s=120.0, retries=0)
            yield cliente
            cliente._channel.close()  # noqa: SLF001 - el ambiente lo cierra el fixture de módulo

    @pytest.fixture
    def neutral_action(self, environment: Environment) -> np.ndarray:
        return _neutra(environment)


def test_cambiar_de_tarea_y_volver(ambiente: Environment) -> None:
    """Recrear el simulador al cambiar de tarea mantiene instrucción y determinismo."""
    tareas = ambiente.spec.tasks
    if len(tareas) < 2:
        pytest.skip("el ambiente tiene una sola tarea")
    a, b = tareas[0], tareas[1]
    primera = ambiente.reset(task=a, seed=3)
    assert ambiente.reset(task=b, seed=3).instruction == b
    vuelta = ambiente.reset(task=a, seed=3)
    assert vuelta.instruction == a
    assert np.array_equal(primera.proprio, vuelta.proprio)
