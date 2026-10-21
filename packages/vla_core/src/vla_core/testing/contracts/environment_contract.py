"""Suite ENV: contrato de ``Environment``.

La cumplen el ``FakeEnvironment`` (en memoria) y el ``EnvironmentClient``
(contra un servidor fake por ZeroMQ). La implementación entrega el fixture
``environment``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_common.types import Action, ActionChunk
from vla_core.environments import Environment


class EnvironmentContract:
    @pytest.fixture
    def environment(self) -> Environment:
        raise NotImplementedError

    def _chunk(self, environment: Environment) -> ActionChunk:
        return [Action(values=np.zeros(environment.spec.action_dim, dtype=np.float32))]

    def test_env01_reset_devuelve_observacion(self, environment: Environment) -> None:
        obs = environment.reset(task=None, seed=1)
        assert obs.proprio.dtype == np.float32
        for camara in environment.spec.cameras:
            assert camara in obs.images

    def test_env02_step_sin_recompensa(self, environment: Environment) -> None:
        environment.reset(task=None, seed=1)
        resultado = environment.step(self._chunk(environment))
        assert resultado.reward is None  # la recompensa la agrega Core
        assert isinstance(resultado.terminated, bool)
        assert isinstance(resultado.truncated, bool)

    def test_env03_spec_valido(self, environment: Environment) -> None:
        assert environment.spec.action_dim > 0
        assert environment.spec.cameras

    def test_env04_close_y_parada_no_fallan(self, environment: Environment) -> None:
        environment.reset(task=None, seed=0)
        environment.emergency_stop()
        environment.close()

    def test_env05_reset_determinista(self, environment: Environment) -> None:
        a = environment.reset(task=None, seed=7)
        b = environment.reset(task=None, seed=7)
        assert np.array_equal(a.proprio, b.proprio)
