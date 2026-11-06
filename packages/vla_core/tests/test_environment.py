"""Suite ENV contra el ``FakeEnvironment`` y contra el ``EnvironmentClient``.

Más las reglas propias del fake que la suite genérica no puede forzar (FAKE-*):
el corte a mitad de chunk y ``info["success"]``. Ver
``docs/environment/Test Environment.md``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from vla_common.environment import Environment
from vla_common.testing.contracts import EnvironmentContract
from vla_common.types import Action
from vla_core.environments import EnvironmentClient
from vla_core.testing.fakes import MAX_STEPS, FakeEnvironment, serve_environment

pytestmark = pytest.mark.fast


class TestFakeEnvironment(EnvironmentContract):
    @pytest.fixture
    def environment(self) -> Environment:
        return FakeEnvironment()


class TestEnvironmentClient(EnvironmentContract):
    @pytest.fixture
    def environment(self) -> Iterator[Environment]:
        server = serve_environment(FakeEnvironment())
        address = server.start()
        client = EnvironmentClient(address, timeout_s=5.0)
        try:
            yield client
        finally:
            client.close()
            server.stop()


def _accion(valores: np.ndarray) -> Action:
    return Action(values=np.asarray(valores, dtype=np.float32))


def test_fake01_corta_el_chunk_al_alcanzar_la_meta() -> None:
    env = FakeEnvironment()
    obs = env.reset(task=None, seed=4)
    pos, goal = obs.proprio[:2], obs.proprio[2:4]
    # Primera acción: lleva el punto exactamente a la meta. Las dos siguientes lo
    # alejarían; si el fake no cortara, la observación final no estaría en la meta.
    lejos = np.where(goal > 0, -0.5, 0.5)
    resultado = env.step([_accion(goal - pos), _accion(lejos), _accion(lejos)])
    assert resultado.terminated and resultado.info["success"]
    assert np.allclose(resultado.observation.proprio[:2], goal, atol=1e-6)


def test_fake02_success_solo_si_alcanza_la_meta() -> None:
    env = FakeEnvironment()
    env.reset(task=None, seed=4)
    quieto = [_accion(np.zeros(2))]
    resultado = env.step(quieto)
    while not (resultado.terminated or resultado.truncated):
        resultado = env.step(quieto)
    assert env._steps == MAX_STEPS  # noqa: SLF001 - cortó justo en el límite
    assert resultado.truncated and not resultado.terminated
    assert resultado.info["success"] is False
