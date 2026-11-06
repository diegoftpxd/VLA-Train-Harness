"""Suite SRVE: un ambiente servido por ``EnvironmentServer`` y consumido por el
``EnvironmentClient`` real de Core (``docs/environment/Test Environment.md``).

``vla_env`` no depende de ``vla_core`` en runtime; en los tests sí lo importa,
para probar el cliente real contra el servidor real.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import zmq

from vla_common.environment import Environment
from vla_common.testing.contracts import EnvironmentContract
from vla_common.types import Action
from vla_core.environments import EnvironmentClient
from vla_env.reach import ReachEnvironment
from vla_env.server import EnvironmentServer

pytestmark = pytest.mark.fast


@pytest.fixture
def servido() -> Iterator[EnvironmentClient]:
    with EnvironmentServer(ReachEnvironment()) as server:
        client = EnvironmentClient(server.address, timeout_s=5.0)
        try:
            yield client
        finally:
            client.close()


class TestReachPorLaRed(EnvironmentContract):
    """SRVE-01: la suite ENV completa, a través del wire."""

    @pytest.fixture
    def environment(self, servido: EnvironmentClient) -> Environment:
        return servido


def test_srve02_sin_perdida_por_el_wire(servido: EnvironmentClient) -> None:
    local = ReachEnvironment()
    remoto, en_memoria = servido.reset(None, 5), local.reset(None, 5)
    assert np.array_equal(remoto.proprio, en_memoria.proprio)
    assert np.array_equal(remoto.images["front"], en_memoria.images["front"])
    accion = [Action(values=np.array([0.123456789, -0.987654321], dtype=np.float32))]
    a, b = servido.step(accion), local.step(accion)
    assert a.observation.proprio.tobytes() == b.observation.proprio.tobytes()
    assert a.info == b.info


def test_srve03_error_del_ambiente_y_el_servidor_sigue(servido: EnvironmentClient) -> None:
    with pytest.raises(RuntimeError, match="tarea desconocida"):
        servido.reset("otra tarea", 0)
    assert servido.reset(None, 0).instruction == "reach the goal"


def test_srve04_stop_libera_el_puerto() -> None:
    server = EnvironmentServer(ReachEnvironment())
    address = server.start()
    server.stop()
    try:
        with EnvironmentServer(ReachEnvironment(), address) as otro:
            assert otro.address == address
    except zmq.ZMQError as error:  # pragma: no cover - solo si el puerto quedó tomado
        pytest.fail(f"el puerto no se liberó: {error}")
