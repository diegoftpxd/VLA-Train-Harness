"""Suite SRV: la capa de transporte de Core (``CoreServer``).

Prueba round-trips de cada llamada del protocolo usando el ``CoreClient`` real
del Learner contra un ``CoreServer`` que envuelve el ``StaticCoreBackend``. Así
se verifica, sin componentes reales, que el transporte y el despacho compartido
(``vla_common.dispatch_request``) funcionan: los arrays sobreviven el wire bit a
bit, el sobre ``ok``/``error`` funciona y el campo ``reward`` opcional se respeta.

Marcados ``fast``: corren siempre, en CPU, sin red externa (``tcp://127.0.0.1``).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from vla_common import Action, ActionChunk
from vla_core.server import CoreServer
from vla_core.testing.static_backend import StaticCoreBackend
from vla_learner.data.core_client import CoreClient

pytestmark = pytest.mark.fast


def _chunk(values: list[float]) -> ActionChunk:
    return [Action(values=np.asarray(values, dtype=np.float32))]


@pytest.fixture
def client() -> Iterator[CoreClient]:
    server = CoreServer(StaticCoreBackend(), "tcp://127.0.0.1:0")
    address = server.start()
    cliente = CoreClient(address, timeout_s=5.0)
    try:
        yield cliente
    finally:
        cliente.close()
        server.stop()


def test_srv01_reset_roundtrip(client: CoreClient) -> None:
    episodios = client.reset(["e0", "e1"], seed=1)
    assert len(episodios) == 2
    (episode_id, obs) = episodios[0]
    assert episode_id.startswith("e0")
    assert obs.proprio.dtype == np.float32
    assert "front" in obs.images


def test_srv02_step_y_get_episode_bit_a_bit(client: CoreClient) -> None:
    ((episode_id, _),) = client.reset(["e0"], seed=0)
    valores = [0.123, -0.456]
    (resultado,) = client.step(["e0"], [_chunk(valores)], with_reward=True)
    assert resultado.reward == pytest.approx(1.0)
    transiciones = client.get_episode(episode_id)
    recibido = transiciones[0].action[0].values
    enviado = np.asarray(valores, dtype=np.float32)
    assert recibido.dtype == np.float32
    assert np.array_equal(recibido, enviado)  # bit a bit a través del wire


def test_srv03_simulate_roundtrip(client: CoreClient) -> None:
    ((_, obs),) = client.reset(["e0"], seed=0)
    secuencias = [[_chunk([0.1, 0.2]), _chunk([0.3, 0.4])], [_chunk([0.0, 0.0])]]
    salida = client.simulate(obs, secuencias, with_reward=True)
    assert [len(seq) for seq in salida] == [2, 1]
    assert salida[0][0].reward == pytest.approx(0.5)


def test_srv04_sample_supervised_roundtrip(client: CoreClient) -> None:
    pares = client.sample_supervised(3)
    assert len(pares) == 3
    obs, chunk = pares[0]
    assert obs.proprio.dtype == np.float32
    assert chunk[0].values.shape == (2,)


def test_srv05_recompensa_opcional(client: CoreClient) -> None:
    client.reset(["e0"], seed=0)
    (resultado,) = client.step(["e0"], [_chunk([0.0, 0.0])], with_reward=False)
    assert resultado.reward is None


def test_srv06_error_se_propaga_como_sobre(client: CoreClient) -> None:
    # El backend levanta para el env reservado "boom"; el servidor lo devuelve
    # como {ok: False, error} y el cliente lo vuelve a levantar.
    with pytest.raises(RuntimeError):
        client.reset(["boom"], seed=0)
