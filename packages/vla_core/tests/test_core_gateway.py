"""Gateway ``Core``: episodios reales (reset/step/get_episode) y contrato CORE.

Dos niveles:

- Pruebas directas del gateway con componentes fake en memoria.
- La **misma** suite CORE que corre contra ``FakeCoreClient`` y contra
  ``CoreClient``+``FakeCoreServer`` (en el Learner), ahora contra el ``Core``
  real expuesto por un ``CoreServer``. Así se verifica que el Core real respeta
  el protocolo.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from vla_common.types import Action, ActionChunk
from vla_core.errors import EpisodeAlreadyDone
from vla_core.server import CoreServer
from vla_core.testing.fakes import make_fake_core
from vla_learner.data.core_client import CoreClient
from vla_learner.testing.contracts.core_client_contract import CoreClientContract

pytestmark = pytest.mark.fast


def _chunk(values: list[float]) -> ActionChunk:
    return [Action(values=np.asarray(values, dtype=np.float32))]


# -- Pruebas directas del gateway -------------------------------------------


def test_reset_abre_episodios_y_devuelve_observaciones() -> None:
    core = make_fake_core()
    episodios = core.reset(["a", "b"], seed=0)
    pares = zip(["a", "b"], episodios, strict=True)
    assert [ep.startswith(env) for env, (ep, _) in pares] == [True, True]
    assert all(obs.proprio.dtype == np.float32 for _, obs in episodios)


def test_step_registra_transicion_con_source_env() -> None:
    core = make_fake_core()
    ((episode_id, _),) = core.reset(["e0"], seed=0)
    core.step(["e0"], [_chunk([0.1, 0.1])], with_reward=True)
    transiciones = core.get_episode(episode_id)
    assert len(transiciones) == 1
    assert transiciones[0].source == "env"
    assert transiciones[0].result.reward is not None  # con recompensa


def test_step_sin_recompensa_deja_reward_none() -> None:
    core = make_fake_core()
    core.reset(["e0"], seed=0)
    (resultado,) = core.step(["e0"], [_chunk([0.0, 0.0])], with_reward=False)
    assert resultado.reward is None


def test_step_sobre_episodio_terminado_es_error() -> None:
    core = make_fake_core()
    core.reset(["e0"], seed=0)
    # Acciones nulas: nunca llega a la meta, trunca al límite de pasos.
    termino = False
    for _ in range(25):
        (resultado,) = core.step(["e0"], [_chunk([0.0, 0.0])], with_reward=False)
        if resultado.terminated or resultado.truncated:
            termino = True
            break
    assert termino
    with pytest.raises(EpisodeAlreadyDone):
        core.step(["e0"], [_chunk([0.0, 0.0])], with_reward=False)


# -- Suite CORE reutilizada contra el Core real -----------------------------


class TestCoreRealContract(CoreClientContract):
    @pytest.fixture
    def client(self) -> Iterator[CoreClient]:
        server = CoreServer(make_fake_core(), "tcp://127.0.0.1:0")
        address = server.start()
        cliente = CoreClient(address, timeout_s=5.0)
        try:
            yield cliente
        finally:
            cliente.close()
            server.stop()
