"""Suite SIM: ``Core.simulate`` sobre el WorldModel.

Verifica el camino RL-en-world-model: resolver el inicio, correr el WorldModel y
puntuar con el Reward Model en un solo batch. Directo sobre el gateway y por la
red (``CoreClient`` ↔ ``CoreServer``).
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_common.types import Action, ActionChunk
from vla_core.server import CoreServer
from vla_core.testing.fakes import make_fake_core
from vla_learner.data.core_client import CoreClient

pytestmark = pytest.mark.fast


def _chunk(values: list[float]) -> ActionChunk:
    return [Action(values=np.asarray(values, dtype=np.float32))]


def test_simulate_desde_observacion_puntua_cada_paso() -> None:
    core = make_fake_core()
    ((_, obs),) = core.reset(["e0"], seed=0)
    secuencias = [[_chunk([0.1, 0.1]), _chunk([0.1, 0.1])], [_chunk([0.0, 0.0])]]
    salida = core.simulate(obs, secuencias, with_reward=True)
    assert len(salida) == 2
    assert len(salida[0]) == 2 and len(salida[1]) == 1
    assert all(paso.reward is not None for seq in salida for paso in seq)


def test_simulate_sin_recompensa_deja_reward_none() -> None:
    core = make_fake_core()
    ((_, obs),) = core.reset(["e0"], seed=0)
    salida = core.simulate(obs, [[_chunk([0.1, 0.1])]], with_reward=False)
    assert salida[0][0].reward is None


def test_simulate_desde_referencia_de_episodio() -> None:
    core = make_fake_core()
    ((episode_id, obs0),) = core.reset(["e0"], seed=0)
    core.step(["e0"], [_chunk([0.1, 0.1])], with_reward=True)
    # El inicio (episode_id, 0) resuelve a la observación previa de la transición 0.
    salida = core.simulate((episode_id, 0), [[_chunk([0.1, 0.1])]], with_reward=True)
    assert len(salida) == 1 and len(salida[0]) == 1
    transicion0 = core.get_episode(episode_id)[0]
    assert np.array_equal(transicion0.observation.proprio, obs0.proprio)


def test_simulate_roundtrip_por_la_red() -> None:
    server = CoreServer(make_fake_core(), "tcp://127.0.0.1:0")
    address = server.start()
    client = CoreClient(address, timeout_s=5.0)
    try:
        ((_, obs),) = client.reset(["e0"], seed=0)
        salida = client.simulate(obs, [[_chunk([0.1, 0.1]), _chunk([0.1, 0.1])]], with_reward=True)
        assert len(salida) == 1 and len(salida[0]) == 2
        assert all(paso.reward is not None for paso in salida[0])
    finally:
        client.close()
        server.stop()
