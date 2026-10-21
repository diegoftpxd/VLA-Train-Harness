"""Fase 5: ``Core.sample_supervised`` (datos del Dataset) y el gancho de
reinyección de episodios exitosos (``Core.ingest_episode``).

Dos niveles: pruebas directas del gateway con fakes en memoria y un round-trip
de ``sample_supervised`` por la red (``CoreClient`` ↔ ``CoreServer``), que la
suite CORE no cubre.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from vla_common.types import Action, ActionChunk, Observation
from vla_core.core import Core
from vla_core.dataset import Dataset
from vla_core.episodes import EpisodeManager, InMemoryEpisodeStore
from vla_core.errors import DatasetNotAppendable
from vla_core.reward import RewardScorer
from vla_core.server import CoreServer
from vla_core.testing.fakes import (
    FakeEnvironmentRegistry,
    FakeRewardModel,
    FakeWorldModel,
    make_fake_core,
)
from vla_learner.data.core_client import CoreClient

pytestmark = pytest.mark.fast


def _chunk(values: list[float]) -> ActionChunk:
    return [Action(values=np.asarray(values, dtype=np.float32))]


def _accion_a_la_meta(obs: Observation) -> ActionChunk:
    """Controlador perfecto del reach: un solo chunk que lleva pos a goal."""
    pos = obs.proprio[:2]
    goal = obs.proprio[2:4]
    return _chunk(list(goal - pos))


# -- sample_supervised ------------------------------------------------------


def test_sample_supervised_delega_en_dataset() -> None:
    core = make_fake_core()
    pares = core.sample_supervised(3)
    assert len(pares) == 3
    obs, chunk = pares[0]
    assert obs.proprio.dtype == np.float32
    assert chunk[0].values.shape == (2,)


def test_sample_supervised_pasa_filtros() -> None:
    core = make_fake_core()
    pares = core.sample_supervised(2, {"task": "stack the cups"})
    assert all(obs.instruction == "stack the cups" for obs, _ in pares)


# -- Gancho de episodios exitosos -------------------------------------------


def test_ingesta_episodio_exitoso_reinyecta_demos() -> None:
    core = make_fake_core()
    ((episode_id, obs),) = core.reset(["e0"], seed=0)
    (resultado,) = core.step(["e0"], [_accion_a_la_meta(obs)], with_reward=True)
    assert resultado.terminated  # alcanzó la meta

    agregados = core.ingest_episode(episode_id)
    assert agregados == 1

    # La demo reinyectada se sirve después (va antes que lo generado al vuelo).
    (par,) = core.sample_supervised(1)
    assert np.array_equal(par[0].proprio, obs.proprio)  # la observación previa


def test_ingesta_episodio_no_exitoso_no_agrega() -> None:
    core = make_fake_core()
    ((episode_id, _),) = core.reset(["e0"], seed=0)
    # Acción nula: un paso que no llega a la meta ni trunca todavía.
    (resultado,) = core.step(["e0"], [_chunk([0.0, 0.0])], with_reward=False)
    assert not resultado.terminated and not resultado.truncated
    assert core.ingest_episode(episode_id) == 0


def test_ingesta_sin_dataset_agregable_es_error() -> None:
    class _SoloLectura(Dataset):
        def sample(
            self, n: int, filters: dict | None = None
        ) -> list[tuple[Observation, ActionChunk]]:
            return []

    core = Core(
        environments=FakeEnvironmentRegistry(),
        episodes=EpisodeManager(),
        store=InMemoryEpisodeStore(),
        scorer=RewardScorer(FakeRewardModel()),
        world_model=FakeWorldModel(),
        dataset=_SoloLectura(),
    )
    ((episode_id, obs),) = core.reset(["e0"], seed=0)
    core.step(["e0"], [_accion_a_la_meta(obs)], with_reward=True)
    with pytest.raises(DatasetNotAppendable):
        core.ingest_episode(episode_id)


# -- Round-trip por la red --------------------------------------------------


@pytest.fixture
def client() -> Iterator[CoreClient]:
    server = CoreServer(make_fake_core(), "tcp://127.0.0.1:0")
    address = server.start()
    cliente = CoreClient(address, timeout_s=5.0)
    try:
        yield cliente
    finally:
        cliente.close()
        server.stop()


def test_sample_supervised_roundtrip_con_filtros(client: CoreClient) -> None:
    pares = client.sample_supervised(2, {"task": "open the drawer"})
    assert len(pares) == 2
    obs, chunk = pares[0]
    assert obs.instruction == "open the drawer"
    assert obs.proprio.dtype == np.float32
    assert chunk[0].values.shape == (2,)
