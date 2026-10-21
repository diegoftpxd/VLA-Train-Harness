"""Test ``distributed``: varios Environment en paralelo atendidos por un Core.

Cada ambiente es un proceso ZeroMQ propio (``serve_environment``); Core los
consume por ``EnvironmentClient`` (``DEALER``). Un solo ``step`` enruta cada
chunk a su ambiente y devuelve los resultados **en el orden de entrada**, con una
transición por ambiente. Verifica el enrutado multi-ambiente de extremo a extremo
sobre el transporte real.

Marcado ``distributed`` (excluido por defecto; se corre con ``-m distributed``).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from vla_common.types import Action, ActionChunk
from vla_core.core import Core
from vla_core.environments import EnvironmentClient, EnvironmentRegistry
from vla_core.episodes import EpisodeManager, InMemoryEpisodeStore
from vla_core.reward import RewardScorer
from vla_core.testing.fakes import (
    FakeDataset,
    FakeEnvironment,
    FakeRewardModel,
    FakeWorldModel,
    serve_environment,
)

pytestmark = pytest.mark.distributed

N = 4


def _chunk(values: list[float]) -> ActionChunk:
    return [Action(values=np.asarray(values, dtype=np.float32))]


@pytest.fixture
def core_multi() -> Iterator[tuple[Core, list[str]]]:
    servers = [serve_environment(FakeEnvironment()) for _ in range(N)]
    registry = EnvironmentRegistry()
    env_ids = [f"env-{i}" for i in range(N)]
    try:
        for env_id, server in zip(env_ids, servers, strict=True):
            address = server.start()
            registry.register(env_id, EnvironmentClient(address, timeout_s=5.0))
        core = Core(
            environments=registry,
            episodes=EpisodeManager(),
            store=InMemoryEpisodeStore(),
            scorer=RewardScorer(FakeRewardModel()),
            world_model=FakeWorldModel(),
            dataset=FakeDataset(),
        )
        yield core, env_ids
        core.close()
    finally:
        for server in servers:
            server.stop()


def test_distributed_step_enruta_en_orden(core_multi: tuple[Core, list[str]]) -> None:
    core, env_ids = core_multi
    episodios = core.reset(env_ids, seed=0)
    assert [ep.startswith(env) for env, (ep, _) in zip(env_ids, episodios, strict=True)] == [
        True
    ] * N

    # Un chunk distinto por ambiente: sirve para comprobar que cada acción llegó
    # al ambiente correcto y en el orden de entrada.
    acciones = [_chunk([0.1 * i, -0.1 * i]) for i in range(N)]
    resultados = core.step(env_ids, acciones, with_reward=True)
    assert len(resultados) == N
    assert all(r.reward is not None for r in resultados)

    for (episode_id, _), accion in zip(episodios, acciones, strict=True):
        transiciones = core.get_episode(episode_id)
        assert len(transiciones) == 1
        assert np.array_equal(transiciones[0].action[0].values, accion[0].values)
