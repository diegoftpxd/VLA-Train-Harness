"""Suite FAIL: robustez de Core ante caídas de componentes.

Dos niveles:

- **Transporte** (``RpcChannel``/``EnvironmentClient``): un componente colgado
  produce ``DownstreamTimeout``, y un componente que se recupera vuelve a
  responder (reconexión).
- **Gateway** (``Core``): un ambiente que no responde se marca no disponible, se
  le invoca la parada de seguridad y el error se escala a
  ``DownstreamUnavailable``, sin romper al resto de los ambientes.

Todo ``fast``: CPU, ``tcp://127.0.0.1`` con puerto efímero, timeouts cortos.
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_common.types import Action, ActionChunk
from vla_core.core import Core
from vla_core.environments import EnvironmentClient, EnvironmentRegistry
from vla_core.episodes import EpisodeManager, InMemoryEpisodeStore
from vla_core.errors import DownstreamTimeout, DownstreamUnavailable
from vla_core.reward import RewardScorer
from vla_core.testing.fakes import (
    FakeDataset,
    FakeEnvironment,
    FakeRewardModel,
    FakeWorldModel,
    FlakyEnvironment,
    SilentRouter,
    serve_environment,
)

pytestmark = pytest.mark.fast


def _chunk(values: list[float]) -> ActionChunk:
    return [Action(values=np.asarray(values, dtype=np.float32))]


def _core_con(registry: EnvironmentRegistry) -> Core:
    return Core(
        environments=registry,
        episodes=EpisodeManager(),
        store=InMemoryEpisodeStore(),
        scorer=RewardScorer(FakeRewardModel()),
        world_model=FakeWorldModel(),
        dataset=FakeDataset(),
    )


# -- Transporte -------------------------------------------------------------


def test_fail_timeout_sin_respuesta() -> None:
    with SilentRouter() as silent:
        client = EnvironmentClient(silent.address, timeout_s=0.2, retries=0)
        try:
            with pytest.raises(DownstreamTimeout):
                client.reset(task=None, seed=0)
        finally:
            client.close()


def test_fail_reconexion_tras_caida() -> None:
    # El componente está colgado: el cliente agota timeout y reintentos.
    silent = SilentRouter()
    address = silent.start()
    client = EnvironmentClient(address, timeout_s=0.2, retries=2)
    try:
        with pytest.raises(DownstreamTimeout):
            client.reset(task=None, seed=0)
        silent.stop()

        # El componente se recupera en la misma dirección; el DEALER reconecta.
        server = serve_environment(FakeEnvironment(), address)
        server.start()
        try:
            obs = client.reset(task=None, seed=0)
            assert obs.proprio.dtype == np.float32
        finally:
            server.stop()
    finally:
        client.close()


# -- Gateway: caída de un ambiente ------------------------------------------


def test_fail_reset_caido_marca_no_disponible_y_para() -> None:
    registry = EnvironmentRegistry()
    robot = FlakyEnvironment(fail_reset=True)
    registry.register("robot", robot)
    registry.register("sim", FakeEnvironment())
    core = _core_con(registry)

    with pytest.raises(DownstreamUnavailable):
        core.reset(["robot"], seed=0)
    assert robot.emergency_stops == 1  # parada de seguridad invocada
    assert "robot" not in registry.available()

    # El resto sigue vivo.
    ((_, obs),) = core.reset(["sim"], seed=0)
    assert obs.proprio.dtype == np.float32
    assert "sim" in registry.available()

    # Reusar el caído da Unavailable directo, sin volver a tocar el ambiente.
    with pytest.raises(DownstreamUnavailable):
        core.reset(["robot"], seed=0)
    assert robot.emergency_stops == 1


def test_fail_step_caido_no_rompe_el_resto() -> None:
    registry = EnvironmentRegistry()
    robot = FlakyEnvironment(fail_reset=False, fail_step=True)
    registry.register("robot", robot)
    registry.register("sim", FakeEnvironment())
    core = _core_con(registry)

    core.reset(["robot"], seed=0)
    ((sim_ep, _),) = core.reset(["sim"], seed=0)

    with pytest.raises(DownstreamUnavailable):
        core.step(["robot"], [_chunk([0.1, 0.1])], with_reward=True)
    assert robot.emergency_stops == 1
    assert "robot" not in registry.available()

    # El ambiente sano sigue operando y registra su transición con normalidad.
    (resultado,) = core.step(["sim"], [_chunk([0.1, 0.1])], with_reward=True)
    assert resultado.reward is not None
    assert len(core.get_episode(sim_ep)) == 1
