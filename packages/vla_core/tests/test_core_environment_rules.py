"""Reglas de Core sobre los ambientes (CENV-*).

Ver ``docs/environment/Test Environment.md`` ("Integración"): Core valida la
forma de cada acción contra el ``spec`` antes de mover ningún ambiente, escala la
caída del ``spec`` como cualquier otra, y reinyecta solo episodios con
``info["success"]``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from vla_common.environment import Environment, EnvSpec
from vla_common.types import Action, ActionChunk, Observation, StepResult
from vla_core.core import Core
from vla_core.environments import EnvironmentRegistry
from vla_core.episodes import EpisodeManager, InMemoryEpisodeStore
from vla_core.errors import DownstreamTimeout, DownstreamUnavailable
from vla_core.reward import RewardScorer
from vla_core.server import CoreServer
from vla_core.testing.fakes import (
    FAKE_SPEC,
    FakeDataset,
    FakeEnvironment,
    FakeRewardModel,
    FakeWorldModel,
    make_fake_core,
)
from vla_learner.data.core_client import CoreClient

pytestmark = pytest.mark.fast


def _chunk(n: int) -> ActionChunk:
    return [Action(values=np.full(n, 0.01, dtype=np.float32))]


def _core_con(**envs: Environment) -> Core:
    registro = EnvironmentRegistry()
    for env_id, env in envs.items():
        registro.register(env_id, env)
    return Core(
        environments=registro,
        episodes=EpisodeManager(),
        store=InMemoryEpisodeStore(),
        scorer=RewardScorer(FakeRewardModel()),
        world_model=FakeWorldModel(),
        dataset=FakeDataset(),
    )


class _ContadorDePasos(FakeEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.pasos = 0

    def step(self, actions: ActionChunk) -> StepResult:
        self.pasos += 1
        return super().step(actions)


def test_cenv01_accion_mal_dimensionada_no_mueve_ningun_ambiente() -> None:
    a, b = _ContadorDePasos(), _ContadorDePasos()
    core = _core_con(a=a, b=b)
    core.reset(["a", "b"], seed=0)
    # El chunk inválido es el segundo: el primero tampoco debe ejecutarse.
    with pytest.raises(ValueError, match="'b'.*forma"):
        core.step(["a", "b"], [_chunk(2), _chunk(3)], with_reward=True)
    assert a.pasos == 0 and b.pasos == 0
    # El episodio sigue intacto y usable.
    (resultado,) = core.step(["a"], [_chunk(2)], with_reward=False)
    assert resultado.reward is None


@pytest.fixture
def cliente_de_core() -> Iterator[CoreClient]:
    server = CoreServer(make_fake_core(), "tcp://127.0.0.1:0")
    address = server.start()
    client = CoreClient(address, timeout_s=5.0)
    try:
        yield client
    finally:
        client.close()
        server.stop()


def test_cenv02_accion_mal_dimensionada_por_la_red(cliente_de_core: CoreClient) -> None:
    cliente_de_core.reset(["e0"], seed=0)
    with pytest.raises(RuntimeError, match="forma"):
        cliente_de_core.step(["e0"], [_chunk(5)], with_reward=False)


class _TerminaPorFallo(Environment):
    """Termina en el primer paso, pero sin éxito (p. ej. una colisión)."""

    def reset(self, task: str | None, seed: int | None) -> Observation:
        return FakeEnvironment().reset(task, seed)

    def step(self, actions: ActionChunk) -> StepResult:
        obs = FakeEnvironment().reset(None, 0)
        return StepResult(obs, None, terminated=True, truncated=False, info={"success": False})

    def close(self) -> None:
        pass

    def emergency_stop(self) -> None:
        pass

    @property
    def spec(self) -> EnvSpec:
        return FAKE_SPEC


def test_cenv03_terminado_sin_exito_no_se_reinyecta() -> None:
    core = _core_con(a=_TerminaPorFallo())
    ((episode_id, _),) = core.reset(["a"], seed=0)
    (resultado,) = core.step(["a"], [_chunk(2)], with_reward=False)
    assert resultado.terminated
    assert core.ingest_episode(episode_id) == 0


class _SpecCaido(FakeEnvironment):
    """El proceso responde ``reset`` pero se cae al pedirle el ``spec``."""

    def __init__(self) -> None:
        super().__init__()
        self.emergency_stops = 0
        self.caido = False

    @property
    def spec(self) -> EnvSpec:
        if self.caido:
            raise DownstreamTimeout("spec sin respuesta")
        return FAKE_SPEC

    def emergency_stop(self) -> None:
        self.emergency_stops += 1


def test_cenv04_caida_del_spec_se_escala() -> None:
    robot, sano = _SpecCaido(), FakeEnvironment()
    core = _core_con(robot=robot, sano=sano)
    core.reset(["robot", "sano"], seed=0)
    robot.caido = True
    with pytest.raises(DownstreamUnavailable):
        core.step(["robot", "sano"], [_chunk(2), _chunk(2)], with_reward=False)
    assert robot.emergency_stops == 1
    # El otro ambiente sigue operando.
    (resultado,) = core.step(["sano"], [_chunk(2)], with_reward=False)
    assert resultado.reward is None
