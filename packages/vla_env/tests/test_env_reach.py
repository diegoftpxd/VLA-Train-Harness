"""``ReachEnvironment``: suite ENV en memoria (REACH-01) y equivalencia con el
``FakeEnvironment`` de Core (REACH-02).

REACH-02 garantiza que la integración con el Learner (fase 4 del plan) mide lo
mismo que la E2E actual. Se compara contra la misma fórmula que usan el fake y el
``ToyReachEnv`` (sin importar ``vla_core``: la regla de arquitectura lo pide).
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_common.environment import Environment
from vla_common.testing.contracts import EnvironmentContract
from vla_common.types import Action
from vla_env.reach import BOUND, THRESHOLD, ReachEnvironment

pytestmark = pytest.mark.fast


class TestReachEnvironment(EnvironmentContract):
    @pytest.fixture
    def environment(self) -> Environment:
        return ReachEnvironment()


def test_reach02_misma_dinamica_que_el_fake() -> None:
    env = ReachEnvironment()
    for seed in range(5):
        obs = env.reset(None, seed)
        rng = np.random.default_rng(seed)
        pos = rng.uniform(-BOUND, BOUND, size=2).astype(np.float32)
        goal = rng.uniform(-BOUND, BOUND, size=2).astype(np.float32)
        assert np.array_equal(obs.proprio, np.concatenate([pos, goal]))
        acciones = np.random.default_rng(100 + seed).uniform(-0.3, 0.3, size=(20, 2))
        for paso, delta in enumerate(acciones.astype(np.float32), start=1):
            resultado = env.step([Action(values=delta)])
            pos = np.clip(pos + delta, -BOUND, BOUND).astype(np.float32)
            llego = float(np.linalg.norm(pos - goal)) < THRESHOLD
            assert np.array_equal(resultado.observation.proprio[:2], pos)
            assert resultado.terminated == llego
            assert resultado.truncated == (paso >= 20 and not llego)
            if resultado.terminated or resultado.truncated:
                break
