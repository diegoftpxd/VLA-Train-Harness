"""Tests de ``FakeCoreClient`` con ``ToyReachEnv`` y ``ToyWorldModel``.

Precursores de las suites CORE y DATA (Fase 5): aquí se verifica el
comportamiento del falso; allá se comparará contra el cliente real.
"""

import numpy as np
import pytest

from vla_common import Action, ActionChunk
from vla_learner.testing.fake_core import FakeCoreClient
from vla_learner.testing.toy_env import GOAL_RADIUS, MAX_STEPS

pytestmark = pytest.mark.fast


def _chunk(delta: list[float]) -> ActionChunk:
    return [Action(values=np.asarray(delta, dtype=np.float32))]


def _fijar(client: FakeCoreClient, env_id: str, pos: list[float], goal: list[float]) -> None:
    env = client._envs[env_id]
    env.pos = np.asarray(pos, dtype=np.float32)
    env.goal = np.asarray(goal, dtype=np.float32)
    env.steps = 0
    env.done = False
    client._last_obs[env_id] = env.observation()


def test_reset_devuelve_episodios_distintos():
    client = FakeCoreClient()
    out = client.reset(["e0", "e1"], seed=0)
    assert len(out) == 2
    ids = [ep for ep, _ in out]
    assert ids[0] != ids[1]
    for _, obs in out:
        assert obs.proprio.shape == (4,)


def test_step_recompensa_y_terminated():
    client = FakeCoreClient()
    client.reset(["e0"], seed=1)
    _fijar(client, "e0", [0.0, 0.0], [0.5, 0.0])
    (res,) = client.step(["e0"], [_chunk([0.5, 0.0])], with_reward=True)
    assert res.terminated
    assert res.reward == pytest.approx(0.0, abs=1e-6)
    assert res.info["distance"] <= GOAL_RADIUS


def test_step_truncated_a_los_max_steps():
    client = FakeCoreClient()
    client.reset(["e0"], seed=2)
    _fijar(client, "e0", [0.0, 0.0], [0.9, 0.9])
    quieto = _chunk([0.0, 0.0])
    res = None
    for _ in range(MAX_STEPS):
        (res,) = client.step(["e0"], [quieto], with_reward=True)
    assert res is not None
    assert res.truncated
    assert not res.terminated
    with pytest.raises(RuntimeError):
        client.step(["e0"], [quieto], with_reward=True)


def test_with_reward_false_da_none():
    client = FakeCoreClient()
    client.reset(["e0"], seed=3)
    (res,) = client.step(["e0"], [_chunk([0.0, 0.0])], with_reward=False)
    assert res.reward is None


def test_orden_de_ambientes_y_registro_de_llamadas():
    client = FakeCoreClient()
    client.reset(["a", "b", "c"], seed=0)
    quieto = _chunk([0.0, 0.0])
    results = client.step(["a", "b", "c"], [quieto, quieto, quieto], with_reward=True)
    assert len(results) == 3
    assert client.calls[0][0] == "reset"
    assert client.calls[-1][0] == "step"
    assert client.calls[-1][1]["env_ids"] == ["a", "b", "c"]


def test_get_episode_registra_transiciones():
    client = FakeCoreClient()
    ((episode_id, _),) = client.reset(["e0"], seed=4)
    quieto = _chunk([0.0, 0.0])
    client.step(["e0"], [quieto], with_reward=True)
    client.step(["e0"], [quieto], with_reward=True)
    transiciones = client.get_episode(episode_id)
    assert len(transiciones) == 2
    assert [t.step for t in transiciones] == [0, 1]
    assert all(t.source == "env" for t in transiciones)


def test_sample_supervised_unidades_reales_y_controlador_perfecto():
    client = FakeCoreClient(demo_chunk_size=2)
    pares = client.sample_supervised(5)
    assert len(pares) == 5
    for obs, chunk in pares:
        assert obs.proprio.shape == (4,)
        assert len(chunk) == 2
        for action in chunk:
            assert action.values.shape == (2,)
    obs, chunk = pares[0]
    pos = obs.proprio[:2].copy()
    goal = obs.proprio[2:4]
    for action in chunk:
        pos = pos + action.values
    assert np.allclose(pos, goal, atol=1e-5)


def test_simulate_reproduce_la_dinamica():
    client = FakeCoreClient()
    ((_, obs),) = client.reset(["e0"], seed=5)
    chunk_a = _chunk([0.1, 0.0])
    out = client.simulate(obs, [[chunk_a], [chunk_a]], with_reward=True)
    assert len(out) == 2
    assert len(out[0]) == 1
    esperado = np.clip(obs.proprio[:2] + np.array([0.1, 0.0], np.float32), -1.0, 1.0)
    assert np.allclose(out[0][0].observation.proprio[:2], esperado, atol=1e-6)


def test_simulate_sin_reward():
    client = FakeCoreClient()
    ((_, obs),) = client.reset(["e0"], seed=6)
    out = client.simulate(obs, [[_chunk([0.0, 0.0])]], with_reward=False)
    assert out[0][0].reward is None


def test_simulate_desde_episodio_registrado():
    client = FakeCoreClient()
    ((episode_id, _),) = client.reset(["e0"], seed=7)
    quieto = _chunk([0.0, 0.0])
    client.step(["e0"], [quieto], with_reward=True)
    out = client.simulate((episode_id, 0), [[quieto]], with_reward=True)
    assert len(out) == 1
    assert len(out[0]) == 1
