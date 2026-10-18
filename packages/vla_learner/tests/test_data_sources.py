"""Suite DATA: fuentes de datos y ``RolloutCollector`` contra ``FakeCoreClient``."""

import numpy as np
import pytest
import torch

from vla_common import Observation
from vla_learner.data.collector import RolloutCollector
from vla_learner.data.sources import (
    EnvRolloutSource,
    MixedSource,
    SupervisedSource,
    WorldModelRolloutSource,
)
from vla_learner.testing.fake_core import FakeCoreClient
from vla_learner.testing.start_selector import ListStartSelector
from vla_learner.testing.stub_policy import StubPolicy
from vla_learner.types import RolloutBatch, SupervisedBatch

pytestmark = pytest.mark.fast


def _policy() -> StubPolicy:
    return StubPolicy(action_dim=2, chunk_size=1)


def test_data01_tipo_declarado():
    fuente = SupervisedSource(FakeCoreClient(), batch_size=3)
    assert fuente.batch_type is SupervisedBatch
    assert isinstance(fuente.next_batch(_policy()), SupervisedBatch)


def test_data02_supervisado_en_unidades_reales():
    fuente = SupervisedSource(FakeCoreClient(demo_chunk_size=2), batch_size=5)
    batch = fuente.next_batch(_policy())
    assert isinstance(batch, SupervisedBatch)
    assert batch.actions.shape == (5, 2, 2)  # [B, H, D]
    proprio = batch.inputs.tensors["proprio"]
    esperado = (proprio[:, 2:4] - proprio[:, :2]) / 2  # controlador perfecto / demo_chunk_size
    for h in range(2):
        assert torch.allclose(batch.actions[:, h, :], esperado, atol=1e-5)


def test_data03_rollouts_completos():
    collector = RolloutCollector(FakeCoreClient())
    fuente = EnvRolloutSource(collector, env_ids=["e0", "e1"], steps_per_batch=6)
    policy = _policy()
    batch = fuente.next_batch(policy)
    assert isinstance(batch, RolloutBatch)
    assert len(batch) == 6
    evaluacion = policy.evaluate_actions(batch.inputs, batch.actions, batch.extras)
    assert torch.allclose(evaluacion.log_probs, batch.old_log_probs, atol=1e-4)
    assert batch.terminated.dtype == torch.bool
    assert batch.truncated.dtype == torch.bool


def test_data04_reinicio_de_ambientes():
    collector = RolloutCollector(FakeCoreClient())
    batch = collector.collect_env(_policy(), ["e0"], num_steps=25, group_size=1)
    assert len(set(batch.episode_ids)) >= 2  # truncó a los 20 pasos y reinició


def test_data05_acciones_enviadas_sin_alterar():
    client = FakeCoreClient()
    collector = RolloutCollector(client)
    batch = collector.collect_env(_policy(), ["e0", "e1"], num_steps=1, group_size=1)
    envio = next(c for c in client.calls if c[0] == "step")[1]["actions"]
    for i in range(2):
        assert np.array_equal(envio[i][0].values, batch.actions[i, 0].numpy())


def test_data06_grupos_en_el_ambiente():
    client = FakeCoreClient()
    collector = RolloutCollector(client)
    batch = collector.collect_env(_policy(), ["e0", "e1", "e2", "e3"], num_steps=1, group_size=4)
    assert batch.group_ids is not None
    assert batch.group_ids.tolist() == [0, 0, 0, 0]
    proprio = batch.inputs.tensors["proprio"]
    for i in range(1, 4):
        assert torch.allclose(proprio[i], proprio[0], atol=0)  # mismo arranque
    reset = next(c for c in client.calls if c[0] == "reset")[1]
    assert reset["env_ids"] == ["e0", "e1", "e2", "e3"]
    assert reset["seed"] is not None  # una sola semilla para todo el grupo


def test_data07_grupos_en_el_world_model():
    collector = RolloutCollector(FakeCoreClient())
    start = Observation(
        images={"front": np.zeros((3, 8, 8), dtype=np.float32)},
        proprio=np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32),
        instruction="reach the goal",
    )
    fuente = WorldModelRolloutSource(
        collector, ListStartSelector([start]), sequences_per_start=4, horizon=2, num_starts=1
    )
    batch = fuente.next_batch(_policy())
    assert isinstance(batch, RolloutBatch)
    assert batch.source == "world_model"
    assert len(batch) == 4
    assert batch.group_ids is not None
    assert batch.group_ids.tolist() == [0, 0, 0, 0]


def test_data08_mezcla_de_fuentes():
    client = FakeCoreClient()
    chico = SupervisedSource(client, batch_size=2)
    grande = SupervisedSource(client, batch_size=4)
    mixed = MixedSource([chico, grande], weights=[0.8, 0.2], seed=0)
    policy = _policy()
    conteo = {2: 0, 4: 0}
    for _ in range(300):
        conteo[len(mixed.next_batch(policy))] += 1
    assert 0.7 < conteo[2] / 300 < 0.9  # respeta los pesos en promedio

    env_source = EnvRolloutSource(RolloutCollector(client), ["e0"], steps_per_batch=1)
    with pytest.raises(ValueError):
        MixedSource([chico, env_source], weights=[0.5, 0.5])


def test_data09_colector_no_deja_rastro():
    policy = _policy()
    policy.train()
    RolloutCollector(FakeCoreClient()).collect_env(policy, ["e0"], num_steps=3, group_size=1)
    assert policy.training is True
    assert all(p.grad is None for p in policy.parameters())
