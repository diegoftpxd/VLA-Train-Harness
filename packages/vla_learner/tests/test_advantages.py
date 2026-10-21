"""Suite ADV: tests numéricos contra valores calculados a mano."""

import pytest
import torch

from vla_learner.algorithms.advantages import (
    GAE,
    GroupRelativeAdvantage,
    MonteCarloReturn,
)
from vla_learner.types import ModelInput, RolloutBatch

pytestmark = pytest.mark.fast


def _batch(
    rewards: list[float],
    *,
    values: list[float] | None = None,
    episode_ids: list[str] | None = None,
    step_ids: list[int] | None = None,
    terminated: list[bool] | None = None,
    truncated: list[bool] | None = None,
    group_ids: list[int] | None = None,
) -> RolloutBatch:
    n = len(rewards)
    episode_ids = episode_ids or ["e"] * n
    step_ids = step_ids if step_ids is not None else list(range(n))
    if terminated is None:
        # por defecto: el último paso de cada episodio termina
        ultimo = {
            eid: max(i for i in range(n) if episode_ids[i] == eid) for eid in set(episode_ids)
        }
        terminated = [i == ultimo[episode_ids[i]] for i in range(n)]
    truncated = truncated or [False] * n
    return RolloutBatch(
        inputs=ModelInput(tensors={"x": torch.zeros(n, 1)}, batch_size=n),
        actions=torch.zeros(n, 1, 1),
        old_log_probs=torch.zeros(n),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        terminated=torch.tensor(terminated),
        truncated=torch.tensor(truncated),
        episode_ids=episode_ids,
        step_ids=torch.tensor(step_ids),
        source="env",
        old_values=None if values is None else torch.tensor(values, dtype=torch.float32),
        group_ids=None if group_ids is None else torch.tensor(group_ids),
        extras={},
    )


def test_adv01_gae_casos_limite():
    batch = _batch([1.0, 2.0, 3.0], values=[0.5, 0.5, 0.5])  # episodio terminado

    adv1, ret1 = GAE(gamma=0.9, lam=1.0).compute(batch)
    mc = torch.tensor([5.23, 4.7, 3.0])  # retornos Monte Carlo a mano
    assert torch.allclose(ret1, mc, atol=1e-4)
    assert torch.allclose(adv1, mc - 0.5, atol=1e-4)  # λ=1 → MC - valores

    adv0, _ = GAE(gamma=0.9, lam=0.0).compute(batch)
    td = torch.tensor([1 + 0.9 * 0.5 - 0.5, 2 + 0.9 * 0.5 - 0.5, 3 - 0.5])  # λ=0 → TD de un paso
    assert torch.allclose(adv0, td, atol=1e-4)


def test_adv02_terminated_vs_truncated():
    gae = GAE(gamma=0.9, lam=0.95)
    terminado = _batch([1.0], values=[2.0], terminated=[True], truncated=[False])
    truncado = _batch([1.0], values=[2.0], terminated=[False], truncated=[True])
    adv_t, _ = gae.compute(terminado)
    adv_u, _ = gae.compute(truncado)
    assert torch.allclose(adv_t, torch.tensor([1.0 - 2.0]), atol=1e-4)  # sin bootstrap
    assert torch.allclose(adv_u, torch.tensor([1.0 + 0.9 * 2.0 - 2.0]), atol=1e-4)  # con bootstrap


def test_adv03_varios_episodios_mezclados():
    batch = _batch(
        [1.0, 10.0, 2.0, 20.0],
        values=[0.0, 0.0, 0.0, 0.0],
        episode_ids=["e0", "e1", "e0", "e1"],
        step_ids=[0, 0, 1, 1],
        terminated=[False, False, True, True],
    )
    adv, _ = GAE(gamma=0.9, lam=1.0).compute(batch)
    # e0: [1+0.9*2, 2] = [2.8, 2]; e1: [10+0.9*20, 20] = [28, 20]
    assert torch.allclose(adv, torch.tensor([2.8, 28.0, 2.0, 20.0]), atol=1e-4)


def test_adv04_group_relative():
    batch = _batch([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], group_ids=[0, 0, 0, 1, 1, 1])
    adv, ret = GroupRelativeAdvantage(normalize_std=True).compute(batch)
    assert torch.allclose(ret, batch.rewards)
    for grupo in (slice(0, 3), slice(3, 6)):
        trozo = adv[grupo]
        assert torch.allclose(trozo.mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(trozo.std(unbiased=False), torch.tensor(1.0), atol=1e-5)


def test_adv05_grupos_degenerados():
    batch = _batch([5.0, 5.0, 5.0, 7.0], group_ids=[0, 0, 0, 1])
    adv, _ = GroupRelativeAdvantage().compute(batch)
    assert torch.allclose(adv, torch.zeros(4))
    assert torch.isfinite(adv).all()


def test_adv06_monte_carlo_return():
    batch = _batch([1.0, 2.0, 4.0])
    _, ret = MonteCarloReturn(gamma=0.5).compute(batch)
    assert torch.allclose(ret, torch.tensor([3.0, 4.0, 4.0]), atol=1e-5)
