"""Suite RM contra el ``FakeRewardModel`` y contra el ``RewardModelClient``."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from vla_common.types import Action
from vla_core.reward import RewardModel, RewardModelClient, ScoreItem
from vla_core.testing.contracts import RewardModelContract
from vla_core.testing.fakes import FakeEnvironment, FakeRewardModel, serve_reward_model

pytestmark = pytest.mark.fast


class _RewardInputs:
    @pytest.fixture
    def sample_item(self) -> ScoreItem:
        env = FakeEnvironment()
        obs = env.reset(task=None, seed=0)
        resultado = env.step([Action(values=np.zeros(2, dtype=np.float32))])
        return ScoreItem(
            instruction=obs.instruction,
            observation=obs,
            action=[Action(values=np.zeros(2, dtype=np.float32))],
            next_observation=resultado.observation,
        )


class TestFakeRewardModel(_RewardInputs, RewardModelContract):
    @pytest.fixture
    def reward_model(self) -> RewardModel:
        return FakeRewardModel()


class TestRewardModelClient(_RewardInputs, RewardModelContract):
    @pytest.fixture
    def reward_model(self) -> Iterator[RewardModel]:
        server = serve_reward_model(FakeRewardModel())
        address = server.start()
        client = RewardModelClient(address, timeout_s=5.0)
        try:
            yield client
        finally:
            client.close()
            server.stop()
