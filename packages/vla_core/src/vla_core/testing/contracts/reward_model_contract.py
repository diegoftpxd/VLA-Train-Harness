"""Suite RM: contrato de ``RewardModel``.

La implementación entrega los fixtures ``reward_model`` y ``sample_item``.
"""

from __future__ import annotations

import pytest

from vla_core.reward import RewardModel, ScoreItem


class RewardModelContract:
    @pytest.fixture
    def reward_model(self) -> RewardModel:
        raise NotImplementedError

    @pytest.fixture
    def sample_item(self) -> ScoreItem:
        raise NotImplementedError

    def test_rm01_una_recompensa_por_item(
        self, reward_model: RewardModel, sample_item: ScoreItem
    ) -> None:
        recompensas = reward_model.score([sample_item, sample_item])
        assert len(recompensas) == 2
        assert all(isinstance(r, float) for r in recompensas)

    def test_rm02_batch_vacio(self, reward_model: RewardModel) -> None:
        assert reward_model.score([]) == []
