"""Suite WM: contrato de ``WorldModel``.

La implementación entrega los fixtures ``world_model``, ``sample_observation`` y
``sample_chunk`` (una observación y un chunk válidos para ese modelo).
"""

from __future__ import annotations

import pytest

from vla_common.types import ActionChunk, Observation
from vla_core.world_model import LatentState, WorldModel


class WorldModelContract:
    @pytest.fixture
    def world_model(self) -> WorldModel:
        raise NotImplementedError

    @pytest.fixture
    def sample_observation(self) -> Observation:
        raise NotImplementedError

    @pytest.fixture
    def sample_chunk(self) -> ActionChunk:
        raise NotImplementedError

    def test_wm01_encode_devuelve_latente(
        self, world_model: WorldModel, sample_observation: Observation
    ) -> None:
        latent = world_model.encode(sample_observation)
        assert isinstance(latent, LatentState)
        assert latent.values.ndim >= 1

    def test_wm02_rollout_una_lista_por_secuencia(
        self, world_model: WorldModel, sample_observation: Observation, sample_chunk: ActionChunk
    ) -> None:
        latent = world_model.encode(sample_observation)
        secuencias = [[sample_chunk, sample_chunk], [sample_chunk]]
        predichas = world_model.rollout(latent, secuencias)
        assert [len(seq) for seq in predichas] == [2, 1]
        for seq in predichas:
            for obs in seq:
                assert isinstance(obs, Observation)
