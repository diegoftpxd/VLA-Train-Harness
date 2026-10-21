"""Suite WM contra el ``FakeWorldModel`` y contra el ``WorldModelClient``."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from vla_common.types import Action, ActionChunk, Observation
from vla_core.testing.contracts import WorldModelContract
from vla_core.testing.fakes import FakeEnvironment, FakeWorldModel, serve_world_model
from vla_core.world_model import WorldModel, WorldModelClient

pytestmark = pytest.mark.fast


class _WorldModelInputs:
    @pytest.fixture
    def sample_observation(self) -> Observation:
        return FakeEnvironment().reset(task=None, seed=0)

    @pytest.fixture
    def sample_chunk(self) -> ActionChunk:
        return [Action(values=np.zeros(2, dtype=np.float32))]


class TestFakeWorldModel(_WorldModelInputs, WorldModelContract):
    @pytest.fixture
    def world_model(self) -> WorldModel:
        return FakeWorldModel()


class TestWorldModelClient(_WorldModelInputs, WorldModelContract):
    @pytest.fixture
    def world_model(self) -> Iterator[WorldModel]:
        server = serve_world_model(FakeWorldModel())
        address = server.start()
        client = WorldModelClient(address, timeout_s=5.0)
        try:
            yield client
        finally:
            client.close()
            server.stop()
