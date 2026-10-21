"""Suite STORE contra el ``InMemoryEpisodeStore``."""

from __future__ import annotations

import pytest

from vla_core.episodes import EpisodeStore, InMemoryEpisodeStore
from vla_core.testing.contracts import EpisodeStoreContract

pytestmark = pytest.mark.fast


class TestInMemoryEpisodeStore(EpisodeStoreContract):
    @pytest.fixture
    def store(self) -> EpisodeStore:
        return InMemoryEpisodeStore()
