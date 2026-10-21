"""Suite DSET contra el ``FakeDataset`` (incluye el gancho de reinyección)."""

from __future__ import annotations

import pytest

from vla_core.dataset import AppendableDataset
from vla_core.testing.contracts import AppendableDatasetContract
from vla_core.testing.fakes import FakeDataset

pytestmark = pytest.mark.fast


class TestFakeDataset(AppendableDatasetContract):
    @pytest.fixture
    def dataset(self) -> AppendableDataset:
        return FakeDataset(seed=0)

    @pytest.fixture
    def appendable(self) -> AppendableDataset:
        return FakeDataset(seed=0)
