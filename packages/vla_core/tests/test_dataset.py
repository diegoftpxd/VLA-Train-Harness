"""Suite DSET contra el ``FakeDataset``."""

from __future__ import annotations

import pytest

from vla_core.dataset import Dataset
from vla_core.testing.contracts import DatasetContract
from vla_core.testing.fakes import FakeDataset

pytestmark = pytest.mark.fast


class TestFakeDataset(DatasetContract):
    @pytest.fixture
    def dataset(self) -> Dataset:
        return FakeDataset(seed=0)
