"""Suite DSET: contrato de ``Dataset``.

La implementación entrega el fixture ``dataset``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_core.dataset import Dataset


class DatasetContract:
    @pytest.fixture
    def dataset(self) -> Dataset:
        raise NotImplementedError

    def test_dset01_sample_devuelve_n_pares(self, dataset: Dataset) -> None:
        pares = dataset.sample(3)
        assert len(pares) == 3
        obs, chunk = pares[0]
        assert obs.proprio.dtype == np.float32
        assert len(chunk) >= 1
        assert isinstance(chunk[0].values, np.ndarray)
