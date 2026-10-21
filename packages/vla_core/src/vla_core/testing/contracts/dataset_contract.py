"""Suites DSET: contrato de ``Dataset`` y de ``AppendableDataset``.

La implementación entrega el fixture ``dataset`` (y ``appendable`` para el
gancho de reinyección).
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_common.types import Action, ActionChunk, Observation
from vla_core.dataset import AppendableDataset, Dataset


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

    def test_dset02_acciones_en_unidades_reales(self, dataset: Dataset) -> None:
        # Las acciones salen sin normalizar: ndarray float32, no un escalar ni
        # un tipo normalizado. (La escala real la fija cada dataset.)
        (_, chunk), *_ = dataset.sample(1)
        assert chunk[0].values.dtype == np.float32
        assert chunk[0].values.ndim == 1

    def test_dset03_filtros_se_respetan(self, dataset: Dataset) -> None:
        pares = dataset.sample(2, {"task": "push the block"})
        assert len(pares) == 2
        assert all(obs.instruction == "push the block" for obs, _ in pares)


class AppendableDatasetContract(DatasetContract):
    """Gancho de episodios exitosos: lo reinyectado se puede volver a muestrear."""

    @pytest.fixture
    def appendable(self) -> AppendableDataset:
        raise NotImplementedError

    def _par(self, instruction: str) -> tuple[Observation, ActionChunk]:
        obs = Observation(
            images={"front": np.zeros((3, 8, 8), dtype=np.float32)},
            proprio=np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32),
            instruction=instruction,
        )
        chunk: ActionChunk = [Action(values=np.array([0.5, 0.5], dtype=np.float32))]
        return obs, chunk

    def test_dset04_add_devuelve_la_cantidad(self, appendable: AppendableDataset) -> None:
        assert appendable.add([self._par("reinyectado")]) == 1

    def test_dset05_lo_reinyectado_se_muestrea(self, appendable: AppendableDataset) -> None:
        appendable.add([self._par("reinyectado")])
        pares = appendable.sample(1, {"task": "reinyectado"})
        assert pares[0][0].instruction == "reinyectado"
