"""Suite DSET contra ``OpenXDataset`` (adaptador Open-X Embodiment / RLDS).

Marcada ``heavy``: no corre en la máquina de desarrollo (requiere el grupo
opcional ``openx`` y un corpus RLDS). Se activa solo si están instalado
``tensorflow_datasets`` y definidas las variables de entorno:

- ``VLA_OPENX_NAME`` o ``VLA_OPENX_BUILDER_DIR``: dataset registrado o snapshot.
- ``VLA_OPENX_DATA_DIR`` (opcional): directorio de datos de tfds.

Corre el contrato de solo-lectura (``DatasetContract``). El filtro de tarea
(DSET03) solo tiene sentido con instrucciones conocidas; se omite si no se
declara ``VLA_OPENX_TASK``.
"""

from __future__ import annotations

import os

import pytest

from vla_core.dataset import Dataset
from vla_core.dataset.openx import OpenXDataset
from vla_core.testing.contracts import DatasetContract

pytestmark = pytest.mark.heavy


class TestOpenXDataset(DatasetContract):
    @pytest.fixture
    def dataset(self) -> Dataset:
        pytest.importorskip("tensorflow_datasets")
        name = os.environ.get("VLA_OPENX_NAME")
        builder_dir = os.environ.get("VLA_OPENX_BUILDER_DIR")
        if not name and not builder_dir:
            pytest.skip("define VLA_OPENX_NAME o VLA_OPENX_BUILDER_DIR para correr la suite Open-X")
        return OpenXDataset(
            name=name,
            builder_dir=builder_dir,
            data_dir=os.environ.get("VLA_OPENX_DATA_DIR"),
        )

    def test_dset03_filtros_se_respetan(self, dataset: Dataset) -> None:
        task = os.environ.get("VLA_OPENX_TASK")
        if not task:
            pytest.skip("define VLA_OPENX_TASK para probar el filtro de tarea")
        pares = dataset.sample(2, {"task": task})
        assert len(pares) == 2
        assert all(obs.instruction == task for obs, _ in pares)
