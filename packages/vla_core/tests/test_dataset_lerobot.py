"""Suite DSET contra ``LeRobotDataset`` (adaptador de corpus LeRobot).

Marcada ``heavy``: no corre en la máquina de desarrollo (requiere el grupo
opcional ``lerobot`` y un corpus/snapshot). Se activa solo si están instalado
``lerobot`` y definidas las variables de entorno:

- ``VLA_LEROBOT_REPO_ID``: el ``repo_id`` del corpus.
- ``VLA_LEROBOT_ROOT`` (opcional): ruta a un snapshot local para trabajar offline.

Corre el contrato de solo-lectura (``DatasetContract``). El filtro de tarea
(DSET03) solo tiene sentido con un corpus cuya instrucción sea conocida; por eso
se omite si no se declara ``VLA_LEROBOT_TASK``.
"""

from __future__ import annotations

import os

import pytest

from vla_core.dataset import Dataset
from vla_core.dataset.lerobot import LeRobotDataset
from vla_core.testing.contracts import DatasetContract

pytestmark = pytest.mark.heavy


class TestLeRobotDataset(DatasetContract):
    @pytest.fixture
    def dataset(self) -> Dataset:
        pytest.importorskip("lerobot")
        repo_id = os.environ.get("VLA_LEROBOT_REPO_ID")
        if not repo_id:
            pytest.skip("define VLA_LEROBOT_REPO_ID para correr la suite LeRobot")
        return LeRobotDataset(repo_id=repo_id, root=os.environ.get("VLA_LEROBOT_ROOT"))

    def test_dset03_filtros_se_respetan(self, dataset: Dataset) -> None:
        task = os.environ.get("VLA_LEROBOT_TASK")
        if not task:
            pytest.skip("define VLA_LEROBOT_TASK para probar el filtro de tarea")
        pares = dataset.sample(2, {"task": task})
        assert len(pares) == 2
        assert all(obs.instruction == task for obs, _ in pares)
