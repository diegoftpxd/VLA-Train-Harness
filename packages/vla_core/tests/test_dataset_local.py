"""Suite DSET contra ``LocalDataset`` (adaptador de solo-lectura sobre disco).

Corre el contrato de solo-lectura (``DatasetContract``) contra un fixture chico
incluido en el repo, más pruebas propias del formato on-disk (chunking, mapeo de
cámaras, muestreo con reemplazo). Todo en ``fast``, offline y determinista.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vla_core.dataset import Dataset, LocalDataset, build_dataset
from vla_core.testing.contracts import DatasetContract

pytestmark = pytest.mark.fast

FIXTURE = Path(__file__).parent / "data" / "local_reach"


class TestLocalDataset(DatasetContract):
    @pytest.fixture
    def dataset(self) -> Dataset:
        # El fixture incluye la tarea "push the block" (DSET03 la filtra).
        return LocalDataset(path=FIXTURE, demo_chunk_size=1)


def test_factory_local_devuelve_localdataset() -> None:
    ds = build_dataset({"type": "local", "path": str(FIXTURE)})
    assert isinstance(ds, LocalDataset)


def test_chunking_agrupa_pasos_consecutivos() -> None:
    # demo_chunk_size=2 sobre episodios de T=4 → chunks de 2 (último grupo lleno).
    ds = LocalDataset(path=FIXTURE, demo_chunk_size=2)
    _, chunk = ds.sample(1)[0]
    assert len(chunk) == 2
    assert chunk[0].values.shape == (2,)


def test_mapeo_camaras_e_imagenes() -> None:
    ds = LocalDataset(path=FIXTURE, demo_chunk_size=1)
    obs, _ = ds.sample(1)[0]
    assert set(obs.images) == {"front"}
    img = obs.images["front"]
    assert img.shape == (3, 4, 4)
    assert img.dtype == np.float32


def test_filtro_sin_coincidencia_es_error() -> None:
    ds = LocalDataset(path=FIXTURE, demo_chunk_size=1)
    with pytest.raises(ValueError):
        ds.sample(1, {"task": "tarea inexistente"})


def test_directorio_inexistente_es_error() -> None:
    with pytest.raises(FileNotFoundError):
        LocalDataset(path=FIXTURE / "no_existe")


def test_muestreo_determinista_por_semilla() -> None:
    a = LocalDataset(path=FIXTURE, seed=3).sample(5)
    b = LocalDataset(path=FIXTURE, seed=3).sample(5)
    for (_, chunk_a), (_, chunk_b) in zip(a, b, strict=True):
        assert (chunk_a[0].values == chunk_b[0].values).all()
