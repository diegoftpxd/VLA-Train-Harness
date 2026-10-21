"""Costura ``build_dataset``: despacho por ``type`` que mantiene a Core agnóstico.

Ver ``docs/dataset/Dataset Design.md`` (arquitectura pluggable).
"""

from __future__ import annotations

import pytest

from vla_core.dataset import AppendableDataset, Dataset, build_dataset
from vla_core.testing.fakes import FakeDataset

pytestmark = pytest.mark.fast


def test_type_fake_devuelve_fakedataset() -> None:
    ds = build_dataset({"type": "fake", "demo_chunk_size": 2})
    assert isinstance(ds, FakeDataset)
    assert isinstance(ds, Dataset)


def test_type_por_defecto_es_fake() -> None:
    assert isinstance(build_dataset({}), FakeDataset)


def test_fake_es_usable_via_abc() -> None:
    # Core solo ve la ABC: muestrear por la interfaz debe devolver n pares.
    ds: Dataset = build_dataset({"type": "fake"})
    pares = ds.sample(3)
    assert len(pares) == 3


def test_fake_respeta_semilla() -> None:
    a = build_dataset({"type": "fake", "seed": 7}).sample(2)
    b = build_dataset({"type": "fake", "seed": 7}).sample(2)
    for (_, chunk_a), (_, chunk_b) in zip(a, b, strict=True):
        for act_a, act_b in zip(chunk_a, chunk_b, strict=True):
            assert (act_a.values == act_b.values).all()


def test_fake_es_appendable() -> None:
    ds = build_dataset({"type": "fake"})
    assert isinstance(ds, AppendableDataset)


def test_tipo_pendiente_error_claro() -> None:
    # Adaptadores anunciados pero aún no implementados dan NotImplementedError.
    for tipo in ("openx",):
        with pytest.raises(NotImplementedError):
            build_dataset({"type": tipo})


def test_tipo_desconocido_valueerror() -> None:
    with pytest.raises(ValueError):
        build_dataset({"type": "no_existe"})


def test_lerobot_requiere_grupo_opcional() -> None:
    # Sin el grupo opcional instalado, construir 'lerobot' da un ImportError claro.
    import importlib.util

    if importlib.util.find_spec("lerobot") is not None:
        pytest.skip("lerobot instalado; no aplica el error de dependencia faltante")
    with pytest.raises(ImportError):
        build_dataset({"type": "lerobot", "repo_id": "x"})
