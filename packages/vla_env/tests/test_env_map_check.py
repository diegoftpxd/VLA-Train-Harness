"""Suite MAP sobre datos sintéticos: ``scripts/check_env_vs_dataset.py`` detecta
cada tipo de desajuste entre el ambiente y el dataset (``fast``).

El chequeo real contra LIBERO corre en el cluster (``slurm/libero_env_check.sbatch``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.fast

REPO = Path(__file__).resolve().parents[3]
CAM = "observation.images.base_0_rgb"
TAREA = "put the bowl on the plate"


def _script():
    ruta = REPO / "scripts" / "check_env_vs_dataset.py"
    spec = importlib.util.spec_from_file_location("check_env_vs_dataset", ruta)
    assert spec and spec.loader
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo  # dataclasses lo busca ahí al definir las clases
    spec.loader.exec_module(modulo)
    return modulo


def _escena(rng: np.random.Generator) -> np.ndarray:
    """Imagen asimétrica (gradiente + mancha arriba a la izquierda), CHW en [0, 1]."""
    imagen = np.tile(np.linspace(0, 1, 16, dtype=np.float32), (3, 16, 1))
    imagen[:, :4, :4] = 1.0
    return np.clip(imagen + rng.normal(0, 0.02, imagen.shape).astype(np.float32), 0, 1)


def _dataset(directorio: Path, rng: np.random.Generator) -> Path:
    for i in range(3):
        episodio = directorio / f"{i:05d}"
        episodio.mkdir(parents=True)
        frames = np.stack([_escena(rng) for _ in range(5)])
        # ``Any``: las claves son nombres de arrays, no parámetros de savez_compressed.
        contenido: dict[str, Any] = {
            CAM: (frames * 255).astype(np.uint8),
            "proprio": rng.uniform(-1, 1, size=(5, 8)).astype(np.float32),
            "actions": np.zeros((5, 7), dtype=np.float32),
        }
        np.savez_compressed(episodio / "data.npz", **contenido)
        meta = {"instruction": TAREA, "source_episode": i // 2}  # 0 y 1 son del mismo
        (episodio / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directorio


def _ambiente(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        CAM: np.stack([_escena(rng) for _ in range(3)]),
        "proprio": rng.uniform(-0.5, 0.5, size=(3, 8)).astype(np.float32),
        "instruction": np.asarray([TAREA] * 3),
    }


@pytest.fixture
def entorno(tmp_path: Path):
    rng = np.random.default_rng(0)
    modulo = _script()
    return modulo, modulo.cargar_dataset(_dataset(tmp_path / "ds", rng)), _ambiente(rng)


def _estado(resultados) -> dict[str, bool]:
    return {r.id: r.ok for r in resultados}


def test_map_todo_coincide(entorno) -> None:
    modulo, dataset, ambiente = entorno
    assert len(dataset.inicios[TAREA][CAM]) == 2  # un inicio por episodio fuente
    assert _estado(modulo.comparar(ambiente, dataset)) == dict.fromkeys(
        ["MAP-01", "MAP-02", "MAP-03", "MAP-04", "MAP-05"], True
    )


def test_map01_camara_o_forma_distinta(entorno) -> None:
    modulo, dataset, ambiente = entorno
    ambiente["otra.camara"] = ambiente.pop(CAM)
    assert not _estado(modulo.comparar(ambiente, dataset))["MAP-01"]


def test_map02_y_03_estado(entorno) -> None:
    modulo, dataset, ambiente = entorno
    fuera = dict(ambiente, proprio=ambiente["proprio"] + 5.0)
    assert not _estado(modulo.comparar(fuera, dataset))["MAP-03"]
    corto = dict(ambiente, proprio=ambiente["proprio"][:, :7])
    estado = _estado(modulo.comparar(corto, dataset))
    assert not estado["MAP-02"] and not estado["MAP-03"]


def test_map04_detecta_la_rotacion(entorno) -> None:
    modulo, dataset, ambiente = entorno
    rotado = dict(ambiente, **{CAM: ambiente[CAM][:, :, ::-1, ::-1].copy()})
    assert not _estado(modulo.comparar(rotado, dataset))["MAP-04"]


def test_map05_instruccion_desconocida(entorno) -> None:
    modulo, dataset, ambiente = entorno
    otra = dict(ambiente, instruction=np.asarray(["open the drawer"] * 3))
    assert not _estado(modulo.comparar(otra, dataset))["MAP-05"]


def test_main_devuelve_1_si_falla(tmp_path: Path, entorno) -> None:
    modulo, _, ambiente = entorno
    smoke = tmp_path / "smoke.npz"
    np.savez(smoke, **ambiente)
    assert modulo.main(["--smoke", str(smoke), "--dataset", str(tmp_path / "ds")]) == 0
    np.savez(smoke, **dict(ambiente, proprio=ambiente["proprio"] + 5.0))
    assert modulo.main(["--smoke", str(smoke), "--dataset", str(tmp_path / "ds")]) == 1
