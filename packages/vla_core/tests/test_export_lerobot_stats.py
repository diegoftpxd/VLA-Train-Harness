"""Suite STATS: ``scripts/export_lerobot_stats.py`` (``docs/environment/Test Environment.md``).

Las estadísticas de un pipeline de LeRobot se traducen al formato que lee
``normalizer.stats: <directorio>`` del builder del Learner. Se prueba con un
pipeline sintético escrito a mano (mismo formato ``.safetensors``), sin red.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vla_learner.policy.normalizer import ActionNormalizer

pytestmark = pytest.mark.fast

REPO = Path(__file__).resolve().parents[3]


def _script():
    ruta = REPO / "scripts" / "export_lerobot_stats.py"
    spec = importlib.util.spec_from_file_location("export_lerobot_stats", ruta)
    assert spec and spec.loader
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    return modulo


EXPORT = _script()
CAMPOS = ("mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99")


def _escribir_safetensors(ruta: Path, tensores: dict[str, np.ndarray]) -> None:
    cabecera, cuerpo, offset = {}, b"", 0
    for nombre, valor in tensores.items():
        datos = np.ascontiguousarray(valor, dtype=np.float32).tobytes()
        cabecera[nombre] = {
            "dtype": "F32",
            "shape": list(valor.shape),
            "data_offsets": [offset, offset + len(datos)],
        }
        cuerpo += datos
        offset += len(datos)
    cabecera["__metadata__"] = {"format": "pt"}
    texto = json.dumps(cabecera).encode()
    ruta.write_bytes(len(texto).to_bytes(8, "little") + texto + cuerpo)


def _pipeline(directorio: Path, norm_map: dict[str, str] | None = None) -> dict:
    rng = np.random.default_rng(0)
    tensores = {}
    for clave, dim in (("action", 7), ("observation.state", 8)):
        for campo in CAMPOS:
            tensores[f"{clave}.{campo}"] = rng.normal(size=dim).astype(np.float32)
        tensores[f"{clave}.std"] = np.abs(tensores[f"{clave}.std"]) + 0.1
    tensores["observation.images.image.mean"] = np.zeros((3, 1, 1), dtype=np.float32)
    _escribir_safetensors(directorio / "norm.safetensors", tensores)
    preprocesador = {
        "steps": [
            {"registry_name": "tokenizer_processor", "config": {}},
            {
                "registry_name": "normalizer_processor",
                "config": {"norm_map": norm_map or {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"}},
                "state_file": "norm.safetensors",
            },
        ]
    }
    (directorio / "policy_preprocessor.json").write_text(json.dumps(preprocesador), "utf-8")
    return tensores


def test_stats01_exporta_accion_y_estado_en_el_formato_del_builder(tmp_path: Path) -> None:
    tensores = _pipeline(tmp_path)
    salida = tmp_path / "stats"
    assert EXPORT.main(["--source", str(tmp_path), "--out", str(salida)]) == 0
    accion = ActionNormalizer.from_file(salida / "action_stats.pt")
    estado = ActionNormalizer.from_file(salida / "state_stats.pt")
    assert (accion.dim(), estado.dim()) == (7, 8)
    assert accion.mode == estado.mode == "mean_std"
    x = np.zeros(7, dtype=np.float32)
    import torch

    normalizada = accion.normalize(torch.from_numpy(x), "default").numpy()
    esperado = (x - tensores["action.mean"]) / tensores["action.std"]
    assert np.allclose(normalizada, esperado, atol=1e-6)


def test_stats02_modo_desde_el_norm_map(tmp_path: Path) -> None:
    _pipeline(tmp_path, {"STATE": "MIN_MAX", "ACTION": "QUANTILES"})
    accion, estado = EXPORT.export_stats(tmp_path, tmp_path / "s")
    assert (accion.mode, estado.mode) == ("quantile", "min_max")


def test_stats03_errores_claros(tmp_path: Path) -> None:
    _pipeline(tmp_path, {"ACTION": "RARO"})
    with pytest.raises(ValueError, match="no soportado"):
        EXPORT.export_stats(tmp_path, tmp_path / "s")
    (tmp_path / "policy_preprocessor.json").write_text('{"steps": []}', "utf-8")
    with pytest.raises(ValueError, match="normalizer_processor"):
        EXPORT.export_stats(tmp_path, tmp_path / "s")
