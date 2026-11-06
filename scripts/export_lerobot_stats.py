"""Estadísticas de normalización de una política de LeRobot → formato de ``Pi0Policy``.

    uv run --extra pi0 python scripts/export_lerobot_stats.py \\
        --source lerobot/pi0_libero_finetuned_v044 --out checkpoints/pi0_libero_finetuned_stats

Un π0 ya ajustado en LeRobot (p. ej. ``lerobot/pi0_libero_finetuned_v044``) trae
sus estadísticas en el pipeline de procesadores (``policy_preprocessor.json`` y
el ``.safetensors`` de su paso ``normalizer_processor``), que ``Pi0Policy`` no
usa: el adaptador normaliza dentro de la política (``Learner Design.md``,
"Normalización de acciones") con estadísticas propias. Este script las traduce a
``action_stats.pt`` y ``state_stats.pt``, los archivos que lee
``normalizer.stats: <directorio>`` del builder del Learner. Así se evalúa el
modelo con **sus** estadísticas, sin re-ajustarlas contra otro dataset.

``--source`` es un id de Hugging Face o una carpeta local con esos archivos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vla_learner.policy.normalizer import ActionNormalizer, ActionStats

ACTION_KEY = "action"
STATE_KEY = "observation.state"
ACTION_STATS_FILE = "action_stats.pt"  # los nombres que lee el builder
STATE_STATS_FILE = "state_stats.pt"
PREPROCESSOR = "policy_preprocessor.json"
# ``norm_map`` de LeRobot → modo de ``ActionNormalizer``.
MODES = {"MEAN_STD": "mean_std", "MIN_MAX": "min_max", "QUANTILES": "quantile"}
_DTYPES = {"F32": np.float32, "F64": np.float64, "F16": np.float16, "I64": np.int64}


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Lee un ``.safetensors`` con numpy (8 bytes de largo + cabecera JSON + datos)."""
    datos = Path(path).read_bytes()
    largo = int.from_bytes(datos[:8], "little")
    cabecera = json.loads(datos[8 : 8 + largo])
    cuerpo = memoryview(datos)[8 + largo :]
    tensores = {}
    for nombre, info in cabecera.items():
        if nombre == "__metadata__":
            continue
        if info["dtype"] not in _DTYPES:
            raise ValueError(f"dtype no soportado en {path}: {info['dtype']} ({nombre})")
        inicio, fin = info["data_offsets"]
        plano = np.frombuffer(cuerpo[inicio:fin], dtype=_DTYPES[info["dtype"]])
        tensores[nombre] = plano.reshape(info["shape"]).astype(np.float32)
    return tensores


def _normalizer_step(preprocessor: dict) -> dict:
    for paso in preprocessor.get("steps", []):
        if paso.get("registry_name") == "normalizer_processor":
            return paso
    raise ValueError(f"{PREPROCESSOR} no tiene un paso normalizer_processor")


def _stats(tensores: dict[str, np.ndarray], clave: str) -> ActionStats:
    campos = {}
    for campo in ("mean", "std", "min", "max", "q01", "q99"):
        nombre = f"{clave}.{campo}"
        if nombre not in tensores:
            raise KeyError(f"faltan las estadísticas {nombre!r}")
        campos[campo] = tensores[nombre].reshape(-1)
    return ActionStats(**campos)


def _mode(paso: dict, tipo: str) -> str:
    norm_map = paso.get("config", {}).get("norm_map", {})
    nombre = norm_map.get(tipo, "MEAN_STD")
    if nombre not in MODES:
        raise ValueError(f"modo de normalización de LeRobot no soportado para {tipo}: {nombre}")
    return MODES[nombre]


def resolve_source(source: str) -> Path:
    """Carpeta local con el pipeline; si ``source`` es un id de HF, se baja lo justo."""
    local = Path(source)
    if local.is_dir():
        return local
    from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]

    json_path = Path(hf_hub_download(source, PREPROCESSOR))
    estado = _normalizer_step(json.loads(json_path.read_text("utf-8")))["state_file"]
    hf_hub_download(source, estado)
    return json_path.parent


def export_stats(source_dir: Path, out: Path) -> tuple[ActionNormalizer, ActionNormalizer]:
    """Escribe ``action_stats.pt`` y ``state_stats.pt`` en ``out``; devuelve ambos."""
    paso = _normalizer_step(json.loads((source_dir / PREPROCESSOR).read_text("utf-8")))
    tensores = read_safetensors(source_dir / paso["state_file"])
    accion = ActionNormalizer(
        {"default": _stats(tensores, ACTION_KEY)},
        mode=_mode(paso, "ACTION"),  # type: ignore[arg-type]
    )
    estado = ActionNormalizer(
        {"default": _stats(tensores, STATE_KEY)},
        mode=_mode(paso, "STATE"),  # type: ignore[arg-type]
    )
    out.mkdir(parents=True, exist_ok=True)
    accion.save_file(out / ACTION_STATS_FILE)
    estado.save_file(out / STATE_STATS_FILE)
    return accion, estado


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="export_lerobot_stats")
    parser.add_argument("--source", required=True, help="Id de HF o carpeta local.")
    parser.add_argument("--out", required=True, type=Path, help="Carpeta de salida.")
    args = parser.parse_args(argv)
    accion, estado = export_stats(resolve_source(args.source), args.out)
    print(
        f"estadísticas en {args.out}: acción {accion.dim()}-D ({accion.mode}), "
        f"estado {estado.dim()}-D ({estado.mode})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
