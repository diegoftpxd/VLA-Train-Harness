"""Suite MAP: ¿la observación del ambiente se parece a la del dataset de entrenamiento?

    uv run vla-env --config configs/env/libero.yaml --smoke 20 --save out/libero_smoke
    uv run python scripts/check_env_vs_dataset.py \\
        --smoke out/libero_smoke/smoke.npz --dataset data/pi0_libero

Definida en ``docs/environment/Test Environment.md`` (MAP-01…05). Una política
entrenada con un dataset solo funciona en el ambiente si ve lo mismo: mismas
cámaras y formas, el estado en el mismo layout y rango, las imágenes con la misma
orientación y las mismas instrucciones. Nada de eso rompe la suite ENV, por eso
es un chequeo aparte. Compara el ``npz`` de ``vla-env --smoke --save`` con un
dataset en formato ``local`` (``<dir>/<episodio>/{meta.json,data.npz}``, lo que
escribe ``scripts/prepare_lerobot_local.py``). Termina con código 1 si algo falla.

Solo usa numpy: no importa el simulador, ni Core, ni el Learner.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NO_CAMARA = frozenset({"proprio", "actions", "instruction"})
MARGEN_RANGO = 0.10  # MAP-03: ± 10 % del rango del dataset, por componente


@dataclass
class Resultado:
    id: str
    ok: bool
    detalle: str


@dataclass
class Dataset:
    camaras: dict[str, tuple[int, ...]]  # nombre → forma de un frame (C, H, W)
    proprio_dim: int
    proprio_min: np.ndarray
    proprio_max: np.ndarray
    instrucciones: set[str]
    inicios: dict[str, dict[str, list[np.ndarray]]]  # instrucción → cámara → frames iniciales


def _a_float(imagen: np.ndarray) -> np.ndarray:
    if imagen.dtype == np.uint8:
        return imagen.astype(np.float32) / 255.0
    return imagen.astype(np.float32)


def cargar_dataset(directorio: Path, max_episodios: int | None = None) -> Dataset:
    """Lee el dataset ``local``: formas, rango del estado, instrucciones y frames iniciales.

    Un frame es "inicial" si es el primero de la primera ventana de su episodio
    fuente (``source_episode`` en ``meta.json``); sin ese dato, el de cada episodio.
    """
    episodios = sorted(p for p in directorio.iterdir() if (p / "data.npz").exists())
    if max_episodios is not None:
        episodios = episodios[:max_episodios]
    if not episodios:
        raise FileNotFoundError(f"no hay episodios en {directorio}")
    camaras: dict[str, tuple[int, ...]] = {}
    minimos, maximos = [], []
    instrucciones: set[str] = set()
    inicios: dict[str, dict[str, list[np.ndarray]]] = {}
    vistos: set[object] = set()
    for episodio in episodios:
        meta = json.loads((episodio / "meta.json").read_text(encoding="utf-8"))
        instruccion = meta["instruction"]
        instrucciones.add(instruccion)
        with np.load(episodio / "data.npz") as datos:
            proprio = datos["proprio"]
            minimos.append(proprio.min(axis=0))
            maximos.append(proprio.max(axis=0))
            nombres = [k for k in datos.files if k not in NO_CAMARA]
            for nombre in nombres:
                camaras.setdefault(nombre, tuple(datos[nombre].shape[1:]))
            fuente = meta.get("source_episode", episodio.name)
            if fuente not in vistos:
                vistos.add(fuente)
                por_camara = inicios.setdefault(instruccion, {})
                for nombre in nombres:
                    por_camara.setdefault(nombre, []).append(_a_float(datos[nombre][0]))
    return Dataset(
        camaras=camaras,
        proprio_dim=int(minimos[0].shape[0]),
        proprio_min=np.min(minimos, axis=0),
        proprio_max=np.max(maximos, axis=0),
        instrucciones=instrucciones,
        inicios=inicios,
    )


def _correlacion(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / den if den > 0 else 0.0


def comparar(ambiente: dict[str, np.ndarray], dataset: Dataset) -> list[Resultado]:
    """MAP-01…05 entre las observaciones del ambiente (``smoke.npz``) y el dataset."""
    camaras_env = sorted(k for k in ambiente if k not in NO_CAMARA)
    resultados: list[Resultado] = []

    faltan = sorted(set(dataset.camaras) - set(camaras_env))
    sobran = sorted(set(camaras_env) - set(dataset.camaras))
    formas = {c: tuple(ambiente[c].shape[1:]) for c in camaras_env if c in dataset.camaras}
    distintas = {c: (f, dataset.camaras[c]) for c, f in formas.items() if f != dataset.camaras[c]}
    rango_ok = all(
        ambiente[c].dtype == np.float32 and ambiente[c].min() >= 0 and ambiente[c].max() <= 1
        for c in camaras_env
    )
    resultados.append(
        Resultado(
            "MAP-01",
            not faltan and not sobran and not distintas and rango_ok,
            f"faltan {faltan}, sobran {sobran}, formas distintas {distintas}, "
            f"float32 en [0, 1]: {rango_ok}",
        )
    )

    dim_env = int(ambiente["proprio"].shape[1])
    resultados.append(
        Resultado(
            "MAP-02",
            dim_env == dataset.proprio_dim,
            f"ambiente {dim_env}-D, dataset {dataset.proprio_dim}-D",
        )
    )

    if dim_env == dataset.proprio_dim:
        inicial = ambiente["proprio"][0]
        margen = MARGEN_RANGO * (dataset.proprio_max - dataset.proprio_min)
        fuera = np.flatnonzero(
            (inicial < dataset.proprio_min - margen) | (inicial > dataset.proprio_max + margen)
        )
        resultados.append(
            Resultado(
                "MAP-03",
                fuera.size == 0,
                f"componentes fuera de rango: {fuera.tolist()} (valores {inicial[fuera].tolist()})",
            )
        )
    else:
        resultados.append(
            Resultado("MAP-03", False, "no se compara: el estado tiene otra dimensión")
        )

    instruccion = str(ambiente["instruction"][0])
    inicios = dataset.inicios.get(instruccion) or next(iter(dataset.inicios.values()))
    detalles, orientacion_ok = [], True
    for camara in camaras_env:
        if camara not in inicios:
            continue
        referencia = np.mean(inicios[camara], axis=0)
        frame = ambiente[camara][0]
        tal_cual = _correlacion(frame, referencia)
        rotada = _correlacion(frame[:, ::-1, ::-1], referencia)
        orientacion_ok &= tal_cual > rotada
        detalles.append(f"{camara}: tal cual {tal_cual:.3f} vs rotada 180° {rotada:.3f}")
    resultados.append(
        Resultado("MAP-04", orientacion_ok and bool(detalles), "; ".join(detalles) or "sin cámaras")
    )

    desconocidas = sorted({str(i) for i in ambiente["instruction"]} - dataset.instrucciones)
    resultados.append(
        Resultado("MAP-05", not desconocidas, f"instrucciones ausentes del dataset: {desconocidas}")
    )
    return resultados


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_env_vs_dataset")
    parser.add_argument("--smoke", required=True, type=Path, help="smoke.npz de vla-env --save")
    parser.add_argument("--dataset", required=True, type=Path, help="dataset en formato local")
    parser.add_argument("--max-episodes", type=int, help="limita los episodios leídos")
    args = parser.parse_args(argv)

    with np.load(args.smoke) as datos:
        ambiente = {k: datos[k] for k in datos.files}
    resultados = comparar(ambiente, cargar_dataset(args.dataset, args.max_episodes))
    for r in resultados:
        print(f"[{'OK ' if r.ok else 'MAL'}] {r.id}: {r.detalle}")
    return 0 if all(r.ok for r in resultados) else 1


if __name__ == "__main__":
    sys.exit(main())
