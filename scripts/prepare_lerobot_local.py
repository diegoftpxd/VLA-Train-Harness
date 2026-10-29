"""Convierte un corpus LeRobot al formato ``local`` que sirve Core.

Es el **paso 1 (offline, una vez)** de ``docs/dataset/Pi0 E2E Plan.md``: traduce un
dataset LeRobot real a episodios on-disk en el "formato canónico" (el que lee
``vla_core.dataset.local.LocalDataset``), troceados en ventanas de **largo fijo**
(50 por defecto) para que π0 pueda apilar las chunks del batch (``n_action_steps``).
Así el proceso Core corre con la base liviana de ``vla_core`` (ZeroMQ + numpy, **sin
torch ni lerobot**) leyendo ``type: local``, y la dependencia pesada queda solo acá,
en un script que se corre una vez.

Sirve para cualquier corpus LeRobot v3 (el que lee lerobot 0.4.4). Los dos usados:

- **pusht** (una cámara, instrucción del corpus)::

      uv run --extra pi0 python -I scripts/prepare_lerobot_local.py

- **LIBERO** (dos cámaras, 40 tareas con su instrucción; ``--max-episodes`` elige
  episodios repartidos por todo el corpus, que viene ordenado por suite y tarea)::

      uv run --extra pi0 python -I scripts/prepare_lerobot_local.py \\
          --repo-id lerobot/libero --out data/pi0_libero --max-episodes 200 \\
          --camera observation.images.image=observation.images.base_0_rgb \\
          --camera observation.images.image2=observation.images.left_wrist_0_rgb

Salida (``--out``)::

    <out>/<idx:05d>/
      meta.json     # {"instruction": "...", "source_episode": N}
      data.npz      # una clave por cámara (H,C,H,W) uint8, proprio (H,P), actions (H,A)

donde ``H`` es ``--chunk`` (50): cada ventana de un episodio del corpus se escribe
como **un** episodio ``local``, que ``LocalDataset`` con ``demo_chunk_size=50`` lee
como un único par ``(Observation, ActionChunk)`` (una transición = una chunk, la
regla de episodio de Core). Las imágenes van en ``uint8`` y comprimidas (4× menos
que ``float32``; ``LocalDataset`` las devuelve ``float32`` en ``[0, 1]``).

El corpus descargado es **dato no confiable**: va a su propio directorio de caché de
Hugging Face y solo se lee por ruta; por eso el script se corre con ``python -I``.
``lerobot`` se importa **dentro de** :func:`main`, así que importar este módulo (lo
hace el test ``fast``) no requiere tenerlo instalado: el test ejercita solo los
helpers puros de abajo sobre arrays sintéticos, sin descargar nada.

Se usa ``--extra pi0`` (no ``--extra lerobot``): ese extra vive en ``vla-core`` y no
está expuesto a nivel de workspace; el root sí expone ``pi0`` (``vla-learner[pi0]``),
que fija ``lerobot>=0.4.4``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

# Cámara canónica que exige el procesador de π0 (una de las tres que
# ``_preprocess_images`` acepta; las que falten las enmascara el modelo).
CAMERA_BASE = "observation.images.base_0_rgb"

# Clave del dict de episodio con la instrucción del corpus (``frame["task"]``).
TASK_KEY = "task"


# -- Helpers puros (sin lerobot; los ejercita el test ``fast``) --------------


def ventanas(n_pasos: int, tamano: int) -> list[tuple[int, int]]:
    """Troza ``[0, n_pasos)`` en ventanas consecutivas de **exactamente** ``tamano``.

    La cola ``< tamano`` se **descarta**, de modo que todas las ventanas miden lo
    mismo (restricción de π0: ``torch.stack`` exige chunks de igual largo). Con
    ``n_pasos=123, tamano=50`` devuelve ``[(0, 50), (50, 100)]`` (se pierde
    ``[100, 123)``).
    """
    if tamano < 1:
        raise ValueError("tamano debe ser >= 1")
    n = (n_pasos // tamano) * tamano
    return [(i, i + tamano) for i in range(0, n, tamano)]


def elegir_episodios(total: int, max_episodes: int | None) -> list[int]:
    """Índices de episodio a convertir: todos, o ``max_episodes`` repartidos parejo.

    Los corpus como LIBERO vienen ordenados por suite y tarea: tomar los primeros
    ``N`` dejaría afuera casi todas las tareas. Repartirlos (``linspace``) cubre
    todo el corpus de forma determinista.
    """
    if max_episodes is None or max_episodes >= total:
        return list(range(total))
    if max_episodes < 1:
        raise ValueError("max_episodes debe ser >= 1")
    return sorted({int(i) for i in np.linspace(0, total - 1, max_episodes).round()})


def parse_camaras(specs: list[str] | None) -> dict[str, str]:
    """``["src=dst", ...]`` → ``{src: dst}`` (clave del corpus → clave canónica)."""
    camaras: dict[str, str] = {}
    for spec in specs or []:
        src, sep, dst = spec.partition("=")
        if not sep or not src or not dst:
            raise ValueError(f"--camera espera 'clave_corpus=clave_canonica', no {spec!r}")
        camaras[src] = dst
    return camaras


def a_uint8(imagenes: np.ndarray) -> np.ndarray:
    """Imágenes ``float`` en ``[0, 1]`` (lo que decodifica lerobot) → ``uint8`` 0–255."""
    arr = np.asarray(imagenes)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(np.rint(arr.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)


def escribir_episodio(
    out_dir: Path,
    idx: int,
    *,
    images: dict[str, np.ndarray],
    proprio: np.ndarray,
    actions: np.ndarray,
    instruction: str,
    source_episode: int | None = None,
) -> Path:
    """Escribe una ventana como un episodio en formato ``local``.

    ``images`` (clave canónica → ``(H,C,H,W)``), ``proprio`` y ``actions`` ya vienen
    recortados a la ventana (mismo número de pasos ``H``). Las acciones y la
    propriocepción se guardan en **unidades reales**, ``float32``, **sin
    normalizar** (normaliza la política). Las imágenes en ``uint8``. No se usa
    ``pickle``: ``np.savez_compressed`` con arrays crudos.
    """
    proprio = np.asarray(proprio, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    h = actions.shape[0]
    largos = {k: v.shape[0] for k, v in images.items()}
    if proprio.shape[0] != h or any(n != h for n in largos.values()):
        raise ValueError(
            f"imágenes, proprio y actions deben tener el mismo largo; "
            f"got {largos}, {proprio.shape[0]}, {h}"
        )
    ep_dir = out_dir / f"{idx:05d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    # ``Any``: las claves son nombres de arrays del npz, no parámetros de
    # ``savez_compressed`` (pyright no puede saber que ninguna es ``allow_pickle``).
    contenido: dict[str, Any] = {clave: a_uint8(img) for clave, img in images.items()}
    contenido.update(proprio=proprio, actions=actions)
    np.savez_compressed(ep_dir / "data.npz", **contenido)
    meta: dict[str, Any] = {"instruction": instruction}
    if source_episode is not None:
        meta["source_episode"] = source_episode
    (ep_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ep_dir


def preparar_desde_episodios(
    episodios: Iterable[dict[str, Any]],
    out_dir: Path,
    *,
    chunk: int,
    cameras: dict[str, str],
    state_key: str,
    action_key: str,
    instruction: str | None = None,
) -> int:
    """Trocea cada episodio en ventanas de ``chunk`` y las escribe como ``local``.

    ``episodios`` es un iterable de dicts con los arrays **completos** del episodio
    (``{cam_corpus: (T,C,H,W), state_key: (T,P), action_key: (T,A)}``), su
    instrucción en ``"task"`` y, opcional, su índice en ``"episode_index"``.
    ``instruction`` la reemplaza para todos los episodios. ``cameras`` mapea clave
    del corpus → clave canónica. Devuelve cuántos episodios ``local`` (ventanas) se
    escribieron. Parte pura: no toca lerobot ni la red.
    """
    if not cameras:
        raise ValueError("hace falta al menos una cámara (--camera)")
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for ep in episodios:
        texto = instruction if instruction is not None else ep.get(TASK_KEY)
        if not texto:
            raise ValueError("el episodio no trae instrucción ('task') y no se pasó --instruction")
        proprio = np.asarray(ep[state_key])
        actions = np.asarray(ep[action_key])
        imagenes = {dst: np.asarray(ep[src]) for src, dst in cameras.items()}
        for inicio, fin in ventanas(actions.shape[0], chunk):
            escribir_episodio(
                out_dir,
                idx,
                images={k: v[inicio:fin] for k, v in imagenes.items()},
                proprio=proprio[inicio:fin],
                actions=actions[inicio:fin],
                instruction=str(texto),
                source_episode=ep.get("episode_index"),
            )
            idx += 1
    return idx


def agrupar_por_episodio(
    frames: Iterable[dict[str, Any]], *, keys: list[str]
) -> Iterator[dict[str, Any]]:
    """Agrupa frames consecutivos por ``episode_index`` y emite un episodio a la vez.

    LeRobot entrega los frames ordenados por episodio, así que basta con cortar
    cuando cambia el índice: en memoria queda **un** episodio, no el corpus entero
    (LIBERO decodificado ocuparía cientos de GB). ``keys`` son los arrays a apilar;
    ``task`` se toma del primer frame del episodio.
    """
    actual: int | None = None
    buffer: dict[str, list[np.ndarray]] = {}
    task: Any = None

    def _emitir() -> dict[str, Any]:
        episodio: dict[str, Any] = {k: np.stack(v) for k, v in buffer.items()}
        episodio[TASK_KEY] = task
        episodio["episode_index"] = actual
        return episodio

    for frame in frames:
        ep = int(frame["episode_index"])
        if ep != actual:
            if actual is not None:
                yield _emitir()
            actual, buffer, task = ep, {k: [] for k in keys}, frame.get(TASK_KEY)
        for k in keys:
            buffer[k].append(np.asarray(frame[k]))
    if actual is not None:
        yield _emitir()


# -- Carga del corpus LeRobot (aislada; solo en main) ------------------------


def _cargar_lerobot(repo_id: str, episodes: list[int] | None) -> Any:
    """Abre un ``LeRobotDataset`` (solo ``episodes``, si se dan).

    ``pusht`` y ``LIBERO`` guardan las cámaras como vídeo y lerobot las decodifica al
    indexar. Su backend por defecto es ``torchcodec``, cuyo wheel (0.10) choca de ABI
    con torch 2.6 (``undefined symbol`` de ``libc10`` al cargar libtorchcodec). Se
    fuerza ``pyav``, que trae su propio FFmpeg (con ``libdav1d`` para el AV1 de
    LIBERO) y decodifica vía torchvision, sin depender de torch.

    lerobot >=0.4 usa ``lerobot.datasets.lerobot_dataset``; versiones previas,
    ``lerobot.common.datasets.lerobot_dataset``.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except ImportError:  # layout viejo (< 0.4)
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    return LeRobotDataset(repo_id, episodes=episodes, video_backend="pyav")


def _total_episodios(repo_id: str) -> int:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore
    except ImportError:  # layout viejo (< 0.4)
        from lerobot.common.datasets.lerobot_dataset import (  # type: ignore
            LeRobotDatasetMetadata,
        )
    return int(LeRobotDatasetMetadata(repo_id).total_episodes)


def _frames_de_lerobot(dataset: Any, keys: list[str]) -> Iterator[dict[str, Any]]:
    """Frames decodificados como numpy (solo ``keys`` + índice de episodio y tarea).

    Las imágenes de lerobot vienen ``CHW`` float en ``[0, 1]``; :func:`a_uint8` las
    pasa a ``uint8`` al escribir. Es la glue específica de lerobot: si la API de tu
    versión cambia, se ajusta acá.
    """
    import torch

    def _np(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    for i in range(len(dataset)):
        frame = dataset[i]
        salida: dict[str, Any] = {k: _np(frame[k]) for k in keys}
        salida["episode_index"] = int(_np(frame["episode_index"]))
        salida[TASK_KEY] = frame.get(TASK_KEY)
        if (i + 1) % 5000 == 0:
            print(f"   {i + 1}/{len(dataset)} frames decodificados", flush=True)
        yield salida


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id", default="lerobot/pusht", help="id de HF o ruta a un snapshot local"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/pi0_pusht"), help="directorio de salida"
    )
    parser.add_argument(
        "--chunk", type=int, default=50, help="pasos por episodio local (= n_action_steps de π0)"
    )
    parser.add_argument(
        "--camera",
        action="append",
        help=(
            "clave_corpus=clave_canonica (repetible). Por defecto, la cámara de pusht: "
            f"observation.image={CAMERA_BASE}"
        ),
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="instrucción fija para todos los episodios (por defecto, la del corpus)",
    )
    parser.add_argument(
        "--state-key", default="observation.state", help="clave de propriocepción en el corpus"
    )
    parser.add_argument("--action-key", default="action", help="clave de acción en el corpus")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="convertir solo N episodios, repartidos por todo el corpus",
    )
    args = parser.parse_args()

    camaras = parse_camaras(args.camera) or {"observation.image": CAMERA_BASE}
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(
            f"{args.out} no está vacío: borralo o usá otro --out (no se mezclan corridas)."
        )

    total = _total_episodios(args.repo_id)
    elegidos = elegir_episodios(total, args.max_episodes)
    print(
        f">> Cargando corpus LeRobot {args.repo_id!r}: {len(elegidos)} de {total} episodios ...",
        flush=True,
    )
    dataset = _cargar_lerobot(args.repo_id, None if len(elegidos) == total else elegidos)
    keys = [*camaras, args.state_key, args.action_key]
    episodios = agrupar_por_episodio(_frames_de_lerobot(dataset, keys), keys=keys)
    escritos = preparar_desde_episodios(
        episodios,
        args.out,
        chunk=args.chunk,
        cameras=camaras,
        state_key=args.state_key,
        action_key=args.action_key,
        instruction=args.instruction,
    )
    print(f">> Listo: {escritos} episodios de {args.chunk} pasos en {args.out}")
    if escritos == 0:
        raise SystemExit(
            "No se escribió ningún episodio: ¿todos los episodios del corpus miden "
            f"menos de --chunk={args.chunk}? Bajá --chunk o revisá --repo-id."
        )


if __name__ == "__main__":
    main()
