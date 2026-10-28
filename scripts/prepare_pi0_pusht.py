"""Convierte el corpus LeRobot ``pusht`` al formato ``local`` que sirve Core.

Es el **paso 1 (offline, una vez)** del plan ``docs/dataset/Pi0 E2E Plan.md``:
traduce un dataset LeRobot real a episodios on-disk en el "formato canónico"
(el que lee ``vla_core.dataset.local.LocalDataset``), troceados en ventanas de
**largo fijo** (50 por defecto) para que π0 pueda apilar las chunks del batch
(``n_action_steps``). Así el proceso Core corre con la base liviana de
``vla_core`` (ZeroMQ + numpy, **sin torch ni lerobot**) leyendo ``type: local``,
y la dependencia pesada queda solo acá, en un script que se corre una vez.

Salida (``--out``, por defecto ``data/pi0_pusht``)::

    <out>/<idx:04d>/
      meta.json     # {"instruction": "..."}
      data.npz      # observation.images.base_0_rgb (H,C,H,W)?  proprio (H,P)  actions (H,A)

donde ``H`` es ``--chunk`` (50) y cada ventana de un episodio de pusht se escribe
como **un** episodio ``local`` de 50 pasos; ``LocalDataset`` con
``demo_chunk_size=50`` lo lee como un único par ``(Observation, ActionChunk)`` de
chunk 50 (una transición = una chunk, la regla de episodio de Core).

Uso (en el cluster o donde esté instalado el extra ``pi0``, que trae lerobot
0.4.4)::

    uv run --extra pi0 python -I scripts/prepare_pi0_pusht.py

El corpus descargado es **dato no confiable**: va a su propio directorio de caché
de Hugging Face y solo se lee por ruta; por eso el script se corre con ``python
-I`` (aislado, sin el cwd ni PYTHONPATH en ``sys.path``). ``lerobot`` se importa
**dentro de** :func:`main`, así que importar este módulo (lo hace el test ``fast``)
no requiere tenerlo instalado: el test ejercita solo los helpers puros de abajo
sobre arrays sintéticos, sin descargar nada.

Nota sobre el extra: el plan menciona ``--extra lerobot``, pero ese extra vive en
``vla-core`` y no está expuesto a nivel de workspace; el root sí expone ``pi0``
(``vla-learner[pi0]``), que fija ``lerobot>=0.4.4`` —la versión cuyo path de
import (``lerobot.datasets.lerobot_dataset``) usa este script—. De ahí
``--extra pi0``. El import de abajo igual prueba el layout viejo como respaldo.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

# Clave de cámara canónica que exige el procesador de π0 (una de las tres que
# ``_preprocess_images`` acepta; las otras dos las enmascara el modelo).
CAMERA_BASE = "observation.images.base_0_rgb"


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


def escribir_episodio(
    out_dir: Path,
    idx: int,
    *,
    images: np.ndarray,
    proprio: np.ndarray,
    actions: np.ndarray,
    instruction: str,
    camera_key: str = CAMERA_BASE,
) -> Path:
    """Escribe una ventana como un episodio en formato ``local``.

    ``images``/``proprio``/``actions`` ya vienen recortados a la ventana (mismo
    número de pasos ``H``). Las acciones y la propriocepción se guardan en
    **unidades reales**, ``float32``, **sin normalizar** (el adaptador de π0
    normaliza y padea a 32 adentro). No se usa ``pickle``: ``np.savez`` con
    arrays crudos.
    """
    images = np.asarray(images, dtype=np.float32)
    proprio = np.asarray(proprio, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    h = actions.shape[0]
    if images.shape[0] != h or proprio.shape[0] != h:
        raise ValueError(
            f"imágenes, proprio y actions deben tener el mismo largo; "
            f"got {images.shape[0]}, {proprio.shape[0]}, {h}"
        )
    ep_dir = out_dir / f"{idx:04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ep_dir / "data.npz", **{camera_key: images, "proprio": proprio, "actions": actions})
    (ep_dir / "meta.json").write_text(
        json.dumps({"instruction": instruction}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ep_dir


def preparar_desde_episodios(
    episodios: Iterable[dict[str, np.ndarray]],
    out_dir: Path,
    *,
    instruction: str,
    chunk: int,
    image_key: str,
    state_key: str,
    action_key: str,
    camera_key: str = CAMERA_BASE,
) -> int:
    """Trocea cada episodio en ventanas de ``chunk`` y las escribe como ``local``.

    ``episodios`` es un iterable de dicts con los arrays **completos** del
    episodio (``{image_key: (T,C,H,W), state_key: (T,P), action_key: (T,A)}``).
    Devuelve cuántos episodios ``local`` (ventanas) se escribieron. Parte pura:
    no toca lerobot ni la red, así que el test la corre con arrays sintéticos.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for ep in episodios:
        images = np.asarray(ep[image_key])
        proprio = np.asarray(ep[state_key])
        actions = np.asarray(ep[action_key])
        for inicio, fin in ventanas(actions.shape[0], chunk):
            escribir_episodio(
                out_dir,
                idx,
                images=images[inicio:fin],
                proprio=proprio[inicio:fin],
                actions=actions[inicio:fin],
                instruction=instruction,
                camera_key=camera_key,
            )
            idx += 1
    return idx


# -- Carga del corpus LeRobot (aislada; solo en main) ------------------------


def _cargar_lerobot(repo_id: str) -> Any:
    """Abre un ``LeRobotDataset`` probando el layout nuevo y, como respaldo, el viejo.

    lerobot >=0.4 usa ``lerobot.datasets.lerobot_dataset``; versiones previas,
    ``lerobot.common.datasets.lerobot_dataset``. Mismo doble-import que el
    adaptador de π0 (``vla_learner/policy/adapters/pi0.py``); este es el único
    punto a tocar si tu versión difiere.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except ImportError:  # layout viejo (< 0.4)
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    return LeRobotDataset(repo_id)


def _episodios_de_lerobot(
    dataset: Any, *, image_key: str, state_key: str, action_key: str, max_episodes: int | None
) -> Iterator[dict[str, np.ndarray]]:
    """Agrupa los frames decodificados del dataset por ``episode_index``.

    Indexa ``dataset[i]`` (decodifica el frame de vídeo) y acumula por episodio.
    Las imágenes de lerobot vienen ``CHW`` float en ``[0, 1]``; se dejan tal cual
    (el procesador de π0 acepta ese formato). Es la glue específica de lerobot:
    si la API de tu versión cambia, se ajusta acá.
    """
    import torch

    def _np(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    buckets: dict[int, dict[str, list[np.ndarray]]] = {}
    orden: list[int] = []
    for i in range(len(dataset)):
        frame = dataset[i]
        ep = int(_np(frame["episode_index"]))
        if ep not in buckets:
            if max_episodes is not None and len(orden) >= max_episodes:
                continue
            buckets[ep] = {image_key: [], state_key: [], action_key: []}
            orden.append(ep)
        buckets[ep][image_key].append(_np(frame[image_key]))
        buckets[ep][state_key].append(_np(frame[state_key]))
        buckets[ep][action_key].append(_np(frame[action_key]))

    for ep in orden:
        b = buckets[ep]
        yield {
            image_key: np.stack(b[image_key]),
            state_key: np.stack(b[state_key]),
            action_key: np.stack(b[action_key]),
        }


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
        "--instruction", default="push the T block", help="instrucción (lenguaje) de la tarea"
    )
    parser.add_argument(
        "--image-key", default="observation.image", help="clave de cámara en el corpus"
    )
    parser.add_argument(
        "--state-key", default="observation.state", help="clave de propriocepción en el corpus"
    )
    parser.add_argument("--action-key", default="action", help="clave de acción en el corpus")
    parser.add_argument(
        "--max-episodes", type=int, default=None, help="tope de episodios del corpus (debug)"
    )
    args = parser.parse_args()

    print(f">> Cargando corpus LeRobot {args.repo_id!r} ...")
    dataset = _cargar_lerobot(args.repo_id)
    episodios = _episodios_de_lerobot(
        dataset,
        image_key=args.image_key,
        state_key=args.state_key,
        action_key=args.action_key,
        max_episodes=args.max_episodes,
    )
    escritos = preparar_desde_episodios(
        episodios,
        args.out,
        instruction=args.instruction,
        chunk=args.chunk,
        image_key=args.image_key,
        state_key=args.state_key,
        action_key=args.action_key,
    )
    print(f">> Listo: {escritos} episodios de {args.chunk} pasos en {args.out}")
    if escritos == 0:
        raise SystemExit(
            "No se escribió ningún episodio: ¿todos los episodios del corpus miden "
            f"menos de --chunk={args.chunk}? Bajá --chunk o revisá --repo-id."
        )


if __name__ == "__main__":
    main()
