"""``LeRobotDataset``: adaptador de solo-lectura sobre un corpus LeRobot.

Definido en ``docs/dataset/Dataset Design.md``. Traduce un dataset en formato
LeRobot (HuggingFace: parquet + vídeos) al tipo canónico del harness
``(Observation, ActionChunk)``, en **unidades reales** y ``float32``, sin
normalizar. Es un corpus congelado: implementa solo ``Dataset`` (no acepta
reinyección).

La dependencia ``lerobot`` es pesada (arrastra ``torch``) y vive en el grupo
opcional uv ``lerobot``. Por eso **no se importa en el top-level**: se importa de
forma perezosa dentro de ``__init__``, igual que el resto de adaptadores en la
factory. Así ``vla_core`` base sigue sin ``torch`` y este módulo se puede
importar aunque el grupo no esté instalado (falla solo al construir la clase).

Supuestos sobre el formato (LeRobot v2). Cada frame del dataset subyacente es un
diccionario con:

- ``observation.images.<cam>`` → imagen ``(C, H, W)`` (una clave por cámara),
- ``observation.state``        → vector de estado (propriocepción),
- ``action``                   → acción de bajo nivel,
- ``episode_index``            → índice de episodio (acota los chunks),
- ``task``                     → instrucción en lenguaje natural.

``cameras`` mapea nombre canónico → clave del frame (p. ej.
``{"front": "observation.images.top"}``); si es ``None`` se toman todas las
claves ``observation.images.*``. Las claves de estado/acción/tarea son
configurables para tolerar variantes del corpus.

Este adaptador se valida con la suite marcada ``heavy`` (no corre en la máquina
de desarrollo): requiere el corpus completo o un snapshot local.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_common.types import Action, ActionChunk, Observation
from vla_core.dataset.base import Dataset

_PREFIJO_IMAGEN = "observation.images."


class LeRobotDataset(Dataset):
    """Corpus LeRobot servido como pares canónicos (solo-lectura)."""

    def __init__(
        self,
        repo_id: str,
        root: str | None = None,
        cameras: dict[str, str] | None = None,
        state_key: str = "observation.state",
        action_key: str = "action",
        task_key: str = "task",
        demo_chunk_size: int = 1,
        seed: int = 0,
    ) -> None:
        if demo_chunk_size < 1:
            raise ValueError("demo_chunk_size debe ser >= 1")
        # Import perezoso: la dependencia pesada solo se exige al construir.
        try:
            # El grupo opcional 'lerobot' no está en el venv base (dep pesada); el
            # import vive aquí dentro y pyright no puede resolverlo sin el extra.
            from lerobot.common.datasets.lerobot_dataset import (  # pyright: ignore[reportMissingImports]
                LeRobotDataset as _LeRobotDataset,
            )
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise ImportError(
                "el adaptador 'lerobot' requiere el grupo opcional uv 'lerobot' "
                "(instálalo con `uv sync --extra lerobot`)"
            ) from exc

        self._ds = _LeRobotDataset(repo_id, root=root)
        self._cameras = cameras
        self._state_key = state_key
        self._action_key = action_key
        self._task_key = task_key
        self._demo_chunk_size = demo_chunk_size
        self._rng = np.random.default_rng(seed)

    # -- utilidades de mapeo --------------------------------------------------

    @staticmethod
    def _np1d(valor: Any) -> np.ndarray:
        """Convierte un tensor/array de un frame a ``ndarray`` 1-D ``float32``."""
        arr = valor.numpy() if hasattr(valor, "numpy") else np.asarray(valor)
        return np.asarray(arr, dtype=np.float32).reshape(-1)

    @staticmethod
    def _np_img(valor: Any) -> np.ndarray:
        arr = valor.numpy() if hasattr(valor, "numpy") else np.asarray(valor)
        return np.asarray(arr, dtype=np.float32)

    def _nombres_camara(self, frame: dict[str, Any]) -> dict[str, str]:
        if self._cameras is not None:
            return self._cameras
        return {
            clave[len(_PREFIJO_IMAGEN) :]: clave
            for clave in frame
            if clave.startswith(_PREFIJO_IMAGEN)
        }

    def _observacion(self, frame: dict[str, Any]) -> Observation:
        camaras = self._nombres_camara(frame)
        return Observation(
            images={nombre: self._np_img(frame[clave]) for nombre, clave in camaras.items()},
            proprio=self._np1d(frame[self._state_key]),
            instruction=str(frame.get(self._task_key, "")),
        )

    def _episodio_de(self, i: int) -> int:
        valor = self._ds[i]["episode_index"]
        return int(valor.item() if hasattr(valor, "item") else valor)

    # -- interfaz Dataset -----------------------------------------------------

    def sample(self, n: int, filters: dict | None = None) -> list[tuple[Observation, ActionChunk]]:
        total = len(self._ds)
        task = (filters or {}).get("task")

        # Índices de frame candidatos (respetan el filtro de tarea si lo hay).
        if task is None:
            candidatos = list(range(total))
        else:
            candidatos = [
                i for i in range(total) if str(self._ds[i].get(self._task_key, "")) == task
            ]
        if not candidatos:
            raise ValueError(f"ningún frame cumple los filtros: {filters!r}")

        elegidos = self._rng.integers(0, len(candidatos), size=n)
        return [self._par_desde(candidatos[i], total) for i in elegidos]

    def _par_desde(self, inicio: int, total: int) -> tuple[Observation, ActionChunk]:
        """Arma el par a partir del frame ``inicio``, agrupando el chunk dentro del episodio."""
        frame0 = self._ds[inicio]
        obs = self._observacion(frame0)
        episodio = self._episodio_de(inicio)
        chunk: ActionChunk = [Action(values=self._np1d(frame0[self._action_key]))]
        # Agrega acciones consecutivas mientras sigamos en el mismo episodio.
        for i in range(inicio + 1, min(inicio + self._demo_chunk_size, total)):
            if self._episodio_de(i) != episodio:
                break
            chunk.append(Action(values=self._np1d(self._ds[i][self._action_key])))
        return obs, chunk
