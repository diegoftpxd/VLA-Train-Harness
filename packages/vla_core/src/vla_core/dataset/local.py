"""``LocalDataset``: adaptador de solo-lectura sobre un layout on-disk simple.

Definido en ``docs/dataset/Dataset Design.md``. Es el **adaptador de referencia**:
no depende de nada externo (solo numpy), así que su suite de contrato corre en
``fast``, offline y determinista. Fija además el "formato canónico en disco" al
que los adaptadores reales (LeRobot, Open-X) pueden exportar.

Layout (``path`` apunta a un directorio de episodios)::

    <path>/
      <episodio>/
        meta.json     # {"instruction": "...", "embodiment": "..."?}
        data.npz      # proprio (T, P), actions (T, A), y una clave por cámara (T, C, H, W)

Cada par ``(Observation, ActionChunk)`` sale de agrupar ``demo_chunk_size`` pasos
consecutivos (sin solape): la observación es el primer paso del grupo y el chunk
son sus acciones, en **unidades reales** y ``float32``, sin normalizar. Si el
episodio termina a mitad de grupo, el chunk se corta ahí.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vla_common.types import Action, ActionChunk, Observation
from vla_core.dataset.base import Dataset

# Claves de ``data.npz`` que no son cámaras.
_NO_CAMARA = frozenset({"proprio", "actions"})


class LocalDataset(Dataset):
    """Dataset de solo-lectura leído desde disco (ver el módulo para el layout)."""

    def __init__(
        self,
        path: str | Path,
        cameras: dict[str, str] | None = None,
        demo_chunk_size: int = 1,
        seed: int = 0,
    ) -> None:
        self._path = Path(path)
        if not self._path.is_dir():
            raise FileNotFoundError(f"no existe el directorio de dataset local: {self._path}")
        if demo_chunk_size < 1:
            raise ValueError("demo_chunk_size debe ser >= 1")
        # ``cameras`` mapea nombre canónico -> clave en el npz. None = autodetectar.
        self._cameras = cameras
        self._demo_chunk_size = demo_chunk_size
        self._rng = np.random.default_rng(seed)
        # Cargamos todos los pares una vez; el fixture es chico y así el muestreo
        # (con reemplazo) es barato y determinista.
        self._pares: list[tuple[Observation, ActionChunk]] = self._cargar()
        if not self._pares:
            raise ValueError(f"el dataset local no tiene pares: {self._path}")

    # -- carga ----------------------------------------------------------------

    def _cargar(self) -> list[tuple[Observation, ActionChunk]]:
        pares: list[tuple[Observation, ActionChunk]] = []
        for ep_dir in sorted(p for p in self._path.iterdir() if p.is_dir()):
            meta_path = ep_dir / "meta.json"
            data_path = ep_dir / "data.npz"
            if not (meta_path.exists() and data_path.exists()):
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            instruction = meta["instruction"]
            # ``allow_pickle=False``: nunca cargamos objetos arbitrarios desde disco.
            with np.load(data_path, allow_pickle=False) as data:
                pares.extend(self._pares_de_episodio(dict(data), instruction))
        return pares

    def _nombres_camara(self, data: dict[str, np.ndarray]) -> dict[str, str]:
        if self._cameras is not None:
            return self._cameras
        return {k: k for k in data if k not in _NO_CAMARA}

    def _pares_de_episodio(
        self, data: dict[str, np.ndarray], instruction: str
    ) -> list[tuple[Observation, ActionChunk]]:
        proprio = np.asarray(data["proprio"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        camaras = self._nombres_camara(data)
        imagenes = {
            nombre: np.asarray(data[clave], dtype=np.float32) for nombre, clave in camaras.items()
        }

        t = actions.shape[0]
        if proprio.shape[0] != t or any(arr.shape[0] != t for arr in imagenes.values()):
            raise ValueError("proprio, actions y cámaras deben tener el mismo número de pasos")

        pares: list[tuple[Observation, ActionChunk]] = []
        for inicio in range(0, t, self._demo_chunk_size):
            fin = min(inicio + self._demo_chunk_size, t)
            obs = Observation(
                images={nombre: arr[inicio] for nombre, arr in imagenes.items()},
                proprio=proprio[inicio],
                instruction=instruction,
            )
            chunk: ActionChunk = [Action(values=actions[i]) for i in range(inicio, fin)]
            pares.append((obs, chunk))
        return pares

    # -- interfaz Dataset -----------------------------------------------------

    def sample(self, n: int, filters: dict | None = None) -> list[tuple[Observation, ActionChunk]]:
        candidatos = self._filtrar(self._pares, filters)
        if not candidatos:
            raise ValueError(f"ningún par cumple los filtros: {filters!r}")
        # Muestreo con reemplazo: siempre devuelve n pares aunque haya menos únicos.
        indices = self._rng.integers(0, len(candidatos), size=n)
        return [candidatos[i] for i in indices]

    def _filtrar(
        self, pares: list[tuple[Observation, ActionChunk]], filters: dict | None
    ) -> list[tuple[Observation, ActionChunk]]:
        if not filters:
            return pares
        task = filters.get("task")
        if task is None:
            return pares
        return [par for par in pares if par[0].instruction == task]
