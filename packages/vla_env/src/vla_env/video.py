"""``EpisodeRecorder``: un video MP4 por episodio, con cada acción del simulador.

Definido en ``docs/environment/Environment Design.md`` (sección "Videos"). Lo usa
``GymEnvironment`` cuando la config trae ``record:`` (o ``vla-env --record DIR``).
Graba **dentro** del ambiente porque solo ahí se ve cada acción de bajo nivel:
Core y el Learner reciben una observación por chunk (50 acciones en π0), que
daría un video a saltos.

Cada cuadro son las cámaras canónicas de la observación (ya orientadas como el
dataset) puestas lado a lado. El archivo se escribe a medida que avanza el
episodio (no se acumulan cuadros en memoria) y al terminar se renombra con el
resultado:

    <dir>/<NNNN>_<tarea>_seed<S>_{success,fail}.mp4

Un episodio abandonado (``reset`` o ``close`` antes del final) queda como
``..._incomplete.mp4``. ``imageio`` (extra ``libero`` de ``vla_env``) se importa
perezosamente, al abrir el primer video.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from vla_common.types import Observation

DEFAULT_FPS = 20  # LIBERO controla a 20 Hz


class FrameWriter(Protocol):
    def append_data(self, frame: np.ndarray) -> None: ...
    def close(self) -> None: ...


WriterFactory = Callable[[Path, int], FrameWriter]


def imageio_writer(path: Path, fps: int) -> FrameWriter:
    """Escritor MP4 (H.264) de ``imageio``; se reproduce en navegadores."""
    import imageio.v2 as imageio  # pyright: ignore[reportMissingImports]

    return imageio.get_writer(  # type: ignore[no-any-return]
        str(path), fps=fps, codec="libx264", quality=7, macro_block_size=1
    )


def frame_from_observation(obs: Observation, cameras: list[str] | None = None) -> np.ndarray:
    """Cámaras canónicas (``float32`` CHW en ``[0, 1]``) → un cuadro HWC ``uint8``.

    Las cámaras van lado a lado, en el orden de ``cameras`` (por defecto, el de la
    observación). Si tienen alturas distintas es error: no se reescala.
    """
    nombres = cameras or list(obs.images)
    faltan = [c for c in nombres if c not in obs.images]
    if faltan:
        raise KeyError(f"cámaras para el video que la observación no trae: {faltan}")
    paneles = [np.transpose(obs.images[c], (1, 2, 0)) for c in nombres]
    if len({p.shape[0] for p in paneles}) != 1:
        raise ValueError(f"cámaras de alturas distintas: {[p.shape for p in paneles]}")
    cuadro = np.concatenate(paneles, axis=1)
    return (np.clip(cuadro, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _slug(texto: str, largo: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "_", texto.lower()).strip("_")[:largo] or "tarea"


class EpisodeRecorder:
    """Graba los episodios de un ambiente en ``directory``, uno por archivo."""

    def __init__(
        self,
        directory: str | Path,
        fps: int = DEFAULT_FPS,
        cameras: list[str] | None = None,
        writer_factory: WriterFactory = imageio_writer,
    ) -> None:
        self.directory = Path(directory)
        self.fps = int(fps)
        self.cameras = list(cameras) if cameras else None
        self._factory = writer_factory
        self._writer: FrameWriter | None = None
        self._ruta: Path | None = None
        self._nombre = ""
        self._cuadros = 0
        self._contador = 0
        self.saved: list[Path] = []  # videos terminados, en orden

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def start(self, task: str, seed: int | None) -> None:
        """Empieza el video de un episodio (cierra el anterior si quedó abierto)."""
        self.abort()
        self.directory.mkdir(parents=True, exist_ok=True)
        semilla = "none" if seed is None else str(seed)
        self._nombre = f"{self._contador:04d}_{_slug(task)}_seed{semilla}"
        self._contador += 1
        self._ruta = self.directory / f"{self._nombre}.partial.mp4"
        self._writer = self._factory(self._ruta, self.fps)
        self._cuadros = 0

    def add(self, obs: Observation) -> None:
        if self._writer is None:
            raise RuntimeError("EpisodeRecorder.add sin start()")
        self._writer.append_data(frame_from_observation(obs, self.cameras))
        self._cuadros += 1

    def finish(self, success: bool) -> Path | None:
        """Cierra el video y lo renombra con el resultado; devuelve su ruta."""
        return self._cerrar("success" if success else "fail")

    def abort(self) -> Path | None:
        """Cierra un episodio sin final (``reset``/``close`` a mitad) como ``incomplete``."""
        return self._cerrar("incomplete")

    def _cerrar(self, resultado: str) -> Path | None:
        if self._writer is None or self._ruta is None:
            return None
        self._writer.close()
        destino = self.directory / f"{self._nombre}_{resultado}.mp4"
        self._ruta.replace(destino)
        self._writer = self._ruta = None
        self.saved.append(destino)
        return destino


def build_recorder(config: dict[str, Any] | str | Path) -> EpisodeRecorder:
    """``record:`` de la config (un directorio, o ``{dir, fps, cameras}``) → grabador."""
    if isinstance(config, (str, Path)):
        return EpisodeRecorder(config)
    opciones = dict(config)
    if "dir" not in opciones:
        raise ValueError("environment.record necesita `dir` (carpeta de los videos)")
    desconocidas = set(opciones) - {"dir", "fps", "cameras"}
    if desconocidas:
        raise ValueError(f"opciones desconocidas en environment.record: {sorted(desconocidas)}")
    return EpisodeRecorder(
        opciones["dir"], fps=opciones.get("fps", DEFAULT_FPS), cameras=opciones.get("cameras")
    )
