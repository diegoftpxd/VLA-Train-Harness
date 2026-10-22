"""``OpenXDataset``: adaptador de solo-lectura sobre Open-X Embodiment (RLDS).

Definido en ``docs/dataset/Dataset Design.md``. Traduce un dataset en formato
RLDS (Open-X Embodiment, servido por ``tensorflow-datasets``) al tipo canónico
del harness ``(Observation, ActionChunk)``, en **unidades reales** y ``float32``,
sin normalizar. Es un corpus congelado: implementa solo ``Dataset``.

La dependencia (``tensorflow-datasets``, que arrastra ``tensorflow``) es pesada y
vive en el grupo opcional uv ``openx``. Por eso **no se importa en el top-level**:
se importa de forma perezosa dentro de ``__init__``. Así ``vla_core`` base sigue
sin ``tensorflow`` y este módulo se puede importar aunque el grupo no esté
instalado (falla solo al construir la clase).

Formato RLDS (supuestos). El dataset es una secuencia de **episodios**; cada
episodio tiene ``steps`` (secuencia de pasos) y cada paso es un diccionario::

    step["observation"][<cam>]        # imagen por cámara
    step["observation"][state_key]    # vector de estado (propriocepción)
    step["action"]                    # acción de bajo nivel
    step["language_instruction"]      # instrucción (bytes/str)

``cameras`` mapea nombre canónico → clave dentro de ``observation`` (p. ej.
``{"front": "image"}``); si es ``None`` se toman las claves de ``observation``
cuyo valor tenga 3+ dimensiones (imágenes). Open-X es multi-embodiment: el
``embodiment`` del corpus se declara en la config y el filtro ``embodiment`` lo
compara (todo-o-nada, porque un builder RLDS suele ser un único embodiment).

Este adaptador se valida con la suite marcada ``heavy`` (no corre en la máquina
de desarrollo): requiere el corpus RLDS completo o un snapshot local.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_common.types import Action, ActionChunk, Observation
from vla_core.dataset.base import Dataset


class OpenXDataset(Dataset):
    """Corpus Open-X Embodiment (RLDS) servido como pares canónicos (solo-lectura)."""

    def __init__(
        self,
        name: str | None = None,
        builder_dir: str | None = None,
        data_dir: str | None = None,
        split: str = "train",
        cameras: dict[str, str] | None = None,
        state_key: str = "state",
        action_key: str = "action",
        instruction_key: str = "language_instruction",
        embodiment: str | None = None,
        demo_chunk_size: int = 1,
        seed: int = 0,
    ) -> None:
        if demo_chunk_size < 1:
            raise ValueError("demo_chunk_size debe ser >= 1")
        if not name and not builder_dir:
            raise ValueError("OpenXDataset requiere 'name' o 'builder_dir'")
        # Import perezoso: la dependencia pesada solo se exige al construir.
        try:
            import tensorflow_datasets as tfds  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise ImportError(
                "el adaptador 'openx' requiere el grupo opcional uv 'openx' "
                "(instálalo con `uv sync --extra openx`)"
            ) from exc

        self._cameras = cameras
        self._state_key = state_key
        self._action_key = action_key
        self._instruction_key = instruction_key
        self._embodiment = embodiment
        self._demo_chunk_size = demo_chunk_size
        self._rng = np.random.default_rng(seed)

        if builder_dir:
            builder = tfds.builder_from_directory(builder_dir)
        else:
            builder = tfds.builder(name, data_dir=data_dir)
            builder.download_and_prepare()
        dataset = builder.as_dataset(split=split)
        # Materializamos los pares una vez (corpus de prueba acotado). Una versión
        # de producción debería indexar/streamear en vez de cargar todo en memoria.
        self._pares: list[tuple[Observation, ActionChunk]] = self._cargar(tfds, dataset)
        if not self._pares:
            raise ValueError("el dataset Open-X no produjo pares")

    # -- utilidades de mapeo --------------------------------------------------

    @staticmethod
    def _np1d(valor: Any) -> np.ndarray:
        return np.asarray(valor, dtype=np.float32).reshape(-1)

    @staticmethod
    def _np_img(valor: Any) -> np.ndarray:
        return np.asarray(valor, dtype=np.float32)

    @staticmethod
    def _texto(valor: Any) -> str:
        if isinstance(valor, bytes):
            return valor.decode("utf-8")
        return str(valor)

    def _nombres_camara(self, observation: dict[str, Any]) -> dict[str, str]:
        if self._cameras is not None:
            return self._cameras
        return {clave: clave for clave, valor in observation.items() if np.asarray(valor).ndim >= 3}

    def _observacion(self, step: dict[str, Any]) -> Observation:
        observation = step["observation"]
        camaras = self._nombres_camara(observation)
        return Observation(
            images={nombre: self._np_img(observation[clave]) for nombre, clave in camaras.items()},
            proprio=self._np1d(observation[self._state_key]),
            instruction=self._texto(step.get(self._instruction_key, b"")),
        )

    # -- carga ----------------------------------------------------------------

    def _cargar(self, tfds: Any, dataset: Any) -> list[tuple[Observation, ActionChunk]]:
        pares: list[tuple[Observation, ActionChunk]] = []
        for episodio in tfds.as_numpy(dataset):
            pasos = list(episodio["steps"])
            for inicio in range(0, len(pasos), self._demo_chunk_size):
                fin = min(inicio + self._demo_chunk_size, len(pasos))
                obs = self._observacion(pasos[inicio])
                chunk: ActionChunk = [
                    Action(values=self._np1d(pasos[i][self._action_key]))
                    for i in range(inicio, fin)
                ]
                pares.append((obs, chunk))
        return pares

    # -- interfaz Dataset -----------------------------------------------------

    def sample(self, n: int, filters: dict | None = None) -> list[tuple[Observation, ActionChunk]]:
        candidatos = self._filtrar(filters)
        if not candidatos:
            raise ValueError(f"ningún par cumple los filtros: {filters!r}")
        elegidos = self._rng.integers(0, len(candidatos), size=n)
        return [candidatos[i] for i in elegidos]

    def _filtrar(self, filters: dict | None) -> list[tuple[Observation, ActionChunk]]:
        if not filters:
            return self._pares
        # embodiment: todo-o-nada contra el declarado en la config.
        emb = filters.get("embodiment")
        if emb is not None and emb != self._embodiment:
            return []
        task = filters.get("task")
        if task is None:
            return self._pares
        return [par for par in self._pares if par[0].instruction == task]
