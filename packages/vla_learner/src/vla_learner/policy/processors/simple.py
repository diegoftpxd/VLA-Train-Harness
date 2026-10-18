"""``SimpleProcessor``: un procesador de observación mínimo pero completo.

Definido en ``docs/Learner Design.md`` (Fase 3, "un procesador simple").
Convierte una lista de ``Observation`` en un ``ModelInput`` con:

- ``image.<cam>``: ``[B, C, H, W]`` por cada cámara requerida.
- ``proprio``: ``[B, P]``.
- ``text``: ``[B, text_dim]``, una codificación de la instrucción por bolsa de
  caracteres (determinista y de tamaño fijo, así soporta instrucciones de largo
  variable sin un tokenizador real).

Es determinista y no modifica las observaciones de entrada.
"""

from __future__ import annotations

import numpy as np
import torch

from vla_common import Observation
from vla_learner.policy.interfaces import ObservationProcessor
from vla_learner.types import ModelInput


class SimpleProcessor(ObservationProcessor):
    def __init__(
        self,
        cameras: list[str] | tuple[str, ...] = ("front",),
        text_dim: int = 16,
    ) -> None:
        self._cameras = list(cameras)
        self._text_dim = text_dim

    @property
    def required_cameras(self) -> list[str]:
        return list(self._cameras)

    def _encode_text(self, instruction: str) -> np.ndarray:
        """Bolsa de caracteres: cuenta cada carácter en ``text_dim`` cubetas.

        Usa ``ord`` (no ``hash``, que está aleatorizado por proceso) para que la
        codificación sea determinista entre corridas.
        """
        vector = np.zeros(self._text_dim, dtype=np.float32)
        for char in instruction:
            vector[ord(char) % self._text_dim] += 1.0
        return vector

    def __call__(self, observations: list[Observation]) -> ModelInput:
        tensors: dict[str, torch.Tensor] = {}
        for camera in self._cameras:
            imagenes = []
            for obs in observations:
                if camera not in obs.images:
                    raise ValueError(
                        f"Falta la cámara requerida {camera!r} en la observación "
                        f"(disponibles: {sorted(obs.images)})."
                    )
                imagenes.append(torch.as_tensor(obs.images[camera], dtype=torch.float32))
            tensors[f"image.{camera}"] = torch.stack(imagenes)

        tensors["proprio"] = torch.stack(
            [torch.as_tensor(obs.proprio, dtype=torch.float32) for obs in observations]
        )
        tensors["text"] = torch.stack(
            [torch.from_numpy(self._encode_text(obs.instruction)) for obs in observations]
        )
        return ModelInput(tensors=tensors, batch_size=len(observations))
