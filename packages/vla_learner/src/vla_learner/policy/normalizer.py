"""``ActionNormalizer``: normaliza acciones dentro de una política.

Definido en ``docs/learner/Learner Design.md`` (sección "Normalización de acciones").
El resto del harness trabaja en unidades reales; cada política normaliza a su
manera. ``ComposedPolicy`` y los adaptadores sin normalización propia usan esta
pieza. Las estadísticas se guardan como buffers, así viajan con los pesos.

Los tres modos se expresan como ``norm = (x - center) / scale`` (y la inversa),
con un ``scale`` protegido contra cero para que las dimensiones constantes no
produzcan NaN:

- ``mean_std``: center = media, scale = desviación.
- ``min_max``: center = (min + max) / 2, scale = (max - min) / 2  → [-1, 1].
- ``quantile``: igual que ``min_max`` pero con los cuantiles 1% y 99%.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from vla_common import ActionChunk, Observation

Mode = Literal["mean_std", "min_max", "quantile"]

_EPS = 1e-6
_FIELDS = ("mean", "std", "min", "max", "q01", "q99")


@dataclass
class ActionStats:
    """Estadísticas por dimensión (cada campo tiene forma ``[D]``)."""

    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    q01: np.ndarray
    q99: np.ndarray


class ActionNormalizer(nn.Module):
    def __init__(self, stats: dict[str, ActionStats], mode: Mode = "mean_std") -> None:
        super().__init__()
        if not stats:
            raise ValueError("ActionNormalizer necesita estadísticas de al menos un embodiment.")
        self.mode: Mode = mode
        self._embodiments = list(stats)
        for embodiment, st in stats.items():
            for field in _FIELDS:
                value = np.asarray(getattr(st, field), dtype=np.float32)
                self.register_buffer(self._key(embodiment, field), torch.from_numpy(value.copy()))

    @staticmethod
    def _key(embodiment: str, field: str) -> str:
        return f"{embodiment}__{field}"

    @property
    def embodiments(self) -> list[str]:
        return list(self._embodiments)

    def default_embodiment(self) -> str:
        return self._embodiments[0]

    def _buffer(self, embodiment: str, field: str) -> Tensor:
        if embodiment not in self._embodiments:
            raise KeyError(
                f"Embodiment {embodiment!r} desconocido. Disponibles: {self._embodiments}"
            )
        return getattr(self, self._key(embodiment, field))

    def _center_scale(self, embodiment: str) -> tuple[Tensor, Tensor]:
        if self.mode == "mean_std":
            center = self._buffer(embodiment, "mean")
            scale = self._buffer(embodiment, "std")
        elif self.mode == "min_max":
            low, high = self._buffer(embodiment, "min"), self._buffer(embodiment, "max")
            center, scale = (low + high) / 2, (high - low) / 2
        elif self.mode == "quantile":
            low, high = self._buffer(embodiment, "q01"), self._buffer(embodiment, "q99")
            center, scale = (low + high) / 2, (high - low) / 2
        else:  # pragma: no cover - el tipo Literal lo evita
            raise ValueError(f"Modo de normalización desconocido: {self.mode!r}")
        safe_scale = torch.where(scale.abs() < _EPS, torch.ones_like(scale), scale)
        return center, safe_scale

    def normalize(self, actions: Tensor, embodiment: str) -> Tensor:
        center, scale = self._center_scale(embodiment)
        return (actions - center) / scale

    def denormalize(self, actions: Tensor, embodiment: str) -> Tensor:
        center, scale = self._center_scale(embodiment)
        return actions * scale + center

    @classmethod
    def fit(
        cls,
        samples: list[tuple[Observation, ActionChunk]],
        mode: Mode = "mean_std",
        embodiment: str = "default",
    ) -> ActionNormalizer:
        """Calcula estadísticas a partir de muestras del Dataset (vía ``sample_supervised``)."""
        valores = np.concatenate(
            [np.stack([np.asarray(a.values) for a in chunk]) for _, chunk in samples], axis=0
        )
        stats = ActionStats(
            mean=valores.mean(axis=0),
            std=valores.std(axis=0),
            min=valores.min(axis=0),
            max=valores.max(axis=0),
            q01=np.quantile(valores, 0.01, axis=0),
            q99=np.quantile(valores, 0.99, axis=0),
        )
        return cls({embodiment: stats}, mode=mode)

    def save_file(self, path: Path) -> None:
        torch.save(
            {"mode": self.mode, "embodiments": self._embodiments, "state": self.state_dict()},
            Path(path),
        )

    @classmethod
    def from_file(cls, path: Path) -> ActionNormalizer:
        data = torch.load(Path(path), weights_only=False)
        state = data["state"]
        stats = {
            embodiment: ActionStats(
                **{field: state[cls._key(embodiment, field)].numpy() for field in _FIELDS}
            )
            for embodiment in data["embodiments"]
        }
        return cls(stats, mode=data["mode"])
