"""Paquete ``dataset`` de Core: la frontera ``Dataset`` y la factory de adaptadores.

Reexporta las ABCs (``Dataset``, ``AppendableDataset``) para que la ruta de
import ``vla_core.dataset`` siga siendo estable, y ``build_dataset`` para armar
el adaptador concreto desde configuración. Ver ``docs/dataset/Dataset Design.md``.
"""

from vla_core.dataset.base import AppendableDataset, Dataset
from vla_core.dataset.factory import build_dataset

__all__ = ["AppendableDataset", "Dataset", "build_dataset"]
