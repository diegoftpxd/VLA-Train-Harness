"""Partir cualquier ``Batch`` en minibatches.

Definido en ``docs/Learner Design.md``. En cada iteración el Learner recibe un
batch y, según el algoritmo, lo recorre en partes más chicas una o varias
veces (PPO: varias pasadas; BC: una). La lógica de partir es la misma sin
importar el tipo de batch, así que vive en una sola clase: solo usa
``__len__`` y ``select``.

No guarda datos entre iteraciones: v1 es síncrono y on-policy, así que cada
batch se usa en su iteración y se descarta. La reproducibilidad del orden
(MB-04) se consigue sembrando el RNG global de torch antes de iterar.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from vla_learner.types import Batch


class MinibatchIterator:
    def __init__(self, batch: Batch, size: int | None, shuffle: bool = True) -> None:
        self.batch = batch
        self.size = size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Batch]:
        n = len(self.batch)
        indices = torch.randperm(n) if self.shuffle else torch.arange(n)
        size = n if self.size is None else self.size
        if size <= 0:
            raise ValueError(f"El tamaño de minibatch debe ser > 0 (recibido {size!r}).")
        for start in range(0, n, size):
            yield self.batch.select(indices[start : start + size])
