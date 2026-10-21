"""Suite de contrato BAT para ``Batch``.

La heredan ``SupervisedBatch``, ``RolloutBatch`` y cualquier batch futuro. La
implementación entrega el fixture ``batch`` con al menos 3 filas, todas
distinguibles fila a fila.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from vla_learner.types import Batch, ModelInput


def _assert_row_equal(original: Batch, i: int, sub: Batch, p: int) -> None:
    """La fila ``p`` de ``sub`` es igual a la fila ``i`` de ``original`` en todo campo."""
    assert dataclasses.is_dataclass(original)
    for f in dataclasses.fields(original):
        a = getattr(original, f.name)
        b = getattr(sub, f.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a[i], b[p]), f"tensor {f.name!r}"
        elif isinstance(a, ModelInput):
            for key, tensor in a.tensors.items():
                assert torch.equal(tensor[i], b.tensors[key][p]), f"inputs.tensors[{key!r}]"
        elif isinstance(a, dict):
            for key, tensor in a.items():
                assert torch.equal(tensor[i], b[key][p]), f"{f.name}[{key!r}]"
        elif isinstance(a, list):
            assert a[i] == b[p], f"lista {f.name!r}"
        # Los escalares (p. ej. ``source``) no dependen de la fila.


class BatchContract:
    @pytest.fixture
    def batch(self) -> Batch:
        raise NotImplementedError

    def test_bat01_select_consistente(self, batch: Batch) -> None:
        n = len(batch)
        assert n >= 3, "el fixture debe tener al menos 3 filas"
        indices = torch.tensor([n - 1, 0, 1])
        sub = batch.select(indices)
        assert type(sub) is type(batch)
        assert len(sub) == len(indices)
        for p, i in enumerate(indices.tolist()):
            _assert_row_equal(batch, i, sub, p)

    def test_bat02_casos_borde(self, batch: Batch) -> None:
        n = len(batch)

        vacio = batch.select(torch.tensor([], dtype=torch.long))
        assert type(vacio) is type(batch)
        assert len(vacio) == 0

        todos = batch.select(torch.arange(n))
        assert len(todos) == n
        for i in range(n):
            _assert_row_equal(batch, i, todos, i)

        perm = torch.randperm(n)
        permutado = batch.select(perm)
        assert type(permutado) is type(batch)
        assert len(permutado) == n
        for p, i in enumerate(perm.tolist()):
            _assert_row_equal(batch, i, permutado, p)
