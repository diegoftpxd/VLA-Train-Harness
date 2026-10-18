"""Suite MB sobre ``MinibatchIterator``.

Se usa un batch cuyas filas llevan su propio índice (``idx``) para poder
recuperar el orden de salida y verificar cobertura y reproducibilidad.
"""

import pytest
import torch

from vla_learner.data.minibatch import MinibatchIterator
from vla_learner.types import ModelInput, SupervisedBatch

pytestmark = pytest.mark.fast


def _index_batch(n: int) -> SupervisedBatch:
    idx = torch.arange(n)
    inputs = ModelInput(tensors={"idx": idx}, batch_size=n)
    actions = idx.to(torch.float32).reshape(n, 1, 1)
    return SupervisedBatch(inputs=inputs, actions=actions)


def _collect_order(iterator: MinibatchIterator) -> list[int]:
    order: list[int] = []
    for mb in iterator:
        order.extend(mb.inputs.tensors["idx"].tolist())  # type: ignore[attr-defined]
    return order


def test_mb01_cubre_todo_una_vez() -> None:
    n = 10
    order = _collect_order(MinibatchIterator(_index_batch(n), size=4))
    assert sorted(order) == list(range(n))  # cada fila exactamente una vez


def test_mb02_ultimo_minibatch_parcial() -> None:
    sizes = [len(mb) for mb in MinibatchIterator(_index_batch(10), size=4)]
    assert sizes == [4, 4, 2]


def test_mb03_size_none_entrega_batch_completo() -> None:
    minibatches = list(MinibatchIterator(_index_batch(7), size=None))
    assert len(minibatches) == 1
    assert len(minibatches[0]) == 7


def test_mb04_reproducible_con_la_misma_semilla() -> None:
    batch = _index_batch(20)

    torch.manual_seed(0)
    primero = _collect_order(MinibatchIterator(batch, size=5))
    torch.manual_seed(0)
    segundo = _collect_order(MinibatchIterator(batch, size=5))
    assert primero == segundo

    torch.manual_seed(1)
    otro = _collect_order(MinibatchIterator(batch, size=5))
    assert primero != otro  # con otra semilla, otro orden (casi seguro)
