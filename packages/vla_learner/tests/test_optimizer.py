"""Suite OPT sobre ``OptimizerManager``."""

import pytest
import torch
from torch import nn

from vla_learner.infra.optimizer import OptimizerConfig, OptimizerManager


def _config(**kwargs) -> OptimizerConfig:
    base = {"name": "sgd", "lr": 0.1, "grad_clip": None, "scheduler": None, "precision": "fp32"}
    base.update(kwargs)
    return OptimizerConfig(**base)


@pytest.mark.fast
def test_opt01_solo_entrena_lo_entrenable():
    a = nn.Parameter(torch.tensor([1.0, 2.0]))
    b = nn.Parameter(torch.tensor([3.0, 4.0]))
    manager = OptimizerManager([a], _config())
    b_previo = b.detach().clone()
    manager.step(a.sum() + b.sum())
    assert not torch.equal(a.detach(), torch.tensor([1.0, 2.0]))
    assert torch.equal(b.detach(), b_previo)  # b no está en el optimizador


@pytest.mark.fast
def test_opt02_clipping():
    p = nn.Parameter(torch.tensor([3.0, 4.0]))
    manager = OptimizerManager([p], _config(lr=1.0, grad_clip=1.0))
    previo = p.detach().clone()
    metrics = manager.step((p * torch.tensor([3.0, 4.0])).sum())  # grad = [3, 4], norma 5
    delta = previo - p.detach()
    assert torch.allclose(delta.norm(), torch.tensor(1.0), atol=1e-5)  # recortado a grad_clip
    assert metrics["grad_norm"] == pytest.approx(5.0, abs=1e-4)  # norma pre-recorte


@pytest.mark.fast
def test_opt03_acumulacion_equivale_a_batch_grande():
    objetivos = [
        torch.tensor([1.0, 1.0]),
        torch.tensor([2.0, 0.0]),
        torch.tensor([0.0, 3.0]),
        torch.tensor([1.0, 2.0]),
    ]
    n = len(objetivos)

    p_acum = nn.Parameter(torch.zeros(2))
    acum = OptimizerManager([p_acum], _config(grad_accumulation=n))
    for objetivo in objetivos:
        acum.step(((p_acum - objetivo) ** 2).sum())

    p_batch = nn.Parameter(torch.zeros(2))
    batch = OptimizerManager([p_batch], _config(grad_accumulation=1))
    batch.step(torch.stack([((p_batch - o) ** 2).sum() for o in objetivos]).mean())

    assert torch.allclose(p_acum.detach(), p_batch.detach(), atol=1e-6)


@pytest.mark.fast
def test_opt04_scheduler_warmup_y_coseno():
    p = nn.Parameter(torch.zeros(2))
    manager = OptimizerManager(
        [p], _config(lr=0.1, scheduler="cosine", warmup_steps=5, total_steps=20)
    )
    lrs = [manager.step(p.sum())["lr"] for _ in range(20)]
    assert lrs[0] < lrs[4]  # warmup creciente
    assert lrs[4] == pytest.approx(0.1, abs=1e-6)  # fin de warmup
    assert lrs[5] > lrs[19]  # coseno decreciente
    assert lrs[19] < 0.02


@pytest.mark.gpu
def test_opt05_fp16_con_desborde():
    p = nn.Parameter(torch.ones(2, device="cuda"))
    manager = OptimizerManager([p], _config(lr=0.1, precision="fp16"))
    loss = (p * torch.tensor([float("inf"), 1.0], device="cuda")).sum()
    manager.step(loss)  # el GradScaler salta el paso con gradiente infinito
    assert torch.isfinite(p.detach()).all()
