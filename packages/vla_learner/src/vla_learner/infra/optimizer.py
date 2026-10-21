"""``OptimizerManager``: todo lo que pasa después de calcular la loss.

Definido en ``docs/learner/Learner Design.md``. Concentra backward (con escala si fp16),
acumulación de gradientes, clipping, paso del optimizador, scheduler y
``zero_grad``. Es la **única** clase que llama a ``backward()`` y a ``step()``
(lo verifica ARCH-02).
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

_PRECISION_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    scheduler: str | None = "cosine"
    warmup_steps: int = 0
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    grad_accumulation: int = 1
    total_steps: int | None = None  # horizonte del scheduler coseno


class OptimizerManager:
    def __init__(self, params: Iterable[nn.Parameter], config: OptimizerConfig) -> None:
        self._params = list(params)
        self.config = config
        self._optimizer = self._build_optimizer()
        self._scheduler = self._build_scheduler()
        self._device_type = self._params[0].device.type if self._params else "cpu"
        usa_fp16 = config.precision == "fp16" and self._device_type == "cuda"
        self._scaler = torch.amp.GradScaler("cuda") if usa_fp16 else None  # type: ignore[attr-defined]
        self._micro = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.config.name == "adamw":
            return torch.optim.AdamW(
                self._params, lr=self.config.lr, weight_decay=self.config.weight_decay
            )
        if self.config.name == "sgd":
            return torch.optim.SGD(
                self._params, lr=self.config.lr, weight_decay=self.config.weight_decay
            )
        raise ValueError(f"Optimizador desconocido: {self.config.name!r}")

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.config.scheduler is None:
            return None
        if self.config.scheduler != "cosine":
            raise ValueError(f"Scheduler desconocido: {self.config.scheduler!r}")
        warmup, total = self.config.warmup_steps, self.config.total_steps

        def factor(paso: int) -> float:
            if warmup > 0 and paso < warmup:
                return (paso + 1) / warmup
            if total is not None and total > warmup:
                progreso = min((paso - warmup) / (total - warmup), 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progreso))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self._optimizer, factor)

    def step(self, loss: torch.Tensor) -> dict[str, float]:
        escalada = loss / self.config.grad_accumulation
        if self._scaler is not None:
            self._scaler.scale(escalada).backward()
        else:
            escalada.backward()
        self._micro += 1
        metrics = {"lr": self._current_lr()}
        if self._micro % self.config.grad_accumulation == 0:
            metrics["grad_norm"] = self._apply_update()
        return metrics

    def _apply_update(self) -> float:
        if self._scaler is not None:
            self._scaler.unscale_(self._optimizer)
        if self.config.grad_clip is not None:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self._params, self.config.grad_clip))
        else:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self._params, float("inf")))
        if self._scaler is not None:
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            self._optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)
        return grad_norm

    def _current_lr(self) -> float:
        return float(self._optimizer.param_groups[0]["lr"])

    def autocast(self) -> contextlib.AbstractContextManager:
        if self.config.precision == "fp32":
            return contextlib.nullcontext()
        return torch.autocast(self._device_type, dtype=_PRECISION_DTYPE[self.config.precision])

    def state_dict(self) -> dict:
        return {
            "optimizer": self._optimizer.state_dict(),
            "scheduler": None if self._scheduler is None else self._scheduler.state_dict(),
            "micro": self._micro,
        }

    def load_state_dict(self, state: dict) -> None:
        self._optimizer.load_state_dict(state["optimizer"])
        if self._scheduler is not None and state["scheduler"] is not None:
            self._scheduler.load_state_dict(state["scheduler"])
        self._micro = state["micro"]
