"""``MetricsLogger`` y sus implementaciones.

Definido en ``docs/Learner Design.md``. Solo el proceso principal publica
(``is_main``); en los demás, ``log`` no hace nada (LOG-01). ``WandbLogger`` y
``TensorBoardLogger`` importan su dependencia de forma perezosa, así la base del
harness no las necesita.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class MetricsLogger(ABC):
    def __init__(self, is_main: bool = True) -> None:
        self.is_main = is_main

    def log(self, metrics: dict[str, float], step: int) -> None:
        if not self.is_main:
            return
        self._log(metrics, step)

    @abstractmethod
    def _log(self, metrics: dict[str, float], step: int) -> None: ...

    def close(self) -> None:  # noqa: B027  (no-op intencional; las subclases lo redefinen)
        pass


class StdoutLogger(MetricsLogger):
    def _log(self, metrics: dict[str, float], step: int) -> None:
        partes = " ".join(f"{clave}={valor:.4f}" for clave, valor in metrics.items())
        print(f"[step {step}] {partes}")


class WandbLogger(MetricsLogger):
    def __init__(self, is_main: bool = True, **kwargs: object) -> None:
        super().__init__(is_main)
        self._run = None
        if self.is_main:
            import wandb  # type: ignore[import-not-found]

            self._run = wandb.init(**kwargs)

    def _log(self, metrics: dict[str, float], step: int) -> None:
        assert self._run is not None
        self._run.log(metrics, step=step)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


class TensorBoardLogger(MetricsLogger):
    def __init__(self, log_dir: str, is_main: bool = True) -> None:
        super().__init__(is_main)
        self._writer = None
        if self.is_main:
            from torch.utils.tensorboard.writer import SummaryWriter

            self._writer = SummaryWriter(log_dir)

    def _log(self, metrics: dict[str, float], step: int) -> None:
        assert self._writer is not None
        for clave, valor in metrics.items():
            self._writer.add_scalar(clave, valor, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
