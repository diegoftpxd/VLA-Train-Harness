"""``CheckpointManager``: guardar y cargar estado de entrenamiento.

Definido en ``docs/Learner Design.md``. Separa lo que necesita la evaluación
(carpeta ``policy/``, escrita por ``policy.save_pretrained``) de lo que necesita
retomar el entrenamiento (``training_state.pt``: pesos, optimizador, paso y
estado del RNG, para que ``resume`` sea exacto).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from vla_learner.policy.interfaces import VLAPolicy

if TYPE_CHECKING:
    from vla_learner.infra.optimizer import OptimizerManager


class CheckpointManager:
    def __init__(self, directory: Path, keep_last: int = 3) -> None:
        self.directory = Path(directory)
        self.keep_last = keep_last
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        step: int,
        policy: VLAPolicy,
        optimizer: OptimizerManager,
        strategy: object,
        config: dict,
    ) -> Path:
        destino = self.directory / f"step_{step:06d}"
        destino.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(destino / "policy")
        torch.save(
            {
                "step": step,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "rng": torch.get_rng_state(),
            },
            destino / "training_state.pt",
        )
        self._prune()
        return destino

    def load_training_state(
        self, path: Path, policy: VLAPolicy, optimizer: OptimizerManager
    ) -> int:
        data = torch.load(Path(path) / "training_state.pt", weights_only=False)
        policy.load_state_dict(data["policy"])
        optimizer.load_state_dict(data["optimizer"])
        torch.set_rng_state(data["rng"])
        return int(data["step"])

    def latest(self) -> Path | None:
        checkpoints = self._checkpoints()
        return checkpoints[-1] if checkpoints else None

    def _checkpoints(self) -> list[Path]:
        return sorted(p for p in self.directory.glob("step_*") if p.is_dir())

    def _prune(self) -> None:
        checkpoints = self._checkpoints()
        for viejo in checkpoints[: -self.keep_last] if self.keep_last > 0 else []:
            for archivo in sorted(viejo.rglob("*"), reverse=True):
                archivo.unlink() if archivo.is_file() else archivo.rmdir()
            viejo.rmdir()
