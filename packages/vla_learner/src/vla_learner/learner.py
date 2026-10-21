"""``Learner``: orquesta todo y corre el loop de entrenamiento.

Definido en ``docs/learner/Learner Design.md``. No tiene lógica de modelo, de loss ni de
red: conecta las piezas y corre el loop. No importa ninguna implementación
concreta de ``VLAPolicy``, ``Backbone`` ni ``ActionHead`` (lo verifica ARCH-03):
``from_config`` delega la construcción en ``builder``, que usa el ``Registry``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vla_learner.algorithms.base import Algorithm
from vla_learner.data.minibatch import MinibatchIterator
from vla_learner.data.sources import DataSource
from vla_learner.infra.checkpoint import CheckpointManager
from vla_learner.infra.distributed import DistributedStrategy
from vla_learner.infra.logging import MetricsLogger
from vla_learner.infra.optimizer import OptimizerManager
from vla_learner.policy.interfaces import VLAPolicy


@dataclass
class LearnerConfig:
    checkpoint_every: int = 100
    seed: int = 0
    available_cameras: list[str] | None = None  # cámaras del embodiment, si se conocen
    raw: dict = field(default_factory=dict)


class Learner:
    def __init__(
        self,
        policy: VLAPolicy,
        algorithm: Algorithm,
        data_source: DataSource,
        optimizer: OptimizerManager,
        checkpoints: CheckpointManager,
        logger: MetricsLogger,
        strategy: DistributedStrategy,
        config: LearnerConfig,
    ) -> None:
        self.policy = policy
        self.algorithm = algorithm
        self.data_source = data_source
        self.optimizer = optimizer
        self.checkpoints = checkpoints
        self.logger = logger
        self.strategy = strategy
        self.config = config
        self.start_iteration = 0
        self._check_compatibility()

    @classmethod
    def from_config(cls, config: dict) -> Learner:
        from vla_learner.builder import build_components

        return cls(**build_components(config))

    def train(self, num_iterations: int) -> None:
        for iteracion in range(self.start_iteration, num_iterations):
            batch = self.data_source.next_batch(self.policy)
            batch = self.algorithm.prepare(batch, self.policy)
            metrics: dict[str, float] = {}
            for _ in range(self.algorithm.update_epochs):
                for minibatch in MinibatchIterator(batch, self.algorithm.minibatch_size):
                    with self.optimizer.autocast():
                        salida = self.algorithm.loss(self.policy, minibatch)
                    metrics = {**salida.metrics, **self.optimizer.step(salida.loss)}
                if self.algorithm.should_stop_epochs(metrics):
                    break  # p. ej. PPO corta si approx_kl > target_kl
            self.logger.log(self.strategy.reduce_metrics(metrics), iteracion)
            if iteracion % self.config.checkpoint_every == 0:
                self.checkpoints.save(
                    iteracion, self.policy, self.optimizer, self.strategy, self.config.raw
                )

    def resume(self, path: Path | None = None) -> None:
        destino = path or self.checkpoints.latest()
        if destino is None:
            return
        paso = self.checkpoints.load_training_state(Path(destino), self.policy, self.optimizer)
        self.start_iteration = paso + 1  # la iteración inicial es la siguiente a la guardada

    def _check_compatibility(self) -> None:
        if not self.algorithm.required_capabilities <= self.policy.capabilities:
            faltan = self.algorithm.required_capabilities - self.policy.capabilities
            raise ValueError(
                f"La política no tiene las capacidades que exige el algoritmo: {faltan}."
            )
        if self.algorithm.batch_type is not self.data_source.batch_type:
            raise ValueError(
                f"El algoritmo consume {self.algorithm.batch_type.__name__} pero la fuente entrega "
                f"{self.data_source.batch_type.__name__}."
            )
        disponibles = self.config.available_cameras
        if disponibles is not None:
            faltantes = set(self.policy.processor.required_cameras) - set(disponibles)
            if faltantes:
                raise ValueError(f"El embodiment no tiene las cámaras requeridas: {faltantes}.")
