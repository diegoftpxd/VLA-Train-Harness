"""Fuentes de datos: entregan el siguiente batch al Learner.

Definidas en ``docs/Learner Design.md``. ``SupervisedSource`` tira de demos;
``EnvRolloutSource`` y ``WorldModelRolloutSource`` usan el ``RolloutCollector``;
``MixedSource`` combina fuentes del mismo tipo de batch.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import torch

from vla_common import Observation
from vla_learner.data.collector import RolloutCollector
from vla_learner.data.core_client import CoreClientProtocol
from vla_learner.policy.interfaces import VLAPolicy
from vla_learner.types import Batch, RolloutBatch, SupervisedBatch


class DataSource(ABC):
    @property
    @abstractmethod
    def batch_type(self) -> type[Batch]: ...

    @abstractmethod
    def next_batch(self, policy: VLAPolicy) -> Batch:
        """Recibe la política porque las fuentes de rollouts la usan para actuar."""


class StartSelector(ABC):
    @abstractmethod
    def select(self, n: int) -> list[Observation | tuple[str, int]]:
        """Desde dónde simular: observaciones reales o (episode_id, paso) registrados."""


class SupervisedSource(DataSource):
    def __init__(
        self, client: CoreClientProtocol, batch_size: int, filters: dict | None = None
    ) -> None:
        self.client = client
        self.batch_size = batch_size
        self.filters = filters

    @property
    def batch_type(self) -> type[Batch]:
        return SupervisedBatch

    def next_batch(self, policy: VLAPolicy) -> Batch:
        pares = self.client.sample_supervised(self.batch_size, self.filters)
        observaciones = [obs for obs, _ in pares]
        inputs = policy.processor(observaciones)
        actions = torch.stack(
            [
                torch.stack([torch.as_tensor(a.values, dtype=torch.float32) for a in chunk])
                for _, chunk in pares
            ]
        )
        return SupervisedBatch(inputs=inputs, actions=actions)


class EnvRolloutSource(DataSource):
    def __init__(
        self,
        collector: RolloutCollector,
        env_ids: list[str],
        steps_per_batch: int,
        group_size: int = 1,
    ) -> None:
        if steps_per_batch % len(env_ids) != 0:
            raise ValueError("steps_per_batch debe ser múltiplo de len(env_ids).")
        self.collector = collector
        self.env_ids = env_ids
        self.steps_per_batch = steps_per_batch
        self.group_size = group_size

    @property
    def batch_type(self) -> type[Batch]:
        return RolloutBatch

    def next_batch(self, policy: VLAPolicy) -> Batch:
        num_steps = self.steps_per_batch // len(self.env_ids)
        return self.collector.collect_env(
            policy, self.env_ids, num_steps, group_size=self.group_size
        )


class WorldModelRolloutSource(DataSource):
    def __init__(
        self,
        collector: RolloutCollector,
        start_selector: StartSelector,
        sequences_per_start: int,
        horizon: int,
        num_starts: int = 1,
    ) -> None:
        self.collector = collector
        self.start_selector = start_selector
        self.sequences_per_start = sequences_per_start
        self.horizon = horizon
        self.num_starts = num_starts

    @property
    def batch_type(self) -> type[Batch]:
        return RolloutBatch

    def next_batch(self, policy: VLAPolicy) -> Batch:
        starts = self.start_selector.select(self.num_starts)
        return self.collector.collect_world_model(
            policy, starts, self.sequences_per_start, self.horizon
        )


class MixedSource(DataSource):
    def __init__(self, sources: list[DataSource], weights: list[float], seed: int = 0) -> None:
        tipos = {source.batch_type for source in sources}
        if len(tipos) != 1:
            raise ValueError("MixedSource exige que todas las fuentes tengan el mismo batch_type.")
        self.sources = sources
        self.weights = weights
        self._rng = random.Random(seed)

    @property
    def batch_type(self) -> type[Batch]:
        return self.sources[0].batch_type

    def next_batch(self, policy: VLAPolicy) -> Batch:
        (fuente,) = self._rng.choices(self.sources, weights=self.weights, k=1)
        return fuente.next_batch(policy)
