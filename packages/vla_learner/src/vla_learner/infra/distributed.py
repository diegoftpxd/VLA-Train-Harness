"""``DistributedStrategy``: una GPU, DDP o FSDP.

Definido en ``docs/learner/Learner Design.md``. Abstrae cómo se reparten modelo y
gradientes entre procesos, para que el resto del Learner no cambie.

``DDPStrategy`` no envuelve la política en ``torch`` DDP (perdería los métodos de
``VLAPolicy``): sincroniza los gradientes con un hook por parámetro que hace
all-reduce (promedio) durante el backward, y difunde los pesos iniciales desde
el proceso principal. Así los métodos del modelo siguen intactos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.distributed as dist

from vla_learner.policy.interfaces import VLAPolicy


class DistributedStrategy(ABC):
    @abstractmethod
    def wrap(self, policy: VLAPolicy) -> VLAPolicy: ...

    @property
    @abstractmethod
    def is_main(self) -> bool: ...

    @abstractmethod
    def barrier(self) -> None: ...

    @abstractmethod
    def reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]: ...

    @abstractmethod
    def full_state_dict(self, policy: VLAPolicy) -> dict:
        """Reúne los pesos completos (necesario con FSDP antes de guardar)."""


class SingleDevice(DistributedStrategy):
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

    def wrap(self, policy: VLAPolicy) -> VLAPolicy:
        policy.to(self.device)
        return policy

    @property
    def is_main(self) -> bool:
        return True

    def barrier(self) -> None:
        pass

    def reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        return dict(metrics)

    def full_state_dict(self, policy: VLAPolicy) -> dict:
        return policy.state_dict()


class DDPStrategy(DistributedStrategy):
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def wrap(self, policy: VLAPolicy) -> VLAPolicy:
        policy.to(self.device)
        for parametro in policy.parameters():
            dist.broadcast(parametro.data, src=0)  # mismo arranque en todos
            if parametro.requires_grad:
                parametro.register_post_accumulate_grad_hook(self._promediar_gradiente)
        return policy

    def _promediar_gradiente(self, parametro: torch.nn.Parameter) -> None:
        if parametro.grad is not None:
            dist.all_reduce(parametro.grad, op=dist.ReduceOp.SUM)
            parametro.grad /= self.world_size

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        dist.barrier()

    def reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        reducidas: dict[str, float] = {}
        for clave, valor in metrics.items():
            tensor = torch.tensor(valor)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reducidas[clave] = float(tensor) / self.world_size
        return reducidas

    def full_state_dict(self, policy: VLAPolicy) -> dict:
        return policy.state_dict()  # en DDP cada proceso tiene los pesos completos


class FSDPStrategy(DistributedStrategy):
    """Implementada para cuando haya varias GPUs; su test va marcado ``heavy``."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.rank = dist.get_rank()

    def wrap(self, policy: VLAPolicy) -> VLAPolicy:
        from torch.distributed.fsdp import FullyShardedDataParallel

        policy.to(self.device)
        return FullyShardedDataParallel(policy)  # type: ignore[return-value]

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        dist.barrier()

    def reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        world_size = dist.get_world_size()
        reducidas: dict[str, float] = {}
        for clave, valor in metrics.items():
            tensor = torch.tensor(valor)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reducidas[clave] = float(tensor) / world_size
        return reducidas

    def full_state_dict(self, policy: VLAPolicy) -> dict:
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel,
            StateDictType,
        )

        config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(
            policy, StateDictType.FULL_STATE_DICT, config
        ):
            return policy.state_dict()
