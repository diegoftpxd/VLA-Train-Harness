"""Suite DIST sobre las estrategias de distribución.

DIST-01 (SingleDevice) corre en el gate ``fast``. DIST-02 (DDP) usa 2 procesos y
backend gloo → marca ``distributed``. DIST-03 (FSDP) necesita varias GPUs →
marca ``heavy``.
"""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vla_learner.infra.distributed import DDPStrategy, SingleDevice
from vla_learner.infra.optimizer import OptimizerConfig, OptimizerManager
from vla_learner.testing.stub_policy import StubPolicy
from vla_learner.types import ModelInput


@pytest.mark.fast
def test_dist01_single_device():
    estrategia = SingleDevice("cpu")
    policy = StubPolicy(action_dim=2, chunk_size=1)
    assert estrategia.wrap(policy) is policy
    assert estrategia.is_main is True
    metrics = {"loss": 1.0, "kl": 0.2}
    assert estrategia.reduce_metrics(metrics) == metrics
    assert set(estrategia.full_state_dict(policy)) == set(policy.state_dict())


def _puerto_libre() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    puerto = sock.getsockname()[1]
    sock.close()
    return puerto


def _ddp_worker(rank: int, world_size: int, puerto: int, cola: mp.Queue) -> None:
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(puerto)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        estrategia = DDPStrategy("cpu")
        policy = StubPolicy(action_dim=2, chunk_size=1)
        estrategia.wrap(policy)  # difunde pesos + hooks de all-reduce
        optimizer = OptimizerManager(
            policy.parameters(),
            OptimizerConfig(name="sgd", lr=0.1, grad_clip=None, scheduler=None, precision="fp32"),
        )
        torch.manual_seed(100 + rank)  # datos distintos por proceso
        actions = torch.randn(4, 1, 2)
        inputs = ModelInput(tensors={"proprio": torch.randn(4, 4)}, batch_size=4)
        optimizer.step(policy.imitation_loss(inputs, actions).loss)
        reducidas = estrategia.reduce_metrics({"x": float(rank)})
        cola.put((rank, float(policy.mean.detach().sum()), reducidas["x"]))
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_dist02_ddp_sincroniza():
    world_size = 2
    contexto = mp.get_context("spawn")
    cola: mp.Queue = contexto.Queue()
    puerto = _puerto_libre()
    procesos = [
        contexto.Process(target=_ddp_worker, args=(rank, world_size, puerto, cola))
        for rank in range(world_size)
    ]
    for proceso in procesos:
        proceso.start()
    resultados = [cola.get(timeout=60) for _ in range(world_size)]
    for proceso in procesos:
        proceso.join(timeout=60)
        assert proceso.exitcode == 0

    por_rango = {rank: (pesos, x) for rank, pesos, x in resultados}
    # los pesos quedan iguales en todos tras un paso
    assert por_rango[0][0] == pytest.approx(por_rango[1][0], abs=1e-6)
    # reduce_metrics promedia entre procesos: (0 + 1) / 2 = 0.5
    assert por_rango[0][1] == pytest.approx(0.5)
    assert por_rango[1][1] == pytest.approx(0.5)


@pytest.mark.heavy
def test_dist03_fsdp_full_state_dict():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("FSDP requiere varias GPUs")
