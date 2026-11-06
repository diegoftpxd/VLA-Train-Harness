"""Suite DIST: Core contra procesos ``vla-env`` reales (``distributed``).

Ver ``docs/environment/Test Environment.md`` ("Integración"). Cada ambiente es un
proceso aparte (``python -m vla_env.cli``) y Core lo consume por la red con su
``EnvironmentClient``, armado desde el bloque ``environments`` de la config.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack

import numpy as np
import pytest
from conftest import lanzar_vla_env

from vla_common.types import Action, ActionChunk
from vla_core.builder import build_core
from vla_core.environments import EnvironmentClient
from vla_core.server import CoreServer
from vla_learner.data.core_client import CoreClient

pytestmark = pytest.mark.distributed


def _chunk(dx: float, dy: float) -> ActionChunk:
    return [Action(values=np.array([dx, dy], dtype=np.float32))]


@pytest.fixture
def core_y_procesos() -> Iterator[tuple[CoreServer, CoreClient, list]]:
    with ExitStack() as pila:
        procesos = pila.enter_context(lanzar_vla_env(2))
        config = {
            "server": {"address": "tcp://127.0.0.1:0"},
            "backend": {"type": "fake"},
            "environments": [
                {"id": f"reach-{i}", "address": address} for i, (address, _) in enumerate(procesos)
            ],
            # Timeout corto: DIST-02 mata un proceso y espera la escalada.
            "timeouts": {"environment_s": 1.0, "retries": 0},
        }
        core = build_core(config)
        core.start()
        pila.callback(core.stop)
        cliente = CoreClient(core.address, timeout_s=10.0)
        pila.callback(cliente.close)
        yield core, cliente, procesos


def test_dist01_enruta_respeta_orden_y_registra(core_y_procesos) -> None:
    _, cliente, _ = core_y_procesos
    episodios = cliente.reset(["reach-1", "reach-0"], seed=3)
    assert [ep.split("-ep")[0] for ep, _ in episodios] == ["reach-1", "reach-0"]
    resultados = cliente.step(
        ["reach-1", "reach-0"], [_chunk(0.1, 0.0), _chunk(0.0, 0.1)], with_reward=True
    )
    assert len(resultados) == 2 and all(r.reward is not None for r in resultados)
    for (episode_id, inicial), resultado, delta in zip(
        episodios, resultados, [(0.1, 0.0), (0.0, 0.1)], strict=True
    ):
        esperado = np.clip(inicial.proprio[:2] + np.array(delta, dtype=np.float32), -1, 1)
        assert np.allclose(resultado.observation.proprio[:2], esperado)
        (transicion,) = cliente.get_episode(episode_id)
        assert transicion.source == "env" and "success" in transicion.result.info


def test_dist02_caida_de_un_proceso_queda_aislada(core_y_procesos) -> None:
    _, cliente, procesos = core_y_procesos
    cliente.reset(["reach-0", "reach-1"], seed=0)
    _, caido = procesos[0]
    caido.kill()
    caido.wait(timeout=10)
    with pytest.raises(RuntimeError, match="DownstreamUnavailable"):
        cliente.step(["reach-0"], [_chunk(0.0, 0.0)], with_reward=False)
    # Reusar el caído falla de inmediato; el otro sigue operando.
    with pytest.raises(RuntimeError, match="no disponible"):
        cliente.step(["reach-0"], [_chunk(0.0, 0.0)], with_reward=False)
    (resultado,) = cliente.step(["reach-1"], [_chunk(0.0, 0.0)], with_reward=False)
    assert resultado.reward is None


def test_dist03_shutdown_de_core_cierra_los_procesos(core_y_procesos) -> None:
    core, cliente, procesos = core_y_procesos
    cliente.reset(["reach-0"], seed=0)
    core.shutdown()  # lo que hace `vla-core` al salir: cada proceso recibe `close`
    address, _ = procesos[0]
    directo = EnvironmentClient(address, timeout_s=5.0, retries=0)
    try:
        with pytest.raises(RuntimeError, match="reset"):
            directo.step(_chunk(0.0, 0.0))  # el ambiente quedó cerrado
        assert directo.reset(None, 0).instruction == "reach the goal"  # y se reabre
    finally:
        directo.close()
