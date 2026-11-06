"""Suite CFG: ``build_core`` arma el registro de ambientes desde la config.

Ver ``docs/environment/Test Environment.md`` ("Integración"). Los ambientes se
sirven en el mismo test con ``serve_environment`` (``RouterServer`` +
``handle_environment``) sobre ``FakeEnvironment``: el cruce con procesos
``vla-env`` reales está en ``test_env_processes.py`` (``distributed``).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack

import numpy as np
import pytest

from vla_common.types import Action
from vla_core.builder import build_core, build_environments
from vla_core.environments import EnvironmentClient
from vla_core.errors import UnknownEnvironment
from vla_core.testing.fakes import FakeEnvironment, serve_environment
from vla_learner.data.core_client import CoreClient

pytestmark = pytest.mark.fast


def test_cfg01_un_cliente_por_entrada_con_sus_timeouts() -> None:
    registro = build_environments(
        [{"id": "a", "address": "tcp://127.0.0.1:1"}, {"id": "b", "address": "tcp://127.0.0.1:2"}],
        {"environment_s": 0.5, "retries": 3},
    )
    assert registro.registered() == ["a", "b"]
    cliente = registro.get("a")
    assert isinstance(cliente, EnvironmentClient)
    canal = cliente._channel  # noqa: SLF001 - se verifica que la config llegó al canal
    assert canal._timeout_ms == 500 and canal._retries == 3  # noqa: SLF001


def test_cfg01_defaults_de_timeout() -> None:
    cliente = build_environments([{"id": "a", "address": "tcp://127.0.0.1:1"}]).get("a")
    assert isinstance(cliente, EnvironmentClient)
    assert cliente._channel._timeout_ms == 5000  # noqa: SLF001


@pytest.mark.parametrize(
    "entradas, mensaje",
    [
        ([], "vacío"),
        ([{"id": "a"}], "address"),
        ([{"address": "tcp://127.0.0.1:1"}], "id"),
        (
            [{"id": "a", "address": "tcp://127.0.0.1:1"}, {"id": "a", "address": "tcp://x:2"}],
            "repetido",
        ),
    ],
)
def test_cfg01_config_invalida(entradas: list[dict], mensaje: str) -> None:
    with pytest.raises(ValueError, match=mensaje):
        build_environments(entradas)


def test_cfg02_sin_autoregistro() -> None:
    registro = build_environments([{"id": "a", "address": "tcp://127.0.0.1:1"}])
    with pytest.raises(UnknownEnvironment):
        registro.get("otro")


@pytest.fixture
def core_con_dos_ambientes() -> Iterator[CoreClient]:
    with ExitStack() as pila:
        direcciones = []
        for _ in range(2):
            servidor = serve_environment(FakeEnvironment())
            direcciones.append(servidor.start())
            pila.callback(servidor.stop)
        config = {
            "server": {"address": "tcp://127.0.0.1:0"},
            "backend": {"type": "fake"},
            "environments": [
                {"id": "izq", "address": direcciones[0]},
                {"id": "der", "address": direcciones[1]},
            ],
            "timeouts": {"environment_s": 5, "retries": 0},
        }
        core = build_core(config)
        core.start()
        pila.callback(core.stop)
        cliente = CoreClient(core.address, timeout_s=10.0)
        pila.callback(cliente.close)
        yield cliente


def test_cfg03_round_trip_contra_ambientes_de_la_config(
    core_con_dos_ambientes: CoreClient,
) -> None:
    cliente = core_con_dos_ambientes
    episodios = cliente.reset(["der", "izq"], seed=1)
    assert [ep.split("-")[0] for ep, _ in episodios] == ["der", "izq"]
    quieto = [Action(values=np.zeros(2, dtype=np.float32))]
    resultados = cliente.step(["der", "izq"], [quieto, quieto], with_reward=True)
    assert all(r.reward is not None for r in resultados)
    assert all(isinstance(r.info["success"], bool) for r in resultados)
    with pytest.raises(RuntimeError, match="no registrado"):
        cliente.reset(["e0"], seed=0)
