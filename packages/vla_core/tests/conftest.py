"""Fixtures compartidas de los tests de Core.

``lanzar_vla_env`` levanta procesos ``vla-env`` reales (``reach``) como
subprocesos, con ``python -m vla_env.cli``: Core habla con ellos solo por la red,
sin importar ``vla_env``. La usan los tests ``distributed`` de la integración con
Core y los ``slow`` de punta a punta con el Learner.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from vla_core.environments import EnvironmentClient
from vla_core.errors import DownstreamTimeout

RAIZ = Path(__file__).resolve().parents[3]
REACH_CONFIG = RAIZ / "configs" / "env" / "reach.yaml"
ESPERA_MAXIMA_S = 60.0


def _puerto_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _esperar_listo(address: str, proceso: subprocess.Popen) -> None:
    """Espera a que el proceso responda un ``spec`` (o falla si muere o tarda)."""
    limite = time.monotonic() + ESPERA_MAXIMA_S
    while time.monotonic() < limite:
        if proceso.poll() is not None:
            raise RuntimeError(f"vla-env terminó al arrancar (código {proceso.returncode})")
        cliente = EnvironmentClient(address, timeout_s=0.5, retries=0)
        try:
            cliente.spec  # noqa: B018 - solo se espera la respuesta
            return
        except DownstreamTimeout:
            continue
        finally:
            cliente._channel.close()  # noqa: SLF001 - sin mandar close al ambiente
    raise TimeoutError(f"vla-env no respondió en {address} tras {ESPERA_MAXIMA_S} s")


@contextmanager
def lanzar_vla_env(
    n: int, config: Path = REACH_CONFIG
) -> Iterator[list[tuple[str, subprocess.Popen]]]:
    """Levanta ``n`` procesos ``vla-env`` y devuelve ``(dirección, proceso)`` de cada uno."""
    procesos: list[tuple[str, subprocess.Popen]] = []
    try:
        for _ in range(n):
            address = f"tcp://127.0.0.1:{_puerto_libre()}"
            comando = [sys.executable, "-m", "vla_env.cli", "--config", str(config)]
            proceso = subprocess.Popen([*comando, "--address", address], cwd=RAIZ)
            procesos.append((address, proceso))
        for address, proceso in procesos:
            _esperar_listo(address, proceso)
        yield procesos
    finally:
        for _, proceso in procesos:
            proceso.kill()
            proceso.wait(timeout=10)
