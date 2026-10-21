"""Los configs de ejemplo de ``configs/core/`` se construyen y atienden.

``build_core`` arma el ``CoreServer`` desde el YAML tal como lo hace la CLI
``vla-core``; estas pruebas levantan cada config y hacen un round-trip real con
el ``CoreClient`` del Learner, para que un ejemplo roto no pase inadvertido.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from vla_core.builder import build_core
from vla_core.server import CoreServer
from vla_learner.data.core_client import CoreClient

pytestmark = pytest.mark.fast

CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "core"


@contextmanager
def _servir(nombre: str) -> Iterator[CoreServer]:
    config = yaml.safe_load((CONFIGS / nombre).read_text(encoding="utf-8"))
    # Puerto efímero para el test, sin pisar la dirección fija del YAML.
    config["server"] = {"address": "tcp://127.0.0.1:0"}
    server = build_core(config)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.mark.parametrize("nombre", ["fake.yaml", "echo.yaml", "local.yaml"])
def test_config_se_construye_y_responde(nombre: str) -> None:
    with _servir(nombre) as server:
        client = CoreClient(server.address, timeout_s=5.0)
        try:
            ((episode_id, obs),) = client.reset(["e0"], seed=0)
            assert episode_id.startswith("e0")
            assert obs.proprio.dtype.kind == "f"
        finally:
            client.close()
