"""Construcción de Core desde configuración (``build_core``).

Definido en ``docs/core/Core Design.md``. Arma el backend que cumple
``CoreInterface`` y lo envuelve en un ``CoreServer``.

En la Fase 1 el único backend disponible es el ``StaticCoreBackend`` de prueba
(``type: static``): sirve para levantar el proceso y validar el transporte. El
gateway ``Core`` real (con su registro de ambientes, world model, reward model,
dataset y almacén de episodios) se registra acá en fases posteriores.
"""

from __future__ import annotations

from typing import Any

from vla_common import CoreInterface
from vla_core.server import CoreServer
from vla_core.testing.static_backend import StaticCoreBackend


def _build_backend(config: dict[str, Any]) -> CoreInterface:
    tipo = config.get("type", "static")
    if tipo == "static":
        return StaticCoreBackend()
    if tipo == "fake":
        # Gateway Core real con todos los componentes fake en memoria. Los
        # clientes DEALER a componentes reales se cablean en fases posteriores.
        from vla_core.testing.fakes import make_fake_core

        return make_fake_core(demo_chunk_size=config.get("demo_chunk_size", 1))
    raise ValueError(
        f"Backend de Core desconocido: {tipo!r}. El gateway con componentes reales "
        "se agrega en fases posteriores."
    )


def build_core(config: dict[str, Any]) -> CoreServer:
    backend = _build_backend(config.get("backend", {"type": "static"}))
    address = config.get("server", {}).get("address", "tcp://127.0.0.1:5555")
    return CoreServer(backend, address)
