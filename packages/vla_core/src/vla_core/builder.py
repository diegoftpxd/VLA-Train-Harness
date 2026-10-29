"""Construcción de Core desde configuración (``build_core``).

Definido en ``docs/core/Core Design.md``. Arma el backend que cumple
``CoreInterface`` y lo envuelve en un ``CoreServer``.

Backends: ``static`` (eco de transporte) y ``fake`` (el gateway ``Core`` con
componentes fake). Sobre ``fake``, dos bloques opcionales reemplazan piezas por
las reales: ``dataset`` (vía ``build_dataset``) y ``environments`` (un
``EnvironmentClient`` por proceso ``vla-env``, con los ``timeouts`` de la
config; ver ``docs/environment/Environment Design.md``).
"""

from __future__ import annotations

from typing import Any

from vla_common import CoreInterface
from vla_core.environments import EnvironmentClient, EnvironmentRegistry
from vla_core.server import CoreServer
from vla_core.testing.static_backend import StaticCoreBackend

# Defaults de "Manejo de fallas" en Core Design.md (Environment: 5 s, 1 reintento).
ENVIRONMENT_TIMEOUT_S = 5.0
ENVIRONMENT_RETRIES = 1


def build_environments(
    entries: list[dict[str, Any]], timeouts: dict[str, Any] | None = None
) -> EnvironmentRegistry:
    """Registro con un ``EnvironmentClient`` por entrada ``{id, address}``.

    No hay autoregistro: un ``env_id`` fuera de la config es
    ``UnknownEnvironment``. Los clientes conectan perezosamente (el ``spec`` se
    pide al primer uso), así que el orden de arranque de los procesos no importa.
    """
    if not entries:
        raise ValueError("el bloque `environments` está vacío")
    timeouts = timeouts or {}
    timeout_s = float(timeouts.get("environment_s", ENVIRONMENT_TIMEOUT_S))
    retries = int(timeouts.get("retries", ENVIRONMENT_RETRIES))
    registro = EnvironmentRegistry()
    for entrada in entries:
        if "id" not in entrada or "address" not in entrada:
            raise ValueError(f"cada ambiente necesita `id` y `address`: {entrada}")
        env_id = str(entrada["id"])
        if env_id in registro.registered():
            raise ValueError(f"ambiente repetido en la config: {env_id!r}")
        registro.register(env_id, EnvironmentClient(entrada["address"], timeout_s, retries))
    return registro


def _build_backend(
    config: dict[str, Any],
    dataset_config: dict[str, Any] | None = None,
    environments_config: list[dict[str, Any]] | None = None,
    timeouts: dict[str, Any] | None = None,
) -> CoreInterface:
    tipo = config.get("type", "static")
    if tipo == "static":
        return StaticCoreBackend()
    if tipo == "fake":
        # Gateway Core real con componentes fake en memoria, salvo lo que la
        # config reemplace: `dataset` elige el Dataset vía build_dataset (Core
        # queda agnóstico del formato) y `environments` conecta procesos vla-env
        # (sin él, cualquier env_id obtiene un FakeEnvironment al vuelo).
        from vla_core.dataset import build_dataset
        from vla_core.testing.fakes import make_fake_core

        dataset = build_dataset(dataset_config) if dataset_config else None
        environments = (
            build_environments(environments_config, timeouts)
            if environments_config is not None
            else None
        )
        return make_fake_core(
            demo_chunk_size=config.get("demo_chunk_size", 1),
            dataset=dataset,
            environments=environments,
        )
    raise ValueError(
        f"Backend de Core desconocido: {tipo!r}. El gateway con componentes reales "
        "se agrega en fases posteriores."
    )


def build_core(config: dict[str, Any]) -> CoreServer:
    backend = _build_backend(
        config.get("backend", {"type": "static"}),
        config.get("dataset"),
        config.get("environments"),
        config.get("timeouts"),
    )
    address = config.get("server", {}).get("address", "tcp://127.0.0.1:5555")
    return CoreServer(backend, address)
