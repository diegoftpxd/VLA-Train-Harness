"""``build_environment(config)``: construye el adaptador según ``type``.

Definido en ``docs/environment/Environment Design.md``. Igual que
``build_dataset``: cada adaptador se importa dentro de su rama, así una
dependencia pesada (MuJoCo, un driver) no se carga salvo que la config la pida,
y la instalación base de ``vla_env`` sigue liviana.
"""

from __future__ import annotations

from typing import Any

from vla_common.environment import Environment

# Grupo opcional de uv que necesita cada adaptador con dependencias.
EXTRAS: dict[str, str] = {}


def build_environment(config: dict[str, Any]) -> Environment:
    tipo = config.get("type", "reach")
    opciones = {k: v for k, v in config.items() if k not in ("type", "smoke_action")}
    try:
        if tipo == "reach":
            from vla_env.reach import ReachEnvironment

            return ReachEnvironment(**opciones)
    except ImportError as error:
        extra = EXTRAS.get(tipo, tipo)
        raise ImportError(
            f"el ambiente {tipo!r} necesita dependencias opcionales: "
            f"`uv sync --extra {extra}` ({error})"
        ) from error
    raise ValueError(f"tipo de ambiente desconocido: {tipo!r}. Disponibles: reach")
