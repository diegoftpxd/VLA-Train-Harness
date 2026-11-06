"""Reglas de arquitectura de ``vla_common`` (ARCH-ENV-01).

``vla_common`` es el punto de encuentro de todos los procesos: no hace I/O ni
depende de ningún otro paquete del harness. ``testing/`` queda afuera porque
solo lo importan los tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

RAIZ = Path(__file__).resolve().parents[1] / "src" / "vla_common"
PROHIBIDOS = {"zmq", "torch", "vla_core", "vla_learner", "vla_env"}


def _modulos_importados(archivo: Path) -> set[str]:
    arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    nombres: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            nombres.update(alias.name.split(".")[0] for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.level == 0:
            nombres.add(nodo.module.split(".")[0])
    return nombres


def test_arch_env01_vla_common_no_importa_io_ni_otros_paquetes() -> None:
    archivos = [p for p in RAIZ.rglob("*.py") if "testing" not in p.relative_to(RAIZ).parts]
    assert archivos
    for archivo in archivos:
        prohibidos = _modulos_importados(archivo) & PROHIBIDOS
        assert not prohibidos, f"{archivo.relative_to(RAIZ)} importa {sorted(prohibidos)}"
