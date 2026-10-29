"""Reglas de arquitectura de ``vla_env`` (ARCH-ENV-02/03).

``vla_env`` no depende de ``vla_core`` ni del Learner: se encuentra con Core solo
en ``vla_common``. Su base tampoco importa ``torch``; un adaptador que lo
necesite lo importa de forma perezosa, dentro de una función.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

RAIZ = Path(__file__).resolve().parents[1] / "src" / "vla_env"


def _imports_de_modulo(archivo: Path) -> set[str]:
    """Imports a nivel de módulo (no los perezosos dentro de funciones)."""
    nombres: set[str] = set()
    for nodo in ast.parse(archivo.read_text(encoding="utf-8")).body:
        if isinstance(nodo, ast.Import):
            nombres.update(alias.name.split(".")[0] for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.level == 0:
            nombres.add(nodo.module.split(".")[0])
    return nombres


def _todos_los_imports(archivo: Path) -> set[str]:
    nombres: set[str] = set()
    for nodo in ast.walk(ast.parse(archivo.read_text(encoding="utf-8"))):
        if isinstance(nodo, ast.Import):
            nombres.update(alias.name.split(".")[0] for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.level == 0:
            nombres.add(nodo.module.split(".")[0])
    return nombres


def test_arch_env02_no_importa_core_ni_learner() -> None:
    for archivo in RAIZ.rglob("*.py"):
        prohibidos = _todos_los_imports(archivo) & {"vla_core", "vla_learner"}
        assert not prohibidos, f"{archivo.name} importa {sorted(prohibidos)}"


def test_arch_env03_torch_solo_perezoso() -> None:
    for archivo in RAIZ.rglob("*.py"):
        assert "torch" not in _imports_de_modulo(archivo), f"{archivo.name} importa torch"
