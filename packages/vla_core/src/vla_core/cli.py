"""Punto de entrada del proceso Core (``uv run vla-core --config ...``).

Carga un YAML, arma el ``CoreServer`` con ``build_core`` y lo deja atendiendo
hasta que se interrumpe con Ctrl-C.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import yaml

from vla_core.builder import build_core


def main() -> None:
    parser = argparse.ArgumentParser(prog="vla-core")
    parser.add_argument("--config", required=True, type=Path, help="Ruta al YAML de configuración.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    server = build_core(config)
    address = server.start()
    print(f"Core atendiendo en {address}. Ctrl-C para salir.")
    try:
        threading.Event().wait()  # bloquea hasta Ctrl-C
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
