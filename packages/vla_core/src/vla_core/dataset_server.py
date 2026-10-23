"""Punto de entrada del proceso Dataset (``uv run vla-core-dataset --config ...``).

En v1 el Dataset vive dentro de Core; este proceso lo expone aparte por ZeroMQ
para el flujo de integración de 3 procesos (Dataset ↔ Core ↔ Learner). Sirve un
``FakeDataset`` —el único dataset disponible en v1— detrás de un socket
``ROUTER``; Core se conecta como ``DEALER`` con un ``DatasetClient``. Ver
``docs/core/Core Design.md`` ("Dataset como proceso aparte").
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import yaml

from vla_core.testing.fakes import FakeDataset, serve_dataset


def main() -> None:
    parser = argparse.ArgumentParser(prog="vla-core-dataset")
    parser.add_argument("--config", required=True, type=Path, help="Ruta al YAML de configuración.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ds_cfg = config.get("dataset", {})
    address = config.get("server", {}).get("address", "tcp://127.0.0.1:5556")

    dataset = FakeDataset(
        seed=ds_cfg.get("seed", 0), demo_chunk_size=ds_cfg.get("demo_chunk_size", 1)
    )
    server = serve_dataset(dataset, address)
    bound = server.start()
    print(f"Dataset atendiendo en {bound}. Ctrl-C para salir.")
    try:
        threading.Event().wait()  # bloquea hasta Ctrl-C
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
