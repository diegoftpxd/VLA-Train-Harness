"""Punto de entrada del proceso Learner (``uv run vla-learner --config ...``).

Carga un YAML, arma el ``Learner`` con ``from_config`` y corre el loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vla_learner.learner import Learner


def main() -> None:
    parser = argparse.ArgumentParser(prog="vla-learner")
    parser.add_argument("--config", required=True, type=Path, help="Ruta al YAML de configuración.")
    parser.add_argument("--iterations", type=int, default=None, help="Sobrescribe num_iterations.")
    parser.add_argument("--resume", action="store_true", help="Retoma desde el último checkpoint.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    learner = Learner.from_config(config)
    if args.resume:
        learner.resume()
    iteraciones = (
        args.iterations if args.iterations is not None else config.get("num_iterations", 1)
    )
    learner.train(iteraciones)
