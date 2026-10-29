"""Lanza el flujo supervisado de punta a punta en 3 procesos independientes.

    uv run python scripts/run_supervised_e2e.py

Arranca, en orden: el proceso **Dataset** (sirve demos por ZeroMQ), el proceso
**Core** (gateway que consume ese Dataset y expone la interfaz al Learner) y el
proceso **Learner** (Behavior Cloning contra Core). El Learner corre en primer
plano; cuando termina, el lanzador baja Core y Dataset.

El flujo es: Learner.sample_supervised -> Core -> Dataset -> pares
(Observation, ActionChunk) -> el Learner calcula la loss de imitación y
actualiza los pesos. Todo pasa por Core; el Learner nunca habla con el Dataset.

La política por defecto es tiny (CPU), para correr en cualquier máquina. En el
cluster, pasá --learner-config apuntando a un YAML con un modelo real
(``policy: {type: openvla, ...}``, ver configs/openvla_bc.yaml); el resto del
flujo de 3 procesos no cambia.
"""



from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import yaml

RAIZ = Path(__file__).resolve().parents[1]


def _address(config_path: Path) -> str:
    """Dirección del socket ``server`` declarada en un YAML de proceso."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return config["server"]["address"]


def _wait_for_port(address: str, timeout_s: float = 30.0) -> None:
    """Espera a que alguien escuche en ``address`` (readiness del proceso).

    Un socket ``ROUTER`` sobre ``tcp://`` acepta conexiones TCP una vez bindeado,
    así que un connect exitoso indica que el proceso está atendiendo. Para
    transportes no-tcp (p. ej. ``ipc://``) cae a una espera fija corta.
    """
    parsed = urlparse(address)
    if parsed.scheme != "tcp" or parsed.hostname is None or parsed.port is None:
        time.sleep(1.0)
        return
    host = "127.0.0.1" if parsed.hostname == "*" else parsed.hostname
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, parsed.port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"nadie escucha en {address} tras {timeout_s}s")


def _spawn(script: str, config: Path) -> subprocess.Popen[bytes]:
    """Lanza ``uv run <script> --config <config>`` desde la raíz del workspace."""
    return subprocess.Popen(["uv", "run", script, "--config", str(config)], cwd=RAIZ)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flujo supervisado en 3 procesos (Dataset/Core/Learner)."
    )
    parser.add_argument("--dataset-config", type=Path, default=RAIZ / "configs/core/dataset.yaml")
    parser.add_argument("--core-config", type=Path, default=RAIZ / "configs/core/supervised.yaml")
    parser.add_argument("--learner-config", type=Path, default=RAIZ / "configs/supervised_e2e.yaml")
    args = parser.parse_args()

    procesos: list[subprocess.Popen[bytes]] = []
    codigo = 1
    try:
        print("[1/3] Levantando el proceso Dataset...", flush=True)
        procesos.append(_spawn("vla-core-dataset", args.dataset_config))
        _wait_for_port(_address(args.dataset_config))

        print("[2/3] Levantando el proceso Core...", flush=True)
        procesos.append(_spawn("vla-core", args.core_config))
        _wait_for_port(_address(args.core_config))

        print("[3/3] Corriendo el Learner (Behavior Cloning) en primer plano...\n", flush=True)
        learner = _spawn("vla-learner", args.learner_config)
        procesos.append(learner)
        codigo = learner.wait()
        print(f"\nLearner terminó con código {codigo}.", flush=True)
    finally:
        # Baja Core y Dataset (y el Learner, si seguía vivo por una interrupción).
        for proceso in reversed(procesos):
            if proceso.poll() is None:
                proceso.terminate()
        for proceso in reversed(procesos):
            try:
                proceso.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proceso.kill()
    sys.exit(codigo)


if __name__ == "__main__":
    main()
