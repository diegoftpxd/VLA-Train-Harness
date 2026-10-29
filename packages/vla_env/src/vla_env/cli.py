"""Punto de entrada de un proceso Environment (``uv run vla-env --config ...``).

Definido en ``docs/environment/Environment Design.md``. Dos modos:

- **Servidor** (por defecto): construye el ambiente del YAML y lo sirve por
  ZeroMQ hasta Ctrl-C. ``--address`` pisa ``server.address``, para levantar N
  procesos con un mismo YAML.
- **``--smoke N``**: prueba el ambiente sin Core ni Learner. Hace ``reset`` y N
  ``step`` con la acción neutra (``environment.smoke_action``, o ceros), informa
  formas, dtypes, rangos y tiempos, y con ``--save DIR`` guarda las
  observaciones en un ``npz`` para inspeccionarlas.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vla_common.environment import Environment
from vla_common.types import Action, Observation
from vla_env.factory import build_environment
from vla_env.server import EnvironmentServer

DEFAULT_ADDRESS = "tcp://127.0.0.1:5601"


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_server(config: dict[str, Any], address: str | None = None) -> EnvironmentServer:
    """Ambiente del bloque ``environment`` servido en ``address`` (o en ``server.address``)."""
    env = build_environment(config.get("environment", {}))
    direccion = address or config.get("server", {}).get("address", DEFAULT_ADDRESS)
    return EnvironmentServer(env, direccion)


@dataclass
class SmokeReport:
    spec: str
    steps: int
    episodes: int
    successes: int
    mean_step_s: float
    cameras: dict[str, str]  # nombre → "forma dtype [mín, máx]"
    proprio: str
    observations: list[Observation] = field(repr=False)

    def text(self) -> str:
        lineas = [
            f"spec: {self.spec}",
            f"pasos: {self.steps}  episodios: {self.episodes}  éxitos: {self.successes}",
            f"tiempo medio por step: {self.mean_step_s * 1000:.2f} ms",
            *(f"cámara {nombre}: {desc}" for nombre, desc in self.cameras.items()),
            f"proprio: {self.proprio}",
        ]
        return "\n".join(lineas)


def _describir(arrays: list[np.ndarray]) -> str:
    pila = np.stack(arrays)
    return f"{pila.shape[1:]} {pila.dtype} [{pila.min():.3f}, {pila.max():.3f}]"


def smoke(env: Environment, steps: int, action: list[float] | None = None) -> SmokeReport:
    """``reset`` y ``steps`` pasos con la acción neutra; reinicia si el episodio termina."""
    valores = np.asarray(
        action if action is not None else np.zeros(env.spec.action_dim), dtype=np.float32
    )
    chunk = [Action(values=valores)]
    observaciones = [env.reset(task=None, seed=0)]
    episodios, exitos, tiempos = 1, 0, []
    for _ in range(steps):
        inicio = time.perf_counter()
        resultado = env.step(chunk)
        tiempos.append(time.perf_counter() - inicio)
        observaciones.append(resultado.observation)
        if resultado.info.get("success"):
            exitos += 1
        if resultado.terminated or resultado.truncated:
            observaciones.append(env.reset(task=None, seed=episodios))
            episodios += 1
    return SmokeReport(
        spec=str(env.spec),
        steps=steps,
        episodes=episodios,
        successes=exitos,
        mean_step_s=float(np.mean(tiempos)) if tiempos else 0.0,
        cameras={c: _describir([o.images[c] for o in observaciones]) for c in env.spec.cameras},
        proprio=_describir([o.proprio for o in observaciones]),
        observations=observaciones,
    )


def save_observations(report: SmokeReport, directory: Path) -> Path:
    """Guarda las observaciones del smoke en ``directory/smoke.npz``."""
    directory.mkdir(parents=True, exist_ok=True)
    destino = directory / "smoke.npz"
    camaras = report.observations[0].images.keys()
    arrays = {c: np.stack([o.images[c] for o in report.observations]) for c in camaras}
    arrays["proprio"] = np.stack([o.proprio for o in report.observations])
    arrays["instruction"] = np.asarray([o.instruction for o in report.observations])
    np.savez_compressed(destino, **arrays)  # type: ignore[arg-type]
    return destino


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vla-env")
    parser.add_argument("--config", required=True, type=Path, help="YAML del proceso de ambiente.")
    parser.add_argument("--address", help="Pisa server.address (para levantar N procesos).")
    parser.add_argument("--smoke", type=int, metavar="N", help="Prueba local de N pasos, sin red.")
    parser.add_argument("--save", type=Path, metavar="DIR", help="Con --smoke: guarda un npz.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.smoke is not None:
        bloque = config.get("environment", {})
        env = build_environment(bloque)
        try:
            reporte = smoke(env, args.smoke, bloque.get("smoke_action"))
            print(reporte.text())
            if args.save is not None:
                print(f"observaciones guardadas en {save_observations(reporte, args.save)}")
        finally:
            env.emergency_stop()
            env.close()
        return 0

    server = build_server(config, args.address)
    direccion = server.start()
    print(f"Environment atendiendo en {direccion}. Ctrl-C para salir.", flush=True)
    try:
        threading.Event().wait()  # bloquea hasta Ctrl-C
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        server.env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
