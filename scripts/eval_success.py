"""Tasa de éxito de una política, por tarea, conduciendo ambientes reales por Core.

    uv run --extra pi0 python scripts/eval_success.py \\
        --learner-config configs/pi0_libero_bc.yaml --checkpoint checkpoints/pi0_libero \\
        --tasks "put the bowl on the plate" "open the top drawer" \\
        --episodes 10 --env-ids libero-0 libero-1 --out out/eval.json

Fase 6 de ``docs/environment/Environment Implementation Plan.md``. La política
se arma con la config del Learner y se carga desde un checkpoint (sin tocar el
Learner). Cada tarea corre ``--episodes`` episodios con una semilla distinta
(``seed``, ``seed + 1``, …), repartidos en tandas entre los ``--env-ids``, que
deben existir en el Core (bloque ``environments``). La política actúa en modo
determinista y sin recompensa (``with_reward=False``). Un episodio cuenta como
éxito por ``info["success"]``, no por ``terminated`` (que también marca fallos).

Es una medición interna del harness, no un benchmark publicable (ver
``Architecture.md``, "Evaluación").
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from vla_common import Action, ActionChunk, CoreInterface, Observation
from vla_learner.data.core_client import CoreClient
from vla_learner.learner import Learner
from vla_learner.policy.interfaces import VLAPolicy

MAX_CHUNKS = 1000  # tope de seguridad por episodio; los ambientes truncan antes


@dataclass
class TaskResult:
    episodes: int = 0
    successes: int = 0

    @property
    def rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0


@dataclass
class Report:
    tasks: dict[str, TaskResult] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        total = sum(r.episodes for r in self.tasks.values())
        return sum(r.successes for r in self.tasks.values()) / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "rate": self.rate,
            "tasks": {t: {**asdict(r), "rate": r.rate} for t, r in self.tasks.items()},
        }

    def text(self) -> str:
        ancho = max((len(t) for t in self.tasks), default=5)
        lineas = [f"{'tarea':<{ancho}}  éxitos  tasa"]
        for tarea, r in self.tasks.items():
            lineas.append(f"{tarea:<{ancho}}  {r.successes:>3}/{r.episodes:<3} {r.rate:.2f}")
        lineas.append(f"{'TOTAL':<{ancho}}  {'':>7} {self.rate:.2f}")
        return "\n".join(lineas)


def _a_chunk(fila: torch.Tensor) -> ActionChunk:
    valores = fila.detach().cpu().numpy().astype(np.float32)
    return [Action(values=valores[h].copy()) for h in range(valores.shape[0])]


@torch.no_grad()
def evaluate(
    policy: VLAPolicy,
    client: CoreInterface,
    tasks: list[str],
    episodes_per_task: int,
    env_ids: list[str],
    seed: int = 0,
) -> Report:
    """Corre ``episodes_per_task`` episodios por tarea y cuenta los exitosos."""
    if not env_ids:
        raise ValueError("hace falta al menos un env_id")
    previo = policy.training
    policy.eval()
    reporte = Report()
    try:
        for tarea in tasks:
            resultado = reporte.tasks.setdefault(tarea, TaskResult())
            semillas = list(range(seed, seed + episodes_per_task))
            for inicio in range(0, len(semillas), len(env_ids)):
                tanda = list(zip(env_ids, semillas[inicio : inicio + len(env_ids)], strict=False))
                exitos = _correr_tanda(policy, client, tarea, tanda)
                resultado.episodes += len(tanda)
                resultado.successes += exitos
    finally:
        if previo:
            policy.train()
    return reporte


def _correr_tanda(
    policy: VLAPolicy, client: CoreInterface, tarea: str, tanda: list[tuple[str, int]]
) -> int:
    observaciones: dict[str, Observation] = {}
    for env_id, semilla in tanda:  # una semilla por ambiente: reset de a uno
        try:
            ((_, obs),) = client.reset([env_id], task=tarea, seed=semilla)
        except RuntimeError as error:
            raise ValueError(f"no se pudo iniciar {tarea!r} en {env_id!r}: {error}") from error
        observaciones[env_id] = obs
    activos = [env_id for env_id, _ in tanda]
    exitos = 0
    for _ in range(MAX_CHUNKS):
        if not activos:
            break
        salida = policy.act(
            policy.processor([observaciones[e] for e in activos]), deterministic=True
        )
        chunks = [_a_chunk(salida.actions[i]) for i in range(len(activos))]
        resultados = client.step(activos, chunks, with_reward=False)
        siguen = []
        for env_id, res in zip(activos, resultados, strict=True):
            observaciones[env_id] = res.observation
            if res.terminated or res.truncated:
                exitos += int(bool(res.info.get("success", False)))
            else:
                siguen.append(env_id)
        activos = siguen
    if activos:
        raise RuntimeError(f"episodios sin terminar tras {MAX_CHUNKS} chunks: {activos}")
    return exitos


def latest_checkpoint(learner_config: dict) -> Path | None:
    """El último ``step_*`` del directorio de checkpoints de la config, si hay."""
    directorio = Path(learner_config.get("checkpoint", {}).get("directory", "checkpoints"))
    pasos = sorted(p for p in directorio.glob("step_*") if p.is_dir())
    return pasos[-1] if pasos else None


def eval_config(
    learner_config: dict, checkpoint: Path | None, core_address: str | None = None
) -> dict:
    """La config del Learner adaptada para evaluar (sin mutar la original).

    - ``core_address`` pisa ``core.address`` (p. ej. el socket ``ipc://`` del job).
    - Con checkpoint, ``normalizer.stats: fit`` pasa a la carpeta ``policy/`` del
      checkpoint: la evaluación usa las estadísticas con las que se entrenó, nunca
      las re-ajusta contra el dataset del Core de evaluación (que puede ser otro).
    """
    config = copy.deepcopy(learner_config)
    if core_address is not None:
        config.setdefault("core", {})["address"] = core_address
    normalizador = config.get("policy", {}).get("normalizer")
    if checkpoint is not None and isinstance(normalizador, dict):
        if normalizador.get("stats") == "fit":
            estadisticas = Path(checkpoint) / "policy"
            if not (estadisticas / "action_stats.pt").exists():
                raise FileNotFoundError(
                    f"el checkpoint no trae estadísticas del normalizador en {estadisticas}"
                )
            normalizador["stats"] = str(estadisticas)
            normalizador.pop("fit_samples", None)
    return config


def load_policy(learner_config: dict, checkpoint: Path | None) -> VLAPolicy:
    """La política de la config, con los pesos de ``checkpoint`` (sin él, la inicial)."""
    learner = Learner.from_config(learner_config)
    if checkpoint is not None:
        learner.resume(checkpoint)
    return learner.policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_success")
    parser.add_argument("--learner-config", required=True, type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, help="step_* a evaluar (por defecto, el último)."
    )
    parser.add_argument("--core-address", help="Pisa core.address de la config del Learner.")
    parser.add_argument("--tasks", nargs="+", required=True, help="Instrucciones a evaluar.")
    parser.add_argument("--episodes", type=int, default=10, help="Episodios por tarea.")
    parser.add_argument("--env-ids", nargs="+", required=True, help="Ambientes del Core.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="JSON con el resultado.")
    args = parser.parse_args(argv)

    original = yaml.safe_load(args.learner_config.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or latest_checkpoint(original)
    if checkpoint is None:
        print("AVISO: sin checkpoint; se evalúa la política inicial.", file=sys.stderr)
    config = eval_config(original, checkpoint, args.core_address)
    policy = load_policy(config, checkpoint)
    core = config["core"]
    client = CoreClient(core["address"], core.get("timeout_s", 60.0))
    try:
        reporte = evaluate(policy, client, args.tasks, args.episodes, args.env_ids, args.seed)
    finally:
        client.close()
    print(reporte.text())
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reporte.to_dict(), indent=2, ensure_ascii=False), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
