"""``RolloutCollector``: el loop de reset/act/step que produce rollouts.

Definido en ``docs/learner/Learner Design.md``. Es la única clase que llama
``policy.act``. Pone la política en modo ``eval`` mientras recolecta y la
devuelve a su modo previo al terminar; todo bajo ``no_grad``, así no deja grafo
ni gradientes (DATA-09). Como ``act`` ya devuelve unidades reales, pasar a
``ActionChunk`` es solo un cambio de formato.

Para rollouts en el WorldModel se genera, por cada punto de partida,
``sequences_per_start`` secuencias de ``horizon`` chunks muestreadas open-loop
desde ese estado; cada secuencia es una fila cuyo reward es el retorno de la
secuencia y cuya acción acreditada es la del primer chunk (vista de bandit
contextual para GRPO). Esto se documenta en el diseño.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from vla_common import Action, ActionChunk, Observation
from vla_learner.data.core_client import CoreClientProtocol
from vla_learner.policy.interfaces import VLAPolicy
from vla_learner.types import ModelInput, RolloutBatch


def _row_to_chunk(action_row: torch.Tensor) -> ActionChunk:
    valores = action_row.detach().cpu().numpy().astype(np.float32)
    return [Action(values=valores[h].copy()) for h in range(valores.shape[0])]


class RolloutCollector:
    def __init__(self, client: CoreClientProtocol) -> None:
        self.client = client

    @torch.no_grad()
    def collect_env(
        self,
        policy: VLAPolicy,
        env_ids: list[str],
        num_steps: int,
        task: str | None = None,
        seed: int | None = None,
        group_size: int = 1,
    ) -> RolloutBatch:
        if len(env_ids) % group_size != 0:
            raise ValueError("len(env_ids) debe ser múltiplo de group_size.")
        previo = policy.training
        policy.eval()
        try:
            observaciones, episode_ids, group_of = self._reset_en_grupos(
                env_ids, task, seed, group_size
            )
            paso_de = dict.fromkeys(env_ids, 0)
            filas: list[dict[str, Any]] = []
            for _ in range(num_steps):
                inputs = policy.processor([observaciones[e] for e in env_ids])
                salida = policy.act(inputs)
                assert salida.log_probs is not None, "los rollouts exigen LOG_PROB"
                chunks = [_row_to_chunk(salida.actions[i]) for i in range(len(env_ids))]
                resultados = self.client.step(env_ids, chunks, with_reward=True)
                for i, env_id in enumerate(env_ids):
                    resultado = resultados[i]
                    filas.append(
                        {
                            "inputs": {k: v[i] for k, v in inputs.tensors.items()},
                            "action": salida.actions[i],
                            "old_log_prob": salida.log_probs[i],
                            "old_value": None if salida.values is None else salida.values[i],
                            "reward": float(resultado.reward or 0.0),
                            "terminated": bool(resultado.terminated),
                            "truncated": bool(resultado.truncated),
                            "episode_id": episode_ids[env_id],
                            "step_id": paso_de[env_id],
                            "group_id": group_of[env_id],
                            "extras": {k: v[i] for k, v in salida.extras.items()},
                        }
                    )
                    paso_de[env_id] += 1
                    observaciones[env_id] = resultado.observation
                    if resultado.terminated or resultado.truncated:
                        ((ep, obs),) = self.client.reset([env_id], task=task, seed=None)
                        observaciones[env_id] = obs
                        episode_ids[env_id] = ep
                        paso_de[env_id] = 0
            return self._assemble(filas, source="env", use_groups=group_size > 1)
        finally:
            if previo:
                policy.train()

    @torch.no_grad()
    def collect_world_model(
        self,
        policy: VLAPolicy,
        starts: list[Observation | tuple[str, int]],
        sequences_per_start: int,
        horizon: int,
    ) -> RolloutBatch:
        previo = policy.training
        policy.eval()
        try:
            filas: list[dict[str, Any]] = []
            for grupo, start in enumerate(starts):
                start_obs = start if isinstance(start, Observation) else self._observacion(start)
                inputs = policy.processor([start_obs] * sequences_per_start)
                acciones_paso: list[torch.Tensor] = []
                logp_paso: list[torch.Tensor] = []
                extras_paso: list[dict[str, torch.Tensor]] = []
                for _ in range(horizon):
                    salida = policy.act(inputs)
                    assert salida.log_probs is not None, "los rollouts exigen LOG_PROB"
                    acciones_paso.append(salida.actions)
                    logp_paso.append(salida.log_probs)
                    extras_paso.append(salida.extras)
                secuencias = [
                    [_row_to_chunk(acciones_paso[h][s]) for h in range(horizon)]
                    for s in range(sequences_per_start)
                ]
                resultados = self.client.simulate(start, secuencias, with_reward=True)
                for s in range(sequences_per_start):
                    pasos = resultados[s]
                    total = float(sum((p.reward or 0.0) for p in pasos))
                    filas.append(
                        {
                            "inputs": {k: v[s] for k, v in inputs.tensors.items()},
                            "action": acciones_paso[0][s],
                            "old_log_prob": logp_paso[0][s],
                            "old_value": None,
                            "reward": total,
                            "terminated": bool(pasos[-1].terminated),
                            "truncated": bool(pasos[-1].truncated),
                            "episode_id": f"wm-g{grupo}-s{s}",
                            "step_id": 0,
                            "group_id": grupo,
                            "extras": {k: v[s] for k, v in extras_paso[0].items()},
                        }
                    )
            return self._assemble(filas, source="world_model", use_groups=True)
        finally:
            if previo:
                policy.train()

    # -- Interno ----------------------------------------------------------

    def _reset_en_grupos(
        self, env_ids: list[str], task: str | None, seed: int | None, group_size: int
    ) -> tuple[dict[str, Observation], dict[str, str], dict[str, int]]:
        observaciones: dict[str, Observation] = {}
        episode_ids: dict[str, str] = {}
        group_of: dict[str, int] = {}
        for grupo in range(len(env_ids) // group_size):
            miembros = env_ids[grupo * group_size : (grupo + 1) * group_size]
            if seed is not None:
                semilla: int | None = seed + grupo
            elif group_size > 1:
                # semilla compartida por el grupo → todos sus miembros arrancan igual
                semilla = random.randint(0, 2**31 - 1)
            else:
                semilla = None
            pares = self.client.reset(miembros, task, semilla)
            for env_id, (ep, obs) in zip(miembros, pares, strict=True):
                observaciones[env_id] = obs
                episode_ids[env_id] = ep
                group_of[env_id] = grupo
        return observaciones, episode_ids, group_of

    def _observacion(self, ref: tuple[str, int]) -> Observation:
        return self.client.get_episode(ref[0])[ref[1]].observation

    def _assemble(self, filas: list[dict[str, Any]], source: str, use_groups: bool) -> RolloutBatch:
        claves = filas[0]["inputs"].keys()
        tensors = {k: torch.stack([f["inputs"][k] for f in filas]) for k in claves}
        hay_valores = all(f["old_value"] is not None for f in filas)
        extras_claves = filas[0]["extras"].keys()
        return RolloutBatch(
            inputs=ModelInput(tensors=tensors, batch_size=len(filas)),
            actions=torch.stack([f["action"] for f in filas]),
            old_log_probs=torch.stack([f["old_log_prob"] for f in filas]),
            rewards=torch.tensor([f["reward"] for f in filas], dtype=torch.float32),
            terminated=torch.tensor([f["terminated"] for f in filas]),
            truncated=torch.tensor([f["truncated"] for f in filas]),
            episode_ids=[f["episode_id"] for f in filas],
            step_ids=torch.tensor([f["step_id"] for f in filas]),
            source=source,  # type: ignore[arg-type]
            old_values=torch.stack([f["old_value"] for f in filas]) if hay_valores else None,
            group_ids=torch.tensor([f["group_id"] for f in filas]) if use_groups else None,
            extras={k: torch.stack([f["extras"][k] for f in filas]) for k in extras_claves},
        )
