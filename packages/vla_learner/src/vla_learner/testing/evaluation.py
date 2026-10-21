"""Evaluación de aprendizaje: tasa de éxito de una política en ``ToyReachEnv``.

La política actúa de forma **determinista** (usa la media, no muestrea), así una
política recién inicializada casi no mueve el punto y su tasa de éxito es ~0.
Cada episodio parte de un estado distinto pero reproducible (una semilla por
episodio).
"""

from __future__ import annotations

import numpy as np
import torch

from vla_common import Action, ActionChunk
from vla_learner.policy.interfaces import VLAPolicy
from vla_learner.testing.fake_core import FakeCoreClient
from vla_learner.testing.toy_env import MAX_STEPS


def _to_chunk(action_row: torch.Tensor) -> ActionChunk:
    valores = action_row.detach().cpu().numpy().astype(np.float32)
    return [Action(values=valores[h].copy()) for h in range(valores.shape[0])]


@torch.no_grad()
def success_rate(
    policy: VLAPolicy,
    n_episodes: int = 50,
    max_steps: int = MAX_STEPS,
    seed: int = 1000,
) -> float:
    """Fracción de episodios que llegan al objetivo con la política determinista."""
    previo = policy.training
    policy.eval()
    try:
        client = FakeCoreClient()
        env_ids = [f"eval-{i}" for i in range(n_episodes)]
        observaciones = {}
        for i, env_id in enumerate(env_ids):
            ((_, obs),) = client.reset([env_id], seed=seed + i)
            observaciones[env_id] = obs
        terminado = dict.fromkeys(env_ids, False)
        exito = dict.fromkeys(env_ids, False)
        for _ in range(max_steps):
            activos = [e for e in env_ids if not terminado[e]]
            if not activos:
                break
            salida = policy.act(
                policy.processor([observaciones[e] for e in activos]), deterministic=True
            )
            chunks = [_to_chunk(salida.actions[i]) for i in range(len(activos))]
            resultados = client.step(activos, chunks, with_reward=False)
            for i, env_id in enumerate(activos):
                observaciones[env_id] = resultados[i].observation
                if resultados[i].terminated:
                    exito[env_id] = True
                    terminado[env_id] = True
                elif resultados[i].truncated:
                    terminado[env_id] = True
        return sum(exito.values()) / n_episodes
    finally:
        if previo:
            policy.train()
