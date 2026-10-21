"""Estimadores de ventaja: de recompensas a ventajas y retornos.

Definidos en ``docs/learner/Learner Design.md``. Trabajan sobre un ``RolloutBatch`` cuyas
filas pueden venir de varios episodios y desordenadas: se agrupan por
``episode_ids`` y se ordenan por ``step_ids`` antes de calcular, y el resultado
se devuelve en el orden original de las filas.

Nota sobre el bootstrap de GAE: el ``RolloutBatch`` solo guarda ``V(s_t)`` en
``old_values``. Para el último paso de un episodio, ``terminated`` corta el
bootstrap (valor siguiente = 0) y ``truncated`` lo mantiene usando ``V(s_t)`` del
propio paso como aproximación de ``V(s_{t+1})`` (el harness aún no almacena la
observación siguiente; la Fase 5 podrá guardar un valor de bootstrap propio).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

import torch
from torch import Tensor

from vla_learner.types import RolloutBatch

_EPS = 1e-8


def _grupos_ordenados(batch: RolloutBatch) -> dict[str, list[int]]:
    """Índices de fila por episodio, ordenados por ``step_ids`` ascendente."""
    pasos = batch.step_ids.tolist()
    por_episodio: dict[str, list[int]] = defaultdict(list)
    for fila, episode_id in enumerate(batch.episode_ids):
        por_episodio[episode_id].append(fila)
    for filas in por_episodio.values():
        filas.sort(key=lambda fila: pasos[fila])
    return por_episodio


class AdvantageEstimator(ABC):
    @abstractmethod
    def compute(self, batch: RolloutBatch) -> tuple[Tensor, Tensor]:
        """Devuelve (advantages ``[B]``, returns ``[B]``)."""


class GAE(AdvantageEstimator):
    def __init__(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        self.gamma = gamma
        self.lam = lam

    def compute(self, batch: RolloutBatch) -> tuple[Tensor, Tensor]:
        if batch.old_values is None:
            raise ValueError("GAE necesita old_values (una cabeza de valor).")
        values = batch.old_values
        rewards = batch.rewards
        terminated = batch.terminated
        advantages = torch.zeros(len(batch), dtype=torch.float32)
        returns = torch.zeros(len(batch), dtype=torch.float32)

        for filas in _grupos_ordenados(batch).values():
            ventaja_siguiente = 0.0
            for posicion in reversed(range(len(filas))):
                fila = filas[posicion]
                no_terminal = 0.0 if bool(terminated[fila]) else 1.0
                if posicion < len(filas) - 1:
                    valor_siguiente = float(values[filas[posicion + 1]])
                elif bool(terminated[fila]):
                    valor_siguiente = 0.0
                else:  # truncated (o fin de recolección): bootstrap con V(s_t)
                    valor_siguiente = float(values[fila])
                delta = (
                    float(rewards[fila])
                    + self.gamma * valor_siguiente * no_terminal
                    - float(values[fila])
                )
                ventaja_siguiente = delta + self.gamma * self.lam * no_terminal * ventaja_siguiente
                advantages[fila] = ventaja_siguiente
                returns[fila] = ventaja_siguiente + float(values[fila])
        return advantages, returns


class MonteCarloReturn(AdvantageEstimator):
    def __init__(self, gamma: float = 0.99) -> None:
        self.gamma = gamma

    def compute(self, batch: RolloutBatch) -> tuple[Tensor, Tensor]:
        rewards = batch.rewards
        returns = torch.zeros(len(batch), dtype=torch.float32)
        for filas in _grupos_ordenados(batch).values():
            acumulado = 0.0
            for posicion in reversed(range(len(filas))):
                fila = filas[posicion]
                acumulado = float(rewards[fila]) + self.gamma * acumulado
                returns[fila] = acumulado
        if batch.old_values is not None:
            advantages = returns - batch.old_values
        else:
            advantages = returns.clone()
        return advantages, returns


class GroupRelativeAdvantage(AdvantageEstimator):
    def __init__(self, normalize_std: bool = True) -> None:
        self.normalize_std = normalize_std

    def compute(self, batch: RolloutBatch) -> tuple[Tensor, Tensor]:
        if batch.group_ids is None:
            raise ValueError("GroupRelativeAdvantage necesita group_ids en el batch.")
        rewards = batch.rewards
        advantages = torch.zeros(len(batch), dtype=torch.float32)
        grupos = batch.group_ids.tolist()
        por_grupo: dict[int, list[int]] = defaultdict(list)
        for fila, group_id in enumerate(grupos):
            por_grupo[group_id].append(fila)
        for filas in por_grupo.values():
            indices = torch.tensor(filas)
            recompensas = rewards[indices]
            media = recompensas.mean()
            desviacion = recompensas.std(unbiased=False)
            if len(filas) <= 1 or float(desviacion) < _EPS:
                continue  # grupo degenerado → ventaja 0, sin NaN
            centrado = recompensas - media
            advantages[indices] = centrado / desviacion if self.normalize_std else centrado
        return advantages, rewards.clone()
