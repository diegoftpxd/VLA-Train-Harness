"""Algoritmos de aprendizaje: BC (SL) y PPO/GRPO (RL), con sus estimadores.

Los algoritmos solo calculan la loss; nunca llaman a ``backward()`` ni a
``step()`` (eso es del ``OptimizerManager``) y solo conocen la interfaz
``VLAPolicy``, no modelos concretos (lo verifican los tests ARCH).
"""

from vla_learner.algorithms.advantages import (
    GAE,
    AdvantageEstimator,
    GroupRelativeAdvantage,
    MonteCarloReturn,
)
from vla_learner.algorithms.base import Algorithm
from vla_learner.algorithms.bc import BehaviorCloning
from vla_learner.algorithms.grpo import GRPO, GRPOConfig
from vla_learner.algorithms.ppo import PPO, PPOConfig
from vla_learner.algorithms.reference import ReferencePolicy

__all__ = [
    "GAE",
    "GRPO",
    "PPO",
    "AdvantageEstimator",
    "Algorithm",
    "BehaviorCloning",
    "GRPOConfig",
    "GroupRelativeAdvantage",
    "MonteCarloReturn",
    "PPOConfig",
    "ReferencePolicy",
]
