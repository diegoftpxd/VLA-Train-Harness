"""Cabezas de acción y de valor."""

from vla_learner.policy.heads.gaussian import GaussianActionHead
from vla_learner.policy.heads.token import TokenActionHead
from vla_learner.policy.heads.value import ValueHead

__all__ = ["GaussianActionHead", "TokenActionHead", "ValueHead"]
