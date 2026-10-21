"""Suites de contrato de los componentes de Core.

Las importan tanto los tests de los *fakes* como, en el futuro, los de las
implementaciones reales (un Environment, un Dataset de verdad). Cada suite deja
abstracto un *fixture* con la instancia bajo prueba.
"""

from vla_core.testing.contracts.dataset_contract import DatasetContract
from vla_core.testing.contracts.environment_contract import EnvironmentContract
from vla_core.testing.contracts.episode_store_contract import EpisodeStoreContract
from vla_core.testing.contracts.reward_model_contract import RewardModelContract
from vla_core.testing.contracts.world_model_contract import WorldModelContract

__all__ = [
    "DatasetContract",
    "EnvironmentContract",
    "EpisodeStoreContract",
    "RewardModelContract",
    "WorldModelContract",
]
