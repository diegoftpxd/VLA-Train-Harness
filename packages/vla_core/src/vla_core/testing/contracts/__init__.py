"""Suites de contrato de los componentes de Core.

Las importan tanto los tests de los *fakes* como, en el futuro, los de las
implementaciones reales (un Dataset de verdad, un WorldModel). Cada suite deja
abstracto un *fixture* con la instancia bajo prueba. La suite ENV vive en
``vla_common.testing.contracts`` (la comparten Core y ``vla_env``).
"""

from vla_core.testing.contracts.dataset_contract import (
    AppendableDatasetContract,
    DatasetContract,
)
from vla_core.testing.contracts.episode_store_contract import EpisodeStoreContract
from vla_core.testing.contracts.reward_model_contract import RewardModelContract
from vla_core.testing.contracts.world_model_contract import WorldModelContract

__all__ = [
    "AppendableDatasetContract",
    "DatasetContract",
    "EpisodeStoreContract",
    "RewardModelContract",
    "WorldModelContract",
]
