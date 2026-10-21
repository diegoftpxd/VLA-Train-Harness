"""Suite STORE: contrato de ``EpisodeStore``.

La implementación entrega el fixture ``store``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_common.types import Action, Observation, StepResult, Transition
from vla_core.episodes import EpisodeStore


def _transition(episode_id: str, step: int) -> Transition:
    obs = Observation(
        images={"front": np.zeros((3, 8, 8), dtype=np.float32)},
        proprio=np.zeros(4, dtype=np.float32),
        instruction="reach the goal",
    )
    result = StepResult(observation=obs, reward=1.0, terminated=False, truncated=False, info={})
    return Transition(
        episode_id=episode_id,
        step=step,
        source="env",
        observation=obs,
        action=[Action(values=np.zeros(2, dtype=np.float32))],
        result=result,
    )


class EpisodeStoreContract:
    @pytest.fixture
    def store(self) -> EpisodeStore:
        raise NotImplementedError

    def test_store01_append_y_get_en_orden(self, store: EpisodeStore) -> None:
        store.append(_transition("ep1", 0))
        store.append(_transition("ep1", 1))
        transiciones = store.get("ep1")
        assert [t.step for t in transiciones] == [0, 1]

    def test_store02_get_desconocido_es_error(self, store: EpisodeStore) -> None:
        with pytest.raises(KeyError):
            store.get("no-existe")
