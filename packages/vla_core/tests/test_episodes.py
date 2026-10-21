"""Suite EP: estado en vuelo de ``EpisodeManager``."""

from __future__ import annotations

import numpy as np
import pytest

from vla_common.types import Observation
from vla_core.episodes import EpisodeManager

pytestmark = pytest.mark.fast


def _obs(valor: float) -> Observation:
    return Observation(
        images={"front": np.zeros((3, 8, 8), dtype=np.float32)},
        proprio=np.full(4, valor, dtype=np.float32),
        instruction="reach the goal",
    )


def test_ep01_open_crea_id_unico_y_guarda_observacion() -> None:
    manager = EpisodeManager()
    ep0 = manager.open("e0", _obs(0.0))
    ep1 = manager.open("e1", _obs(1.0))
    assert ep0 != ep1
    assert manager.episode_of("e0") == ep0
    assert np.array_equal(manager.last_observation("e1").proprio, _obs(1.0).proprio)
    assert not manager.is_done("e0")


def test_ep02_advance_incrementa_y_actualiza() -> None:
    manager = EpisodeManager()
    manager.open("e0", _obs(0.0))
    paso0 = manager.advance("e0", _obs(1.0), terminated=False, truncated=False)
    paso1 = manager.advance("e0", _obs(2.0), terminated=False, truncated=False)
    assert (paso0, paso1) == (0, 1)
    assert np.array_equal(manager.last_observation("e0").proprio, _obs(2.0).proprio)


def test_ep03_is_done_refleja_fin() -> None:
    manager = EpisodeManager()
    manager.open("e0", _obs(0.0))
    manager.advance("e0", _obs(1.0), terminated=True, truncated=False)
    assert manager.is_done("e0")
    # Reabrir el episodio lo deja activo otra vez.
    manager.open("e0", _obs(0.0))
    assert not manager.is_done("e0")
