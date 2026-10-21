"""Tests de los generadores ``make_observations`` y ``make_action_chunks``."""

import numpy as np
import pytest

from vla_learner.testing.data import make_action_chunks, make_observations

pytestmark = pytest.mark.fast


def test_make_observations_formas_y_camaras():
    obs = make_observations(n=4, cameras=("front", "wrist"), seed=0, proprio_dim=7, image_size=32)
    assert len(obs) == 4
    for o in obs:
        assert set(o.images) == {"front", "wrist"}
        for img in o.images.values():
            assert img.shape == (3, 32, 32)
            assert img.dtype == np.float32
        assert o.proprio.shape == (7,)
        assert isinstance(o.instruction, str)
        assert len(o.instruction) > 0


def test_make_observations_determinista():
    a = make_observations(3, seed=42)
    b = make_observations(3, seed=42)
    for oa, ob in zip(a, b, strict=True):
        assert np.array_equal(oa.images["front"], ob.images["front"])
        assert np.array_equal(oa.proprio, ob.proprio)
        assert oa.instruction == ob.instruction


def test_make_observations_instrucciones_largo_variable():
    obs = make_observations(6, seed=1)
    largos = {len(o.instruction.split()) for o in obs}
    assert len(largos) > 1


def test_make_action_chunks_formas():
    chunks = make_action_chunks(n=5, chunk_size=3, action_dim=4, seed=0)
    assert len(chunks) == 5
    for chunk in chunks:
        assert len(chunk) == 3
        for action in chunk:
            assert action.values.shape == (4,)


def test_make_action_chunks_escalas_distintas():
    chunks = make_action_chunks(n=200, chunk_size=1, action_dim=3, seed=0)
    valores = np.stack([chunk[0].values for chunk in chunks])  # [200, 3]
    stds = valores.std(axis=0)
    # escalas 10**(j - 1): dim0 ~0.1, dim1 ~1, dim2 ~10
    assert stds[0] < stds[1] < stds[2]


def test_make_action_chunks_determinista():
    a = make_action_chunks(3, 2, 2, seed=7)
    b = make_action_chunks(3, 2, 2, seed=7)
    for ca, cb in zip(a, b, strict=True):
        for aa, bb in zip(ca, cb, strict=True):
            assert np.array_equal(aa.values, bb.values)
