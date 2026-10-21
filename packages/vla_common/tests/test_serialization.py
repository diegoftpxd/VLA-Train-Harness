"""Round-trip de la serialización: dtype, forma y valores bit a bit.

Precursor de la suite CORE (Fase 5), que exigirá lo mismo sobre el protocolo
completo hacia Core.
"""

import numpy as np
import pytest

from vla_common.serialization import decode, encode

pytestmark = pytest.mark.fast


def test_roundtrip_conserva_arrays_y_metadatos():
    payload = {
        "call": "step",
        "env_ids": ["a", "b"],
        "arrays": [
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.array([[1, 2], [3, 4]], dtype=np.int64),
        ],
        "flag": True,
        "reward": None,
    }

    frames = encode(payload)
    assert isinstance(frames, list)
    assert all(isinstance(frame, (bytes, bytearray)) for frame in frames)

    out = decode(frames)
    assert out["call"] == "step"
    assert out["env_ids"] == ["a", "b"]
    assert out["flag"] is True
    assert out["reward"] is None
    for original, recovered in zip(payload["arrays"], out["arrays"], strict=True):
        assert original.dtype == recovered.dtype
        assert original.shape == recovered.shape
        assert np.array_equal(original, recovered)


def test_cada_array_va_en_un_frame_separado():
    payload = {
        "x": np.zeros(4, dtype=np.float32),
        "y": np.ones(3, dtype=np.float64),
    }
    frames = encode(payload)
    assert len(frames) == 3  # 1 encabezado + 2 buffers


def test_escalares_de_numpy_se_convierten():
    out = decode(encode({"r": np.float32(1.5), "n": np.int64(7)}))
    assert out == {"r": 1.5, "n": 7}
