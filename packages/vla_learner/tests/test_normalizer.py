"""Suite NRM sobre ``ActionNormalizer``."""

import numpy as np
import pytest
import torch

from vla_learner.policy.normalizer import ActionNormalizer, ActionStats
from vla_learner.testing.data import make_observations

pytestmark = pytest.mark.fast

_MODES = ("mean_std", "min_max", "quantile")


def _stats_de(data: np.ndarray) -> ActionStats:
    return ActionStats(
        mean=data.mean(axis=0),
        std=data.std(axis=0),
        min=data.min(axis=0),
        max=data.max(axis=0),
        q01=np.quantile(data, 0.01, axis=0),
        q99=np.quantile(data, 0.99, axis=0),
    )


def _dos_embodiments() -> dict[str, ActionStats]:
    rng = np.random.default_rng(0)
    return {
        "robotA": _stats_de(rng.standard_normal((100, 3)) * np.array([0.1, 1.0, 10.0])),
        "robotB": _stats_de(rng.standard_normal((100, 3)) + 5.0),
    }


def test_nrm01_ida_y_vuelta_en_los_tres_modos():
    stats = _dos_embodiments()
    x = torch.randn(5, 4, 3)
    for mode in _MODES:
        norm = ActionNormalizer(stats, mode=mode)
        for embodiment in stats:
            recuperado = norm.denormalize(norm.normalize(x, embodiment), embodiment)
            assert torch.allclose(recuperado, x, atol=1e-4)


def test_nrm02_estadisticas_por_embodiment():
    stats = _dos_embodiments()
    norm = ActionNormalizer(stats, mode="mean_std")
    x = torch.randn(4, 2, 3)
    assert not torch.allclose(norm.normalize(x, "robotA"), norm.normalize(x, "robotB"))
    with pytest.raises(KeyError):
        norm.normalize(x, "desconocido")


def test_nrm03_fit_coincide_con_numpy():
    rng = np.random.default_rng(1)
    obs = make_observations(1, seed=0)[0]
    from vla_common import Action

    muestras = []
    todos = []
    for _ in range(20):
        valores = rng.standard_normal((3, 2)).astype(np.float32)
        muestras.append((obs, [Action(values=valores[t]) for t in range(3)]))
        todos.append(valores)
    referencia = np.concatenate(todos, axis=0)

    norm = ActionNormalizer.fit(muestras, mode="mean_std")
    assert np.allclose(norm._buffer("default", "mean").numpy(), referencia.mean(axis=0), atol=1e-5)
    assert np.allclose(norm._buffer("default", "std").numpy(), referencia.std(axis=0), atol=1e-5)


def test_nrm04_dimension_constante_sin_nan():
    stats = ActionStats(
        mean=np.array([1.0, 5.0], np.float32),
        std=np.array([0.0, 2.0], np.float32),
        min=np.array([1.0, 0.0], np.float32),
        max=np.array([1.0, 9.0], np.float32),
        q01=np.array([1.0, 0.1], np.float32),
        q99=np.array([1.0, 8.0], np.float32),
    )
    x = torch.tensor([[[1.0, 3.0]]])  # [1, 1, 2]; la dim 0 es constante
    for mode in _MODES:
        norm = ActionNormalizer({"c": stats}, mode=mode)
        normalizado = norm.normalize(x, "c")
        assert torch.isfinite(normalizado).all()
        assert torch.allclose(norm.denormalize(normalizado, "c"), x, atol=1e-4)


def test_nrm05_sobrevive_state_dict():
    stats = _dos_embodiments()
    norm = ActionNormalizer(stats, mode="quantile")
    otra = ActionNormalizer(stats, mode="quantile")
    otra.load_state_dict(norm.state_dict())
    x = torch.randn(3, 1, 3)
    assert torch.allclose(norm.normalize(x, "robotA"), otra.normalize(x, "robotA"))


def test_from_values_coincide_con_fit_y_expone_la_dimension():
    muestras = list(zip(make_observations(4), _chunks_de_prueba(), strict=True))
    por_fit = ActionNormalizer.fit(muestras, mode="mean_std")
    valores = np.concatenate([np.stack([a.values for a in chunk]) for _, chunk in muestras])
    por_valores = ActionNormalizer.from_values(valores, mode="mean_std")
    x = torch.randn(2, 3, 2)
    assert torch.allclose(por_fit.normalize(x, "default"), por_valores.normalize(x, "default"))
    assert por_valores.dim() == 2


def test_from_values_rechaza_formas_invalidas():
    with pytest.raises(ValueError, match=r"\[N, D\]"):
        ActionNormalizer.from_values(np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[N, D\]"):
        ActionNormalizer.from_values(np.zeros((0, 2), dtype=np.float32))


def _chunks_de_prueba():
    from vla_common import Action

    rng = np.random.default_rng(1)
    return [
        [Action(values=(rng.standard_normal(2) * 50 + 300).astype(np.float32)) for _ in range(3)]
        for _ in range(4)
    ]
