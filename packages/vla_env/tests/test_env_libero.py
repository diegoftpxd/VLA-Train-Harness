"""LIBERO sin el simulador (``fast``): mapeo al tipo canónico, el envoltorio del
simulador, la config de LIBERO y la factory sin el extra.

Lo que necesita MuJoCo (ENV/SRVE contra LIBERO real y MAP) va en
``test_env_from_config.py`` (``heavy``) y en ``slurm/libero_env_check.sbatch``.
Ver ``docs/environment/Test Environment.md``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from vla_env import libero as lib
from vla_env.factory import build_environment

pytestmark = pytest.mark.fast

S = float(np.sqrt(0.5))


@pytest.mark.parametrize(
    "quat, esperado",
    [
        ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),  # identidad (w≈1, sin eje)
        ((S, 0.0, 0.0, S), (np.pi / 2, 0.0, 0.0)),  # 90° en x
        ((0.0, S, 0.0, S), (0.0, np.pi / 2, 0.0)),  # 90° en y
        ((0.0, 0.0, S, S), (0.0, 0.0, np.pi / 2)),  # 90° en z
        ((1.0, 0.0, 0.0, 0.0), (np.pi, 0.0, 0.0)),  # 180° en x
    ],
)
def test_lib01_quat2axisangle(quat: tuple, esperado: tuple) -> None:
    resultado = lib.quat2axisangle(np.array(quat))
    assert resultado.dtype == np.float32
    assert np.allclose(resultado, esperado, atol=1e-5)


def test_lib02_proprio_en_el_orden_del_dataset() -> None:
    raw = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, S, S]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }
    proprio = lib.libero_proprio(raw)
    assert proprio.dtype == np.float32 and proprio.shape == (8,)
    assert np.allclose(proprio, [0.1, 0.2, 0.3, 0.0, 0.0, np.pi / 2, 0.04, -0.04], atol=1e-6)


def test_lib03_imagenes_rotadas_180_y_canonicas() -> None:
    imagen = np.zeros((4, 6, 3), dtype=np.uint8)
    imagen[0, 0] = [255, 255, 255]  # arriba a la izquierda
    imagenes = lib.libero_images({"agentview_image": imagen}, {"base": "agentview_image"})
    base = imagenes["base"]
    assert base.shape == (3, 4, 6) and base.dtype == np.float32
    assert base[:, 3, 5].tolist() == [1.0, 1.0, 1.0]  # abajo a la derecha
    assert base.sum() == 3.0


def test_lib04_indice_de_tareas_y_limites() -> None:
    indice = lib.index_tasks({"libero_goal": ["abrir", "cerrar"], "libero_10": ["apilar"]})
    assert indice == {
        "abrir": ("libero_goal", 0),
        "cerrar": ("libero_goal", 1),
        "apilar": ("libero_10", 0),
    }
    with pytest.raises(ValueError, match="repetida"):
        lib.index_tasks({"a": ["x"], "b": ["x"]})
    assert lib.MAX_STEPS["libero_10"] == 520 and lib.MAX_STEPS["libero_spatial"] == 280


class _OffScreenFalso:
    """Lo que ``_LiberoSim`` usa de ``OffScreenRenderEnv``, con registro de llamadas."""

    def __init__(self, exito_en: int = 3) -> None:
        self.llamadas: list[Any] = []
        self.robots = [SimpleNamespace(controller=SimpleNamespace(use_delta=False))]
        self._pasos = 0
        self._exito_en = exito_en

    def seed(self, seed: int) -> None:
        self.llamadas.append(("seed", seed))

    def reset(self) -> dict:
        self.llamadas.append("reset")
        self._pasos = 0
        return {}

    def set_init_state(self, estado: np.ndarray) -> dict:
        self.llamadas.append(("init", float(estado[0])))
        return {"estado": float(estado[0])}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        self.llamadas.append(("step", action.tolist()))
        self._pasos += 1
        return {"paso": self._pasos}, 0.0, False, {}

    def check_success(self) -> bool:
        return self._pasos >= self._exito_en

    def close(self) -> None:
        self.llamadas.append("close")


def test_lib08_envoltorio_del_simulador() -> None:
    falso = _OffScreenFalso(exito_en=4)
    sim = lib._LiberoSim(falso, np.arange(5, dtype=np.float32)[:, None], num_steps_wait=2)
    raw, info = sim.reset(seed=7)
    noop = lib.NOOP_ACTION.tolist()
    assert falso.llamadas == [("seed", 7), "reset", ("init", 2.0), ("step", noop), ("step", noop)]
    assert raw == {"paso": 2} and info == {}
    assert falso.robots[0].controller.use_delta is True
    _, _, terminated, truncated, info = sim.step(np.ones(7, dtype=np.float32))
    assert (terminated, truncated, info) == (False, False, {"is_success": False})
    _, _, _, _, info = sim.step(np.ones(7, dtype=np.float32))
    assert info == {"is_success": True}


def test_lib08_semilla_none_elige_un_estado_valido() -> None:
    falso = _OffScreenFalso()
    sim = lib._LiberoSim(falso, np.arange(3, dtype=np.float32)[:, None], num_steps_wait=0)
    sim.reset(seed=None)
    assert ("seed", None) not in falso.llamadas
    (inicial,) = [c for c in falso.llamadas if isinstance(c, tuple) and c[0] == "init"]
    assert inicial[1] in (0.0, 1.0, 2.0)


def _spec_falsa(raiz: Path) -> SimpleNamespace:
    return SimpleNamespace(submodule_search_locations=[str(raiz)])


def test_lib09_config_de_libero_sin_preguntar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "cfg"))
    paquete = tmp_path / "site" / "libero"
    monkeypatch.setattr(importlib.util, "find_spec", lambda nombre: _spec_falsa(paquete))
    archivo = lib.ensure_libero_config()
    config = yaml.safe_load(archivo.read_text(encoding="utf-8"))
    assert Path(config["benchmark_root"]) == paquete / "libero"
    assert Path(config["bddl_files"]) == paquete / "libero" / "bddl_files"
    assert Path(config["init_states"]) == paquete / "libero" / "init_files"
    assert Path(config["datasets"]) == paquete / "datasets"
    assert Path(config["assets"]) == paquete / "libero" / "assets"
    # Si ya existe, no se toca (respeta una config hecha a mano).
    archivo.write_text("benchmark_root: propia\n", encoding="utf-8")
    assert lib.ensure_libero_config().read_text(encoding="utf-8") == "benchmark_root: propia\n"


def test_fact03_libero_sin_el_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path))
    monkeypatch.setattr(importlib.util, "find_spec", lambda nombre: None)
    with pytest.raises(ImportError, match="uv sync --extra libero"):
        build_environment({"type": "libero", "suites": ["libero_goal"], "cameras": {}})
