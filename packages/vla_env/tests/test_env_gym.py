"""Suite GYM: reglas de la base ``GymEnvironment`` (``docs/environment/Test Environment.md``).

Se prueban sobre un simulador de juguete que registra cada acción recibida, para
verificar no solo el resultado sino qué llegó al simulador.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vla_common.environment import EnvSpec
from vla_common.types import Action, ActionChunk, Observation
from vla_env.gym import GymEnvironment, to_canonical_image

pytestmark = pytest.mark.fast


class _SimContador:
    """Simulador 1-D: suma la acción a ``x``. Puede terminar, cortar o marcar éxito
    al llegar a ciertos valores de ``x``, y siempre devuelve una recompensa."""

    def __init__(self, success_at: float | None = None, terminate_at: float | None = None):
        self.success_at = success_at
        self.terminate_at = terminate_at
        self.x = 0.0
        self.recibidas: list[float] = []
        self.semillas: list[int | None] = []

    def reset(self, seed: int | None = None) -> tuple[float, dict]:
        self.semillas.append(seed)
        self.x = 0.0
        return self.x, {}

    def step(self, action: np.ndarray) -> tuple[float, float, bool, bool, dict[str, Any]]:
        self.recibidas.append(float(action[0]))
        self.x += float(action[0])
        terminated = self.terminate_at is not None and self.x >= self.terminate_at
        exito = self.success_at is not None and self.x >= self.success_at
        return self.x, 123.0, terminated, False, {"is_success": exito}


class _EnvDePrueba(GymEnvironment):
    def __init__(self, sim: _SimContador, max_steps: int = 100) -> None:
        super().__init__(EnvSpec(action_dim=1, cameras=["cam"], tasks=["t0", "t1"]), max_steps)
        self.sim = sim
        self.tareas_pedidas: list[str] = []

    def _make(self, task: str) -> _SimContador:
        self.tareas_pedidas.append(task)
        return self.sim

    def _observation(self, raw: Any, task: str) -> Observation:
        return Observation(
            images={"cam": np.zeros((3, 2, 2), dtype=np.float32)},
            proprio=np.array([raw], dtype=np.float32),
            instruction=task,
        )


def _chunk(*valores: float) -> ActionChunk:
    return [Action(values=np.array([v], dtype=np.float32)) for v in valores]


def test_gym01_corte_a_mitad_de_chunk() -> None:
    sim = _SimContador(success_at=2.0)
    env = _EnvDePrueba(sim)
    env.reset(None, 0)
    resultado = env.step(_chunk(1.0, 1.0, 1.0, 1.0))
    assert sim.recibidas == [1.0, 1.0]  # las dos últimas no llegan al simulador
    assert resultado.observation.proprio[0] == 2.0


def test_gym02_max_steps_en_acciones() -> None:
    sim = _SimContador()
    env = _EnvDePrueba(sim, max_steps=5)
    env.reset(None, 0)
    assert not env.step(_chunk(0.0, 0.0, 0.0)).truncated
    resultado = env.step(_chunk(0.0, 0.0, 0.0))  # cruza el límite en la 2ª acción
    assert len(sim.recibidas) == 5
    assert resultado.truncated and not resultado.terminated


def test_gym03_exito_del_simulador_termina_el_episodio() -> None:
    env = _EnvDePrueba(_SimContador(success_at=1.0))  # el sim nunca marca terminated
    env.reset(None, 0)
    resultado = env.step(_chunk(1.0))
    assert resultado.terminated and resultado.info["success"] is True
    assert not resultado.truncated


def test_gym03_terminado_sin_exito() -> None:
    env = _EnvDePrueba(_SimContador(terminate_at=1.0))
    env.reset(None, 0)
    resultado = env.step(_chunk(1.0))
    assert resultado.terminated and resultado.info["success"] is False


def test_gym04_recompensa_del_simulador_ignorada() -> None:
    env = _EnvDePrueba(_SimContador())
    env.reset(None, 0)
    assert env.step(_chunk(0.1)).reward is None


def test_gym05_step_tras_final_o_sin_reset() -> None:
    env = _EnvDePrueba(_SimContador(success_at=1.0))
    with pytest.raises(RuntimeError, match="reset"):
        env.step(_chunk(0.0))
    env.reset(None, 0)
    env.step(_chunk(1.0))
    with pytest.raises(RuntimeError, match="reset"):
        env.step(_chunk(0.0))


def test_gym06_accion_mal_dimensionada_no_llega_al_simulador() -> None:
    sim = _SimContador()
    env = _EnvDePrueba(sim)
    env.reset(None, 0)
    malo = [Action(values=np.zeros(1, dtype=np.float32)), Action(values=np.zeros(2))]
    with pytest.raises(ValueError, match="forma"):
        env.step(malo)
    assert sim.recibidas == []


def test_gym07_tareas() -> None:
    env = _EnvDePrueba(_SimContador())
    assert env.reset(None, 0).instruction == "t0"
    assert env.reset("t1", 0).instruction == "t1"
    assert env.tareas_pedidas == ["t0", "t1"]
    with pytest.raises(ValueError, match="tarea desconocida"):
        env.reset("t9", 0)


def test_gym08_to_canonical_image() -> None:
    hwc = np.zeros((4, 5, 3), dtype=np.uint8)
    hwc[0, 0] = [255, 0, 128]
    chw = to_canonical_image(hwc)
    assert chw.shape == (3, 4, 5) and chw.dtype == np.float32
    assert chw[:, 0, 0].tolist() == pytest.approx([1.0, 0.0, 128 / 255])
    ya_canonica = np.full((3, 4, 5), 0.5, dtype=np.float32)
    assert np.array_equal(to_canonical_image(ya_canonica), ya_canonica)
    with pytest.raises(ValueError, match="RGB"):
        to_canonical_image(np.zeros((4, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match="RGB"):
        to_canonical_image(np.zeros((4, 5, 4), dtype=np.uint8))


def test_gym09_reset_pasa_la_semilla_y_reinicia_el_contador() -> None:
    sim = _SimContador()
    env = _EnvDePrueba(sim, max_steps=2)
    env.reset(None, 11)
    assert env.step(_chunk(0.0, 0.0)).truncated
    env.reset(None, 12)
    assert not env.step(_chunk(0.0)).truncated  # el contador volvió a cero
    assert sim.semillas == [11, 12]
