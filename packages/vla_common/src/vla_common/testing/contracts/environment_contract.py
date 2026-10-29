"""Suite ENV: contrato de ``Environment``.

Definida en ``docs/environment/Test Environment.md``. La cumplen el
``FakeEnvironment`` y el ``EnvironmentClient`` de Core y todo adaptador de
``vla_env``. La implementación entrega el fixture ``environment``; si los ceros
no son una acción neutra para ese ambiente, redefine también ``neutral_action``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_common.environment import Environment
from vla_common.types import Action, ActionChunk, StepResult

# Tope de pasos para llevar un episodio a un final con la acción neutra (ENV-04).
MAX_CONTRACT_STEPS = 1000


class EnvironmentContract:
    @pytest.fixture
    def environment(self) -> Environment:
        raise NotImplementedError

    @pytest.fixture
    def neutral_action(self, environment: Environment) -> np.ndarray:
        return np.zeros(environment.spec.action_dim, dtype=np.float32)

    @staticmethod
    def _chunk(values: np.ndarray) -> ActionChunk:
        return [Action(values=np.asarray(values, dtype=np.float32))]

    def _hasta_final(self, environment: Environment, neutral_action: np.ndarray) -> StepResult:
        environment.reset(task=None, seed=0)
        for _ in range(MAX_CONTRACT_STEPS):
            resultado = environment.step(self._chunk(neutral_action))
            if resultado.terminated or resultado.truncated:
                return resultado
        pytest.fail(f"el episodio no terminó en {MAX_CONTRACT_STEPS} pasos con la acción neutra")

    def test_env01_reset_devuelve_observacion(self, environment: Environment) -> None:
        obs = environment.reset(task=None, seed=1)
        assert obs.proprio.dtype == np.float32
        assert obs.proprio.ndim == 1
        for camara in environment.spec.cameras:
            assert camara in obs.images

    def test_env02_step_sin_recompensa(
        self, environment: Environment, neutral_action: np.ndarray
    ) -> None:
        environment.reset(task=None, seed=1)
        resultado = environment.step(self._chunk(neutral_action))
        assert resultado.reward is None  # la recompensa la agrega Core
        assert isinstance(resultado.terminated, bool)
        assert isinstance(resultado.truncated, bool)

    def test_env03_spec_valido(self, environment: Environment) -> None:
        spec = environment.spec
        assert spec.action_dim > 0
        assert spec.cameras and len(set(spec.cameras)) == len(spec.cameras)
        assert spec.tasks and len(set(spec.tasks)) == len(spec.tasks)

    def test_env04_close_y_parada_tras_un_final(
        self, environment: Environment, neutral_action: np.ndarray
    ) -> None:
        self._hasta_final(environment, neutral_action)
        environment.emergency_stop()
        environment.close()

    def test_env05_reset_determinista(self, environment: Environment) -> None:
        a = environment.reset(task=None, seed=7)
        b = environment.reset(task=None, seed=7)
        assert np.array_equal(a.proprio, b.proprio)

    def test_env06_tareas(self, environment: Environment) -> None:
        tarea = environment.spec.tasks[0]
        assert environment.reset(task=tarea, seed=0).instruction == tarea
        assert environment.reset(task=None, seed=0).instruction == tarea
        with pytest.raises(Exception, match="tarea desconocida"):
            environment.reset(task="__tarea_que_no_existe__", seed=0)

    def test_env07_accion_mal_dimensionada(self, environment: Environment) -> None:
        environment.reset(task=None, seed=0)
        mala = np.zeros(environment.spec.action_dim + 1, dtype=np.float32)
        with pytest.raises(Exception, match="forma"):
            environment.step(self._chunk(mala))

    def test_env08_info_con_exito_y_escalares(
        self, environment: Environment, neutral_action: np.ndarray
    ) -> None:
        environment.reset(task=None, seed=0)
        info = environment.step(self._chunk(neutral_action)).info
        assert isinstance(info.get("success"), bool)
        for clave, valor in info.items():
            assert isinstance(valor, (bool, int, float, str)), f"info[{clave!r}] no es escalar"

    def test_env09_imagenes_canonicas_y_deterministas(self, environment: Environment) -> None:
        a = environment.reset(task=None, seed=3)
        b = environment.reset(task=None, seed=3)
        for camara in environment.spec.cameras:
            imagen = a.images[camara]
            assert imagen.dtype == np.float32
            assert imagen.ndim == 3 and imagen.shape[0] == 3, f"{camara}: {imagen.shape} no es CHW"
            assert float(imagen.min()) >= 0.0 and float(imagen.max()) <= 1.0
            assert np.array_equal(imagen, b.images[camara])

    def test_env10_banderas_de_final_coherentes(
        self, environment: Environment, neutral_action: np.ndarray
    ) -> None:
        final = self._hasta_final(environment, neutral_action)
        assert not (final.terminated and final.truncated)
        if final.info["success"]:
            assert final.terminated

    def test_env11_step_tras_final_es_error(
        self, environment: Environment, neutral_action: np.ndarray
    ) -> None:
        self._hasta_final(environment, neutral_action)
        with pytest.raises(Exception, match="reset"):
            environment.step(self._chunk(neutral_action))
