"""Suite EVAL: ``scripts/eval_success.py`` mide la tasa de éxito por Core.

Ver ``docs/environment/Test Environment.md`` ("Integración"). En ``fast``, la
lógica de conteo y reparto contra un ``Core`` en proceso (y una vez por la red);
en ``slow``, una política entrenada con BC supera a una sin entrenar contra
procesos ``vla-env`` reales.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from conftest import core_con_procesos

from vla_common.environment import Environment, EnvSpec
from vla_common.types import ActionChunk, Observation, StepResult
from vla_core.environments import EnvironmentRegistry
from vla_core.server import CoreServer
from vla_core.testing.fakes import FAKE_SPEC, TASK, FakeEnvironment, make_fake_core
from vla_learner.data.core_client import CoreClient
from vla_learner.learner import Learner

REPO = Path(__file__).resolve().parents[3]


def _script():
    ruta = REPO / "scripts" / "eval_success.py"
    spec = importlib.util.spec_from_file_location("eval_success", ruta)
    assert spec and spec.loader
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo  # dataclasses lo busca ahí al definir las clases
    spec.loader.exec_module(modulo)
    return modulo


EVAL = _script()


def _learner_config(core: dict, tmp_path: Path) -> dict:
    return {
        "seed": 0,
        "core": core,
        "policy": {
            "type": "composed",
            "action_dim": 2,
            "chunk_size": 1,
            "feature_dim": 32,
            "backbone": {"type": "tiny", "seq_len": 1},
            "action_head": {"type": "gaussian", "init_zero": True, "init_log_std": -1.0},
            "processor": {"type": "simple", "cameras": ["front"]},
            "normalizer": {"mode": "mean_std"},
        },
        "algorithm": {"type": "bc"},
        "data": {"type": "supervised", "batch_size": 64},
        "optimizer": {
            "name": "adamw",
            "lr": 1e-3,
            "grad_clip": 1.0,
            "scheduler": None,
            "precision": "fp32",
        },
        "checkpoint": {"directory": str(tmp_path / "ckpt"), "every": 10_000},
    }


def _politica_sin_entrenar(tmp_path: Path):
    learner = Learner.from_config(_learner_config({"type": "fake"}, tmp_path))
    return learner.policy


class _Contador(FakeEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.resets: list[int | None] = []

    def reset(self, task: str | None, seed: int | None) -> Observation:
        self.resets.append(seed)
        return super().reset(task, seed)


class _TerminaEnUnPaso(Environment):
    """Termina en el primer paso con el ``success`` dado (un fallo o un éxito)."""

    def __init__(self, success: bool) -> None:
        self._success = success

    def reset(self, task: str | None, seed: int | None) -> Observation:
        return FakeEnvironment().reset(task, seed)

    def step(self, actions: ActionChunk) -> StepResult:
        obs = FakeEnvironment().reset(None, 0)
        return StepResult(obs, None, True, False, {"success": self._success})

    def close(self) -> None:
        pass

    def emergency_stop(self) -> None:
        pass

    @property
    def spec(self) -> EnvSpec:
        return FAKE_SPEC


def _core_con(**envs: Environment):
    registro = EnvironmentRegistry()
    for env_id, env in envs.items():
        registro.register(env_id, env)
    return make_fake_core(environments=registro)


@pytest.mark.fast
def test_eval01_reparte_episodios_y_semillas(tmp_path: Path) -> None:
    a, b = _Contador(), _Contador()
    reporte = EVAL.evaluate(
        _politica_sin_entrenar(tmp_path), _core_con(a=a, b=b), [TASK], 5, ["a", "b"], seed=10
    )
    resultado = reporte.tasks[TASK]
    assert resultado.episodes == 5 and 0.0 <= resultado.rate <= 1.0
    semillas = [s for s in a.resets + b.resets if s is not None]
    assert sorted(semillas) == [10, 11, 12, 13, 14]  # una semilla por episodio
    assert (len(a.resets), len(b.resets)) == (3, 2)


@pytest.mark.fast
def test_eval01_cuenta_por_success_no_por_terminated(tmp_path: Path) -> None:
    politica = _politica_sin_entrenar(tmp_path)
    fallo = EVAL.evaluate(politica, _core_con(f=_TerminaEnUnPaso(False)), [TASK], 3, ["f"])
    assert fallo.tasks[TASK].episodes == 3 and fallo.rate == 0.0
    exito = EVAL.evaluate(politica, _core_con(e=_TerminaEnUnPaso(True)), [TASK], 3, ["e"])
    assert exito.rate == 1.0
    assert exito.to_dict()["tasks"][TASK] == {"episodes": 3, "successes": 3, "rate": 1.0}


@pytest.mark.fast
def test_eval01_por_la_red(tmp_path: Path) -> None:
    server = CoreServer(make_fake_core(), "tcp://127.0.0.1:0")
    address = server.start()
    cliente = CoreClient(address, timeout_s=10.0)
    try:
        reporte = EVAL.evaluate(_politica_sin_entrenar(tmp_path), cliente, [TASK], 4, ["x", "y"])
        assert reporte.tasks[TASK].episodes == 4
        assert "TOTAL" in reporte.text()
    finally:
        cliente.close()
        server.stop()


@pytest.mark.fast
def test_eval02_tarea_desconocida_da_error_claro(tmp_path: Path) -> None:
    core = CoreServer(_core_con(a=FakeEnvironment()), "tcp://127.0.0.1:0")
    address = core.start()
    cliente = CoreClient(address, timeout_s=10.0)
    try:
        with pytest.raises(ValueError, match="no se pudo iniciar 'abrir la puerta' en 'a'"):
            EVAL.evaluate(_politica_sin_entrenar(tmp_path), cliente, ["abrir la puerta"], 1, ["a"])
    finally:
        cliente.close()
        core.stop()


@pytest.mark.slow
def test_eval03_politica_entrenada_supera_a_la_inicial(tmp_path: Path) -> None:
    with core_con_procesos(2) as server:
        core = {"type": "zmq", "address": server.address, "timeout_s": 30.0}
        learner = Learner.from_config(_learner_config(core, tmp_path))
        learner.logger.is_main = False
        cliente = CoreClient(server.address, timeout_s=30.0)
        try:
            ids = ["reach-0", "reach-1"]
            antes = EVAL.evaluate(learner.policy, cliente, [TASK], 20, ids, seed=100).rate
            learner.train(250)  # BC con las demos del Dataset, servidas por Core
            despues = EVAL.evaluate(learner.policy, cliente, [TASK], 20, ids, seed=100).rate
        finally:
            cliente.close()
    assert despues > antes + 0.3 and despues > 0.6


@pytest.mark.fast
def test_eval05_config_de_evaluacion(tmp_path: Path) -> None:
    original = {
        "core": {"type": "zmq", "address": "tcp://localhost:5555"},
        "policy": {"normalizer": {"mode": "mean_std", "stats": "fit", "fit_samples": 256}},
        "checkpoint": {"directory": str(tmp_path / "ckpt")},
    }
    assert EVAL.latest_checkpoint(original) is None
    for paso in ("step_000010", "step_000200"):
        (tmp_path / "ckpt" / paso / "policy").mkdir(parents=True)
    ultimo = EVAL.latest_checkpoint(original)
    assert ultimo == tmp_path / "ckpt" / "step_000200"
    with pytest.raises(FileNotFoundError, match="estadísticas"):
        EVAL.eval_config(original, ultimo)
    (ultimo / "policy" / "action_stats.pt").write_bytes(b"")
    config = EVAL.eval_config(original, ultimo, "ipc:///tmp/core.ipc")
    assert config["core"]["address"] == "ipc:///tmp/core.ipc"
    assert config["policy"]["normalizer"] == {"mode": "mean_std", "stats": str(ultimo / "policy")}
    assert original["policy"]["normalizer"]["stats"] == "fit"  # la original no se toca
