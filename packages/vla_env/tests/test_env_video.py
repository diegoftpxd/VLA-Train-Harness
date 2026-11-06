"""Suite VID: videos por episodio (``docs/environment/Test Environment.md``).

Se prueban con un escritor falso que guarda los cuadros en memoria (y crea el
archivo, para que el renombrado sea real): ``imageio`` es del extra ``libero`` y
``fast`` no lo necesita. El MP4 de verdad lo verifica
``slurm/pi0_libero_video.sbatch`` en el cluster.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vla_common.environment import Environment, EnvSpec
from vla_common.types import Action, ActionChunk, Observation, StepResult
from vla_env.cli import build_server, environment_config, load_config
from vla_env.factory import build_environment
from vla_env.reach import ReachEnvironment
from vla_env.video import EpisodeRecorder, build_recorder, frame_from_observation

pytestmark = pytest.mark.fast

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "env" / "reach.yaml"


class _Escritor:
    def __init__(self, path: Path, fps: int) -> None:
        self.path, self.fps = path, fps
        self.cuadros: list[np.ndarray] = []
        self.cerrado = False
        path.write_bytes(b"")

    def append_data(self, frame: np.ndarray) -> None:
        assert not self.cerrado
        self.cuadros.append(frame)

    def close(self) -> None:
        self.cerrado = True


def _grabador(directorio: Path, **opciones) -> tuple[EpisodeRecorder, list[_Escritor]]:
    escritores: list[_Escritor] = []

    def fabrica(path: Path, fps: int) -> _Escritor:
        escritores.append(_Escritor(path, fps))
        return escritores[-1]

    return EpisodeRecorder(directorio, writer_factory=fabrica, **opciones), escritores


def _obs(**imagenes: np.ndarray) -> Observation:
    return Observation(images=imagenes, proprio=np.zeros(1, dtype=np.float32), instruction="t")


def _chunk(*valores: tuple[float, float]) -> ActionChunk:
    return [Action(values=np.array(v, dtype=np.float32)) for v in valores]


def test_vid01_cuadro_con_las_camaras_lado_a_lado() -> None:
    a = np.zeros((3, 2, 3), dtype=np.float32)
    b = np.ones((3, 2, 1), dtype=np.float32)
    b[0] = 0.5
    cuadro = frame_from_observation(_obs(a=a, b=b))
    assert cuadro.shape == (2, 4, 3) and cuadro.dtype == np.uint8  # HWC
    assert cuadro[0, 0].tolist() == [0, 0, 0]
    assert cuadro[0, 3].tolist() == [128, 255, 255]
    assert frame_from_observation(_obs(a=a, b=b), ["b"]).shape == (2, 1, 3)
    with pytest.raises(KeyError, match="no trae"):
        frame_from_observation(_obs(a=a), ["c"])
    with pytest.raises(ValueError, match="alturas"):
        frame_from_observation(_obs(a=a, b=np.zeros((3, 5, 1), dtype=np.float32)))


def test_vid02_un_cuadro_por_accion_y_nombre_con_resultado(tmp_path: Path) -> None:
    env = ReachEnvironment(max_steps=5)
    env.recorder, escritores = _grabador(tmp_path, fps=7)
    env.reset(None, 3)
    env.step(_chunk((0, 0), (0, 0)))
    resultado = env.step(_chunk((0, 0), (0, 0), (0, 0), (0, 0)))  # se corta en la 5ª
    assert resultado.truncated
    (escritor,) = escritores
    assert escritor.fps == 7 and escritor.cerrado
    assert len(escritor.cuadros) == 1 + 5  # el reset + una por acción, no por chunk
    assert escritor.cuadros[0].shape == (8, 8, 3)
    assert [p.name for p in tmp_path.iterdir()] == ["0000_reach_the_goal_seed3_fail.mp4"]
    assert env.recorder.saved == [tmp_path / "0000_reach_the_goal_seed3_fail.mp4"]


def test_vid02_exito_en_el_nombre(tmp_path: Path) -> None:
    env = ReachEnvironment()
    env.recorder, _ = _grabador(tmp_path)
    obs = env.reset(None, 0)
    meta, pos = obs.proprio[2:], obs.proprio[:2]
    assert env.step([Action(values=(meta - pos).astype(np.float32))]).info["success"]
    assert env.recorder.saved[-1].name == "0000_reach_the_goal_seed0_success.mp4"


def test_vid03_episodio_abandonado_queda_incompleto(tmp_path: Path) -> None:
    env = ReachEnvironment()
    env.recorder, escritores = _grabador(tmp_path)
    env.reset(None, 1)
    env.step(_chunk((0, 0)))
    env.reset(None, 2)  # a mitad del episodio anterior
    env.close()
    assert all(e.cerrado for e in escritores)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "0000_reach_the_goal_seed1_incomplete.mp4",
        "0001_reach_the_goal_seed2_incomplete.mp4",
    ]


def test_vid04_grabar_no_cambia_lo_que_devuelve_el_ambiente(tmp_path: Path) -> None:
    con, sin = ReachEnvironment(), ReachEnvironment()
    con.recorder, _ = _grabador(tmp_path)
    assert np.array_equal(con.reset(None, 5).proprio, sin.reset(None, 5).proprio)
    for _ in range(3):
        a, b = con.step(_chunk((0.1, 0), (0, 0.1))), sin.step(_chunk((0.1, 0), (0, 0.1)))
        assert np.array_equal(a.observation.proprio, b.observation.proprio)
        assert np.array_equal(a.observation.images["front"], b.observation.images["front"])
        assert (a.terminated, a.truncated, a.info) == (b.terminated, b.truncated, b.info)


def test_vid05_record_desde_la_config_y_la_cli(tmp_path: Path) -> None:
    env = build_environment({"type": "reach", "record": {"dir": str(tmp_path), "fps": 5}})
    assert isinstance(env, ReachEnvironment) and env.recorder is not None
    assert env.recorder.directory == tmp_path and env.recorder.fps == 5
    assert build_environment({"type": "reach"}).recorder is None  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="dir"):
        build_recorder({"fps": 5})
    with pytest.raises(ValueError, match="desconocidas"):
        build_recorder({"dir": "x", "codec": "vp9"})
    # --record pisa record.dir y conserva el resto de las opciones.
    config = {"environment": {"type": "reach", "record": {"dir": "otra", "fps": 5}}}
    assert environment_config(config, tmp_path)["record"] == {"dir": str(tmp_path), "fps": 5}
    assert environment_config(config) == config["environment"]
    servidor = build_server(load_config(CONFIG), "tcp://127.0.0.1:0", tmp_path)
    assert servidor.env.recorder.directory == tmp_path  # type: ignore[attr-defined]


def test_vid05_record_solo_en_simuladores(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Robot(Environment):
        @property
        def spec(self) -> EnvSpec:
            return EnvSpec(action_dim=1, cameras=["c"], tasks=["t"])

        def reset(self, task: str | None, seed: int | None) -> Observation:
            raise NotImplementedError

        def step(self, actions: ActionChunk) -> StepResult:
            raise NotImplementedError

        def close(self) -> None:
            pass

        def emergency_stop(self) -> None:
            pass

    monkeypatch.setattr("vla_env.factory._build_adapter", lambda config: _Robot())
    with pytest.raises(ValueError, match="GymEnvironment"):
        build_environment({"type": "robot", "record": "x"})
