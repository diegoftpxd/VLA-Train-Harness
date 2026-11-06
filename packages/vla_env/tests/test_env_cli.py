"""Suites CLI y de la factory de ``vla-env`` (``docs/environment/Test Environment.md``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vla_env.cli import build_server, load_config, main, save_observations, smoke
from vla_env.factory import build_environment
from vla_env.reach import ReachEnvironment

pytestmark = pytest.mark.fast

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "env" / "reach.yaml"


def test_fact01_reach_desde_config() -> None:
    env = build_environment(load_config(CONFIG)["environment"])
    assert isinstance(env, ReachEnvironment)
    assert env.max_steps == 20


def test_fact02_tipo_desconocido() -> None:
    with pytest.raises(ValueError, match="desconocido"):
        build_environment({"type": "marte"})


def test_cli01_smoke_reporta_y_termina_bien(capsys: pytest.CaptureFixture[str]) -> None:
    reporte = smoke(ReachEnvironment(), 25, [0.05, 0.05])
    assert reporte.steps == 25 and reporte.episodes >= 2  # 20 acciones por episodio
    assert "(3, 8, 8) float32" in reporte.cameras["front"]
    assert reporte.proprio.startswith("(4,) float32")
    assert main(["--config", str(CONFIG), "--smoke", "5"]) == 0
    salida = capsys.readouterr().out
    assert "spec:" in salida and "cámara front" in salida


def test_cli02_save_escribe_npz(tmp_path: Path) -> None:
    assert main(["--config", str(CONFIG), "--smoke", "3", "--save", str(tmp_path)]) == 0
    datos = np.load(tmp_path / "smoke.npz")
    assert datos["front"].shape == (4, 3, 8, 8)  # reset + 3 pasos
    assert datos["proprio"].shape == (4, 4)
    assert save_observations(smoke(ReachEnvironment(), 1), tmp_path / "otra").exists()


def test_cli03_address_pisa_la_config() -> None:
    config = load_config(CONFIG)
    assert build_server(config).address == config["server"]["address"]
    assert build_server(config, "tcp://127.0.0.1:0").address == "tcp://127.0.0.1:0"


def test_cli04_tipo_desconocido_da_mensaje_claro(tmp_path: Path) -> None:
    yaml = tmp_path / "malo.yaml"
    yaml.write_text("environment: { type: marte }\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Disponibles"):
        main(["--config", str(yaml), "--smoke", "1"])
