"""Valida el paso 1 del plan π0 E2E (``docs/dataset/Pi0 E2E Plan.md``) en ``fast``.

Ejercita los helpers **puros** de ``scripts/prepare_pi0_pusht.py`` (troceo a 50 y
escritura en formato ``local``) sobre arrays sintéticos, sin descargar pusht ni
importar lerobot, y comprueba que ``LocalDataset`` lee lo escrito como chunks de
largo fijo. Además chequea los configs committeados y que la cámara/chunk de Core
y Learner coinciden (el desajuste que el plan advierte como riesgo).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from vla_core.builder import build_core
from vla_core.dataset.local import LocalDataset
from vla_learner.data.core_client import CoreClient

pytestmark = pytest.mark.fast

REPO = Path(__file__).resolve().parents[3]
CAMERA = "observation.images.base_0_rgb"


def _cargar_script():
    """Importa ``scripts/prepare_pi0_pusht.py`` como módulo (lerobot se importa
    solo dentro de ``main``, así que esto no requiere el extra ``pi0``)."""
    ruta = REPO / "scripts" / "prepare_pi0_pusht.py"
    spec = importlib.util.spec_from_file_location("prepare_pi0_pusht", ruta)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PREP = _cargar_script()


def _episodio(t: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(t)
    return {
        "observation.image": rng.random((t, 3, 8, 8), dtype=np.float32),
        "observation.state": rng.random((t, 2), dtype=np.float32),
        "action": rng.random((t, 2), dtype=np.float32),
    }


# -- Helpers puros -----------------------------------------------------------


def test_ventanas_trocea_a_largo_fijo_y_descarta_cola() -> None:
    # 123 pasos, ventana 50 -> dos ventanas completas; [100, 123) se descarta.
    assert PREP.ventanas(123, 50) == [(0, 50), (50, 100)]
    assert PREP.ventanas(49, 50) == []  # episodio más corto que un chunk: nada
    assert PREP.ventanas(100, 50) == [(0, 50), (50, 100)]


def test_preparar_escribe_formato_local_leible(tmp_path: Path) -> None:
    # Dos episodios (123 y 100 pasos) -> 2 + 2 = 4 episodios locales de 50 pasos.
    escritos = PREP.preparar_desde_episodios(
        [_episodio(123), _episodio(100)],
        tmp_path,
        instruction="push the T block",
        chunk=50,
        image_key="observation.image",
        state_key="observation.state",
        action_key="action",
    )
    assert escritos == 4
    assert sorted(p.name for p in tmp_path.iterdir()) == ["0000", "0001", "0002", "0003"]

    ds = LocalDataset(tmp_path, cameras={CAMERA: CAMERA}, demo_chunk_size=50)
    pares = ds.sample(3)
    assert len(pares) == 3
    obs, chunk = pares[0]
    assert len(chunk) == 50  # chunk de largo fijo (restricción de torch.stack de π0)
    assert CAMERA in obs.images and obs.images[CAMERA].shape == (3, 8, 8)
    assert obs.proprio.shape == (2,)
    assert chunk[0].values.shape == (2,)  # acción en unidades reales, sin padear
    assert obs.instruction == "push the T block"


# -- Configs committeados ----------------------------------------------------


def test_config_core_pi0_pusht_invariantes() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "core" / "pi0_pusht.yaml").read_text(encoding="utf-8"))
    ds = cfg["dataset"]
    assert ds["type"] == "local"
    assert ds["path"] == "data/pi0_pusht"
    assert ds["demo_chunk_size"] == 50  # = n_action_steps de π0
    assert CAMERA in ds["cameras"]  # la cámara canónica que exige el procesador


def test_configs_core_y_learner_coinciden_en_camara() -> None:
    # El riesgo nº1 del plan: que la cámara del dataset y la del Learner no sean la
    # misma clave exacta, o que el chunk no case con lo que π0 apila.
    core = yaml.safe_load(
        (REPO / "configs" / "core" / "pi0_pusht.yaml").read_text(encoding="utf-8")
    )
    learner = yaml.safe_load((REPO / "configs" / "pi0_pusht_bc.yaml").read_text(encoding="utf-8"))
    assert set(core["dataset"]["cameras"]) == set(learner["policy"]["cameras"])
    assert learner["policy"]["cameras"] == [CAMERA]
    assert learner["available_cameras"] == [CAMERA]
    assert learner["policy"]["type"] == "pi0"
    assert learner["algorithm"]["type"] == "bc"


def test_core_sirve_demos_pi0_por_la_red(tmp_path: Path) -> None:
    # El bloque `dataset` de pi0_pusht.yaml, servido por un Core real y consumido
    # por el CoreClient del Learner, entrega chunks de 50 con la cámara correcta.
    PREP.preparar_desde_episodios(
        [_episodio(100)],
        tmp_path,
        instruction="push the T block",
        chunk=50,
        image_key="observation.image",
        state_key="observation.state",
        action_key="action",
    )
    cfg = yaml.safe_load((REPO / "configs" / "core" / "pi0_pusht.yaml").read_text(encoding="utf-8"))
    cfg["server"] = {"address": "tcp://127.0.0.1:0"}  # puerto efímero
    cfg["dataset"]["path"] = str(tmp_path)  # fixture sintético en vez del corpus real
    server = build_core(cfg)
    server.start()
    try:
        client = CoreClient(server.address, timeout_s=5.0)
        try:
            pares = client.sample_supervised(2)
            assert len(pares) == 2
            obs, chunk = pares[0]
            assert CAMERA in obs.images
            assert len(chunk) == 50
            assert chunk[0].values.shape == (2,)
        finally:
            client.close()
    finally:
        server.stop()
