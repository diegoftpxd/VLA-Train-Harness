"""Valida el paso 1 del plan π0 E2E (``docs/dataset/Pi0 E2E Plan.md``) en ``fast``.

Ejercita los helpers **puros** de ``scripts/prepare_lerobot_local.py`` (troceo a 50,
selección de episodios, agrupado por episodio y escritura en formato ``local``) sobre
arrays sintéticos, sin descargar ningún corpus ni importar lerobot, y comprueba que
``LocalDataset`` lee lo escrito como chunks de largo fijo. Además chequea los configs
committeados de pusht y LIBERO, y que las cámaras/chunk de Core y Learner coinciden
(el desajuste que el plan advierte como riesgo).
"""

from __future__ import annotations

import importlib.util
import json
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
WRIST = "observation.images.left_wrist_0_rgb"
PUSHT_CAMS = {"observation.image": CAMERA}
LIBERO_CAMS = {"observation.images.image": CAMERA, "observation.images.image2": WRIST}


def _cargar_script():
    """Importa ``scripts/prepare_lerobot_local.py`` como módulo (lerobot se importa
    solo dentro de ``main``, así que esto no requiere el extra ``pi0``)."""
    ruta = REPO / "scripts" / "prepare_lerobot_local.py"
    spec = importlib.util.spec_from_file_location("prepare_lerobot_local", ruta)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PREP = _cargar_script()


def _episodio(t: int, cams: dict[str, str] = PUSHT_CAMS, task: str = "push the T block") -> dict:
    """Episodio sintético con imágenes ``float`` en ``[0, 1]`` (lo que decodifica lerobot)."""
    rng = np.random.default_rng(t)
    ep: dict = {src: rng.random((t, 3, 8, 8), dtype=np.float32) for src in cams}
    ep["observation.state"] = rng.random((t, 2), dtype=np.float32)
    ep["action"] = rng.random((t, 2), dtype=np.float32)
    ep["task"] = task
    return ep


def _preparar(episodios: list[dict], out: Path, cams: dict[str, str] = PUSHT_CAMS, **kw) -> int:
    return PREP.preparar_desde_episodios(
        episodios,
        out,
        chunk=50,
        cameras=cams,
        state_key="observation.state",
        action_key="action",
        **kw,
    )


# -- Helpers puros -----------------------------------------------------------


def test_ventanas_trocea_a_largo_fijo_y_descarta_cola() -> None:
    # 123 pasos, ventana 50 -> dos ventanas completas; [100, 123) se descarta.
    assert PREP.ventanas(123, 50) == [(0, 50), (50, 100)]
    assert PREP.ventanas(49, 50) == []  # episodio más corto que un chunk: nada
    assert PREP.ventanas(100, 50) == [(0, 50), (50, 100)]


def test_elegir_episodios_reparte_por_todo_el_corpus() -> None:
    assert PREP.elegir_episodios(5, None) == [0, 1, 2, 3, 4]
    assert PREP.elegir_episodios(5, 10) == [0, 1, 2, 3, 4]  # tope mayor que el total
    elegidos = PREP.elegir_episodios(1693, 200)
    assert len(elegidos) == 200 and elegidos[0] == 0 and elegidos[-1] == 1692
    assert elegidos == sorted(set(elegidos))  # sin repetidos y en orden
    with pytest.raises(ValueError):
        PREP.elegir_episodios(10, 0)


def test_parse_camaras() -> None:
    assert PREP.parse_camaras(None) == {}
    assert PREP.parse_camaras([f"observation.images.image={CAMERA}"]) == {
        "observation.images.image": CAMERA
    }
    with pytest.raises(ValueError, match="clave_corpus=clave_canonica"):
        PREP.parse_camaras(["sin_igual"])


def test_a_uint8_redondea_float_en_unidad() -> None:
    x = np.array([0.0, 0.5, 1.0, 1.2, -0.1], dtype=np.float32)
    np.testing.assert_array_equal(PREP.a_uint8(x), np.array([0, 128, 255, 255, 0], np.uint8))
    ya = np.array([7, 200], dtype=np.uint8)
    assert PREP.a_uint8(ya) is ya


def test_agrupar_por_episodio_emite_uno_a_la_vez() -> None:
    frames = [
        {"episode_index": ep, "task": f"t{ep}", "x": np.full(2, i, np.float32)}
        for i, ep in enumerate([3, 3, 3, 7, 7])
    ]
    episodios = list(PREP.agrupar_por_episodio(iter(frames), keys=["x"]))
    assert [e["episode_index"] for e in episodios] == [3, 7]
    assert [e["task"] for e in episodios] == ["t3", "t7"]
    assert episodios[0]["x"].shape == (3, 2) and episodios[1]["x"].shape == (2, 2)
    assert list(PREP.agrupar_por_episodio(iter([]), keys=["x"])) == []


def test_preparar_escribe_formato_local_leible(tmp_path: Path) -> None:
    # Dos episodios (123 y 100 pasos) -> 2 + 2 = 4 episodios locales de 50 pasos.
    assert _preparar([_episodio(123), _episodio(100)], tmp_path) == 4
    assert sorted(p.name for p in tmp_path.iterdir()) == ["00000", "00001", "00002", "00003"]

    ds = LocalDataset(tmp_path, cameras={CAMERA: CAMERA}, demo_chunk_size=50)
    pares = ds.sample(3)
    assert len(pares) == 3
    obs, chunk = pares[0]
    assert len(chunk) == 50  # chunk de largo fijo (restricción de torch.stack de π0)
    assert CAMERA in obs.images and obs.images[CAMERA].shape == (3, 8, 8)
    assert obs.proprio.shape == (2,)
    assert chunk[0].values.shape == (2,)  # acción en unidades reales, sin padear
    assert obs.instruction == "push the T block"


def test_imagenes_en_uint8_en_disco_y_unidad_al_leer(tmp_path: Path) -> None:
    ep = _episodio(50)
    _preparar([ep], tmp_path)
    with np.load(tmp_path / "00000" / "data.npz") as data:
        assert data[CAMERA].dtype == np.uint8  # 4× menos disco que float32
    obs, _ = LocalDataset(tmp_path, demo_chunk_size=50).sample(1)[0]
    imagen = obs.images[CAMERA]
    assert imagen.dtype == np.float32 and 0.0 <= imagen.min() and imagen.max() <= 1.0
    # Ida y vuelta con el error del redondeo a 8 bits (≤ 0.5/255).
    np.testing.assert_allclose(imagen, ep["observation.image"][0], atol=0.5 / 255 + 1e-6)


def test_libero_dos_camaras_e_instruccion_por_episodio(tmp_path: Path) -> None:
    episodios = [
        _episodio(60, LIBERO_CAMS, task="put the bowl on the plate"),
        _episodio(110, LIBERO_CAMS, task="open the top drawer"),
    ]
    for i, ep in enumerate(episodios):
        ep["episode_index"] = 10 + i
    assert _preparar(episodios, tmp_path, cams=LIBERO_CAMS) == 3  # 1 + 2 ventanas
    meta = json.loads((tmp_path / "00002" / "meta.json").read_text(encoding="utf-8"))
    assert meta == {"instruction": "open the top drawer", "source_episode": 11}
    ds = LocalDataset(tmp_path, cameras={CAMERA: CAMERA, WRIST: WRIST}, demo_chunk_size=50)
    instrucciones = {obs.instruction for obs, _ in ds.sample(30)}
    assert instrucciones == {"put the bowl on the plate", "open the top drawer"}
    obs, _ = ds.sample(1)[0]
    assert set(obs.images) == {CAMERA, WRIST}


def test_instruccion_fija_reemplaza_la_del_corpus(tmp_path: Path) -> None:
    _preparar([_episodio(50, task="la del corpus")], tmp_path, instruction="fija")
    meta = json.loads((tmp_path / "00000" / "meta.json").read_text(encoding="utf-8"))
    assert meta["instruction"] == "fija"


def test_sin_instruccion_es_error(tmp_path: Path) -> None:
    ep = _episodio(50)
    del ep["task"]
    with pytest.raises(ValueError, match="instrucción"):
        _preparar([ep], tmp_path)


def test_local_dataset_no_retiene_el_episodio_entero(tmp_path: Path) -> None:
    # Cada observación debe ser un frame propio, no una vista del arreglo del
    # episodio (si no, el Core retiene todos los frames: cientos de GB con LIBERO).
    _preparar([_episodio(100)], tmp_path)
    for obs, _ in LocalDataset(tmp_path, demo_chunk_size=50).sample(4):
        assert obs.images[CAMERA].base is None
        assert obs.proprio.base is None


# -- Configs committeados ----------------------------------------------------


def _yaml(*ruta: str) -> dict:
    return yaml.safe_load((REPO.joinpath(*ruta)).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("core", "learner", "path", "cams"),
    [
        ("pi0_pusht.yaml", "pi0_pusht_bc.yaml", "data/pi0_pusht", [CAMERA]),
        ("pi0_libero.yaml", "pi0_libero_bc.yaml", "data/pi0_libero", [CAMERA, WRIST]),
    ],
)
def test_configs_core_y_learner_coinciden(core: str, learner: str, path: str, cams) -> None:
    # El riesgo nº1 del plan: que las cámaras del dataset y las del Learner no sean
    # las mismas claves exactas, o que el chunk no case con lo que π0 apila.
    c = _yaml("configs", "core", core)
    ds = c["dataset"]
    assert ds["type"] == "local" and ds["path"] == path
    assert ds["demo_chunk_size"] == 50  # = n_action_steps de π0
    lrn = _yaml("configs", learner)
    assert list(ds["cameras"]) == cams
    assert lrn["policy"]["cameras"] == cams
    assert lrn["available_cameras"] == cams
    assert lrn["policy"]["type"] == "pi0" and lrn["algorithm"]["type"] == "bc"
    assert lrn["policy"]["normalizer"]["stats"] == "fit"  # pi0_base no trae estadísticas


@pytest.mark.parametrize(
    ("core", "cams", "corpus_cams"),
    [("pi0_pusht.yaml", [CAMERA], PUSHT_CAMS), ("pi0_libero.yaml", [CAMERA, WRIST], LIBERO_CAMS)],
)
def test_core_sirve_demos_pi0_por_la_red(tmp_path: Path, core: str, cams, corpus_cams) -> None:
    # El bloque `dataset` del config, servido por un Core real y consumido por el
    # CoreClient del Learner, entrega chunks de 50 con las cámaras correctas.
    _preparar([_episodio(100, corpus_cams)], tmp_path, cams=corpus_cams)
    cfg = _yaml("configs", "core", core)
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
            assert sorted(obs.images) == sorted(cams)
            assert len(chunk) == 50
            assert chunk[0].values.shape == (2,)
        finally:
            client.close()
    finally:
        server.stop()
