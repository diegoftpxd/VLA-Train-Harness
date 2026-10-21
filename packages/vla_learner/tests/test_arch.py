"""Suite ARCH: tests estáticos que protegen los principios del diseño.

No ejecutan nada del modelo; revisan el código fuente. ``testing/`` queda fuera
de ARCH-01 porque ``FakeCoreServer`` es el lado servidor del protocolo (usa zmq
legítimamente), no parte de la lógica del Learner.
"""

import pathlib
import re

import pytest

pytestmark = pytest.mark.fast

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "vla_learner"

# importa el módulo zmq o lo usa como `zmq.algo` (no el literal de config "zmq")
_USA_ZMQ = re.compile(r"\bimport\s+zmq\b|\bzmq\s*\.")


def test_arch01_solo_core_client_importa_zmq():
    infractores = []
    for archivo in _SRC.rglob("*.py"):
        if "testing" in archivo.parts or archivo.name == "core_client.py":
            continue
        if _USA_ZMQ.search(archivo.read_text(encoding="utf-8")):
            infractores.append(archivo.name)
    assert infractores == []


def test_arch02_los_algoritmos_no_tocan_el_optimizador():
    for archivo in (_SRC / "algorithms").glob("*.py"):
        texto = archivo.read_text(encoding="utf-8")
        assert ".backward(" not in texto, archivo.name
        assert ".step(" not in texto, archivo.name


def test_arch03_el_learner_no_conoce_modelos_concretos():
    texto = (_SRC / "learner.py").read_text(encoding="utf-8")
    for prohibido in (
        "policy.composed",
        "policy.heads",
        "policy.processors",
        "policy.normalizer",
        "ComposedPolicy",
        "GaussianActionHead",
        "TokenActionHead",
        "TinyBackbone",
    ):
        assert prohibido not in texto, prohibido


def test_arch04_los_algoritmos_solo_conocen_la_interfaz():
    for archivo in (_SRC / "algorithms").glob("*.py"):
        texto = archivo.read_text(encoding="utf-8")
        for prohibido in (
            "policy.composed",
            "policy.heads",
            "policy.processors",
            "policy.normalizer",
        ):
            assert prohibido not in texto, (archivo.name, prohibido)
