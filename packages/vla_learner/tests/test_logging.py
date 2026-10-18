"""Suite LOG sobre ``MetricsLogger``."""

import pytest

from vla_learner.infra.logging import StdoutLogger

pytestmark = pytest.mark.fast


def test_log01_solo_el_principal_publica(capsys):
    StdoutLogger(is_main=False).log({"loss": 1.0}, 0)
    assert capsys.readouterr().out == ""  # un proceso no principal no publica

    StdoutLogger(is_main=True).log({"loss": 1.0}, 3)
    salida = capsys.readouterr().out
    assert "loss" in salida
    assert "step 3" in salida
