"""Suite REG sobre ``Registry``.

Se registran dos factories de juguete para ejercitar el registro sin depender
de piezas reales (que llegan en fases posteriores).
"""

import pytest

from vla_learner.registry import Registry
from vla_learner.testing.contracts.registry_contract import RegistryContract

pytestmark = pytest.mark.fast

_DEMO = "__phase1_demo__"


@Registry.register(_DEMO, "suma")
def _build_suma(a: int = 1, b: int = 2) -> int:
    return a + b


@Registry.register(_DEMO, "texto")
def _build_texto(valor: str = "hola") -> str:
    return valor


class TestRegistry(RegistryContract):
    @pytest.fixture
    def registered(self) -> list[tuple[str, str, dict]]:
        return [
            (_DEMO, "suma", {"a": 3, "b": 4}),
            (_DEMO, "texto", {"valor": "x"}),
        ]
