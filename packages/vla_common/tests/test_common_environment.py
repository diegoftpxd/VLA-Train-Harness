"""Reglas del contrato del Environment en ``vla_common`` (COM-*).

Ver ``docs/environment/Test Environment.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_common import wire
from vla_common.environment import (
    EnvSpec,
    check_action_chunk,
    handle_environment,
    resolve_task,
    spec_from_wire,
    spec_to_wire,
)
from vla_common.types import Action

pytestmark = pytest.mark.fast

SPEC = EnvSpec(action_dim=3, cameras=["front"], tasks=["a", "b"])


def _accion(n: int) -> Action:
    return Action(values=np.zeros(n, dtype=np.float32))


def test_com01_check_action_chunk() -> None:
    check_action_chunk([_accion(3), _accion(3)], 3)
    with pytest.raises(ValueError, match="forma"):
        check_action_chunk([_accion(3), _accion(2)], 3)
    with pytest.raises(ValueError, match="forma"):
        check_action_chunk([Action(values=np.zeros((1, 3), dtype=np.float32))], 3)
    with pytest.raises(ValueError, match="vacío"):
        check_action_chunk([], 3)


def test_com01_resolve_task() -> None:
    assert resolve_task(None, SPEC) == "a"
    assert resolve_task("b", SPEC) == "b"
    with pytest.raises(ValueError, match="tarea desconocida"):
        resolve_task("c", SPEC)


class _EnvMinimo:
    """Solo lo que ``handle_environment`` toca para ``spec``."""

    spec = SPEC


def test_com02_spec_por_el_wire_y_llamada_desconocida() -> None:
    respuesta = handle_environment({"call": "spec"}, _EnvMinimo())  # type: ignore[arg-type]
    assert spec_from_wire(respuesta["spec"]) == SPEC
    assert spec_from_wire(spec_to_wire(SPEC)) == SPEC
    with pytest.raises(ValueError, match="desconocida"):
        handle_environment({"call": "volar"}, _EnvMinimo())  # type: ignore[arg-type]


def test_com02_step_usa_las_conversiones_de_wire() -> None:
    # El chunk que llega por el wire se reconstruye igual (bit a bit).
    chunk = [Action(values=np.array([0.1, -0.2, 0.3], dtype=np.float32))]
    vuelta = wire.action_chunk_from_wire(wire.action_chunk_to_wire(chunk))
    assert np.array_equal(vuelta[0].values, chunk[0].values)
