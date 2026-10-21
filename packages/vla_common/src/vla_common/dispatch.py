"""Despacho del protocolo de Core: traduce wire↔tipos y llama al backend.

Definido en ``docs/core/Core Design.md``. Es una función pura, sin ``zmq``:
recibe el request ya decodificado y un backend que cumple ``CoreInterface``, lo
traduce a los tipos del dominio con ``vla_common.wire``, llama al método que
corresponde y vuelve a traducir el resultado a wire. Envuelve la respuesta en
``{"ok": True, ...}`` o, ante cualquier excepción, en ``{"ok": False, "error": ...}``.

La usan los dos servidores (``CoreServer`` real y ``FakeCoreServer`` de prueba),
para que el mapeo llamada→método y el manejo de errores vivan en un solo lugar.
"""

from __future__ import annotations

from typing import Any

from vla_common import wire
from vla_common.protocol import CoreInterface


def dispatch_request(request: dict[str, Any], backend: CoreInterface) -> dict[str, Any]:
    """Despacha un request decodificado al backend y devuelve la respuesta en wire."""
    try:
        response = _handle(request, backend)
        response["ok"] = True
    except Exception as error:  # noqa: BLE001 - se reporta al cliente por el wire
        response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return response


def _handle(request: dict[str, Any], backend: CoreInterface) -> dict[str, Any]:
    call = request["call"]
    if call == "reset":
        episodios = backend.reset(request["env_ids"], request["task"], request["seed"])
        return {"episodes": [[ep, wire.observation_to_wire(o)] for ep, o in episodios]}
    if call == "step":
        acciones = [wire.action_chunk_from_wire(c) for c in request["actions"]]
        resultados = backend.step(request["env_ids"], acciones, request["with_reward"])
        return {"results": [wire.step_result_to_wire(r) for r in resultados]}
    if call == "simulate":
        if "start_obs" in request:
            start: Any = wire.observation_from_wire(request["start_obs"])
        else:
            ref = request["start_ref"]
            start = (ref[0], int(ref[1]))
        secuencias = [
            [wire.action_chunk_from_wire(c) for c in seq] for seq in request["action_sequences"]
        ]
        salida = backend.simulate(start, secuencias, request["with_reward"])
        return {"sequences": [[wire.step_result_to_wire(r) for r in seq] for seq in salida]}
    if call == "sample_supervised":
        pares = backend.sample_supervised(request["n"], request["filters"])
        return {
            "pairs": [[wire.observation_to_wire(o), wire.action_chunk_to_wire(c)] for o, c in pares]
        }
    if call == "get_episode":
        transiciones = backend.get_episode(request["episode_id"])
        return {"transitions": [wire.transition_to_wire(t) for t in transiciones]}
    if call == "close":
        # El ``close`` del cliente no tumba el backend del servidor: el ciclo de
        # vida del servidor es aparte (``CoreServer.stop``). Solo se confirma.
        return {}
    raise ValueError(f"Llamada desconocida: {call!r}")
