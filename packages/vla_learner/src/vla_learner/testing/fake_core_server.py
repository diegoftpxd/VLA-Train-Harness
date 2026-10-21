"""``FakeCoreServer``: servidor ZeroMQ mínimo que envuelve a ``FakeCoreClient``.

Sirve para probar el ``CoreClient`` real sin que exista Core (ver CLAUDE.md).
Usa un socket ``ROUTER`` en un hilo propio y despacha cada llamada al
``FakeCoreClient`` en memoria, con el mismo formato de wire que usará Core.

Importa ``zmq`` por ser el lado servidor del protocolo; es una herramienta de
prueba, no parte de la lógica del Learner (ARCH-01 solo exige que, dentro del
Learner, únicamente ``CoreClient`` lo importe).
"""

from __future__ import annotations

import threading
from typing import Any

import zmq

from vla_common import wire
from vla_common.serialization import decode, encode
from vla_learner.testing.fake_core import FakeCoreClient


class FakeCoreServer:
    def __init__(self, address: str = "tcp://127.0.0.1:0", client: FakeCoreClient | None = None):
        self._requested_address = address
        self.address = address
        self.client = client or FakeCoreClient()
        self._context = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> str:
        socket = self._context.socket(zmq.ROUTER)
        socket.bind(self._requested_address)
        self.address = socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self._socket = socket
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self.address

    def _serve(self) -> None:
        assert self._socket is not None
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            if dict(poller.poll(timeout=100)).get(self._socket) == zmq.POLLIN:
                frames = self._socket.recv_multipart()
                identity, mensaje = frames[0], frames[1:]
                respuesta = self._dispatch(mensaje)
                self._socket.send_multipart([identity, *respuesta])

    def _dispatch(self, mensaje: list[bytes]) -> list[bytes]:
        try:
            request = decode(mensaje)
            response = self._handle(request)
            response["ok"] = True
        except Exception as error:  # noqa: BLE001 - se reporta al cliente por el wire
            response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        return encode(response)

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        call = request["call"]
        if call == "reset":
            episodios = self.client.reset(request["env_ids"], request["task"], request["seed"])
            return {"episodes": [[ep, wire.observation_to_wire(o)] for ep, o in episodios]}
        if call == "step":
            acciones = [wire.action_chunk_from_wire(c) for c in request["actions"]]
            resultados = self.client.step(request["env_ids"], acciones, request["with_reward"])
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
            salida = self.client.simulate(start, secuencias, request["with_reward"])
            return {"sequences": [[wire.step_result_to_wire(r) for r in seq] for seq in salida]}
        if call == "sample_supervised":
            pares = self.client.sample_supervised(request["n"], request["filters"])
            return {
                "pairs": [
                    [wire.observation_to_wire(o), wire.action_chunk_to_wire(c)] for o, c in pares
                ]
            }
        if call == "get_episode":
            transiciones = self.client.get_episode(request["episode_id"])
            return {"transitions": [wire.transition_to_wire(t) for t in transiciones]}
        if call == "close":
            return {}
        raise ValueError(f"Llamada desconocida: {call!r}")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None

    def __enter__(self) -> FakeCoreServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
