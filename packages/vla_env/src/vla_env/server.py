"""``EnvironmentServer``: socket ``ROUTER`` que sirve un ``Environment``.

Definido en ``docs/environment/Environment Design.md`` (decisión E2b). El
servidor es parte del Environment: atiende el protocolo de
``vla_common.environment.handle_environment`` con el sobre ``ok``/``error`` que
espera el ``EnvironmentClient`` de Core. Mensajes con
``vla_common.serialization`` (msgpack + buffers crudos, sin pickle).

Atiende de a un pedido por vez, en un hilo propio. Por eso un robot real no puede
depender de la red para frenar: lleva un watchdog local (ver "Robot real").
"""

from __future__ import annotations

import threading
from typing import Any

import zmq

from vla_common.environment import Environment, handle_environment
from vla_common.serialization import decode, encode


def _responder(request: dict[str, Any], env: Environment) -> dict[str, Any]:
    try:
        response = handle_environment(request, env)
        response["ok"] = True
    except Exception as error:  # noqa: BLE001 - se reporta al cliente por el wire
        response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return response


class EnvironmentServer:
    def __init__(self, env: Environment, address: str = "tcp://127.0.0.1:0") -> None:
        self.env = env
        self._requested_address = address
        self.address = address
        self._context = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> str:
        """Bindea, empieza a atender en un hilo y devuelve el endpoint real."""
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
                respuesta = encode(_responder(decode(mensaje), self.env))
                self._socket.send_multipart([identity, *respuesta])

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None

    def __enter__(self) -> EnvironmentServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
