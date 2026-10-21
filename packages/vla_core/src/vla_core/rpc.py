"""Primitivas de transporte ZeroMQ reutilizables por Core y sus componentes.

Definidas en ``docs/core/Core Design.md``:

- ``RouterServer``: loop de un socket ``ROUTER`` que delega cada request en un
  *handler*. Lo usan el ``CoreServer`` (hacia el Learner) y los servidores fake
  de los componentes (Environment, WorldModel, Reward Model).
- ``RpcChannel``: request/response confiable a un proceso downstream con un
  socket ``DEALER``, timeout y reintentos. Es el gemelo hacia abajo del
  ``CoreClient``: encapsula la mecánica de ZeroMQ para no repetirla en cada
  cliente de componente.
- ``make_handler``: envuelve un ``handle(request, impl)`` en el sobre
  ``ok``/``error`` que esperan el ``RouterServer`` y los clientes.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import zmq

from vla_common.serialization import decode, encode
from vla_core.errors import DownstreamTimeout

Handler = Callable[[dict[str, Any]], dict[str, Any]]


def make_handler(impl: Any, handle: Callable[[dict[str, Any], Any], dict[str, Any]]) -> Handler:
    """Envuelve ``handle(request, impl)`` agregando el sobre ``ok``/``error``."""

    def handler(request: dict[str, Any]) -> dict[str, Any]:
        try:
            response = handle(request, impl)
            response["ok"] = True
        except Exception as error:  # noqa: BLE001 - se reporta al cliente por el wire
            response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        return response

    return handler


class RouterServer:
    """Socket ``ROUTER`` en un hilo propio que responde con el ``handler`` dado.

    Su única responsabilidad es mover frames: recibe, decodifica, llama al
    handler (que ya devuelve la respuesta con su sobre ``ok``/``error``) y
    responde a la identidad del remitente.
    """

    def __init__(self, handler: Handler, address: str = "tcp://127.0.0.1:0") -> None:
        self._handler = handler
        self._requested_address = address
        self.address = address
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
                respuesta = encode(self._handler(decode(mensaje)))
                self._socket.send_multipart([identity, *respuesta])

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None

    def __enter__(self) -> RouterServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class RpcChannel:
    """Request/response confiable a un proceso downstream (``DEALER`` + timeout + reintentos)."""

    def __init__(self, address: str, timeout_s: float, retries: int = 1) -> None:
        self._address = address
        self._timeout_ms = int(timeout_s * 1000)
        self._retries = retries
        self._context = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self._poller = zmq.Poller()
        self._connect()

    def _connect(self) -> None:
        socket = self._context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self._address)
        self._socket = socket
        self._poller.register(socket, zmq.POLLIN)

    def _reconnect(self) -> None:
        # Tras un timeout se recrea el socket: con ``DEALER`` una respuesta tardía
        # desalinearía las siguientes. Por eso ``DEALER`` y no ``REQ``/``REP``.
        if self._socket is not None:
            self._poller.unregister(self._socket)
            self._socket.close(0)
        self._connect()

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self._socket is not None
        for _ in range(self._retries + 1):
            self._socket.send_multipart(encode(request))
            if dict(self._poller.poll(self._timeout_ms)).get(self._socket) == zmq.POLLIN:
                response = decode(self._socket.recv_multipart())
                if not response.get("ok", False):
                    raise RuntimeError(response.get("error", "error desconocido del componente"))
                return response
            self._reconnect()
        raise DownstreamTimeout(
            f"sin respuesta de {self._address} tras {self._retries + 1} intento(s)"
        )

    def close(self) -> None:
        if self._socket is not None:
            self._poller.unregister(self._socket)
            self._socket.close(0)
            self._socket = None
