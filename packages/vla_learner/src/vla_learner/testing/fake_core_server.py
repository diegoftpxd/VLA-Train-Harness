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

import zmq

from vla_common import dispatch_request
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
        # El despacho (wire↔tipos, mapeo llamada→método y manejo de errores) es
        # compartido con el ``CoreServer`` real; vive en ``vla_common``.
        return encode(dispatch_request(decode(mensaje), self.client))

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
