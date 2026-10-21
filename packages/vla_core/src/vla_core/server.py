"""``CoreServer``: la capa de transporte de Core (socket ``ROUTER``).

Definido en ``docs/core/Core Design.md``. Su única responsabilidad es mover
frames: recibe un mensaje multiparte del Learner, separa la identidad del
remitente del resto, delega el despacho en ``vla_common.dispatch_request`` y
devuelve la respuesta a esa identidad. No conoce la lógica de Core ni los tipos
del dominio.

Envuelve *cualquier* backend que cumpla ``CoreInterface``: el ``Core`` real o un
doble de prueba. Por eso comparte forma con el ``FakeCoreServer`` del Learner,
que envuelve un ``FakeCoreClient``.
"""

from __future__ import annotations

import threading

import zmq

from vla_common import CoreInterface, dispatch_request
from vla_common.serialization import decode, encode


class CoreServer:
    def __init__(self, backend: CoreInterface, address: str = "tcp://127.0.0.1:0") -> None:
        self.backend = backend
        self._requested_address = address
        self.address = address
        self._context = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> str:
        """Bindea el socket ``ROUTER``, arranca el hilo de atención y devuelve el endpoint real."""
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
                respuesta = encode(dispatch_request(decode(mensaje), self.backend))
                self._socket.send_multipart([identity, *respuesta])

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None

    def __enter__(self) -> CoreServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
