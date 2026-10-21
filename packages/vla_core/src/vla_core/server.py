"""``CoreServer``: la capa de transporte de Core hacia el Learner (socket ``ROUTER``).

Definido en ``docs/core/Core Design.md``. Es un ``RouterServer`` cuyo handler es
el despacho compartido ``vla_common.dispatch_request``: traduce wire↔tipos, llama
al backend y devuelve el sobre ``ok``/``error``. Envuelve *cualquier* backend que
cumpla ``CoreInterface`` (el ``Core`` real o un doble de prueba), igual que el
``FakeCoreServer`` del Learner envuelve un ``FakeCoreClient``.
"""

from __future__ import annotations

from vla_common import CoreInterface, dispatch_request
from vla_core.rpc import RouterServer


class CoreServer(RouterServer):
    def __init__(self, backend: CoreInterface, address: str = "tcp://127.0.0.1:0") -> None:
        self.backend = backend
        super().__init__(
            handler=lambda request: dispatch_request(request, backend), address=address
        )
