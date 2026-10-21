"""Serialización de mensajes para el protocolo entre procesos.

Un mensaje se codifica como una lista de *frames* (``bytes``), tal como los
espera un mensaje multiparte de ZeroMQ (ver ``docs/Architecture.md``):

- El primer frame es un encabezado en msgpack. Describe la estructura del
  mensaje; cada array de numpy se reemplaza por un marcador con su ``dtype``,
  su ``shape`` y el índice del frame que lo contiene.
- Los frames siguientes son los buffers crudos de cada array, en orden.

No se usa ``pickle``: es inseguro entre procesos y se rompe entre versiones de
Python o de librerías. Este módulo no importa ``zmq``; solo arma y desarma los
frames. Quien los envía por la red es ``CoreClient``.
"""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np

# Clave que marca un array dentro del encabezado.
_ND = "__ndarray__"


def encode(payload: Any) -> list[bytes]:
    """Codifica un payload (dicts, listas, escalares y arrays) en frames.

    Devuelve ``[encabezado_msgpack, buffer_0, buffer_1, ...]``.
    """
    buffers: list[bytes] = []

    def replace(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            arr = np.ascontiguousarray(obj)
            index = len(buffers)
            buffers.append(arr.tobytes())
            return {_ND: index, "dtype": str(arr.dtype), "shape": list(arr.shape)}
        if isinstance(obj, np.generic):
            # Escalares de numpy: a su equivalente de Python.
            return obj.item()
        if isinstance(obj, dict):
            return {key: replace(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [replace(value) for value in obj]
        return obj

    header = replace(payload)
    header_bytes = msgpack.packb(header, use_bin_type=True)
    if header_bytes is None:  # el stub de msgpack lo admite; no ocurre en la práctica
        raise RuntimeError("msgpack.packb devolvió None al serializar el encabezado.")
    return [header_bytes, *buffers]


def decode(frames: list[bytes]) -> Any:
    """Reconstruye el payload a partir de los frames producidos por ``encode``."""
    header = msgpack.unpackb(frames[0], raw=False)
    buffers = frames[1:]

    def restore(obj: Any) -> Any:
        if isinstance(obj, dict):
            if _ND in obj:
                buffer = buffers[obj[_ND]]
                array = np.frombuffer(buffer, dtype=np.dtype(obj["dtype"]))
                return array.reshape(obj["shape"])
            return {key: restore(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [restore(value) for value in obj]
        return obj

    return restore(header)
