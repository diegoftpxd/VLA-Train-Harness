"""Proceso Core del harness VLA.

Core es el gateway entre el Learner y el resto del sistema (ver
``docs/Architecture.md`` y ``docs/core/Core Design.md``). No tiene modelo ni GPU
y no depende de ``torch``.

En la Fase 1 contiene la capa de transporte (``CoreServer``) y sus errores de
dominio; el gateway ``Core`` y sus colaboradores llegan en fases posteriores.
"""

from vla_core.errors import (
    CoreError,
    DatasetNotAppendable,
    DownstreamTimeout,
    DownstreamUnavailable,
    EpisodeAlreadyDone,
    UnknownEnvironment,
)
from vla_core.server import CoreServer

__all__ = [
    "CoreError",
    "CoreServer",
    "DatasetNotAppendable",
    "DownstreamTimeout",
    "DownstreamUnavailable",
    "EpisodeAlreadyDone",
    "UnknownEnvironment",
]
