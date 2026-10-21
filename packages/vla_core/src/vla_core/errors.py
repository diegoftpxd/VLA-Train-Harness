"""Errores de dominio de Core.

Definidos en ``docs/core/Core Design.md``. ``dispatch_request`` (en
``vla_common``) captura cualquier excepción y la devuelve al cliente como
``{"ok": False, "error": ...}``; el ``CoreClient`` la vuelve a levantar como
``RuntimeError`` en el Learner. Estas subclases dan nombres claros a las fallas
previstas.
"""

from __future__ import annotations


class CoreError(Exception):
    """Base de las fallas de dominio de Core."""


class UnknownEnvironment(CoreError):
    """Se pidió un ``env_id`` que no está en el registro de ambientes."""


class EpisodeAlreadyDone(CoreError):
    """``step`` sobre un episodio ya ``terminated``/``truncated`` (falta ``reset``)."""


class DownstreamTimeout(CoreError):
    """Un componente (Environment, WorldModel, Reward Model) no respondió a tiempo."""


class DownstreamUnavailable(CoreError):
    """Un componente se cayó o no está disponible tras agotar los reintentos."""


class DatasetNotAppendable(CoreError):
    """Se intentó reinyectar episodios en un ``Dataset`` que no es ``AppendableDataset``."""
