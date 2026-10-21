"""Construcción de piezas del Learner a partir de su nombre en la configuración.

Definido en ``docs/learner/Learner Design.md`` (sección "Construcción desde
configuración"). Cambiar de modelo o de algoritmo es cambiar un nombre en el
YAML: cada pieza se registra con un ``kind`` (``policy``, ``algorithm``, ...) y
un ``name``, y se construye con ``Registry.build``.

Nota de implementación: el documento describía un diccionario por ``kind``
(``policies``, ``algorithms``, ...). Aquí se usa un único registro anidado
``_registries[kind][name]``, equivalente pero abierto a nuevos ``kind`` sin
tocar la clase. ``Learner Design.md`` quedó actualizado con esta forma.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Factory = Callable[..., Any]


class Registry:
    """Registro global de factories, con un espacio de nombres por ``kind``."""

    _registries: dict[str, dict[str, Factory]] = {}

    @classmethod
    def register(cls, kind: str, name: str) -> Callable[[Factory], Factory]:
        """Decorador que registra una factory bajo ``(kind, name)``.

        Registrar dos veces el mismo ``(kind, name)`` lanza ``ValueError``.
        """
        table = cls._registries.setdefault(kind, {})

        def decorator(factory: Factory) -> Factory:
            if name in table:
                raise ValueError(
                    f"Ya hay algo registrado como {kind!r}/{name!r}. "
                    f"Nombres en {kind!r}: {sorted(table)}"
                )
            table[name] = factory
            return factory

        return decorator

    @classmethod
    def build(cls, kind: str, config: dict[str, Any]) -> Any:
        """Construye la pieza nombrada en ``config['type']``.

        El resto de las claves de ``config`` se pasan como argumentos de la
        factory. Un ``type`` desconocido lanza ``KeyError`` listando los
        nombres disponibles para ese ``kind``.
        """
        table = cls._registries.get(kind, {})
        if "type" not in config:
            raise KeyError(f"La configuración de {kind!r} no tiene clave 'type': {config!r}")
        name = config["type"]
        if name not in table:
            raise KeyError(
                f"{kind!r}/{name!r} no está registrado. Disponibles en {kind!r}: {sorted(table)}"
            )
        params = {key: value for key, value in config.items() if key != "type"}
        return table[name](**params)

    @classmethod
    def available(cls, kind: str) -> list[str]:
        """Nombres registrados para un ``kind``, ordenados."""
        return sorted(cls._registries.get(kind, {}))
