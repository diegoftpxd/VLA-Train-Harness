"""Construcción de un ``Dataset`` desde configuración (``build_dataset``).

Definido en ``docs/dataset/Dataset Design.md``. Es la costura que mantiene a
Core agnóstico del formato: despacha por el campo ``type`` del bloque
``dataset`` de la config y devuelve el adaptador concreto, que ya cumple la ABC
``Dataset``. Core solo ve la ABC.

Los adaptadores se importan **perezosamente** dentro de su rama (igual que
``builder._build_backend`` con los fakes), para que un adaptador con
dependencias pesadas no se importe salvo que la config lo pida y ``vla_core``
base siga sin ``torch`` ni ``tensorflow``.

En esta fase solo está disponible el adaptador ``fake`` (en memoria). Los
adaptadores ``local``, ``lerobot`` y ``openx`` se suman en fases posteriores
(ver el plan de acción en ``Dataset Design.md``).
"""

from __future__ import annotations

from typing import Any

from vla_core.dataset.base import Dataset

# Tipos conocidos que aún no están implementados; se enumeran para dar un error
# claro (y no un "desconocido") mientras llegan en fases posteriores.
_PENDIENTES = {"local", "lerobot", "openx"}


def build_dataset(config: dict[str, Any]) -> Dataset:
    """Construye el ``Dataset`` indicado por ``config`` (``type``, por defecto ``fake``)."""
    tipo = config.get("type", "fake")

    if tipo == "fake":
        # Import perezoso: ``FakeDataset`` vive en ``testing`` y es solo para pruebas.
        from vla_core.testing.fakes import FakeDataset

        return FakeDataset(
            seed=config.get("seed", 0),
            demo_chunk_size=config.get("demo_chunk_size", 1),
        )

    if tipo in _PENDIENTES:
        raise NotImplementedError(
            f"el adaptador de dataset '{tipo}' aún no está implementado "
            "(ver el plan de acción en docs/dataset/Dataset Design.md)"
        )

    raise ValueError(f"tipo de dataset desconocido: {tipo!r}")
