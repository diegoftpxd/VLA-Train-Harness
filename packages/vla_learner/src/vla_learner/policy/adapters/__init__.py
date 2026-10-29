"""Adaptadores de modelos VLA preentrenados a la interfaz ``VLAPolicy``.

Un adaptador envuelve un modelo que viene como un bloque y no conviene partir en
backbone + cabeza (ver ``docs/learner/Learner Design.md``, "Modelos que no se pueden
separar"). El resto del Learner no nota la diferencia.

Cada adaptador importa su dependencia pesada solo dentro de ``from_pretrained``
(``transformers`` en ``openvla``, ``lerobot`` en ``pi0``), así que importar este
paquete —o usar cualquier otro modelo— no requiere tener instalados los grupos
opcionales ``openvla`` ni ``pi0``.
"""

from vla_learner.policy.adapters.openvla import OpenVLAPolicy, OpenVLAProcessor
from vla_learner.policy.adapters.pi0 import Pi0Policy, Pi0Processor

__all__ = ["OpenVLAPolicy", "OpenVLAProcessor", "Pi0Policy", "Pi0Processor"]
