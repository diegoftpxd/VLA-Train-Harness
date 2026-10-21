"""Adaptadores de modelos VLA preentrenados a la interfaz ``VLAPolicy``.

Un adaptador envuelve un modelo que viene como un bloque y no conviene partir en
backbone + cabeza (ver ``docs/Learner Design.md``, "Modelos que no se pueden
separar"). El resto del Learner no nota la diferencia.

El módulo ``openvla`` importa ``transformers`` solo dentro de
``OpenVLAPolicy.from_pretrained``, así que importar este paquete no requiere
tener instalado el grupo opcional ``openvla``.
"""

from vla_learner.policy.adapters.openvla import OpenVLAPolicy, OpenVLAProcessor

__all__ = ["OpenVLAPolicy", "OpenVLAProcessor"]
