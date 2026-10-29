"""Suites de contrato compartidas entre paquetes.

Viven en ``vla_common`` las suites de los contratos que también viven aquí: las
importan tanto Core como los paquetes que implementan el componente (p. ej.
``vla_env``), que no dependen entre sí.
"""

from vla_common.testing.contracts.environment_contract import EnvironmentContract

__all__ = ["EnvironmentContract"]
