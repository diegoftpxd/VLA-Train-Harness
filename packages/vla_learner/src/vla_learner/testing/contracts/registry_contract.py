"""Suite de contrato REG para ``Registry``.

La hereda cada conjunto de piezas registradas: la implementación entrega, vía
el fixture ``registered``, la lista de ``(kind, name, config_mínima)`` que
deben poder construirse.
"""

from __future__ import annotations

import pytest

from vla_learner.registry import Registry


class RegistryContract:
    @pytest.fixture
    def registered(self) -> list[tuple[str, str, dict]]:
        raise NotImplementedError

    def test_reg01_todo_lo_registrado_se_construye(
        self, registered: list[tuple[str, str, dict]]
    ) -> None:
        assert registered, "la implementación debe declarar al menos una pieza"
        for kind, name, config in registered:
            obj = Registry.build(kind, {"type": name, **config})
            assert obj is not None

    def test_reg02_nombre_desconocido_lista_disponibles(
        self, registered: list[tuple[str, str, dict]]
    ) -> None:
        kind = registered[0][0]
        with pytest.raises(KeyError) as excinfo:
            Registry.build(kind, {"type": "__no_existe__"})
        mensaje = str(excinfo.value)
        for _, name, _ in registered:
            assert name in mensaje

    def test_reg03_nombre_duplicado(self) -> None:
        kind, name = "__reg_contract_dup__", "x"

        @Registry.register(kind, name)
        def _primero(**_kwargs: object) -> object:
            return object()

        with pytest.raises(ValueError):

            @Registry.register(kind, name)
            def _segundo(**_kwargs: object) -> object:
                return object()

        # Limpieza: no dejar el nombre de prueba en el registro global.
        Registry._registries.get(kind, {}).pop(name, None)
