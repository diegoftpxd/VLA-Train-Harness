"""Suite de contrato CORE para la interfaz de Core.

La misma suite corre contra ``FakeCoreClient`` (en memoria) y contra el
``CoreClient`` real conectado a un ``FakeCoreServer``, para garantizar que el
falso se comporta como el real. CORE-04 (timeout) es propio del cliente real y
vive en su módulo de test. La implementación entrega el fixture ``client``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_common import Action, ActionChunk


def _chunk(values: list[float]) -> ActionChunk:
    return [Action(values=np.asarray(values, dtype=np.float32))]


def _quieto() -> ActionChunk:
    return _chunk([0.0, 0.0])


class CoreClientContract:
    @pytest.fixture
    def client(self):
        raise NotImplementedError

    def test_core01_serializacion_exacta(self, client):
        ((episode_id, obs),) = client.reset(["e0"], seed=1)
        assert obs.proprio.dtype == np.float32
        valores = [0.123, -0.456]
        client.step(["e0"], [_chunk(valores)], with_reward=True)
        transiciones = client.get_episode(episode_id)
        enviado = np.asarray(valores, dtype=np.float32)
        recibido = transiciones[0].action[0].values
        assert recibido.dtype == np.float32
        assert recibido.shape == enviado.shape
        assert np.array_equal(recibido, enviado)  # bit a bit

    def test_core02_orden_de_ambientes(self, client):
        env_ids = ["a", "b", "c"]
        episodios = client.reset(env_ids, seed=0)
        assert len(episodios) == 3
        for env_id, (episode_id, _) in zip(env_ids, episodios, strict=True):
            assert episode_id.startswith(env_id)

    def test_core03_episodio_terminado(self, client):
        client.reset(["e0"], seed=7)
        termino = False
        for _ in range(25):
            (resultado,) = client.step(["e0"], [_quieto()], with_reward=True)
            if resultado.terminated or resultado.truncated:
                termino = True
                break
        assert termino
        with pytest.raises(RuntimeError):
            client.step(["e0"], [_quieto()], with_reward=True)

    def test_core05_recompensa_opcional(self, client):
        client.reset(["e0"], seed=3)
        (resultado,) = client.step(["e0"], [_quieto()], with_reward=False)
        assert resultado.reward is None
