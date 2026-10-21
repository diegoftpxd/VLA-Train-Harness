"""Suite de contrato OBS para ``ObservationProcessor``.

La implementación entrega el fixture ``processor``. Las observaciones se generan
con ``make_observations`` usando las cámaras requeridas por el procesador.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from vla_common import Observation
from vla_learner.policy.interfaces import ObservationProcessor
from vla_learner.testing.data import make_observations


class ObservationProcessorContract:
    @pytest.fixture
    def processor(self) -> ObservationProcessor:
        raise NotImplementedError

    @pytest.fixture
    def observations(self, processor: ObservationProcessor) -> list[Observation]:
        cameras = processor.required_cameras or ["front"]
        return make_observations(4, cameras=cameras, seed=0)

    def test_obs01_formas_consistentes(self, processor, observations):
        model_input = processor(observations)
        assert model_input.batch_size == len(observations)
        for tensor in model_input.tensors.values():
            assert tensor.shape[0] == len(observations)

    def test_obs02_camaras_faltantes_y_sobrantes(self, processor, observations):
        required = processor.required_cameras
        if not required:
            pytest.skip("el procesador no requiere cámaras")

        sobrantes = [
            Observation(
                images={**o.images, "camara_extra": np.zeros((3, 8, 8), dtype=np.float32)},
                proprio=o.proprio,
                instruction=o.instruction,
            )
            for o in observations
        ]
        processor(sobrantes)  # las cámaras extra se ignoran, no falla

        faltantes = [
            Observation(
                images={k: v for k, v in o.images.items() if k != required[0]},
                proprio=o.proprio,
                instruction=o.instruction,
            )
            for o in observations
        ]
        with pytest.raises(ValueError) as excinfo:
            processor(faltantes)
        assert required[0] in str(excinfo.value)

    def test_obs03_determinista(self, processor, observations):
        a = processor(observations)
        b = processor(observations)
        assert set(a.tensors) == set(b.tensors)
        for key in a.tensors:
            import torch

            assert torch.equal(a.tensors[key], b.tensors[key])

    def test_obs04_instrucciones_largo_variable(self, processor, observations):
        base = observations[0]
        corta = Observation(images=base.images, proprio=base.proprio, instruction="ir")
        larga = Observation(
            images=base.images, proprio=base.proprio, instruction=" ".join(["palabra"] * 50)
        )
        model_input = processor([corta, larga])
        assert model_input.batch_size == 2

    def test_obs05_no_modifica_la_entrada(self, processor, observations):
        copia = copy.deepcopy(observations)
        processor(observations)
        for antes, ahora in zip(copia, observations, strict=True):
            assert np.array_equal(antes.proprio, ahora.proprio)
            assert antes.instruction == ahora.instruction
            assert set(antes.images) == set(ahora.images)
            for key in antes.images:
                assert np.array_equal(antes.images[key], ahora.images[key])
