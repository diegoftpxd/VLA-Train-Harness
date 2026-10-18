"""Suite de contrato POL para ``VLAPolicy``.

La más importante: la debe pasar todo modelo (adaptador o ``ComposedPolicy``).
La implementación entrega el fixture ``policy`` (ya materializada, si usa capas
perezosas). Las observaciones salen de ``make_observations`` con las cámaras que
pida el procesador. Cubre POL-01..15 (los de CPU); POL-16/17 son ``gpu`` y
POL-18 es ``slow``/``heavy``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vla_common import Observation
from vla_learner.policy.interfaces import Capability, VLAPolicy
from vla_learner.testing.data import make_observations
from vla_learner.types import ModelInput


class VLAPolicyContract:
    @pytest.fixture
    def policy(self) -> VLAPolicy:
        raise NotImplementedError

    @pytest.fixture
    def observations(self, policy: VLAPolicy) -> list[Observation]:
        cameras = policy.processor.required_cameras or ["front"]
        return make_observations(4, cameras=cameras, seed=0)

    @pytest.fixture
    def inputs(self, policy: VLAPolicy, observations: list[Observation]) -> ModelInput:
        return policy.processor(observations)

    def test_pol01_capacidades_honestas(self, policy, inputs):
        caps = policy.capabilities
        out = policy.act(inputs)
        if Capability.LOG_PROB not in caps:
            with pytest.raises(NotImplementedError):
                policy.evaluate_actions(inputs, out.actions, out.extras)
            return
        ev = policy.evaluate_actions(inputs, out.actions, out.extras)
        assert torch.isfinite(ev.log_probs).all()
        if Capability.VALUE in caps:
            assert ev.values is not None and torch.isfinite(ev.values).all()
        else:
            assert ev.values is None
        if Capability.ENTROPY in caps:
            assert ev.entropy is not None and torch.isfinite(ev.entropy).all()
        else:
            assert ev.entropy is None

    def test_pol02_formas_de_act(self, policy, inputs):
        out = policy.act(inputs)
        batch = inputs.batch_size
        assert out.actions.shape == (batch, policy.chunk_size, policy.action_dim)
        assert torch.isfinite(out.actions).all()
        if Capability.LOG_PROB in policy.capabilities:
            assert out.log_probs is not None and out.log_probs.shape == (batch,)
        else:
            assert out.log_probs is None
        if Capability.VALUE in policy.capabilities:
            assert out.values is not None and out.values.shape == (batch,)
        else:
            assert out.values is None

    def test_pol03_distintos_tamanos_de_batch(self, policy, observations):
        uno = policy.act(policy.processor(observations[:1]))
        cuatro = policy.act(policy.processor(observations[:4]))
        assert uno.actions.shape[0] == 1
        assert cuatro.actions.shape[0] == 4
        batched = policy.act(policy.processor(observations[:4]), deterministic=True).actions
        for i in range(4):
            single = policy.act(
                policy.processor(observations[i : i + 1]), deterministic=True
            ).actions
            assert torch.allclose(batched[i : i + 1], single, atol=1e-5)

    def test_pol04_determinista_reproducible(self, policy, inputs):
        a = policy.act(inputs, deterministic=True).actions
        b = policy.act(inputs, deterministic=True).actions
        assert torch.equal(a, b)

    def test_pol05_estocastico_depende_de_semilla(self, policy, inputs):
        torch.manual_seed(0)
        a = policy.act(inputs).actions
        torch.manual_seed(0)
        b = policy.act(inputs).actions
        assert torch.allclose(a, b)
        torch.manual_seed(1)
        c = policy.act(inputs).actions
        assert not torch.allclose(a, c)

    def test_pol06_act_sin_efectos_secundarios(self, policy, inputs):
        modo = policy.training
        snapshot = [p.detach().clone() for p in policy.parameters()]
        out = policy.act(inputs)
        assert not out.actions.requires_grad
        assert policy.training == modo
        for parametro, previo in zip(policy.parameters(), snapshot, strict=True):
            assert torch.equal(parametro, previo)

    def test_pol07_acciones_en_unidades_reales(self, policy, inputs):
        out = policy.act(inputs, deterministic=True)
        assert out.actions.abs().max() > 1.5  # no confinadas a [-1, 1]

    def test_pol08_consistencia_con_extras(self, policy, inputs):
        if Capability.LOG_PROB not in policy.capabilities:
            pytest.skip("sin LOG_PROB")
        out = policy.act(inputs)
        assert out.log_probs is not None
        ev = policy.evaluate_actions(inputs, out.actions, out.extras)
        assert torch.allclose(out.log_probs, ev.log_probs, atol=1e-4)

    def test_pol09_consistencia_sin_extras(self, policy, inputs):
        if Capability.LOG_PROB not in policy.capabilities:
            pytest.skip("sin LOG_PROB")
        out = policy.act(inputs)
        assert out.log_probs is not None
        ev = policy.evaluate_actions(inputs, out.actions, None)
        assert torch.allclose(out.log_probs, ev.log_probs, atol=1e-2)

    def test_pol10_evaluate_actions_produce_gradientes(self, policy, inputs):
        if Capability.LOG_PROB not in policy.capabilities:
            pytest.skip("sin LOG_PROB")
        out = policy.act(inputs)
        policy.zero_grad(set_to_none=True)
        ev = policy.evaluate_actions(inputs, out.actions, out.extras)
        ev.log_probs.sum().backward()
        con_gradiente = [p for p in policy.parameters() if p.grad is not None]
        assert con_gradiente  # hay señal de aprendizaje
        for parametro in con_gradiente:
            grad = parametro.grad
            assert grad is not None and torch.isfinite(grad).all()
        for parametro in policy.parameters():
            if not parametro.requires_grad:
                assert parametro.grad is None  # los congelados no reciben gradiente

    def test_pol11_imitation_loss_valida(self, policy, inputs):
        out = policy.act(inputs)
        policy.zero_grad(set_to_none=True)
        resultado = policy.imitation_loss(inputs, out.actions)
        assert resultado.loss.ndim == 0
        assert torch.isfinite(resultado.loss)
        assert all(isinstance(v, float) for v in resultado.metrics.values())
        resultado.loss.backward()
        assert any(p.grad is not None for p in policy.parameters())

    def test_pol12_congelado_se_respeta(self, policy, inputs):
        congelados = list(policy.parameters())[::2]
        for parametro in congelados:
            parametro.requires_grad_(False)
        snapshot = [p.detach().clone() for p in congelados]
        entrenables = [p for p in policy.parameters() if p.requires_grad]
        optimizador = torch.optim.SGD(entrenables, lr=0.1)
        out = policy.act(inputs)
        policy.zero_grad(set_to_none=True)
        policy.imitation_loss(inputs, out.actions).loss.backward()
        optimizador.step()
        for parametro, previo in zip(congelados, snapshot, strict=True):
            assert torch.equal(parametro, previo)

    def test_pol13_guardar_cargar_conserva(self, policy, observations, tmp_path):
        inputs = policy.processor(observations)
        antes = policy.act(inputs, deterministic=True).actions
        policy.save_pretrained(Path(tmp_path))
        recargada = type(policy).from_pretrained(Path(tmp_path))
        despues = recargada.act(recargada.processor(observations), deterministic=True).actions
        assert torch.allclose(antes, despues, atol=1e-5)

    def test_pol14_clone_frozen_independiente(self, policy, inputs):
        clon = policy.clone_frozen()
        assert all(not p.requires_grad for p in clon.parameters())
        a = policy.act(inputs, deterministic=True).actions
        b = clon.act(inputs, deterministic=True).actions
        assert torch.allclose(a, b, atol=1e-5)
        previo = clon.act(inputs, deterministic=True).actions.clone()
        with torch.no_grad():
            for parametro in policy.parameters():
                parametro.add_(0.1)
        posterior = clon.act(inputs, deterministic=True).actions
        assert torch.allclose(previo, posterior, atol=1e-6)

    def test_pol15_extras_seleccionable(self, policy, inputs):
        out = policy.act(inputs)
        for tensor in out.extras.values():
            assert tensor.shape[0] == inputs.batch_size
