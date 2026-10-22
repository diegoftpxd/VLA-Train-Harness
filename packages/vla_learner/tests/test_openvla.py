"""Tests del adaptador ``OpenVLAPolicy`` con un modelo falso diminuto.

No descargan pesos ni usan ``transformers``: inyectan un modelo y un procesador
falsos para ejercitar en CPU la lógica de normalización, tokenización, log-probs
y loss. Los pesos reales de OpenVLA (~16 GB) no caben en esta máquina; su
verificación va en el cluster (ver ``examples/openvla_smoke.py``).

Marcados ``fast``: corren siempre.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vla_common import Observation
from vla_learner.policy.adapters import OpenVLAPolicy, OpenVLAProcessor
from vla_learner.policy.interfaces import Capability
from vla_learner.types import ModelInput

pytestmark = pytest.mark.fast

VOCAB = 64
N_BINS = 8
ACTION_DIM = 3
# Dimensiones 0 y 1 se normalizan; la 2 (p. ej. pinza) pasa sin tocar.
Q01 = np.array([-2.0, -1.0, 0.0], dtype=np.float32)
Q99 = np.array([2.0, 1.0, 1.0], dtype=np.float32)
MASK = np.array([True, True, False])


class _FakeVLA(nn.Module):
    """Modelo mínimo: embedding + cabeza por posición, con tokens de visión al frente.

    Los tokens de visión van adelante para comprobar que el adaptador localiza los
    logits de acción indexando desde el final, sin conocer el desfase de visión.
    """

    def __init__(self, vocab_size: int = VOCAB, hidden: int = 16, n_vision: int = 3) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_vision = n_vision
        self.embed = nn.Embedding(vocab_size, hidden)
        self.vision = nn.Linear(3 * 8 * 8, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def _merged(self, input_ids: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        tok = self.embed(input_ids)
        vis = self.vision(pixel_values.flatten(1)).unsqueeze(1).expand(-1, self.n_vision, -1)
        return torch.cat([vis, tok], dim=1)

    def forward(self, input_ids, attention_mask=None, pixel_values=None):  # noqa: ANN001
        assert pixel_values is not None
        return SimpleNamespace(logits=self.head(self._merged(input_ids, pixel_values)))

    @torch.no_grad()
    def generate(  # noqa: ANN001
        self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        max_new_tokens=1,
        do_sample=False,
        **_,
    ):
        ids = input_ids
        for _step in range(max_new_tokens):
            logits = self.forward(ids, None, pixel_values).logits[:, -1, :]
            nxt = (
                torch.multinomial(logits.softmax(-1), 1)
                if do_sample
                else logits.argmax(-1, keepdim=True)
            )
            ids = torch.cat([ids, nxt], dim=1)
        return ids


class _FakeHFProcessor:
    """Doble del ``AutoProcessor`` de HF: arma tensores de forma válida."""

    def __init__(self, vocab_size: int = VOCAB) -> None:
        self.tokenizer = SimpleNamespace(vocab_size=vocab_size, padding_side="right")

    def __call__(self, text, images, return_tensors="pt", padding=True):  # noqa: ANN001
        batch = len(text)
        length = 5
        return {
            "input_ids": torch.randint(0, self.tokenizer.vocab_size, (batch, length)),
            "attention_mask": torch.ones(batch, length, dtype=torch.long),
            "pixel_values": torch.randn(batch, 3, 8, 8),
        }

    def save_pretrained(self, directory):  # noqa: ANN001
        pass


def _policy(seed: int = 0) -> OpenVLAPolicy:
    torch.manual_seed(seed)
    return OpenVLAPolicy(
        _FakeVLA(),
        _FakeHFProcessor(),
        unnorm_key="fake",
        q01=Q01,
        q99=Q99,
        mask=MASK,
        n_bins=N_BINS,
        llm_vocab_size=VOCAB,
    )


def _inputs(batch: int = 4, length: int = 5) -> ModelInput:
    return ModelInput(
        tensors={
            "input_ids": torch.randint(0, VOCAB, (batch, length)),
            "attention_mask": torch.ones(batch, length, dtype=torch.long),
            "pixel_values": torch.randn(batch, 3, 8, 8),
        },
        batch_size=batch,
    )


def test_propiedades_basicas() -> None:
    policy = _policy()
    assert policy.capabilities == {Capability.LOG_PROB, Capability.ENTROPY}
    assert policy.chunk_size == 1
    assert policy.action_dim == ACTION_DIM


def test_tokens_en_rango_y_roundtrip_de_centros() -> None:
    policy = _policy()
    # Los ids de acción ocupan los últimos ``n_bins`` del vocabulario.
    norm = torch.tensor([[-1.0, 0.0, 1.0]])
    ids = policy._to_token_ids(norm)
    assert ids.min() >= VOCAB - N_BINS
    assert ids.max() <= VOCAB - 1
    # Un centro de bin vuelve exacto al pasar a token y de vuelta.
    center = policy._bin_centers[3].item()
    recovered = policy._from_token_ids(policy._to_token_ids(torch.tensor([[center] * ACTION_DIM])))
    assert torch.allclose(recovered, torch.full((1, ACTION_DIM), center), atol=1e-6)


def test_normalize_respeta_la_mascara() -> None:
    policy = _policy()
    actions = torch.tensor([[1.5, 0.5, 0.7]])
    norm = policy._normalize(actions)
    # Dim 2 (mask=False) pasa sin tocar; dims 0 y 1 caen en [-1, 1].
    assert norm[0, 2].item() == pytest.approx(0.7)
    assert -1.0 <= norm[0, 0].item() <= 1.0
    assert -1.0 <= norm[0, 1].item() <= 1.0
    # Round-trip real -> normalizado -> real en las dims normalizadas.
    back = policy._unnormalize(norm)
    assert torch.allclose(back[:, :2], actions[:, :2], atol=1e-5)


def test_act_formas_y_determinismo() -> None:
    policy = _policy()
    inputs = _inputs(batch=4)
    out1 = policy.act(inputs, deterministic=True)
    out2 = policy.act(inputs, deterministic=True)
    assert out1.actions.shape == (4, 1, ACTION_DIM)
    assert out1.log_probs is not None and out1.log_probs.shape == (4,)
    assert out1.values is None
    assert out1.extras["action_token_ids"].shape == (4, ACTION_DIM)
    # POL-04: determinista => dos llamadas idénticas.
    assert out2.log_probs is not None
    assert torch.equal(out1.actions, out2.actions)
    assert torch.allclose(out1.log_probs, out2.log_probs)


def test_evaluate_coincide_con_act_usando_extras() -> None:
    policy = _policy()
    inputs = _inputs(batch=3)
    out = policy.act(inputs, deterministic=True)
    evaluation = policy.evaluate_actions(inputs, out.actions, out.extras)
    assert evaluation.log_probs.shape == (3,)
    assert evaluation.entropy is not None and evaluation.entropy.shape == (3,)
    assert (evaluation.entropy >= 0).all()
    assert evaluation.values is None
    # Reusar los ids elegidos al actuar da exactamente las mismas log-probs.
    assert out.log_probs is not None
    assert torch.allclose(evaluation.log_probs, out.log_probs, atol=1e-5)


def test_evaluate_sin_extras_tokeniza_las_acciones() -> None:
    policy = _policy()
    inputs = _inputs(batch=3)
    actions = torch.randn(3, 1, ACTION_DIM)
    evaluation = policy.evaluate_actions(inputs, actions)
    assert evaluation.log_probs.shape == (3,)
    assert torch.isfinite(evaluation.log_probs).all()


def test_imitation_loss_propaga_gradientes() -> None:
    policy = _policy()
    inputs = _inputs(batch=4)
    actions = torch.randn(4, 1, ACTION_DIM)
    salida = policy.imitation_loss(inputs, actions)
    assert salida.loss.ndim == 0
    assert salida.loss.requires_grad
    assert "token_accuracy" in salida.metrics
    salida.loss.backward()
    grads = [p.grad for p in policy.vla.parameters() if p.grad is not None]
    assert grads, "la loss de imitación no generó gradientes en el modelo"


def test_freeze_except_deja_solo_lo_entrenable() -> None:
    # Fine-tuning parcial: congela todo salvo ``head`` (análogo a ``lm_head`` en el
    # modelo real). ``trainable_parameters()`` debe devolver solo esos parámetros.
    policy = _policy()
    policy.freeze_except(["head"])

    entrenables = {nombre for nombre, p in policy.vla.named_parameters() if p.requires_grad}
    assert entrenables == {"head.weight", "head.bias"}
    n_entrenables = sum(1 for _ in policy.trainable_parameters())
    assert n_entrenables == 2


def test_freeze_except_un_paso_no_mueve_lo_congelado() -> None:
    # POL-12 para OpenVLA: tras un paso del optimizador, solo cambia lo entrenable.
    policy = _policy()
    policy.freeze_except(["head"])
    antes = {nombre: p.detach().clone() for nombre, p in policy.vla.named_parameters()}

    optim = torch.optim.SGD(policy.trainable_parameters(), lr=0.1)
    salida = policy.imitation_loss(_inputs(batch=4), torch.randn(4, 1, ACTION_DIM))
    salida.loss.backward()
    optim.step()

    for nombre, p in policy.vla.named_parameters():
        cambio = not torch.equal(antes[nombre], p.detach())
        if nombre.startswith("head."):
            assert cambio, f"{nombre} entrenable debió cambiar"
        else:
            assert not cambio, f"{nombre} congelado no debió cambiar"


def test_freeze_except_patron_sin_coincidencia_es_error() -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="Ningún parámetro coincide"):
        policy.freeze_except(["no_existe"])


def test_registrado_en_builder() -> None:
    # Importar el builder registra la policy "openvla" sin cargar transformers.
    import vla_learner.builder  # noqa: F401
    from vla_learner.registry import Registry

    assert "openvla" in Registry.available("policy")


def test_processor_arma_model_input() -> None:
    pytest.importorskip("PIL")
    processor = OpenVLAProcessor(_FakeHFProcessor(), camera="primary")
    assert processor.required_cameras == ["primary"]
    obs = [
        Observation(
            images={"primary": np.zeros((8, 8, 3), dtype=np.uint8)},
            proprio=np.zeros(3, dtype=np.float32),
            instruction="pick up the cube",
        )
        for _ in range(2)
    ]
    model_input = processor(obs)
    assert model_input.batch_size == 2
    assert set(model_input.tensors) == {"input_ids", "attention_mask", "pixel_values"}
