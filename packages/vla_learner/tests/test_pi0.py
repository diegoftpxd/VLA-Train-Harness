"""Tests del adaptador ``Pi0Policy`` con un modelo falso diminuto.

No descargan pesos ni usan ``lerobot``: inyectan un modelo falso que cumple el
contrato (``compute_loss``/``predict_chunk``) para ejercitar en CPU la lógica del
adaptador (procesado de observaciones, formas, capacidades, loss y congelado).
Los pesos reales de π0 (~7 GB) no caben en esta máquina; su verificación va en el
cluster (ver ``examples/pi0_smoke.py``).

Marcados ``fast``: corren siempre.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from vla_common import Observation
from vla_learner.policy.adapters import Pi0Policy, Pi0Processor
from vla_learner.policy.interfaces import Capability
from vla_learner.policy.normalizer import ActionNormalizer

pytestmark = pytest.mark.fast

ACTION_DIM = 7
CHUNK = 4
CAMERA = "primary"


class _FakePi0Model(nn.Module):
    """Doble mínimo del modelo π0: cumple el contrato que usa ``Pi0Policy``.

    Predice un chunk a partir de la media de la imagen y el estado, y como loss usa
    un MSE simple. No es flow matching de verdad; solo ejercita el adaptador.
    """

    def __init__(self, action_dim: int = ACTION_DIM, chunk_size: int = CHUNK) -> None:
        super().__init__()
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        self.action_proj = nn.LazyLinear(chunk_size * action_dim)

    def _features(self, cond: dict[str, Tensor]) -> Tensor:
        imagen = cond[f"image.{CAMERA}"]
        plano = imagen.reshape(imagen.shape[0], -1).float()
        estado = cond["state"].float()
        return torch.cat([plano.mean(dim=1, keepdim=True), estado], dim=1)

    def predict_chunk(self, cond: dict[str, Tensor]) -> Tensor:
        batch = cond["state"].shape[0]
        salida = self.action_proj(self._features(cond))
        # Escalar para que las acciones no queden confinadas a valores chicos.
        return 5.0 * salida.reshape(batch, self._chunk_size, self._action_dim)

    def compute_loss(self, cond: dict[str, Tensor], actions: Tensor) -> Tensor:
        pred = self.predict_chunk(cond)
        return nn.functional.mse_loss(pred, actions)


def _policy() -> Pi0Policy:
    torch.manual_seed(0)
    model = _FakePi0Model()
    policy = Pi0Policy(
        model,
        Pi0Processor([CAMERA]),
        action_dim=ACTION_DIM,
        chunk_size=CHUNK,
    )
    # Materializa la capa perezosa del modelo falso.
    policy.act(policy.processor(_observations(2)), deterministic=True)
    return policy


def _observations(n: int) -> list[Observation]:
    rng = np.random.default_rng(0)
    return [
        Observation(
            images={CAMERA: rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)},
            proprio=rng.standard_normal(7).astype(np.float32),
            instruction="pick up the red block",
        )
        for _ in range(n)
    ]


def test_propiedades_basicas() -> None:
    policy = _policy()
    assert policy.action_dim == ACTION_DIM
    assert policy.chunk_size == CHUNK
    assert policy.capabilities == set()  # flow matching: sin log-prob
    assert policy.processor.required_cameras == [CAMERA]


def test_procesador_arma_cond() -> None:
    processor = Pi0Processor([CAMERA])
    inputs = processor(_observations(3))
    assert inputs.batch_size == 3
    assert inputs.tensors[f"image.{CAMERA}"].shape == (3, 3, 8, 8)  # HWC → CHW
    assert inputs.tensors[f"image.{CAMERA}"].max() <= 1.0  # 0..255 → 0..1
    assert inputs.tensors["state"].shape == (3, 7)
    assert "lang_tokens" in inputs.tensors and "lang_masks" in inputs.tensors


def test_act_formas() -> None:
    policy = _policy()
    obs = _observations(5)
    out = policy.act(policy.processor(obs), deterministic=True)
    assert out.actions.shape == (5, CHUNK, ACTION_DIM)
    assert out.log_probs is None and out.values is None  # sin log-prob ni valor
    assert torch.isfinite(out.actions).all()


def test_evaluate_actions_no_implementado() -> None:
    policy = _policy()
    inputs = policy.processor(_observations(2))
    actions = torch.zeros(2, CHUNK, ACTION_DIM)
    with pytest.raises(NotImplementedError):
        policy.evaluate_actions(inputs, actions)


def test_imitation_loss_propaga_gradientes() -> None:
    policy = _policy()
    inputs = policy.processor(_observations(4))
    actions = torch.randn(4, CHUNK, ACTION_DIM)
    salida = policy.imitation_loss(inputs, actions)
    assert salida.loss.ndim == 0 and torch.isfinite(salida.loss)
    assert "fm_mse" in salida.metrics
    salida.loss.backward()
    grads = [p.grad for p in policy.model.parameters() if p.grad is not None]
    assert grads, "la loss de imitación no generó gradientes en el modelo"


def test_freeze_except_deja_solo_lo_entrenable() -> None:
    policy = _policy()
    policy.freeze_except(["action_proj"])
    entrenables = {n for n, p in policy.model.named_parameters() if p.requires_grad}
    assert entrenables and all("action_proj" in n for n in entrenables)


def test_freeze_except_patron_sin_coincidencia_es_error() -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="Ningún parámetro coincide"):
        policy.freeze_except(["no_existe"])


def test_registrado_en_builder() -> None:
    # Importar el builder registra la policy "pi0" sin cargar lerobot.
    import vla_learner.builder  # noqa: F401
    from vla_learner.registry import Registry

    assert "pi0" in Registry.available("policy")


def test_capacidad_no_es_log_prob() -> None:
    policy = _policy()
    assert Capability.LOG_PROB not in policy.capabilities
    assert Capability.VALUE not in policy.capabilities


# -- Normalización (acciones en la política, estado en el procesador) -------


def _normalizer(dim: int, centro: float, escala: float) -> ActionNormalizer:
    valores = np.array([[centro - escala] * dim, [centro + escala] * dim], dtype=np.float32)
    return ActionNormalizer.from_values(valores, mode="mean_std")


class _RecordingPi0Model(nn.Module):
    """Doble que registra lo que recibe y predice un chunk fijo **normalizado**.

    ``predict_chunk`` devuelve ``padded_dim`` dims (como π0, que padea a 32) para
    comprobar que el adaptador se queda con las reales.
    """

    def __init__(self, prediccion: Tensor) -> None:
        super().__init__()
        self.peso = nn.Parameter(torch.zeros(1))
        self.prediccion = prediccion
        self.acciones_recibidas: Tensor | None = None
        self.cond_recibido: dict[str, Tensor] | None = None

    def predict_chunk(self, cond: dict[str, Tensor]) -> Tensor:
        self.cond_recibido = cond
        batch = cond["state"].shape[0]
        return self.prediccion.expand(batch, -1, -1) + 0 * self.peso

    def compute_loss(self, cond: dict[str, Tensor], actions: Tensor) -> Tensor:
        self.cond_recibido = cond
        self.acciones_recibidas = actions
        return (actions**2).mean() + 0 * self.peso.sum()


def test_imitation_loss_normaliza_las_acciones() -> None:
    norm = _normalizer(ACTION_DIM, centro=300.0, escala=100.0)
    model = _RecordingPi0Model(torch.zeros(1, CHUNK, ACTION_DIM))
    policy = Pi0Policy(
        model,
        Pi0Processor([CAMERA]),
        action_dim=ACTION_DIM,
        chunk_size=CHUNK,
        action_normalizer=norm,
    )
    reales = torch.full((2, CHUNK, ACTION_DIM), 400.0)  # unidades reales (píxeles)
    policy.imitation_loss(policy.processor(_observations(2)), reales)
    assert model.acciones_recibidas is not None
    # (400 - 300) / 100 = 1: al modelo le llegan normalizadas, no en píxeles.
    torch.testing.assert_close(model.acciones_recibidas, torch.ones_like(reales))


def test_act_recorta_dims_padeadas_y_desnormaliza() -> None:
    padded = 32  # π0 predice max_action_dim dims
    prediccion = torch.zeros(1, CHUNK, padded)
    prediccion[..., :ACTION_DIM] = 1.0  # normalizado
    prediccion[..., ACTION_DIM:] = 99.0  # padding: no debe salir
    norm = _normalizer(ACTION_DIM, centro=300.0, escala=100.0)
    policy = Pi0Policy(
        _RecordingPi0Model(prediccion),
        Pi0Processor([CAMERA]),
        action_dim=ACTION_DIM,
        chunk_size=CHUNK,
        action_normalizer=norm,
    )
    out = policy.act(policy.processor(_observations(3)), deterministic=True)
    assert out.actions.shape == (3, CHUNK, ACTION_DIM)
    # 1 * 100 + 300 = 400: sale en unidades reales (POL-07).
    torch.testing.assert_close(out.actions, torch.full((3, CHUNK, ACTION_DIM), 400.0))


def test_action_dim_debe_coincidir_con_el_normalizador() -> None:
    with pytest.raises(ValueError, match="no coincide"):
        Pi0Policy(
            _RecordingPi0Model(torch.zeros(1, CHUNK, ACTION_DIM)),
            Pi0Processor([CAMERA]),
            action_dim=ACTION_DIM,
            chunk_size=CHUNK,
            action_normalizer=_normalizer(2, 0.0, 1.0),
        )


def test_procesador_normaliza_la_propiocepcion() -> None:
    norm = _normalizer(7, centro=10.0, escala=2.0)
    obs = _observations(2)
    inputs = Pi0Processor([CAMERA], state_normalizer=norm)(obs)
    esperado = (torch.as_tensor(np.stack([o.proprio for o in obs])) - 10.0) / 2.0
    torch.testing.assert_close(inputs.tensors["state"], esperado)


def test_normalizadores_viajan_en_el_state_dict() -> None:
    acciones, estado = _normalizer(ACTION_DIM, 300.0, 100.0), _normalizer(7, 10.0, 2.0)
    policy = Pi0Policy(
        _RecordingPi0Model(torch.zeros(1, CHUNK, ACTION_DIM)),
        Pi0Processor([CAMERA], state_normalizer=estado),
        action_dim=ACTION_DIM,
        chunk_size=CHUNK,
        action_normalizer=acciones,
    )
    claves = set(policy.state_dict())
    assert any(k.startswith("action_normalizer.") for k in claves)
    assert any(k.startswith("state_normalizer.") for k in claves)


def test_save_pretrained_guarda_las_estadisticas(tmp_path) -> None:
    from vla_learner.builder import _resolve_normalizer
    from vla_learner.testing.fake_core import FakeCoreClient

    acciones, estado = _normalizer(ACTION_DIM, 300.0, 100.0), _normalizer(7, 10.0, 2.0)
    policy = Pi0Policy(
        _RecordingPi0Model(torch.zeros(1, CHUNK, ACTION_DIM)),
        Pi0Processor([CAMERA], state_normalizer=estado),
        action_dim=ACTION_DIM,
        chunk_size=CHUNK,
        action_normalizer=acciones,
    )
    policy.save_pretrained(tmp_path)
    # ``normalizer.stats: <directorio>`` recupera las mismas estadísticas.
    resuelto = _resolve_normalizer({"mode": "mean_std", "stats": str(tmp_path)}, FakeCoreClient())
    assert resuelto is not None
    x = torch.randn(2, CHUNK, ACTION_DIM)
    torch.testing.assert_close(
        resuelto["action"].normalize(x, "default"), acciones.normalize(x, "default")
    )
    s = torch.randn(2, 7)
    torch.testing.assert_close(
        resuelto["state"].normalize(s, "default"), estado.normalize(s, "default")
    )


# -- Tokenizer de PaliGemma ------------------------------------------------


class _RecordingTokenizer:
    """Doble del tokenizer: registra la llamada y devuelve ids de largo fijo."""

    def __init__(self) -> None:
        self.textos: list[str] = []
        self.kwargs: dict = {}

    def __call__(self, textos: list[str], **kwargs) -> dict[str, Tensor]:
        self.textos, self.kwargs = list(textos), kwargs
        largo = kwargs["max_length"]
        return {
            "input_ids": torch.ones(len(textos), largo, dtype=torch.long),
            "attention_mask": torch.ones(len(textos), largo, dtype=torch.long),
        }


def test_tokenizacion_replica_el_pipeline_de_lerobot() -> None:
    tokenizer = _RecordingTokenizer()
    inputs = Pi0Processor([CAMERA], tokenizer=tokenizer, max_lang_len=48)(_observations(2))
    # Pi0NewLineProcessor: PaliGemma espera la instrucción terminada en "\n".
    assert tokenizer.textos == ["pick up the red block\n"] * 2
    assert tokenizer.kwargs["padding"] == "max_length"
    assert tokenizer.kwargs["padding_side"] == "right"
    assert tokenizer.kwargs["max_length"] == 48 and tokenizer.kwargs["truncation"] is True
    assert inputs.tensors["lang_tokens"].shape == (2, 48)
    assert inputs.tensors["lang_masks"].dtype == torch.bool


def test_tokenizer_sin_acceso_da_un_error_claro(monkeypatch) -> None:
    import sys
    import types

    from vla_learner.policy.adapters.pi0 import load_paligemma_tokenizer

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(source: str):
            raise OSError("Cannot access gated repo")

    falso = types.ModuleType("transformers")
    falso.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", falso)
    with pytest.raises(RuntimeError, match="acceso restringido"):
        load_paligemma_tokenizer()
