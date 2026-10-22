"""``OpenVLAPolicy``: adaptador del modelo OpenVLA-7B a la interfaz ``VLAPolicy``.

OpenVLA (https://huggingface.co/openvla/openvla-7b) es un VLA autoregresivo
(Llama-2 7B + SigLIP/DINOv2) que predice **una** acción de 7 dimensiones por
paso, discretizando cada dimensión en 256 bins mapeados a los últimos 256 ids
del vocabulario de Llama. No se puede ver como "backbone + cabeza", así que se
implementa ``VLAPolicy`` directo (``docs/learner/Learner Design.md``, sección "Modelos
que no se pueden separar": se anticipaba justamente ``OpenVLAPolicy``).

Capacidades: ``LOG_PROB`` y ``ENTROPY``. **No** tiene cabeza de valor, así que es
compatible con BC y GRPO, no con PPO (que necesita ``VALUE``).

**Normalización dentro de la política.** Como fija ``docs/learner/Learner Design.md``
(sección "Normalización de acciones"), el resto del harness trabaja en unidades
reales; OpenVLA normaliza cada dimensión a ``[-1, 1]`` con las estadísticas
``q01``/``q99`` del dataset elegido (``unnorm_key``) y una máscara que indica qué
dimensiones se normalizan (la pinza suele pasar sin tocar). Esa normalización y
la discretización en tokens viven acá adentro.

**Hardware.** Los pesos reales (~16 GB en bf16) no caben en la RTX 2050 de
desarrollo; este adaptador está pensado para correrse en un cluster. Los tests
de esta máquina inyectan un modelo falso diminuto (sin descargar pesos) para
ejercitar la lógica de tokens, log-probs y loss en CPU. ``transformers`` se
importa solo al cargar pesos reales (``from_pretrained``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vla_common import Observation
from vla_learner.policy.interfaces import Capability, ObservationProcessor, VLAPolicy
from vla_learner.types import ActionEvaluation, LossOutput, ModelInput, PolicyOutput

# Plantilla de prompt oficial de OpenVLA; ``{instruction}`` se reemplaza por la tarea.
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


class OpenVLAProcessor(ObservationProcessor):
    """Convierte observaciones en el ``ModelInput`` que OpenVLA espera.

    Envuelve el ``AutoProcessor`` de Hugging Face (el que trae el tokenizador de
    Llama y el extractor de imágenes). Produce un ``ModelInput`` con tensores
    ``input_ids`` ``[B, L]``, ``attention_mask`` ``[B, L]`` y ``pixel_values``
    ``[B, ...]``. El tokenizador usa *padding* a la izquierda para que, al pegar
    los tokens de acción al final (en entrenamiento), queden alineados a la
    derecha en todas las filas del batch.
    """

    def __init__(
        self,
        hf_processor: Any,
        *,
        camera: str = "primary",
        prompt_template: str = PROMPT_TEMPLATE,
    ) -> None:
        self._hf = hf_processor
        self._camera = camera
        self._prompt_template = prompt_template
        # OpenVLA genera los tokens de acción al final de la secuencia; el padding
        # a la izquierda mantiene esa alineación en batches de prompts de distinto largo.
        tokenizer = getattr(hf_processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

    @property
    def required_cameras(self) -> list[str]:
        return [self._camera]

    def _image(self, obs: Observation) -> Any:
        if self._camera not in obs.images:
            raise ValueError(
                f"Falta la cámara requerida {self._camera!r} en la observación "
                f"(disponibles: {sorted(obs.images)})."
            )
        from PIL import Image

        array = np.asarray(obs.images[self._camera])
        # OpenVLA espera imágenes RGB uint8 [H, W, 3]; convertimos si hace falta.
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    def __call__(self, observations: list[Observation]) -> ModelInput:
        prompts = [
            self._prompt_template.format(instruction=obs.instruction) for obs in observations
        ]
        images = [self._image(obs) for obs in observations]
        encoded = self._hf(text=prompts, images=images, return_tensors="pt", padding=True)
        tensors = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "pixel_values": encoded["pixel_values"],
        }
        return ModelInput(tensors=tensors, batch_size=len(observations))

    def to(self, device: torch.device) -> OpenVLAProcessor:
        return self  # el procesador de HF arma tensores en CPU; la política los mueve.


class OpenVLAPolicy(VLAPolicy):
    """Adaptador de OpenVLA a ``VLAPolicy``.

    Se construye normalmente con :meth:`from_pretrained` (carga pesos reales). El
    constructor recibe las piezas ya armadas por inyección de dependencias, para
    poder probarlo con un modelo falso diminuto sin descargar nada.

    Parámetros
    ----------
    vla:
        Modelo de Hugging Face (``OpenVLAForActionPrediction``) o un doble de
        prueba. Debe exponer ``forward(input_ids, attention_mask, pixel_values)``
        devolviendo un objeto con ``.logits`` ``[B, L, V]`` y ``generate(...)``.
    hf_processor:
        ``AutoProcessor`` de OpenVLA (o un doble). Debe tener ``.tokenizer``.
    unnorm_key:
        Clave de las estadísticas de desnormalización (p. ej. ``"bridge_orig"``).
    q01, q99, mask:
        Estadísticas de normalización por dimensión. Si no se pasan, se leen de
        ``vla.norm_stats[unnorm_key]["action"]``.
    """

    # Anotaciones a nivel de clase: el modelo de HF es dinámico (``Any``), y los
    # buffers se declaran ``Tensor`` para que pyright no los infiera como
    # ``Tensor | Module`` vía ``nn.Module.__getattr__``.
    vla: Any
    _q01: Tensor
    _q99: Tensor
    _mask: Tensor
    _bins: Tensor
    _bin_centers: Tensor

    def __init__(
        self,
        vla: nn.Module,
        hf_processor: Any,
        *,
        unnorm_key: str,
        q01: np.ndarray | None = None,
        q99: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        n_bins: int = 256,
        llm_vocab_size: int | None = None,
        camera: str = "primary",
        prompt_template: str = PROMPT_TEMPLATE,
    ) -> None:
        super().__init__()
        self.vla = vla
        self._hf_processor = hf_processor
        self.processor: ObservationProcessor = OpenVLAProcessor(
            hf_processor, camera=camera, prompt_template=prompt_template
        )
        self._unnorm_key = unnorm_key
        self._prompt_template = prompt_template
        self._camera = camera

        if q01 is None or q99 is None:
            stats = _read_norm_stats(vla, unnorm_key)
            q01 = np.asarray(stats["q01"], dtype=np.float32)
            q99 = np.asarray(stats["q99"], dtype=np.float32)
            if mask is None:
                mask = np.asarray(stats.get("mask", np.ones_like(q01, dtype=bool)), dtype=bool)
        q01 = np.asarray(q01, dtype=np.float32)
        q99 = np.asarray(q99, dtype=np.float32)
        if mask is None:
            mask = np.ones_like(q01, dtype=bool)
        mask = np.asarray(mask, dtype=bool)

        self._action_dim = int(q01.shape[0])
        self._chunk_size = 1  # OpenVLA predice una acción por paso.
        self._n_bins = int(n_bins)
        self._llm_vocab_size = int(
            llm_vocab_size if llm_vocab_size is not None else hf_processor.tokenizer.vocab_size
        )

        # Buffers de normalización/discretización (se mueven con ``.to(device)``).
        self.register_buffer("_q01", torch.tensor(q01))
        self.register_buffer("_q99", torch.tensor(q99))
        self.register_buffer("_mask", torch.tensor(mask))
        # ``bins``: 256 bordes en [-1, 1]; ``bin_centers``: 255 puntos medios.
        bins = torch.linspace(-1.0, 1.0, self._n_bins)
        self.register_buffer("_bins", bins)
        self.register_buffer("_bin_centers", (bins[:-1] + bins[1:]) / 2.0)

    # -- Fine-tuning parcial --------------------------------------------

    def freeze_except(self, trainable: list[str]) -> None:
        """Congela todos los pesos salvo los que entrenamos (fine-tuning parcial).

        Deja ``requires_grad=True`` solo en los parámetros cuyo nombre calificado
        contenga alguno de los substrings de ``trainable`` (p. ej. ``["lm_head"]``
        para entrenar únicamente la proyección de salida a tokens de acción). El
        resto queda congelado. Como fija ``docs/learner/Learner Design.md``
        ("Congelar y LoRA"), esto se decide al construir la política: ni el
        algoritmo ni el optimizador se enteran, porque ``trainable_parameters()``
        (heredado) ya devuelve solo lo que tiene ``requires_grad``.

        Entrenar una parte chica recorta la memoria de gradientes y del estado del
        optimizador; los pesos congelados igual se cargan, así que OpenVLA-7B sigue
        necesitando el cluster. Lanza ``ValueError`` si ningún parámetro coincide,
        para que un patrón mal escrito no deje la política entera congelada.
        """
        coincide = False
        for nombre, parametro in self.vla.named_parameters():
            entrenar = any(patron in nombre for patron in trainable)
            parametro.requires_grad_(entrenar)
            coincide = coincide or entrenar
        if not coincide:
            nombres = (nombre for nombre, _ in self.vla.named_parameters())
            disponibles = sorted({nombre.split(".")[0] for nombre in nombres})
            raise ValueError(
                f"Ningún parámetro coincide con trainable={trainable!r}; "
                f"módulos de nivel superior disponibles: {disponibles}"
            )

    # -- Propiedades de la interfaz -------------------------------------

    @property
    def capabilities(self) -> set[Capability]:
        return {Capability.LOG_PROB, Capability.ENTROPY}

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def _n_action_tokens(self) -> int:
        """Cantidad de tokens de acción por muestra: ``chunk_size * action_dim``."""
        return self._chunk_size * self._action_dim

    @property
    def _device(self) -> torch.device:
        return next(self.vla.parameters()).device

    @property
    def _dtype(self) -> torch.dtype:
        return next(self.vla.parameters()).dtype

    # -- Normalización y tokenización (unidades reales <-> tokens) -------

    def _normalize(self, actions: Tensor) -> Tensor:
        """Acciones reales ``[..., D]`` -> normalizadas a ``[-1, 1]`` (según la máscara)."""
        q01, q99, mask = self._q01, self._q99, self._mask
        norm = 2.0 * (actions - q01) / (q99 - q01) - 1.0
        norm = norm.clamp(-1.0, 1.0)
        return torch.where(mask, norm, actions)

    def _unnormalize(self, norm: Tensor) -> Tensor:
        """Normalizadas ``[-1, 1]`` -> acciones reales ``[..., D]`` (según la máscara)."""
        q01, q99, mask = self._q01, self._q99, self._mask
        real = 0.5 * (norm + 1.0) * (q99 - q01) + q01
        return torch.where(mask, real, norm)

    def _to_token_ids(self, norm: Tensor) -> Tensor:
        """Normalizadas ``[..., D]`` -> ids de token ``[..., D]`` (como OpenVLA).

        Replica ``np.digitize`` con ``right=False`` vía ``bucketize(right=True)``:
        ``idx`` cae en ``[1, n_bins]`` y ``token_id = vocab_size - idx``, que ocupa
        los últimos ``n_bins`` ids del vocabulario.
        """
        clipped = norm.clamp(-1.0, 1.0)
        idx = torch.bucketize(clipped, self._bins, right=True).clamp(1, self._n_bins)
        return self._llm_vocab_size - idx

    def _from_token_ids(self, token_ids: Tensor) -> Tensor:
        """Ids de token ``[..., D]`` -> normalizadas ``[..., D]`` (centro del bin)."""
        idx = self._llm_vocab_size - token_ids
        center = (idx - 1).clamp(0, self._bin_centers.shape[0] - 1)
        return self._bin_centers[center]

    # -- Paso por el modelo ---------------------------------------------

    def _move_inputs(self, inputs: ModelInput) -> dict[str, Tensor]:
        """Mueve los tensores del ``ModelInput`` al device (y dtype) del modelo."""
        tensors = inputs.tensors
        return {
            "input_ids": tensors["input_ids"].to(self._device, dtype=torch.long),
            "attention_mask": tensors["attention_mask"].to(self._device, dtype=torch.long),
            "pixel_values": tensors["pixel_values"].to(self._device, dtype=self._dtype),
        }

    def _action_logits(self, moved: dict[str, Tensor], token_ids: Tensor) -> Tensor:
        """Logits que predicen cada token de acción, ``[B, n_action, V]`` (float32).

        Arma la secuencia completa ``[prompt, tokens de acción]`` y corre el
        modelo. Como los tokens de acción son los últimos de la secuencia, sus
        predicciones son las últimas ``n_action`` posiciones de ``logits`` (un
        desfase de 1), **sin depender** de cuántos tokens de visión inserte el
        modelo al principio.
        """
        input_ids = torch.cat([moved["input_ids"], token_ids], dim=1)
        ones = torch.ones_like(token_ids)
        attention_mask = torch.cat([moved["attention_mask"], ones], dim=1)
        output = self.vla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=moved["pixel_values"],
        )
        n = self._n_action_tokens
        logits = output.logits[:, -n - 1 : -1, :]
        return logits.float()

    # -- Interfaz VLAPolicy ---------------------------------------------

    def act(self, inputs: ModelInput, deterministic: bool = False) -> PolicyOutput:
        with torch.no_grad():
            moved = self._move_inputs(inputs)
            generated = self.vla.generate(
                input_ids=moved["input_ids"],
                attention_mask=moved["attention_mask"],
                pixel_values=moved["pixel_values"],
                max_new_tokens=self._n_action_tokens,
                do_sample=not deterministic,
            )
            # Los tokens de acción son los recién generados, al final de la secuencia.
            token_ids = generated[:, -self._n_action_tokens :]
            batch = token_ids.shape[0]
            norm = self._from_token_ids(token_ids)
            actions = self._unnormalize(norm).view(batch, self._chunk_size, self._action_dim)
            log_probs = self._score(moved, token_ids)
        return PolicyOutput(
            actions=actions,
            log_probs=log_probs,
            values=None,
            extras={"action_token_ids": token_ids},
        )

    def _score(self, moved: dict[str, Tensor], token_ids: Tensor) -> Tensor:
        """Log-prob del chunk completo ``[B]``: suma de log-probs de sus tokens."""
        logits = self._action_logits(moved, token_ids)
        log_probs = F.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        return gathered.sum(dim=-1)

    def evaluate_actions(
        self,
        inputs: ModelInput,
        actions: Tensor,
        extras: dict[str, Tensor] | None = None,
    ) -> ActionEvaluation:
        moved = self._move_inputs(inputs)
        if extras is not None and "action_token_ids" in extras:
            # Reusa los ids elegidos al actuar: evita el redondeo de re-tokenizar
            # (misma idea que ``extras`` en ``docs/learner/Learner Design.md``).
            token_ids = extras["action_token_ids"].to(self._device, dtype=torch.long)
        else:
            norm = self._normalize(actions.to(self._device))
            token_ids = self._to_token_ids(norm).reshape(actions.shape[0], -1)

        logits = self._action_logits(moved, token_ids)
        log_probs_all = F.log_softmax(logits, dim=-1)
        gathered = log_probs_all.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        log_probs = gathered.sum(dim=-1)
        # Entropía de la distribución de cada token de acción, sumada en el chunk.
        probs = log_probs_all.exp()
        entropy = -(probs * log_probs_all).sum(dim=-1).sum(dim=-1)
        return ActionEvaluation(log_probs=log_probs, entropy=entropy, values=None)

    def imitation_loss(self, inputs: ModelInput, actions: Tensor) -> LossOutput:
        moved = self._move_inputs(inputs)
        norm = self._normalize(actions.to(self._device))
        token_ids = self._to_token_ids(norm).reshape(actions.shape[0], -1)
        logits = self._action_logits(moved, token_ids)
        vocab = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(-1, vocab), token_ids.reshape(-1))
        # Precisión de tokens: cuántas dimensiones cae en el bin correcto.
        with torch.no_grad():
            accuracy = (logits.argmax(dim=-1) == token_ids).float().mean()
        return LossOutput(
            loss=loss,
            metrics={"ce": float(loss.detach()), "token_accuracy": float(accuracy)},
        )

    # -- Persistencia ----------------------------------------------------

    def save_pretrained(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.vla.save_pretrained(directory)
        self._hf_processor.save_pretrained(directory)
        meta = {
            "adapter": "openvla",
            "unnorm_key": self._unnorm_key,
            "camera": self._camera,
            "prompt_template": self._prompt_template,
            "n_bins": self._n_bins,
            "llm_vocab_size": self._llm_vocab_size,
            "q01": self._q01.tolist(),
            "q99": self._q99.tolist(),
            "mask": self._mask.tolist(),
        }
        (directory / "openvla_adapter.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        unnorm_key: str | None = None,
        camera: str = "primary",
        attn_implementation: str = "flash_attention_2",
        torch_dtype: Any = None,
        device_map: Any = None,
        trust_remote_code: bool = True,
    ) -> OpenVLAPolicy:
        """Carga pesos reales de OpenVLA desde Hugging Face o un directorio local.

        ``transformers``, ``timm`` y ``tokenizers`` deben estar instalados (grupo
        opcional ``openvla``). Si ``flash_attn`` no está disponible en el cluster,
        pasá ``attn_implementation="eager"``.
        """
        from transformers import AutoModelForVision2Seq, AutoProcessor

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        source = str(directory)
        hf_processor = AutoProcessor.from_pretrained(source, trust_remote_code=trust_remote_code)
        vla = AutoModelForVision2Seq.from_pretrained(
            source,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )

        meta_path = Path(source) / "openvla_adapter.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            unnorm_key = str(unnorm_key or meta["unnorm_key"])
            return cls(
                vla,
                hf_processor,
                unnorm_key=unnorm_key,
                q01=np.asarray(meta["q01"], dtype=np.float32),
                q99=np.asarray(meta["q99"], dtype=np.float32),
                mask=np.asarray(meta["mask"], dtype=bool),
                n_bins=meta["n_bins"],
                llm_vocab_size=meta["llm_vocab_size"],
                camera=meta.get("camera", camera),
                prompt_template=meta.get("prompt_template", PROMPT_TEMPLATE),
            )

        if unnorm_key is None:
            unnorm_key = _default_unnorm_key(vla)
        return cls(vla, hf_processor, unnorm_key=unnorm_key, camera=camera)


def _read_norm_stats(vla: Any, unnorm_key: str) -> dict[str, Any]:
    """Lee ``vla.norm_stats[unnorm_key]['action']`` con un error claro si falta."""
    norm_stats = getattr(vla, "norm_stats", None)
    if not norm_stats:
        raise ValueError(
            "El modelo no expone 'norm_stats'; pasá q01/q99/mask explícitos al construir "
            "OpenVLAPolicy."
        )
    if unnorm_key not in norm_stats:
        raise ValueError(
            f"unnorm_key {unnorm_key!r} no está en norm_stats. Disponibles: {sorted(norm_stats)}."
        )
    return norm_stats[unnorm_key]["action"]


def _default_unnorm_key(vla: Any) -> str:
    """Si hay una sola clave de normalización, la usa; si no, exige elegir."""
    norm_stats = getattr(vla, "norm_stats", {}) or {}
    if len(norm_stats) == 1:
        return next(iter(norm_stats))
    raise ValueError(
        "El modelo tiene varias claves de normalización; indicá unnorm_key. "
        f"Disponibles: {sorted(norm_stats)}."
    )
