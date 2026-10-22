"""``Pi0Policy``: adaptador del modelo **π0 completo** (pesos reales) a ``VLAPolicy``.

π0 (Physical Intelligence) es un VLA de *flow matching*: un backbone PaliGemma-3B
más un "action expert" que, por transporte de ruido, predice un **chunk** de
acciones continuas. Acá se integra la **versión completa descargada desde Hugging
Face** vía LeRobot (``lerobot/pi0_base`` y similares), como un bloque que no
conviene partir: se implementa ``VLAPolicy`` directo con un adaptador, igual que
``OpenVLAPolicy`` (``docs/learner/Learner Design.md``, "Modelos que no se pueden
separar"). Para experimentar en CPU con la *mecánica* de flow matching sin pesos,
está la ``FlowMatchingActionHead`` sobre ``ComposedPolicy``; esto de acá es el π0
**pesado** de verdad.

**Aislado y opcional.** Los pesos (~7 GB en bf16, PaliGemma-3B + expert) y
``lerobot`` viven en el grupo opcional de uv ``pi0``; ``lerobot`` se importa solo
dentro de :meth:`from_pretrained`. Así, importar este módulo —y usar cualquier
otro modelo— no requiere tener ``lerobot`` instalado ni interfiere con el resto.

**Capacidades: ninguna.** El flow matching no da una log-prob tratable, así que no
declara ``LOG_PROB`` ni ``ENTROPY`` (``evaluate_actions`` lanza
``NotImplementedError``) ni cabeza de valor: compatible con **Behavior Cloning**
(y un futuro ``RewardWeightedBC``), no con PPO/GRPO, que el Learner descarta al
chequear capacidades.

**Hardware.** No cabe en la RTX 2050 de desarrollo: se corre en el cluster. Los
tests de esta máquina inyectan un modelo falso diminuto (sin ``lerobot``, sin
pesos) para ejercitar la lógica del adaptador en CPU; la verificación con pesos
reales va marcada ``heavy`` y se corre en el cluster (ver ``examples/pi0_smoke.py``).

**Contrato del modelo envuelto.** ``Pi0Policy`` delega el flow matching en
``self.model`` (LeRobot en producción, un doble en los tests), que debe exponer:

- ``compute_loss(cond, actions) -> Tensor``: loss de flow matching (escalar);
  ``actions`` ``[B, H, D]`` en **unidades reales** (el modelo normaliza adentro).
- ``predict_chunk(cond) -> Tensor``: chunk de acciones ``[B, H, D]`` en unidades
  reales.

donde ``cond`` es el diccionario de tensores que arma :class:`Pi0Processor`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from vla_common import Observation
from vla_learner.policy.interfaces import Capability, ObservationProcessor, VLAPolicy
from vla_learner.types import ActionEvaluation, LossOutput, ModelInput, PolicyOutput


class Pi0Processor(ObservationProcessor):
    """Observaciones → ``ModelInput`` para π0 (imágenes, estado y lenguaje).

    Produce tensores:

    - ``image.<cam>`` ``[B, C, H, W]`` float en ``[0, 1]`` por cada cámara.
    - ``state`` ``[B, P]`` (propiocepción).
    - ``lang_tokens`` ``[B, L]`` y ``lang_masks`` ``[B, L]`` si hay tokenizador
      (PaliGemma, inyectado en el modelo real); si no (tests), un placeholder.

    El tokenizador se inyecta —en producción sale del checkpoint de LeRobot— para
    no acoplar el procesador a ``lerobot``.
    """

    def __init__(
        self,
        cameras: list[str] | tuple[str, ...] = ("primary",),
        *,
        tokenizer: Any | None = None,
        image_size: int | None = None,
        max_lang_len: int = 48,
    ) -> None:
        self._cameras = list(cameras)
        self._tokenizer = tokenizer
        self._image_size = image_size
        self._max_lang_len = max_lang_len

    @property
    def required_cameras(self) -> list[str]:
        return list(self._cameras)

    def _image(self, obs: Observation, camera: str) -> Tensor:
        if camera not in obs.images:
            raise ValueError(
                f"Falta la cámara requerida {camera!r} en la observación "
                f"(disponibles: {sorted(obs.images)})."
            )
        array = np.asarray(obs.images[camera], dtype=np.float32)
        if array.ndim == 3 and array.shape[-1] in (1, 3):
            array = np.transpose(array, (2, 0, 1))  # HWC → CHW
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if tensor.max() > 1.5:  # píxeles en 0..255 → 0..1
            tensor = tensor / 255.0
        if self._image_size is not None:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self._image_size, self._image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor

    def _language(self, instructions: list[str]) -> tuple[Tensor, Tensor]:
        if self._tokenizer is None:
            # Placeholder determinista para los tests (el modelo falso lo ignora).
            unos = torch.ones(len(instructions), 1, dtype=torch.long)
            return unos, unos.bool()
        encoded = self._tokenizer(
            instructions,
            padding="max_length",
            max_length=self._max_lang_len,
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"].bool()

    def __call__(self, observations: list[Observation]) -> ModelInput:
        tensors: dict[str, Tensor] = {}
        for camera in self._cameras:
            tensors[f"image.{camera}"] = torch.stack(
                [self._image(obs, camera) for obs in observations]
            )
        tensors["state"] = torch.stack(
            [torch.as_tensor(obs.proprio, dtype=torch.float32) for obs in observations]
        )
        lang_tokens, lang_masks = self._language([obs.instruction for obs in observations])
        tensors["lang_tokens"] = lang_tokens
        tensors["lang_masks"] = lang_masks
        return ModelInput(tensors=tensors, batch_size=len(observations))

    def to(self, device: torch.device) -> Pi0Processor:
        return self  # arma tensores en CPU; la política los mueve.


class Pi0Policy(VLAPolicy):
    """Adaptador de π0 a ``VLAPolicy`` (ver el módulo para el contexto).

    Se construye normalmente con :meth:`from_pretrained` (carga pesos reales desde
    Hugging Face vía LeRobot). El constructor recibe las piezas por inyección, para
    poder probarlo con un modelo falso diminuto sin descargar nada.
    """

    model: Any

    def __init__(
        self,
        model: nn.Module,
        processor: ObservationProcessor,
        *,
        action_dim: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self._action_dim = int(action_dim)
        self._chunk_size = int(chunk_size)

    # -- Propiedades de la interfaz -------------------------------------

    @property
    def capabilities(self) -> set[Capability]:
        return set()  # flow matching: sin log-prob tratable

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def _dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    # -- Fine-tuning parcial --------------------------------------------

    def freeze_except(self, trainable: list[str]) -> None:
        """Congela todo salvo los módulos cuyo nombre contenga un substring de
        ``trainable`` (fine-tuning parcial, p. ej. ``["action"]`` para entrenar
        solo el *action expert* y dejar PaliGemma congelado). ``trainable_parameters()``
        (heredado) devuelve entonces solo lo entrenable. Lanza ``ValueError`` si
        ningún parámetro coincide. Misma mecánica que ``OpenVLAPolicy.freeze_except``.
        """
        coincide = False
        for nombre, parametro in self.model.named_parameters():
            entrenar = any(patron in nombre for patron in trainable)
            parametro.requires_grad_(entrenar)
            coincide = coincide or entrenar
        if not coincide:
            nombres = (nombre for nombre, _ in self.model.named_parameters())
            disponibles = sorted({nombre.split(".")[0] for nombre in nombres})
            raise ValueError(
                f"Ningún parámetro coincide con trainable={trainable!r}; "
                f"módulos de nivel superior disponibles: {disponibles}"
            )

    # -- Interno ---------------------------------------------------------

    def _move_cond(self, inputs: ModelInput) -> dict[str, Tensor]:
        """Mueve el ``cond`` al device del modelo; los flotantes también al dtype."""
        moved: dict[str, Tensor] = {}
        for nombre, tensor in inputs.tensors.items():
            tensor = tensor.to(self._device)
            if tensor.is_floating_point():
                tensor = tensor.to(self._dtype)
            moved[nombre] = tensor
        return moved

    # -- Interfaz VLAPolicy ---------------------------------------------

    def act(self, inputs: ModelInput, deterministic: bool = False) -> PolicyOutput:
        with torch.no_grad():
            cond = self._move_cond(inputs)
            actions = self.model.predict_chunk(cond)
            actions = actions.reshape(inputs.batch_size, self._chunk_size, self._action_dim)
        # Flow matching: sin log-prob ni valor.
        return PolicyOutput(actions=actions, log_probs=None, values=None, extras={})

    def evaluate_actions(
        self,
        inputs: ModelInput,
        actions: Tensor,
        extras: dict[str, Tensor] | None = None,
    ) -> ActionEvaluation:
        raise NotImplementedError(
            "Pi0Policy usa flow matching: no expone log-prob (capabilities no incluye "
            "LOG_PROB). Entrená con Behavior Cloning; PPO/GRPO no son compatibles."
        )

    def imitation_loss(self, inputs: ModelInput, actions: Tensor) -> LossOutput:
        cond = self._move_cond(inputs)
        loss = self.model.compute_loss(cond, actions.to(self._device, self._dtype))
        return LossOutput(loss=loss, metrics={"fm_mse": float(loss.detach())})

    # -- Persistencia ----------------------------------------------------

    def save_pretrained(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # El modelo real (LeRobot) sabe guardarse; el doble de prueba puede no hacerlo.
        guardar = getattr(self.model, "save_pretrained", None)
        if callable(guardar):
            guardar(directory)
        meta = {
            "adapter": "pi0",
            "action_dim": self._action_dim,
            "chunk_size": self._chunk_size,
            "cameras": self.processor.required_cameras,
        }
        (directory / "pi0_adapter.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path = "lerobot/pi0_base",
        *,
        cameras: list[str] | None = None,
        image_size: int = 224,
        torch_dtype: Any = None,
        device_map: Any = None,
    ) -> Pi0Policy:
        """Carga el **π0 completo** desde Hugging Face vía LeRobot (grupo opcional ``pi0``).

        ``lerobot`` se importa solo acá. ``directory`` es un id de HF (p. ej.
        ``lerobot/pi0_base``) o una ruta local. **Cluster-only:** no cabe en la GPU
        de desarrollo y no se ejercita en los tests ``fast``.

        El envoltorio :class:`_LeRobotPi0Model` apunta a la API de LeRobot de la
        versión fijada en el grupo ``pi0`` del ``pyproject``; si tu versión difiere,
        es el único punto a ajustar (igual que el ``trust_remote_code`` de OpenVLA).
        """
        model = _LeRobotPi0Model.from_pretrained(directory, torch_dtype=torch_dtype)
        processor = Pi0Processor(
            cameras or model.cameras,
            tokenizer=model.tokenizer,
            image_size=image_size,
        )
        return cls(
            model,
            processor,
            action_dim=model.action_dim,
            chunk_size=model.chunk_size,
        )


class _LeRobotPi0Model(nn.Module):
    """Envoltorio del ``PI0Policy`` de LeRobot al contrato que usa ``Pi0Policy``.

    Aísla acá todo lo específico de ``lerobot`` (nombres de clase, claves del batch,
    API interna de flow matching), que cambia entre versiones. Apunta a la versión
    fijada en el grupo opcional ``pi0``; ``lerobot`` se importa perezosamente. **No
    se corre en la máquina de desarrollo** (necesita pesos y GPU): su verificación
    va en el cluster, marcada ``heavy``. Si la API de tu ``lerobot`` difiere, este
    es el único lugar a tocar.
    """

    def __init__(self, lerobot_policy: Any) -> None:
        super().__init__()
        self.lerobot = lerobot_policy
        config = lerobot_policy.config
        # Nombres habituales en la config de π0 de LeRobot; con respaldos razonables.
        self.action_dim = int(getattr(config, "max_action_dim", getattr(config, "action_dim", 7)))
        self.chunk_size = int(getattr(config, "n_action_steps", getattr(config, "chunk_size", 50)))
        self.tokenizer = getattr(lerobot_policy, "language_tokenizer", None)
        cams = getattr(config, "image_features", None) or getattr(config, "cameras", None)
        self.cameras = list(cams) if cams else ["primary"]

    @classmethod
    def from_pretrained(cls, source: str | Path, *, torch_dtype: Any = None) -> _LeRobotPi0Model:
        try:
            # Layout de lerobot 0.4.x (el fijado en el grupo opcional ``pi0``).
            from lerobot.common.policies.pi0.modeling_pi0 import (  # pyright: ignore[reportMissingImports]
                PI0Policy,
            )
        except ImportError:  # pragma: no cover - layout de versiones más nuevas
            from lerobot.policies.pi0.modeling_pi0 import (  # pyright: ignore[reportMissingImports]
                PI0Policy,
            )

        lerobot_policy = PI0Policy.from_pretrained(str(source))
        if torch_dtype is not None:
            lerobot_policy = lerobot_policy.to(dtype=torch_dtype)
        lerobot_policy.eval()
        return cls(lerobot_policy)

    def _batch(self, cond: dict[str, Tensor], actions: Tensor | None) -> dict[str, Any]:
        """Arma el batch que espera ``PI0Policy`` de LeRobot a partir del ``cond``."""
        batch: dict[str, Any] = {k: v for k, v in cond.items() if k.startswith("image.")}
        batch["observation.state"] = cond["state"]
        batch["observation.language_tokens"] = cond["lang_tokens"]
        batch["observation.language_attention_mask"] = cond["lang_masks"]
        if actions is not None:
            batch["action"] = actions
        return batch

    def compute_loss(self, cond: dict[str, Tensor], actions: Tensor) -> Tensor:
        salida: Any = self.lerobot.forward(self._batch(cond, actions))
        # ``forward`` devuelve (loss, dict) o un dict con "loss" según la versión.
        if isinstance(salida, tuple):
            salida = salida[0]
        if isinstance(salida, dict):
            salida = salida.get("loss", next(iter(salida.values())))
        return salida

    def predict_chunk(self, cond: dict[str, Tensor]) -> Tensor:
        predecir = getattr(self.lerobot, "predict_action_chunk", None)
        if callable(predecir):
            salida: Any = predecir(self._batch(cond, None))
        else:
            # Respaldo: una sola acción si la versión no expone el chunk completo.
            salida = self.lerobot.select_action(self._batch(cond, None))
        return salida
