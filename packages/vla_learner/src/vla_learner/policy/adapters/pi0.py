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

**Normalización (dentro de la política).** En LeRobot 0.4.4 el ``PI0Policy`` no
normaliza: eso lo hace su pipeline de procesadores (``make_pi0_pre_post_processors``),
que este adaptador no usa. Por eso la normalización la hace el adaptador, como pide
``Learner Design.md`` ("Normalización de acciones"): las acciones con un
``ActionNormalizer`` propio (normaliza antes de la loss y desnormaliza en ``act``) y
la propiocepción en :class:`Pi0Processor`. Mismo modo que LeRobot para π0:
``mean_std`` en estado y acción. ``pi0_base`` no trae estadísticas del dataset, así
que salen del config (``normalizer.stats``: ``fit`` o una ruta; ver el builder).
Ambos normalizadores son submódulos, así que sus buffers viajan en el ``state_dict``.

**Contrato del modelo envuelto.** ``Pi0Policy`` delega el flow matching en
``self.model`` (LeRobot en producción, un doble en los tests), que debe exponer:

- ``compute_loss(cond, actions) -> Tensor``: loss de flow matching (escalar);
  ``actions`` ``[B, H, D]`` **normalizadas** (el adaptador ya las normalizó).
- ``predict_chunk(cond) -> Tensor``: chunk ``[B, H, D']`` normalizado, con
  ``D' >= D`` (π0 devuelve las ``max_action_dim`` dims padeadas; el adaptador se
  queda con las ``D`` reales y desnormaliza).

donde ``cond`` es el diccionario de tensores que arma :class:`Pi0Processor`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from vla_common import Observation
from vla_learner.policy.interfaces import Capability, ObservationProcessor, VLAPolicy
from vla_learner.policy.normalizer import ActionNormalizer
from vla_learner.types import ActionEvaluation, LossOutput, ModelInput, PolicyOutput

# Tokenizer que usa π0 en LeRobot 0.4.4 (``processor_pi0.py``). Es un repo de HF
# con acceso restringido: hay que aceptar la licencia y autenticarse (o pasar una
# ruta local con el tokenizer ya descargado).
PALIGEMMA_TOKENIZER = "google/paligemma-3b-pt-224"

# Archivos de estadísticas que escribe ``save_pretrained`` (y que el builder lee
# con ``normalizer.stats: <directorio>``).
ACTION_STATS_FILE = "action_stats.pt"
STATE_STATS_FILE = "state_stats.pt"


def load_paligemma_tokenizer(source: str = PALIGEMMA_TOKENIZER) -> Any:
    """Carga el tokenizer de PaliGemma (id de HF o ruta local).

    ``transformers`` se importa acá (viene con el grupo opcional ``pi0``). Si falla
    —típicamente porque el repo es de acceso restringido y no hay sesión de HF—, se
    relanza con qué hacer, en vez de caer al placeholder de los tests: sin el
    tokenizer real, π0 entrenaría sin ver la instrucción.
    """
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    try:
        return AutoTokenizer.from_pretrained(source)
    except OSError as error:
        raise RuntimeError(
            f"No se pudo cargar el tokenizer de PaliGemma {source!r}. Es un repo de HF con "
            "acceso restringido: aceptá la licencia en https://huggingface.co/"
            f"{PALIGEMMA_TOKENIZER} y autenticate (`hf auth login` o HF_TOKEN), o pasá "
            "en `policy.tokenizer` una ruta local con el tokenizer descargado."
        ) from error


class Pi0Processor(ObservationProcessor):
    """Observaciones → ``ModelInput`` para π0 (imágenes, estado y lenguaje).

    Produce tensores:

    - ``image.<cam>`` ``[B, C, H, W]`` float en ``[0, 1]`` por cada cámara.
    - ``state`` ``[B, P]`` (propiocepción), **normalizada** con ``state_normalizer``
      si se da (``Learner Design.md``: la propiocepción se normaliza en el procesador).
    - ``lang_tokens`` ``[B, L]`` y ``lang_masks`` ``[B, L]`` si hay tokenizador
      (PaliGemma, inyectado en el modelo real); si no (tests), un placeholder.

    El tokenizador y el normalizador se inyectan —en producción los arma
    :meth:`Pi0Policy.from_pretrained`— para no acoplar el procesador a ``lerobot``.
    La tokenización replica el pipeline de LeRobot 0.4.4 (``processor_pi0.py``):
    la instrucción termina en ``"\\n"`` (``Pi0NewLineProcessor``, lo exige
    PaliGemma) y se rellena a la derecha hasta ``max_lang_len``.
    """

    def __init__(
        self,
        cameras: list[str] | tuple[str, ...] = ("primary",),
        *,
        tokenizer: Any | None = None,
        image_size: int | None = None,
        max_lang_len: int = 48,
        state_normalizer: ActionNormalizer | None = None,
    ) -> None:
        self._cameras = list(cameras)
        self._tokenizer = tokenizer
        self._image_size = image_size
        self._max_lang_len = max_lang_len
        self.state_normalizer = state_normalizer

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
            [texto if texto.endswith("\n") else f"{texto}\n" for texto in instructions],
            padding="max_length",
            max_length=self._max_lang_len,
            truncation=True,
            padding_side="right",
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"].bool()

    def __call__(self, observations: list[Observation]) -> ModelInput:
        tensors: dict[str, Tensor] = {}
        for camera in self._cameras:
            tensors[f"image.{camera}"] = torch.stack(
                [self._image(obs, camera) for obs in observations]
            )
        state = torch.stack(
            [torch.as_tensor(obs.proprio, dtype=torch.float32) for obs in observations]
        )
        if self.state_normalizer is not None:
            state = self.state_normalizer.normalize(
                state, self.state_normalizer.default_embodiment()
            )
        tensors["state"] = state
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

    Sin ``action_normalizer`` las acciones pasan sin tocar (identidad): sirve para
    los tests, pero π0 entrenaría en unidades crudas. Con uno, ``action_dim`` debe
    coincidir con su dimensión. El normalizador de estado lo usa el procesador; la
    política solo lo registra como submódulo para que se guarde con los pesos.
    """

    model: Any

    def __init__(
        self,
        model: nn.Module,
        processor: ObservationProcessor,
        *,
        action_dim: int,
        chunk_size: int,
        action_normalizer: ActionNormalizer | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self._action_dim = int(action_dim)
        self._chunk_size = int(chunk_size)
        if action_normalizer is not None and action_normalizer.dim() != self._action_dim:
            raise ValueError(
                f"action_dim={self._action_dim} no coincide con la dimensión del "
                f"normalizador de acciones ({action_normalizer.dim()})."
            )
        self.action_normalizer = action_normalizer
        self.state_normalizer = getattr(processor, "state_normalizer", None)

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
        """Mueve el ``cond`` al device del modelo (NO castea el dtype).

        π0 tiene precisión MIXTA (PaliGemma bf16, action expert y proyecciones de flow
        matching float32), así que no hay un dtype único al que castear: se alimentan
        los tensores en float32 (como un dataloader de LeRobot) y el modelo castea
        internamente lo que cada submódulo necesita.
        """
        return {nombre: tensor.to(self._device) for nombre, tensor in inputs.tensors.items()}

    def _normalize(self, actions: Tensor) -> Tensor:
        if self.action_normalizer is None:
            return actions
        return self.action_normalizer.normalize(
            actions, self.action_normalizer.default_embodiment()
        )

    def _denormalize(self, actions: Tensor) -> Tensor:
        if self.action_normalizer is None:
            return actions
        return self.action_normalizer.denormalize(
            actions, self.action_normalizer.default_embodiment()
        )

    # -- Interfaz VLAPolicy ---------------------------------------------

    def act(self, inputs: ModelInput, deterministic: bool = False) -> PolicyOutput:
        with torch.no_grad():
            cond = self._move_cond(inputs)
            salida = self.model.predict_chunk(cond)
            if salida.shape[-1] < self._action_dim:
                raise ValueError(
                    f"El modelo predijo {salida.shape[-1]} dims de acción; se esperaban "
                    f"al menos action_dim={self._action_dim}."
                )
            # π0 predice las dims padeadas a ``max_action_dim``: se queda con las reales.
            actions = salida[..., : self._action_dim].reshape(
                inputs.batch_size, self._chunk_size, self._action_dim
            )
            actions = self._denormalize(actions.float())
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
        # Se mueve solo al device; el dtype queda en float32 (ver ``_move_cond``): el
        # flow matching de π0 mezcla estas acciones con ruido float32. Las acciones
        # llegan en unidades reales y se normalizan acá, como en ``act``.
        loss = self.model.compute_loss(cond, self._normalize(actions.to(self._device)))
        return LossOutput(loss=loss, metrics={"fm_mse": float(loss.detach())})

    # -- Persistencia ----------------------------------------------------

    def save_pretrained(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # El modelo real (LeRobot) sabe guardarse; el doble de prueba puede no hacerlo.
        guardar = getattr(self.model, "save_pretrained", None)
        if callable(guardar):
            guardar(directory)
        # Estadísticas aparte de los pesos de LeRobot, para poder reusarlas con
        # ``normalizer.stats: <este directorio>`` sin cargar el checkpoint entero.
        if self.action_normalizer is not None:
            self.action_normalizer.save_file(directory / ACTION_STATS_FILE)
        if self.state_normalizer is not None:
            self.state_normalizer.save_file(directory / STATE_STATS_FILE)
        meta = {
            "adapter": "pi0",
            "action_dim": self._action_dim,
            "chunk_size": self._chunk_size,
            "cameras": self.processor.required_cameras,
            "action_normalizer": getattr(self.action_normalizer, "mode", None),
            "state_normalizer": getattr(self.state_normalizer, "mode", None),
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
        tokenizer: str = PALIGEMMA_TOKENIZER,
        action_normalizer: ActionNormalizer | None = None,
        state_normalizer: ActionNormalizer | None = None,
    ) -> Pi0Policy:
        """Carga el **π0 completo** desde Hugging Face vía LeRobot (grupo opcional ``pi0``).

        ``lerobot`` se importa solo acá. ``directory`` es un id de HF (p. ej.
        ``lerobot/pi0_base``) o una ruta local. **Cluster-only:** no cabe en la GPU
        de desarrollo y no se ejercita en los tests ``fast``.

        ``tokenizer`` es el id de HF o la ruta local del tokenizer de PaliGemma.
        ``action_normalizer``/``state_normalizer`` traen las estadísticas del dataset
        (las arma el builder con ``normalizer.stats``); con ``action_normalizer``,
        ``action_dim`` pasa a ser el del dataset (p. ej. 2 en pusht), no las
        ``max_action_dim`` (32) padeadas de π0.

        El envoltorio :class:`_LeRobotPi0Model` apunta a la API de LeRobot de la
        versión fijada en el grupo ``pi0`` del ``pyproject``; si tu versión difiere,
        es el único punto a ajustar (igual que el ``trust_remote_code`` de OpenVLA).
        """
        model = _LeRobotPi0Model.from_pretrained(
            directory, torch_dtype=torch_dtype, tokenizer=tokenizer
        )
        processor = Pi0Processor(
            cameras or model.cameras,
            tokenizer=model.tokenizer,
            image_size=image_size,
            max_lang_len=model.tokenizer_max_length,
            state_normalizer=state_normalizer,
        )
        action_dim = action_normalizer.dim() if action_normalizer is not None else model.action_dim
        return cls(
            model,
            processor,
            action_dim=action_dim,
            chunk_size=model.chunk_size,
            action_normalizer=action_normalizer,
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

    def __init__(self, lerobot_policy: Any, tokenizer: Any) -> None:
        super().__init__()
        self.lerobot = lerobot_policy
        config = lerobot_policy.config
        # Nombres habituales en la config de π0 de LeRobot; con respaldos razonables.
        self.action_dim = int(getattr(config, "max_action_dim", getattr(config, "action_dim", 7)))
        self.chunk_size = int(getattr(config, "n_action_steps", getattr(config, "chunk_size", 50)))
        self.tokenizer_max_length = int(getattr(config, "tokenizer_max_length", 48))
        # El ``PI0Policy`` de LeRobot 0.4.4 no trae tokenizer (lo arma su pipeline de
        # procesadores, que no usamos): se carga aparte y se inyecta.
        self.tokenizer = tokenizer
        cams = getattr(config, "image_features", None) or getattr(config, "cameras", None)
        self.cameras = list(cams) if cams else ["primary"]

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path,
        *,
        torch_dtype: Any = None,
        tokenizer: str = PALIGEMMA_TOKENIZER,
    ) -> _LeRobotPi0Model:
        try:
            # Layout de lerobot >=0.4.x (el fijado en el grupo opcional ``pi0``).
            from lerobot.configs.policies import (  # pyright: ignore[reportMissingImports]
                PreTrainedConfig,
            )
            from lerobot.policies.pi0.configuration_pi0 import (  # pyright: ignore[reportMissingImports]
                PI0Config,
            )
            from lerobot.policies.pi0.modeling_pi0 import (  # pyright: ignore[reportMissingImports]
                PI0Policy,
            )
        except ImportError:  # pragma: no cover - layout antiguo (lerobot.common.*)
            from lerobot.common.configs.policies import (  # pyright: ignore[reportMissingImports]
                PreTrainedConfig,
            )
            from lerobot.common.policies.pi0.configuration_pi0 import (  # pyright: ignore[reportMissingImports]
                PI0Config,
            )
            from lerobot.common.policies.pi0.modeling_pi0 import (  # pyright: ignore[reportMissingImports]
                PI0Policy,
            )

        # π0 usa precisión MIXTA por diseño: PaliGemma en bf16 pero el action expert
        # y las proyecciones del flow matching (``action_in_proj``/``state_proj``) en
        # float32, y el ruido se samplea float32. LeRobot gobierna ese esquema con
        # ``config.dtype``. Un ``.to(dtype)`` GLOBAL lo rompería: convertiría esas
        # proyecciones a bf16 y chocarían con el ruido float32 ("mat1 and mat2 must
        # have the same dtype"). Por eso la precisión se fija vía config, no con ``.to``.
        config = None
        if torch_dtype is not None:
            precision = {torch.bfloat16: "bfloat16", torch.float32: "float32"}.get(torch_dtype)
            if precision is None:
                raise ValueError(f"π0 solo soporta bfloat16 o float32, no {torch_dtype}.")
            # Se carga por la clase BASE del registro (``PreTrainedConfig``), no por
            # la subclase ``PI0Config``: ``PreTrainedConfig.from_pretrained`` consume
            # el discriminador ``"type": "pi0"`` del ``config.json`` para elegir la
            # subclase y devuelve igual un ``PI0Config``. Llamar ``from_pretrained``
            # sobre la subclase pasa por un atajo de LeRobot (``draccus.parse(cls)``)
            # que con draccus 0.8 no consume ese ``type`` y revienta con
            # ``DecodingError: The fields 'type' are not valid for PI0Config``.
            # ``PreTrainedConfig.from_pretrained`` devuelve en runtime un ``PI0Config``
            # (lo elige por el ``type``); el ``cast`` es solo para el tipado de ``dtype``.
            config = cast(PI0Config, PreTrainedConfig.from_pretrained(str(source)))
            config.dtype = precision

        # El tokenizer primero: si falta el acceso a HF, falla antes de cargar ~7 GB.
        paligemma = load_paligemma_tokenizer(tokenizer)
        lerobot_policy = PI0Policy.from_pretrained(str(source), config=config)
        lerobot_policy.eval()
        return cls(lerobot_policy, paligemma)

    def _batch(self, cond: dict[str, Tensor], actions: Tensor | None) -> dict[str, Any]:
        """Arma el batch que espera ``PI0Policy`` de LeRobot a partir del ``cond``.

        Las claves son EXACTAMENTE las que lee ``modeling_pi0.py`` (en
        ``predict_action_chunk`` y ``forward``): las imágenes por la clave cruda de
        ``config.image_features`` (``observation.images.<cam>``), y estado, lenguaje y
        acción por las constantes ``OBS_STATE``/``OBS_LANGUAGE_*``/``ACTION`` de
        LeRobot. Se importan de ``lerobot`` (perezosamente, como el resto del wrapper)
        en vez de escribir los strings a mano, para no desincronizarnos si LeRobot
        los renombra: ``observation.language.tokens`` lleva PUNTO, no guion bajo.
        """
        from lerobot.utils.constants import (  # pyright: ignore[reportMissingImports]
            ACTION,
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
            OBS_STATE,
        )

        # El procesador nombra las imágenes ``image.<cam>``; LeRobot las busca por la
        # clave cruda de ``config.image_features``, así que se le quita el prefijo.
        batch: dict[str, Any] = {
            k.removeprefix("image."): v for k, v in cond.items() if k.startswith("image.")
        }
        batch[OBS_STATE] = cond["state"]
        batch[OBS_LANGUAGE_TOKENS] = cond["lang_tokens"]
        batch[OBS_LANGUAGE_ATTENTION_MASK] = cond["lang_masks"]
        if actions is not None:
            batch[ACTION] = actions
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
