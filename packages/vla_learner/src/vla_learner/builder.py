"""Construcción del Learner y sus piezas desde configuración (``from_config``).

Registra cada pieza en el ``Registry`` y arma los componentes. Vive aparte de
``learner.py`` para que este no importe implementaciones concretas (ARCH-03).

Las piezas tiny de prueba (``TinyBackbone``) se registran aquí porque los configs
de ejemplo son tiny; los backbones reales (PaliGemma, ...) se registrarían del
mismo modo cuando se implementen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_learner.algorithms.advantages import GAE, GroupRelativeAdvantage, MonteCarloReturn
from vla_learner.algorithms.bc import BehaviorCloning
from vla_learner.algorithms.grpo import GRPO, GRPOConfig
from vla_learner.algorithms.ppo import PPO, PPOConfig
from vla_learner.algorithms.reference import ReferencePolicy
from vla_learner.data.collector import RolloutCollector
from vla_learner.data.core_client import CoreClient, CoreClientProtocol
from vla_learner.data.sources import (
    DataSource,
    EnvRolloutSource,
    MixedSource,
    SupervisedSource,
    WorldModelRolloutSource,
)
from vla_learner.infra.checkpoint import CheckpointManager
from vla_learner.infra.distributed import DDPStrategy, SingleDevice
from vla_learner.infra.logging import StdoutLogger
from vla_learner.infra.optimizer import OptimizerConfig, OptimizerManager
from vla_learner.learner import LearnerConfig
from vla_learner.policy.composed import ComposedPolicy
from vla_learner.policy.heads import (
    FlowMatchingActionHead,
    GaussianActionHead,
    TokenActionHead,
    ValueHead,
)
from vla_learner.policy.normalizer import ActionNormalizer, ActionStats
from vla_learner.policy.processors import SimpleProcessor
from vla_learner.registry import Registry
from vla_learner.testing.fake_core import FakeCoreClient
from vla_learner.testing.start_selector import ListStartSelector, ResetStartSelector
from vla_learner.testing.tiny_backbone import TinyBackbone

# --- Registro de piezas -------------------------------------------------


@Registry.register("backbone", "tiny")
def _build_tiny_backbone(feature_dim: int = 16, seq_len: int = 1) -> TinyBackbone:
    return TinyBackbone(feature_dim=feature_dim, seq_len=seq_len)


@Registry.register("action_head", "gaussian")
def _build_gaussian(
    feature_dim: int,
    action_dim: int,
    chunk_size: int,
    init_zero: bool = False,
    init_log_std: float = 0.0,
) -> GaussianActionHead:
    return GaussianActionHead(feature_dim, action_dim, chunk_size, init_zero, init_log_std)


@Registry.register("action_head", "token")
def _build_token(
    feature_dim: int, action_dim: int, chunk_size: int, bins: int = 51
) -> TokenActionHead:
    return TokenActionHead(feature_dim, action_dim, chunk_size, bins=bins)


@Registry.register("action_head", "flow_matching")
def _build_flow_matching(
    feature_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dim: int = 256,
    num_sampling_steps: int = 10,
    time_dim: int = 32,
) -> FlowMatchingActionHead:
    return FlowMatchingActionHead(
        feature_dim,
        action_dim,
        chunk_size,
        hidden_dim=hidden_dim,
        num_sampling_steps=num_sampling_steps,
        time_dim=time_dim,
    )


@Registry.register("processor", "simple")
def _build_simple_processor(
    cameras: list[str] | None = None, text_dim: int = 16
) -> SimpleProcessor:
    return SimpleProcessor(cameras=cameras or ["front"], text_dim=text_dim)


def _unit_normalizer(action_dim: int, mode: str) -> ActionNormalizer:
    cero = np.zeros(action_dim, dtype=np.float32)
    uno = np.ones(action_dim, dtype=np.float32)
    stats = ActionStats(mean=cero, std=uno, min=-uno, max=uno, q01=-uno, q99=uno)
    return ActionNormalizer({"default": stats}, mode=mode)  # type: ignore[arg-type]


# Archivos que lee ``normalizer.stats: <directorio>``. Son los que escribe
# ``Pi0Policy.save_pretrained`` (carpeta ``policy/`` de un checkpoint); se repiten
# acá para no importar el adaptador (y con él su grupo opcional) al resolverlos.
_ACTION_STATS_FILE = "action_stats.pt"
_STATE_STATS_FILE = "state_stats.pt"


def _resolve_normalizer(spec: dict | None, client: CoreClientProtocol) -> dict | None:
    """Resuelve ``normalizer.stats`` del config a normalizadores ya armados.

    De dónde salen las estadísticas (``Learner Design.md``, "``ActionNormalizer``"):

    - ``stats: fit``: se calculan al arrancar con una muestra de ``fit_samples``
      pares pedida a Core (``sample_supervised``): acciones con
      ``ActionNormalizer.fit`` y propiocepción con ``ActionNormalizer.from_values``.
    - ``stats: <directorio>``: se cargan de ``action_stats.pt`` (y
      ``state_stats.pt`` si existe), p. ej. la carpeta ``policy/`` de un checkpoint.
    - sin ``stats``: se devuelve el spec tal cual (la factory decide).

    Devuelve el spec con ``action`` y ``state`` (``ActionNormalizer`` o ``None``),
    sin mutar el original (``LearnerConfig.raw`` guarda el config tal como vino).
    """
    if not spec or "stats" not in spec:
        return spec
    mode = spec.get("mode", "mean_std")
    fuente = spec["stats"]
    if fuente == "fit":
        muestras = client.sample_supervised(int(spec.get("fit_samples", 256)))
        if not muestras:
            raise ValueError("normalizer.stats=fit: Core no devolvió muestras supervisadas.")
        action = ActionNormalizer.fit(muestras, mode=mode)
        proprios = [obs.proprio for obs, _ in muestras if obs.proprio is not None]
        state = ActionNormalizer.from_values(np.stack(proprios), mode=mode) if proprios else None
    else:
        directorio = Path(fuente)
        action = ActionNormalizer.from_file(directorio / _ACTION_STATS_FILE)
        archivo_estado = directorio / _STATE_STATS_FILE
        state = ActionNormalizer.from_file(archivo_estado) if archivo_estado.exists() else None
    resuelto = {k: v for k, v in spec.items() if k not in ("stats", "fit_samples")}
    return {**resuelto, "action": action, "state": state}


@Registry.register("policy", "composed")
def _build_composed(
    action_dim: int,
    chunk_size: int,
    feature_dim: int,
    backbone: dict,
    action_head: dict,
    processor: dict,
    normalizer: dict | None = None,
    value_head: dict | None = None,
) -> ComposedPolicy:
    bb = Registry.build("backbone", {**backbone, "feature_dim": feature_dim})
    head = Registry.build(
        "action_head",
        {
            **action_head,
            "feature_dim": feature_dim,
            "action_dim": action_dim,
            "chunk_size": chunk_size,
        },
    )
    proc = Registry.build("processor", dict(processor))
    # Con ``normalizer.stats`` el builder ya trae el normalizador armado; sin él se
    # usa uno unitario (las heads tiny trabajan en unidades de la escala de prueba).
    norm = (normalizer or {}).get("action")
    if norm is None:
        norm = _unit_normalizer(action_dim, (normalizer or {}).get("mode", "mean_std"))
    cabeza_valor = ValueHead(feature_dim, value_head["hidden_dim"]) if value_head else None
    return ComposedPolicy(bb, head, proc, norm, cabeza_valor)


@Registry.register("policy", "openvla")
def _build_openvla(
    pretrained: str,
    unnorm_key: str | None = None,
    camera: str = "primary",
    attn_implementation: str = "flash_attention_2",
    torch_dtype: str | None = None,
    device_map: Any = None,
    freeze: dict | None = None,
) -> Any:
    # Importación perezosa: el adaptador carga ``transformers`` solo acá, así que
    # el resto del harness no depende del grupo opcional ``openvla``.
    from vla_learner.policy.adapters import OpenVLAPolicy

    dtype = getattr(torch, torch_dtype) if torch_dtype else None
    policy = OpenVLAPolicy.from_pretrained(
        pretrained,
        unnorm_key=unnorm_key,
        camera=camera,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        device_map=device_map,
    )
    # Fine-tuning parcial (``docs/learner/Learner Design.md``, "Congelar y LoRA"):
    # el factory decide qué se entrena. ``freeze: {trainable: [lm_head]}`` congela
    # todo salvo esos módulos.
    if freeze and freeze.get("trainable"):
        policy.freeze_except(list(freeze["trainable"]))
    return policy


@Registry.register("policy", "pi0")
def _build_pi0(
    pretrained: str = "lerobot/pi0_base",
    cameras: list[str] | None = None,
    image_size: int = 224,
    torch_dtype: str | None = None,
    device_map: Any = None,
    freeze: dict | None = None,
    tokenizer: str | None = None,
    normalizer: dict | None = None,
) -> Any:
    # Importación perezosa: el adaptador carga ``lerobot`` solo acá (grupo opcional
    # ``pi0``), así que el resto del harness no depende de él.
    from vla_learner.policy.adapters import Pi0Policy
    from vla_learner.policy.adapters.pi0 import PALIGEMMA_TOKENIZER

    if normalizer and "action" not in normalizer:
        raise ValueError(
            "policy.normalizer de π0 necesita `stats` (fit o un directorio): pi0_base no "
            "trae estadísticas del dataset."
        )
    dtype = getattr(torch, torch_dtype) if torch_dtype else None
    policy = Pi0Policy.from_pretrained(
        pretrained,
        cameras=cameras,
        image_size=image_size,
        torch_dtype=dtype,
        device_map=device_map,
        tokenizer=tokenizer or PALIGEMMA_TOKENIZER,
        action_normalizer=(normalizer or {}).get("action"),
        state_normalizer=(normalizer or {}).get("state"),
    )
    if freeze and freeze.get("trainable"):
        policy.freeze_except(list(freeze["trainable"]))
    return policy


@Registry.register("advantage", "gae")
def _build_gae(gamma: float = 0.99, lam: float = 0.95) -> GAE:
    return GAE(gamma=gamma, lam=lam)


@Registry.register("advantage", "group_relative")
def _build_group_relative(normalize_std: bool = True) -> GroupRelativeAdvantage:
    return GroupRelativeAdvantage(normalize_std=normalize_std)


@Registry.register("advantage", "monte_carlo")
def _build_monte_carlo(gamma: float = 0.99) -> MonteCarloReturn:
    return MonteCarloReturn(gamma=gamma)


@Registry.register("algorithm", "bc")
def _build_bc() -> BehaviorCloning:
    return BehaviorCloning()


@Registry.register("algorithm", "ppo")
def _build_ppo(advantage: dict | None = None, **config: Any) -> PPO:
    estimador = Registry.build("advantage", advantage or {"type": "gae"})
    return PPO(PPOConfig(**config), estimador)


@Registry.register("algorithm", "grpo")
def _build_grpo(advantage: dict | None = None, **config: Any) -> GRPO:
    estimador = Registry.build("advantage", advantage or {"type": "group_relative"})
    return GRPO(GRPOConfig(**config), estimador)


# --- Construcción de componentes ---------------------------------------


def _build_client(core: dict) -> CoreClientProtocol:
    tipo = core.get("type", "fake")
    if tipo == "fake":
        return FakeCoreClient(
            seed=core.get("seed", 0), demo_chunk_size=core.get("demo_chunk_size", 1)
        )
    if tipo in ("zmq", "real"):
        return CoreClient(core["address"], core["timeout_s"])
    raise ValueError(f"Tipo de core desconocido: {tipo!r}")


def _build_data_source(data: dict, client: CoreClientProtocol, policy) -> DataSource:
    tipo = data["type"]
    if tipo == "supervised":
        return SupervisedSource(client, data["batch_size"], data.get("filters"))
    if tipo == "env_rollout":
        return EnvRolloutSource(
            RolloutCollector(client),
            data["env_ids"],
            data["steps_per_batch"],
            data.get("group_size", 1),
        )
    if tipo == "world_model":
        if data.get("start_selector") == "reset":
            selector = ResetStartSelector(client)
        else:
            inicios = [obs for _, obs in client.reset(data["env_ids"])]
            selector = ListStartSelector(list(inicios))
        return WorldModelRolloutSource(
            RolloutCollector(client),
            selector,
            data["sequences_per_start"],
            data["horizon"],
            data.get("num_starts", 1),
        )
    if tipo == "mixed":
        fuentes = [_build_data_source(s, client, policy) for s in data["sources"]]
        return MixedSource(fuentes, data["weights"])
    raise ValueError(f"Tipo de fuente de datos desconocido: {tipo!r}")


def build_components(config: dict) -> dict[str, Any]:
    torch.manual_seed(config.get("seed", 0))
    client = _build_client(config.get("core", {"type": "fake"}))
    policy_config = dict(config["policy"])
    if "normalizer" in policy_config:
        policy_config["normalizer"] = _resolve_normalizer(policy_config["normalizer"], client)
    policy = Registry.build("policy", policy_config)

    strategy = SingleDevice(config.get("device", "cpu"))
    if config.get("strategy") == "ddp":
        strategy = DDPStrategy(config.get("device", "cpu"))
    strategy.wrap(policy)

    algorithm = Registry.build("algorithm", dict(config["algorithm"]))
    if isinstance(algorithm, GRPO) and algorithm.config.kl_coef > 0 and algorithm.reference is None:
        algorithm.reference = ReferencePolicy(policy)

    data_source = _build_data_source(dict(config["data"]), client, policy)
    optimizer = OptimizerManager(
        policy.trainable_parameters(), OptimizerConfig(**config.get("optimizer", {}))
    )
    checkpoint = config.get("checkpoint", {})
    checkpoints = CheckpointManager(
        Path(checkpoint.get("directory", "checkpoints")), checkpoint.get("keep_last", 3)
    )
    logger = StdoutLogger(is_main=strategy.is_main)
    learner_config = LearnerConfig(
        checkpoint_every=checkpoint.get("every", 100),
        seed=config.get("seed", 0),
        available_cameras=config.get("available_cameras"),
        raw=config,
    )
    return {
        "policy": policy,
        "algorithm": algorithm,
        "data_source": data_source,
        "optimizer": optimizer,
        "checkpoints": checkpoints,
        "logger": logger,
        "strategy": strategy,
        "config": learner_config,
    }
