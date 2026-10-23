"""Prueba de humo de π0 con pesos reales: cargar, actuar y (opcional) entrenar.

Pensado para correr en el cluster (los pesos ~7 GB no caben en la RTX 2050 de
desarrollo). Carga ``lerobot/pi0_base`` con el adaptador ``Pi0Policy``, arma una
observación sintética y muestra el chunk de acciones que predice. Con
``--train-steps N`` hace además unos pasos de Behavior Cloning sobre datos
sintéticos usando la misma infraestructura del Learner (``OptimizerManager``).

Requiere el grupo opcional ``pi0``:

    uv sync --extra pi0
    uv run python examples/pi0_smoke.py

Para fine-tuning parcial (solo el action expert, PaliGemma congelado):

    uv run python examples/pi0_smoke.py --freeze action --train-steps 5
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from vla_common import Observation
from vla_learner.infra.optimizer import OptimizerConfig, OptimizerManager
from vla_learner.policy.adapters import Pi0Policy
from vla_learner.types import SupervisedBatch


def _fake_observation(
    instruction: str,
    cameras: list[str] | tuple[str, ...] = ("primary",),
    size: int = 224,
) -> Observation:
    """Una observación sintética: imagen RGB aleatoria por cámara + propiocepción + instrucción.

    Las cámaras se pasan desde ``processor.required_cameras``: el π0 real de LeRobot
    las nombra según su config (p. ej. ``observation.images.base_0_rgb``), no
    ``"primary"``, así que la observación falsa tiene que usar esos mismos nombres.
    """
    images = {
        cam: np.random.randint(0, 256, (size, size, 3), dtype=np.uint8) for cam in cameras
    }
    return Observation(
        images=images,
        proprio=np.zeros(7, dtype=np.float32),
        instruction=instruction,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba de humo de π0 (LeRobot).")
    parser.add_argument("--model", default="lerobot/pi0_base", help="Ruta o id de HF.")
    parser.add_argument("--instruction", default="pick up the red block")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--freeze", default=None, help="Substring de módulos entrenables (p. ej. action)."
    )
    parser.add_argument("--train-steps", type=int, default=0, help="Pasos de BC sintético.")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Cargando {args.model} en {device} ({dtype})...")
    policy = Pi0Policy.from_pretrained(args.model, torch_dtype=dtype)
    if args.freeze:
        policy.freeze_except([args.freeze])
    policy.to(device)
    policy.eval()
    print(
        f"Listo. action_dim={policy.action_dim}, chunk_size={policy.chunk_size}, "
        f"capacidades={{{', '.join(c.value for c in policy.capabilities) or '∅'}}}"
    )

    # --- Inferencia: un chunk de acciones a partir de una observación sintética ---
    cameras = policy.processor.required_cameras
    obs = [_fake_observation(args.instruction, cameras) for _ in range(args.batch_size)]
    model_input = policy.processor(obs)
    output = policy.act(model_input, deterministic=True)
    print("\n=== act() ===")
    print(f"acciones (unidades reales) [B, H, D] = {tuple(output.actions.shape)}:")
    print(output.actions.float().cpu().numpy())

    # --- (Opcional) unos pasos de Behavior Cloning sobre datos sintéticos ----
    if args.train_steps > 0:
        print(f"\n=== Behavior Cloning sintético: {args.train_steps} pasos ===")
        policy.train()
        optimizer = OptimizerManager(
            policy.trainable_parameters(),
            OptimizerConfig(name="adamw", lr=5e-5, grad_clip=1.0, scheduler=None, precision="fp32"),
        )
        target = torch.randn(args.batch_size, policy.chunk_size, policy.action_dim)
        batch = SupervisedBatch(inputs=model_input, actions=target)
        for paso in range(args.train_steps):
            salida = policy.imitation_loss(batch.inputs, batch.actions)
            metrics = optimizer.step(salida.loss)
            print(
                f"paso {paso:02d}  fm_mse={salida.metrics['fm_mse']:.4f}  "
                f"grad_norm={metrics.get('grad_norm', float('nan')):.3f}"
            )


if __name__ == "__main__":
    main()
