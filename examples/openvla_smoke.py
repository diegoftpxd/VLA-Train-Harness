"""Prueba de humo de OpenVLA con pesos reales: cargar, actuar y (opcional) entrenar.

Pensado para correr en el cluster (los pesos ~16 GB no caben en la RTX 2050 de
desarrollo). Carga ``openvla/openvla-7b`` con el adaptador ``OpenVLAPolicy``,
arma una observación sintética y muestra la acción de 7 dimensiones que predice.
Con ``--train-steps N`` hace además unos pasos de Behavior Cloning sobre datos
sintéticos para ver bajar la loss, usando la misma infraestructura del Learner
(``OptimizerManager``).

Requiere el grupo opcional ``openvla``:

    uv sync --extra openvla
    uv run python examples/openvla_smoke.py --unnorm-key bridge_orig

Si el cluster no tiene ``flash-attn``:

    uv run python examples/openvla_smoke.py --attn eager --unnorm-key bridge_orig
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from vla_common import Observation
from vla_learner.infra.optimizer import OptimizerConfig, OptimizerManager
from vla_learner.policy.adapters import OpenVLAPolicy
from vla_learner.types import SupervisedBatch


def _fake_observation(instruction: str, size: int = 224) -> Observation:
    """Una observación sintética: imagen RGB aleatoria + propiocepción + instrucción."""
    image = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Observation(
        images={"primary": image},
        proprio=np.zeros(7, dtype=np.float32),
        instruction=instruction,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba de humo de OpenVLA.")
    parser.add_argument("--model", default="openvla/openvla-7b", help="Ruta o id de HF.")
    parser.add_argument("--unnorm-key", default=None, help="Clave de desnormalización.")
    parser.add_argument("--instruction", default="pick up the red block")
    parser.add_argument("--attn", default="flash_attention_2", help="flash_attention_2 | eager")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-steps", type=int, default=0, help="Pasos de BC sintético.")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Cargando {args.model} en {device} ({dtype})...")
    policy = OpenVLAPolicy.from_pretrained(
        args.model,
        unnorm_key=args.unnorm_key,
        attn_implementation=args.attn,
        torch_dtype=dtype,
    )
    policy.to(device)
    policy.eval()
    print(
        f"Listo. action_dim={policy.action_dim}, chunk_size={policy.chunk_size}, "
        f"capacidades={{{', '.join(c.value for c in policy.capabilities)}}}"
    )

    # --- Inferencia: una acción a partir de una observación sintética --------
    obs = [_fake_observation(args.instruction) for _ in range(args.batch_size)]
    model_input = policy.processor(obs)
    output = policy.act(model_input, deterministic=True)
    print("\n=== act() ===")
    print(f"acciones (unidades reales) [B, H, D] = {tuple(output.actions.shape)}:")
    print(output.actions.float().cpu().numpy())
    if output.log_probs is not None:
        print(f"log_probs por chunk = {output.log_probs.float().cpu().numpy()}")

    # --- (Opcional) unos pasos de Behavior Cloning sobre datos sintéticos ----
    if args.train_steps > 0:
        print(f"\n=== Behavior Cloning sintético: {args.train_steps} pasos ===")
        policy.train()
        optimizer = OptimizerManager(
            policy.trainable_parameters(),
            OptimizerConfig(name="adamw", lr=5e-5, grad_clip=1.0, scheduler=None, precision="fp32"),
        )
        # Acciones objetivo fijas: deben volverse más probables paso a paso.
        target = torch.randn(args.batch_size, policy.chunk_size, policy.action_dim)
        batch = SupervisedBatch(inputs=model_input, actions=target)
        for paso in range(args.train_steps):
            salida = policy.imitation_loss(batch.inputs, batch.actions)
            metrics = optimizer.step(salida.loss)
            print(
                f"paso {paso:02d}  ce={salida.metrics['ce']:.4f}  "
                f"token_acc={salida.metrics['token_accuracy']:.3f}  "
                f"grad_norm={metrics.get('grad_norm', float('nan')):.3f}"
            )


if __name__ == "__main__":
    main()
