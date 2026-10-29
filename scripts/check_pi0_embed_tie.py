"""Verifica que el ``embed_tokens`` de PaliGemma en π0 quede cargado.

Al cargar ``lerobot/pi0_base``, LeRobot avisa ``Could not remap state dict keys``
porque falta ``...language_model.embed_tokens.weight``: el checkpoint no la trae y
solo guarda ``paligemma.lm_head.weight``. En Gemma esas dos matrices están atadas
(``tie_word_embeddings``), así que el aviso es inofensivo **si** el modelo de
LeRobot las ata. Este script lo comprueba con el mismo camino de carga que usa el
Learner (``_LeRobotPi0Model.from_pretrained``):

1. ``embed_tokens.weight`` y ``lm_head.weight`` son el mismo tensor (``data_ptr``).
2. ``embed_tokens.weight`` tiene los valores de ``lm_head`` del checkpoint (y no
   los de una inicialización aleatoria).

Sale con código 0 si todo se cumple y 1 si no. Necesita GPU o ~16 GB de RAM y el
acceso a HF del tokenizer (ver ``slurm/README.md``); se corre con
``slurm/pi0_check_embed.sbatch``.
"""

from __future__ import annotations

import argparse
import sys

import torch
from safetensors import safe_open
from transformers.utils.hub import cached_file

from vla_learner.policy.adapters.pi0 import _LeRobotPi0Model

LM_HEAD_KEY = "paligemma_with_expert.paligemma.lm_head.weight"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifica el embed_tokens de π0.")
    parser.add_argument("--model", default="lerobot/pi0_base")
    args = parser.parse_args()

    model = _LeRobotPi0Model.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    paligemma = model.lerobot.model.paligemma_with_expert.paligemma
    embed = paligemma.model.language_model.embed_tokens.weight
    lm_head = paligemma.lm_head.weight

    archivo = cached_file(args.model, "model.safetensors")
    assert archivo is not None
    with safe_open(archivo, framework="pt") as f:
        con_embed = [k for k in f.keys() if "embed_tokens" in k]
        print(f">> claves con embed_tokens en el checkpoint: {con_embed}")
        ckpt = f.get_tensor(LM_HEAD_KEY)

    mismo_tensor = embed.data_ptr() == lm_head.data_ptr()
    iguales = torch.equal(embed.detach().cpu().to(ckpt.dtype), ckpt)
    print(f">> embed_tokens {tuple(embed.shape)} {embed.dtype}; lm_head {tuple(lm_head.shape)}")
    atado = paligemma.config.get_text_config().tie_word_embeddings
    print(f">> config.tie_word_embeddings: {atado}")
    print(f">> mismo tensor (data_ptr): {mismo_tensor}")
    print(f">> embed_tokens == lm_head del checkpoint: {iguales}")
    std_embed, std_ckpt = embed.float().std().item(), ckpt.float().std().item()
    print(f">> desvío de embed_tokens: {std_embed:.5f} (checkpoint: {std_ckpt:.5f})")

    ok = mismo_tensor and iguales
    if ok:
        print(">> OK: embed_tokens está atado a lm_head y cargado")
    else:
        print(">> FALLA: embed_tokens no quedó cargado")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
