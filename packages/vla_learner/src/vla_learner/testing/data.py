"""Generadores de datos sintéticos y reproducibles para los tests.

Definidos en ``docs/learner/Test Learners.md``. No dependen de ningún modelo: producen
los tipos de ``vla_common`` con numpy.
"""

from __future__ import annotations

import numpy as np

from vla_common import Action, ActionChunk, Observation

# Vocabulario mínimo para instrucciones de largo variable.
_PALABRAS = (
    "pick up the red block and place it on the table near the plate quickly please now"
).split()


def make_observations(
    n: int,
    cameras: list[str] | tuple[str, ...] = ("front",),
    seed: int = 0,
    proprio_dim: int = 7,
    image_size: int = 32,
) -> list[Observation]:
    """``n`` observaciones reproducibles con imágenes, propiocepción e instrucciones.

    Las instrucciones tienen largo variable (de 1 a muchas palabras) para
    ejercitar el procesamiento de lenguaje.
    """
    rng = np.random.default_rng(seed)
    observations: list[Observation] = []
    for i in range(n):
        images = {cam: rng.random((3, image_size, image_size), dtype=np.float32) for cam in cameras}
        proprio = rng.standard_normal(proprio_dim).astype(np.float32)
        n_palabras = 1 + (i % len(_PALABRAS))  # de 1 en adelante, variable por fila
        instruction = " ".join(_PALABRAS[:n_palabras])
        observations.append(Observation(images=images, proprio=proprio, instruction=instruction))
    return observations


def make_action_chunks(
    n: int,
    chunk_size: int,
    action_dim: int,
    seed: int = 0,
) -> list[ActionChunk]:
    """``n`` chunks de acciones en unidades reales, con escalas distintas por dimensión.

    La dimensión ``j`` se escala por ``10**(j - action_dim//2)``, de modo que las
    columnas tienen magnitudes muy diferentes (útil para probar normalización).
    """
    rng = np.random.default_rng(seed)
    escalas = np.array([10.0 ** (j - action_dim // 2) for j in range(action_dim)], dtype=np.float32)
    chunks: list[ActionChunk] = []
    for _ in range(n):
        bruto = rng.standard_normal((chunk_size, action_dim)).astype(np.float32)
        valores = bruto * escalas
        chunk: ActionChunk = [Action(values=valores[t].copy()) for t in range(chunk_size)]
        chunks.append(chunk)
    return chunks
