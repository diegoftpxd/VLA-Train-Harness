"""``LiberoEnvironment``: el benchmark LIBERO (``type: libero``).

Definido en ``docs/environment/Environment Design.md`` (sección "LIBERO"). Usa
``hf-libero`` (extra ``libero`` de ``vla_env``) directamente, sin ``lerobot`` ni
``torch`` propios: replica en numpy lo que hacen el ``LiberoEnv`` y el
``LiberoProcessorStep`` de lerobot 0.4.4, para que la observación coincida con la
del dataset ``lerobot/libero`` con el que se entrena π0.

Mapeo simulador → canónico:

- **Tareas:** el ``language`` de cada tarea de las ``suites`` configuradas.
- **Cámaras:** ``cameras: {canónico: cámara de LIBERO}``; cada imagen se **rota
  180°** (se invierten H y W, como ``LiberoProcessorStep``) y pasa por
  ``to_canonical_image``.
- **Estado:** 8-D = ``eef_pos`` (3) + ``quat2axisangle(eef_quat)`` (3) +
  ``gripper_qpos`` (2).
- **Acción:** 7-D en unidades reales, control relativo (``use_delta=True``).
- **Reset:** semilla → estado inicial ``init_states[seed % N]`` del benchmark y
  10 pasos con la acción no-op para que los objetos se asienten.
- **Éxito:** ``check_success()`` del simulador. **Límite:** por suite.

Las funciones de mapeo (``quat2axisangle``, ``libero_proprio``,
``libero_images``, ``index_tasks``) no importan LIBERO, así se prueban en
``fast`` sin el simulador.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vla_common.environment import EnvSpec
from vla_common.types import Observation
from vla_env.gym import GymEnvironment, to_canonical_image

ACTION_DIM = 7
NOOP_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)  # gripper abierto
# Límite de acciones por suite (los de lerobot: un margen sobre la demo más larga).
MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# -- Mapeo puro (sin LIBERO) --------------------------------------------------


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Cuaternión ``(x, y, z, w)`` → eje-ángulo 3-D (misma fórmula que lerobot)."""
    q = np.asarray(quat, dtype=np.float32)
    w = float(np.clip(q[3], -1.0, 1.0))
    den = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    return (q[:3] / den * (2.0 * np.arccos(w))).astype(np.float32)


def libero_proprio(raw: dict[str, Any]) -> np.ndarray:
    """Estado 8-D en el orden del dataset: posición, eje-ángulo y gripper."""
    return np.concatenate(
        [
            np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(raw["robot0_eef_quat"]),
            np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).astype(np.float32)


def libero_images(raw: dict[str, Any], cameras: dict[str, str]) -> dict[str, np.ndarray]:
    """Cada cámara de LIBERO, rotada 180° y en formato canónico, con su nombre canónico."""
    return {
        canonico: to_canonical_image(np.asarray(raw[clave])[::-1, ::-1])
        for canonico, clave in cameras.items()
    }


def index_tasks(suites: dict[str, list[str]]) -> dict[str, tuple[str, int]]:
    """``{suite: [language por task_id]}`` → ``{language: (suite, task_id)}``."""
    indice: dict[str, tuple[str, int]] = {}
    for suite, idiomas in suites.items():
        for task_id, idioma in enumerate(idiomas):
            if idioma in indice:
                otra = indice[idioma][0]
                raise ValueError(f"tarea repetida en {otra} y {suite}: {idioma!r}")
            indice[idioma] = (suite, task_id)
    return indice


def ensure_libero_config() -> Path:
    """Escribe la config de LIBERO si falta, **antes** de importarlo.

    Sin ``config.yaml``, el primer ``import libero.libero`` la pide con
    ``input()``, lo que cuelga un proceso sin terminal (un job de Slurm). Se
    escriben los mismos valores por defecto que LIBERO, calculados desde la
    ubicación del paquete sin importarlo.
    """
    directorio = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    config = directorio / "config.yaml"
    if config.exists():
        return config
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("no se encontró el paquete `libero` (hf-libero)")
    raiz = Path(next(iter(spec.submodule_search_locations))) / "libero"
    directorio.mkdir(parents=True, exist_ok=True)
    rutas = {
        "benchmark_root": raiz,
        "bddl_files": raiz / "bddl_files",
        "init_states": raiz / "init_files",
        "datasets": raiz.parent / "datasets",
        "assets": raiz / "assets",
    }
    config.write_text(yaml.safe_dump({k: str(v) for k, v in rutas.items()}), encoding="utf-8")
    return config


# -- Adaptador ---------------------------------------------------------------


class _LiberoSim:
    """Lleva el ``OffScreenRenderEnv`` de una tarea a la API que espera la base."""

    def __init__(self, env: Any, init_states: np.ndarray, num_steps_wait: int) -> None:
        self._env = env
        self._init_states = init_states
        self._num_steps_wait = num_steps_wait

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is None:
            indice = int(np.random.default_rng().integers(len(self._init_states)))
        else:
            self._env.seed(seed)
            indice = seed % len(self._init_states)
        self._env.reset()
        raw = self._env.set_init_state(self._init_states[indice])
        for _ in range(self._num_steps_wait):  # los objetos se asientan
            raw, _, _, _ = self._env.step(NOOP_ACTION)
        for robot in self._env.robots:
            robot.controller.use_delta = True  # control relativo, como el dataset
        return raw, {}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict]:
        raw, recompensa, done, _ = self._env.step(action)
        return raw, recompensa, bool(done), False, {"is_success": bool(self._env.check_success())}

    def close(self) -> None:
        self._env.close()


class LiberoEnvironment(GymEnvironment):
    def __init__(
        self,
        suites: list[str],
        cameras: dict[str, str],
        image_size: int = 256,
        num_steps_wait: int = 10,
        max_steps: int | None = None,
    ) -> None:
        ensure_libero_config()
        from libero.libero import (  # pyright: ignore[reportMissingImports]
            benchmark,
            get_libero_path,
        )

        self._get_libero_path = get_libero_path
        disponibles = benchmark.get_benchmark_dict()
        desconocidas = [s for s in suites if s not in disponibles]
        if desconocidas:
            raise ValueError(
                f"suites desconocidas {desconocidas}. Disponibles: {list(disponibles)}"
            )
        self._suites = {nombre: disponibles[nombre]() for nombre in suites}
        self._tareas = index_tasks(
            {n: [t.language for t in s.tasks] for n, s in self._suites.items()}
        )
        self._cameras = dict(cameras)
        self._image_size = image_size
        self._num_steps_wait = num_steps_wait
        self._max_steps_fijo = max_steps
        self._task_actual: str | None = None
        super().__init__(
            EnvSpec(action_dim=ACTION_DIM, cameras=list(cameras), tasks=list(self._tareas)),
            max_steps or max(MAX_STEPS.get(s, 400) for s in suites),
        )

    def _make(self, task: str) -> _LiberoSim:
        suite, task_id = self._tareas[task]
        self.max_steps = self._max_steps_fijo or MAX_STEPS.get(suite, 400)
        if self._sim is not None and task == self._task_actual:
            return self._sim  # misma tarea: se reusa el simulador
        if self._sim is not None:
            self._sim.close()
        from libero.libero.envs import (  # pyright: ignore[reportMissingImports]
            OffScreenRenderEnv,
        )

        suite_obj = self._suites[suite]
        tarea = suite_obj.get_task(task_id)
        bddl = os.path.join(
            self._get_libero_path("bddl_files"), tarea.problem_folder, tarea.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=self._image_size,
            camera_widths=self._image_size,
        )
        self._task_actual = task
        return _LiberoSim(env, suite_obj.get_task_init_states(task_id), self._num_steps_wait)

    def _observation(self, raw: dict[str, Any], task: str) -> Observation:
        return Observation(
            images=libero_images(raw, self._cameras),
            proprio=libero_proprio(raw),
            instruction=task,
        )

    def close(self) -> None:
        super().close()
        self._task_actual = None
