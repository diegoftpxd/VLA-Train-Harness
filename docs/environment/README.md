# Environment

Ambiente real o simulado, con API tipo Gymnasium (`reset`, `step`, `close`). Un proceso por ambiente (`vla-env`). Declara su espacio de acciones, sus cámaras y sus tareas (`EnvSpec`). Ver [`../Architecture.md`](../Architecture.md) (sección "Environment, WorldModel, Reward Model y Dataset").

El diseño está en [`Environment Design.md`](./Environment%20Design.md):

- el paquete `packages/vla_env`;
- la base `GymEnvironment` para simuladores;
- la CLI `vla-env` con modo `--smoke` para probar un ambiente sin Core;
- las convenciones canónicas (imágenes, estado, `info["success"]`);
- cómo agregar un ambiente o un robot real;
- el primer ambiente real, LIBERO.

Los tests que debe pasar todo ambiente (ENV, GYM, SRVE, CLI, MAP, ROBOT, ARCH) están en [`Test Environment.md`](./Test%20Environment.md). El plan de implementación, con los tests de cada fase y la integración con los demás componentes, está en [`Environment Implementation Plan.md`](./Environment%20Implementation%20Plan.md).

El contrato (`Environment`, `EnvSpec`), el protocolo (`handle_environment`) y la suite ENV viven en `vla_common`. Así `vla_env` y `vla_core` no dependen entre sí. Core se sigue probando contra el `FakeEnvironment` en CPU (`vla_core/testing/fakes.py`).

## Cómo se usa

```bash
uv run vla-env --config configs/env/reach.yaml --smoke 20          # probar un ambiente sin Core
uv run vla-env --config configs/env/reach.yaml --address tcp://127.0.0.1:5601
uv run vla-core --config configs/core/env_reach.yaml               # Core con ambientes vla-env
VLA_ENV_CONFIG=configs/env/<nombre>.yaml uv run pytest -m heavy \
    packages/vla_env/tests/test_env_from_config.py                 # suites ENV/SRVE de un ambiente
```

Ambientes disponibles: `reach` (referencia, sin dependencias) y `libero` (extra `libero`, en
su propio venv; ver `slurm/libero_env_check.sbatch`). Estado de la implementación: tabla
"Mapa de fases" del plan.
