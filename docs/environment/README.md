# Environment

Ambiente real o simulado, con API tipo Gymnasium (`reset`, `step`, `close`). Un proceso por ambiente (`vla-env`). Declara su espacio de acciones, sus cámaras y sus tareas (`EnvSpec`). Ver [`../Architecture.md`](../Architecture.md) (sección "Environment, WorldModel, Reward Model y Dataset").

El diseño está en [`Environment Design.md`](./Environment%20Design.md):

- el paquete `packages/vla_env`;
- la base `GymEnvironment` para simuladores;
- la CLI `vla-env` con modo `--smoke` para probar un ambiente sin Core;
- las convenciones canónicas (imágenes, estado, `info["success"]`);
- cómo agregar un ambiente o un robot real;
- el primer ambiente real, LIBERO.

El plan de implementación, con los tests de cada fase y la integración con los demás componentes, está en [`Environment Implementation Plan.md`](./Environment%20Implementation%20Plan.md).

El contrato (`Environment`, `EnvSpec`), el protocolo (`handle_environment`) y la suite ENV viven en `vla_common`. Así `vla_env` y `vla_core` no dependen entre sí. Core se sigue probando contra el `FakeEnvironment` en CPU (`vla_core/testing/fakes.py`).
