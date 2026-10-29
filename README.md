# VLA-Train-Harness

Harness para entrenar modelos VLA (vision-language-action) con aprendizaje supervisado y RL, sobre ambientes reales o simulados y sobre un world model.

## Procesos

Cada pieza es un proceso aparte; se hablan solo por mensajes ZeroMQ (msgpack + buffers de numpy, sin pickle).

| Proceso | Paquete | Qué hace |
|---|---|---|
| Learner | `packages/vla_learner` | Tiene el modelo y conduce el loop: BC, PPO, GRPO |
| Core | `packages/vla_core` | Gateway en CPU: enruta pedidos, pide recompensas, registra episodios y sirve el Dataset |
| Environment | `packages/vla_env` | Ambientes reales o simulados (`reach`, LIBERO) servidos por `vla-env` |

`packages/vla_common` tiene lo que comparten: tipos, protocolo y el contrato del Environment.

## Cómo se levanta

Todo con [uv](https://docs.astral.sh/uv/). Ejemplo mínimo en CPU (cada proceso en su propia terminal): PPO contra dos ambientes `reach`, a través de Core.

```bash
uv sync
uv run vla-env --config configs/env/reach.yaml --address tcp://127.0.0.1:5601
uv run vla-env --config configs/env/reach.yaml --address tcp://127.0.0.1:5602
uv run vla-core --config configs/core/env_reach.yaml
uv run vla-learner --config configs/ppo_env_reach.yaml
```

Tests: `uv run pytest -m fast` (siempre), `-m slow` (integración de punta a punta), `-m distributed` (varios procesos). En el cluster, ver `slurm/README.md`.

## Documentación

- `docs/Architecture.md`: visión general, protocolo y flujos.
- `docs/learner/`, `docs/core/`, `docs/environment/`, `docs/dataset/`: diseño y suites de test de cada componente.
- `CLAUDE.md`: estado del proyecto y reglas de trabajo.
