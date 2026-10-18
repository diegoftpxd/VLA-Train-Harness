# CLAUDE.md

Harness para entrenar modelos VLA (vision-language-action) con aprendizaje supervisado y RL. Este repositorio implementa, por ahora, **solo el proceso Learner** y sus herramientas de prueba.

## Documentos de diseño (fuente de verdad)

Léelos antes de empezar cualquier tarea. Ante cualquier duda, manda el documento.

- `docs/Architecture.md`: procesos, tipos compartidos, interfaz con Core, protocolo ZeroMQ.
- `docs/Learner Design.md`: clases e interfaces del Learner. Es la especificación a implementar.
- `docs/Test Learners.md`: suites de contrato que toda implementación debe pasar.

Si al implementar necesitas desviarte del diseño (una firma, un tipo, una clase nueva), actualiza el documento en el mismo cambio y explica el motivo en tu resumen. No dejes código y documento en desacuerdo.

## Alcance

**Dentro:**

- Todas las clases de `Learner Design.md`: interfaces, `ComposedPolicy`, `TokenActionHead`, `GaussianActionHead`, `ActionNormalizer`, `BehaviorCloning`, `PPO`, `GRPO`, estimadores de ventaja, fuentes de datos, `CoreClient`, infraestructura, `Learner`, `Registry` y CLI.
- Herramientas de prueba de `Test Learners.md`: `FakeCoreClient`, `ToyReachEnv`, `ToyWorldModel`, `TinyBackbone`, `StubPolicy`, generadores de datos y las suites de contrato.
- `FakeCoreServer`: un servidor ZeroMQ mínimo que envuelve a `FakeCoreClient`, para probar el `CoreClient` real sin que exista Core.

**Fuera (no implementar sin que se pida):**

- Core, Environment, WorldModel y Reward Model reales.
- Adaptadores de modelos reales (OpenVLA, π0, Octo) y `DiffusionActionHead` / `FlowMatchingActionHead`. El diseño debe permitirlos, pero no se escriben todavía.
- `EvalModelServer`: solo si se pide explícitamente, como dependencia opcional.

## Hardware: GPU limitada

La máquina tiene una **RTX 2050 con 4 GB de VRAM**. Por eso:

- **Nunca descargues ni cargues pesos de modelos reales** (OpenVLA 7B, PaliGemma 3B, π0, etc.). No caben y no son necesarios para el alcance actual.
- Todos los tests usan configuraciones **tiny**: pocas capas, pesos aleatorios, imágenes de 32×32. Un test no debe usar más de ~2 GB de VRAM.
- El dispositivo por defecto de los tests es **CPU**. La GPU se usa solo en tests marcados `gpu`.
- Antes de correr tests `gpu`, revisa con `nvidia-smi` que la GPU esté libre.

## Reproducibilidad con uv

**Todo proceso del harness debe poder levantarse con `uv`.** No uses `pip`, `conda` ni entornos creados a mano.

- El repositorio es un **workspace de uv**. `uv.lock` y `.python-version` van en el control de versiones.
- Dependencias: siempre con `uv add` (o `uv add --dev`, o `uv add --optional <grupo>`). Nunca edites versiones a mano sin regenerar el lock.
- Cada proceso tiene un punto de entrada en `[project.scripts]` y se levanta así:

  ```bash
  uv run vla-learner --config configs/<archivo>.yaml
  ```

  Cuando existan Core u otros procesos, siguen el mismo patrón (`uv run vla-core ...`).
- PyTorch con CUDA se instala desde el índice oficial, configurado en `pyproject.toml`:

  ```toml
  [tool.uv.sources]
  torch = [{ index = "pytorch-cu124" }]

  [[tool.uv.index]]
  name = "pytorch-cu124"
  url = "https://download.pytorch.org/whl/cu124"
  explicit = true
  ```

  Si el driver de la máquina no soporta esa versión de CUDA, ajusta el índice y avisa.
- Dependencias pesadas o específicas de un modelo van en grupos opcionales (por ejemplo `openvla`, `lerobot`, `eval`), para que la instalación base sea liviana.
- Cada corrida es reproducible: la configuración incluye `seed`, y cada checkpoint guarda la configuración resuelta, el commit de git y el hash de `uv.lock`.

## Estructura del repositorio

```
pyproject.toml              # raíz del workspace
uv.lock
.python-version
docs/                       # documentos de diseño
configs/                    # YAML de ejemplo (tiny, para pruebas locales)
packages/
  vla_common/               # tipos compartidos (Observation, ActionChunk, ...) y serialización ZeroMQ
  vla_learner/
    src/vla_learner/
      policy/               # VLAPolicy, ComposedPolicy, backbones/, heads/, processors/, normalizer.py
      algorithms/           # base.py, bc.py, ppo.py, grpo.py, advantages.py, reference.py
      data/                 # core_client.py, sources.py, collector.py, minibatch.py
      infra/                # optimizer.py, distributed.py, checkpoint.py, logging.py
      testing/              # fakes, ToyReachEnv, modelos tiny y suites de contrato
      learner.py
      registry.py
      cli.py
    tests/
```

`vla_common` existe porque Core y los demás procesos usarán los mismos tipos y el mismo protocolo. Nada de `vla_common` puede depender de `torch`.

Las suites de contrato viven en `testing/contracts/` y no en `tests/`, porque las implementaciones futuras las importan.

## Tests

Marcas registradas en `pyproject.toml` con `--strict-markers`:

| Marca | Qué incluye | ¿Correr en esta máquina? |
|---|---|---|
| `fast` | Contratos con modelos tiny, en CPU | Siempre, antes de terminar cualquier tarea |
| `slow` | Tests de aprendizaje en `ToyReachEnv` (LEARN) | Sí, al cerrar cada fase. Tardan minutos en CPU |
| `gpu` | Dispositivo y precisión mixta con modelos tiny | Sí, cuando toques código de dispositivo o precisión |
| `distributed` | DDP con 2 procesos y backend gloo en CPU | Sí, cuando toques `infra/distributed.py` |
| `heavy` | Pesos reales, FSDP con varias GPUs o más de 2 GB de VRAM | **No.** Se escriben, pero no se corren aquí |

Comandos:

```bash
uv run pytest -m fast                     # siempre
uv run pytest -m slow                     # al cerrar una fase
uv run pytest -m "gpu and not heavy"      # código de dispositivo o precisión
uv run pytest -m "distributed and not gpu and not heavy"
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

La configuración por defecto de pytest excluye `heavy`, `gpu` y `distributed` para que `uv run pytest` sea seguro de correr sin pensar.

Reglas:

- **No desactives, saltes ni relajes un test para que pase.** Si un test de contrato parece incorrecto, detente y explica por qué en vez de cambiarlo.
- Las tolerancias son las de `Test Learners.md`. No las cambies sin actualizar el documento.
- Todo código nuevo llega con sus tests. Toda implementación de una interfaz hereda su suite de contrato.

## Orden de implementación

Cada fase termina con `uv run pytest -m fast`, ruff y pyright en verde, y un commit. No empieces una fase con la anterior en rojo.

1. **Base.** Workspace de uv, `vla_common` (tipos y serialización), `Batch`, `MinibatchIterator`, `Registry`. Tests BAT, MB, REG.
2. **Herramientas de prueba.** `make_observations`, `make_action_chunks`, `FakeCoreClient`, `ToyReachEnv`, `ToyWorldModel`, `StubPolicy`, `TinyBackbone`. Como `StubPolicy` y `TinyBackbone` implementan las interfaces del modelo (`VLAPolicy`, `Backbone`), estas se adelantan a esta fase (en `policy/interfaces.py`); la Fase 3 agrega las implementaciones concretas.
3. **Modelo.** Implementaciones concretas sobre las interfaces de la Fase 2: `ActionNormalizer`, `ComposedPolicy`, `GaussianActionHead`, `TokenActionHead`, un procesador simple. Suites POL, HEAD, NRM, OBS.
4. **Algoritmos.** Estimadores de ventaja, `BehaviorCloning`, `PPO`, `GRPO`, `ReferencePolicy`. Suites ADV, ALG y específicas.
5. **Datos.** `CoreClient` con ZeroMQ, `FakeCoreServer`, fuentes de datos, `RolloutCollector`. Suites CORE (contra fake y contra el cliente real con `FakeCoreServer`) y DATA.
6. **Infraestructura.** `OptimizerManager`, `SingleDevice`, `DDPStrategy`, `CheckpointManager`, loggers. Suites OPT, DIST-01/02, CKPT, LOG. `FSDPStrategy` se implementa, pero su test va marcado `heavy`.
7. **Learner y CLI.** `Learner`, `from_config`, `uv run vla-learner`, configs tiny de ejemplo. Suites LRN y ARCH.
8. **Aprendizaje.** Tests LEARN en `ToyReachEnv`. Si un algoritmo no aprende, es un bug, no un problema de hiperparámetros, hasta demostrar lo contrario.

## Decisiones abiertas ya resueltas

`Test Learners.md` planteaba dos pendientes, ya resueltas e incorporadas a `Learner Design.md` y `Test Learners.md`. Impleméntalas así:

- **Corte de épocas por KL:** agrega `Algorithm.should_stop_epochs(metrics: dict[str, float]) -> bool`, que por defecto devuelve `False`. PPO devuelve `True` si `approx_kl > target_kl`. El `Learner` lo consulta después de cada época.
- **`deterministic=True` en modelos con ruido:** el ruido inicial se genera con un generador de semilla fija propio de la política, sin tocar el estado aleatorio global.

## Estilo de código

- Python con type hints completos. Dataclasses para los tipos de datos; `ABC` para las interfaces.
- Nombres de clases y métodos exactamente como en el documento de diseño.
- Nada de `pickle` entre procesos (ver protocolo en el documento de arquitectura).
- Solo `CoreClient` importa `zmq`; solo `OptimizerManager` llama a `backward()` y `step()`. Los tests ARCH lo verifican.
- Docstrings y comentarios en español; identificadores en inglés.
- Pregunta antes de agregar una dependencia que no esté implícita en los documentos.