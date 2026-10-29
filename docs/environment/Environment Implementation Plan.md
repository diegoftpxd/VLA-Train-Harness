# Plan de implementación: Environment

Plan para implementar el diseño de [`Environment Design.md`](./Environment%20Design.md) e
integrarlo con el resto del harness (Core, Learner, Dataset, WorldModel, Reward Model y
el cluster). Ante una duda de diseño, manda ese documento. Si al implementar hace falta
desviarse, se actualiza el documento de diseño en el mismo commit.

## Reglas de trabajo

- **Rama `Environment`**, creada desde `main`. Un commit por fase, al cerrarla.
- **No se empieza una fase con la anterior en rojo.** Cada fase cierra con:

  ```bash
  uv run pytest -m fast
  uv run ruff check . && uv run ruff format --check .
  uv run pyright
  ```

  Además corre `-m slow` y/o `-m distributed` cuando la fase lo indica.
- **Todo código nuevo llega con sus tests.** No se relaja ni se salta ningún test para
  que pase. La suite CORE del Learner sigue verde en todas las fases: es el contrato del
  protocolo.
- **Al cerrar cada fase:** se quitan las marcas *pendiente* que la fase resuelve en
  `Core Design.md`, `Core Reference.md` y `Test Core.md`, y se actualiza "Estado del
  proyecto" en `CLAUDE.md`.
- **Dependencias con `uv add`** (nunca a mano), y el `uv.lock` va en el commit.
- **Pesos y simuladores reales solo en `heavy`**, y no se corren en la máquina de
  desarrollo: se corren en el cluster (fases 5 y 6).

## Mapa de fases

| Fase | Qué entrega | Toca | Marca de cierre | Dónde corre |
|---|---|---|---|---|
| 1 | Contrato en `vla_common` y convenciones | `vla_common`, `vla_core` | `fast` + `slow` | local |
| 2 | Paquete `vla_env` con el ambiente `reach` | `vla_env` (nuevo), raíz | `fast` | local |
| 3 | Core con ambientes por config, contra procesos `vla-env` | `vla_core` | `fast` + `distributed` | local |
| 4 | Learner → Core → `vla-env`, de punta a punta | tests, configs | `slow` | local |
| 5 | Adaptador LIBERO | `vla_env`, slurm | `fast` + `heavy` | local + cluster |
| 6 | Tasa de éxito de π0 en LIBERO | `scripts`, slurm | `fast` + `slow` + cluster | local + cluster |
| — | Prerrequisitos de RL sobre LIBERO (solo decisiones) | — | — | — |

Las fases 1 a 4 se cierran por completo en la máquina de desarrollo (CPU, sin pesos). Las
fases 5 y 6 tienen una parte local (`fast`) y una de cluster (`heavy`/sbatch).

---

## Fase 1 — Contrato en `vla_common` y convenciones

**Objetivo.** Mover el contrato del Environment a `vla_common` y aplicar las convenciones
del diseño (`tasks`, `info["success"]`, imágenes canónicas, corte a mitad de chunk,
validación de la acción) sin cambiar el comportamiento que ve el Learner.

### Cambios

1. **`packages/vla_common/src/vla_common/environment.py`** (nuevo):
   - `EnvSpec(action_dim, cameras, tasks)`, `Environment` (ABC) y `handle_environment`, que
     se mueven desde `vla_core/environments.py`. `spec` agrega `tasks` en el wire.
   - `check_action_chunk(actions: ActionChunk, action_dim: int) -> None`: levanta
     `ValueError` si alguna acción no es 1-D con largo `action_dim`. Es la **única**
     definición de esa regla: la usan Core (pre-chequeo de `step`) y `GymEnvironment`
     (fase 2).
2. **`packages/vla_common/src/vla_common/testing/contracts/environment_contract.py`** (se
   mueve desde `vla_core`) con ENV-01 a ENV-11 (ver [`Test Environment.md`](./Test%20Environment.md)). Se borra la copia de
   `vla_core/testing/contracts/` y se actualiza el `__init__` de esa carpeta.
3. **`vla_core/environments.py`** queda solo con `EnvironmentRegistry` y
   `EnvironmentClient`, e importa el contrato desde `vla_common.environment`, sin
   reexportar. `EnvironmentClient.spec` lee `tasks`.
4. **Imports actualizados** en `core.py`, `testing/fakes.py` y los tests
   `test_environment.py`, `test_env_registry.py`, `test_failures.py` y
   `test_core_distributed.py`.
5. **`FakeEnvironment`**:
   - `spec.tasks = ["reach the goal"]`; `reset(task=None)` usa `tasks[0]` y una tarea
     desconocida da `ValueError`.
   - `step` deja de aplicar el chunk completo: revisa la meta después de **cada** acción y
     corta ahí (hoy aplica todo el chunk y recién después mira).
   - `info = {"success": reached, "distance": ...}`.
   - El límite sigue contando chunks, para seguir idéntico a `ToyReachEnv` (excepción
     documentada en el diseño).
   - `FlakyEnvironment` agrega `tasks` a su `spec` y `success` a su `info`.
6. **`Core.step`**: en el pre-chequeo, por cada `env_id`, valida
   `check_action_chunk(chunk, spec.action_dim)`. Lee el `spec` dentro de `_usar_ambiente`,
   para que un timeout al pedirlo se escale igual que en `reset`/`step` (no disponible +
   parada de seguridad). Si algún chunk es inválido, ningún ambiente se mueve.
7. **`Core._episode_succeeded`**: usa `info.get("success", False)` de la última transición,
   en lugar de `terminated`.
8. **Documentos**: se quitan las marcas *pendiente* de la fase 1.

### Tests

| ID | Archivo | Qué verifica | Marca |
|---|---|---|---|
| ENV-01…11 | `vla_common/testing/contracts/environment_contract.py` | Contrato completo (lista en `Test Environment.md`); lo heredan `TestFakeEnvironment` y `TestEnvironmentClient` en `vla_core/tests/test_environment.py`, sin cambios en esas clases | `fast` |
| FAKE-01 | `vla_core/tests/test_environment.py` | Un chunk que alcanza la meta en la acción *k* devuelve la observación de esa acción, no la del final del chunk | `fast` |
| FAKE-02 | idem | `info["success"]` es verdadero si y solo si se alcanzó la meta; un `truncated` tiene `success=False` | `fast` |
| COM-01 | `vla_common/tests/test_environment.py` | `check_action_chunk` acepta la forma correcta y rechaza dimensión distinta, acción 2-D y chunk vacío | `fast` |
| COM-02 | idem | `handle_environment` devuelve `spec` con `tasks` y una llamada desconocida da error | `fast` |
| ARCH-ENV-01 | `vla_common/tests/test_arch.py` | Ningún módulo de `vla_common` (salvo `testing/`) importa `zmq`, `torch`, `vla_core`, `vla_learner` ni `vla_env` | `fast` |
| CENV-01 | `vla_core/tests/test_core_gateway.py` | `step` con una acción de dimensión incorrecta es error y **ningún** ambiente avanza (paso y observación del `EpisodeManager` intactos), también si el chunk inválido es el segundo de la lista | `fast` |
| CENV-02 | idem, por la red | Lo mismo con `CoreClient` → `CoreServer`: llega como `RuntimeError` | `fast` |
| CENV-03 | `vla_core/tests/test_core_dataset.py` | Un episodio `terminated` con `success=False` no se reinyecta (stub de ambiente que termina por fallo); los tests de ingesta existentes pasan sin tocar sus aserciones | `fast` |
| CENV-04 | `vla_core/tests/test_failures.py` | Si el `spec` no responde en el pre-chequeo, el ambiente se marca no disponible y se invoca su parada | `fast` |

### Integración verificada en esta fase

- **Learner (protocolo):** la suite CORE contra el Core real sigue verde (usa acciones de
  dimensión 2, igual que el `spec` del fake, y `reset` sin tarea).
- **Learner (aprendizaje):** `uv run pytest -m slow`. El corte a mitad de chunk cambia el
  comportamiento del fake solo con `chunk_size > 1`, y en la dirección favorable (se
  detiene en la meta). Si E2E cambia, se investiga antes de cerrar. El fallo intermitente
  conocido de `test_learn05` no cuenta como regresión.
- **Reward Model / WorldModel:** sin cambios. `FakeRewardModel` sigue leyendo
  `proprio[:2]` y `proprio[2:4]`.

### Cierre

`fast` + `slow` + ruff + pyright en verde. Commit: `feat(env): contrato del Environment en vla_common y convenciones`.

---

## Fase 2 — Paquete `vla_env`

**Objetivo.** Un proceso `vla-env` que sirve cualquier adaptador, con la base
`GymEnvironment` y el adaptador de referencia `reach`, más el modo `--smoke` para probar un
ambiente sin Core.

### Cambios

1. **Paquete** `packages/vla_env/`:
   - `pyproject.toml`: `name = "vla-env"`; dependencias `vla-common`, `pyzmq`, `numpy`,
     `pyyaml`, todas con `uv add --package vla-env ...` (ya están en el workspace); script
     `vla-env = "vla_env.cli:main"`; `[tool.uv.sources] vla-common = { workspace = true }`.
   - Raíz: `vla-env` en `dependencies` y en `[tool.uv.sources]`, y
     `packages/vla_env/tests` en `testpaths`. pyright ya incluye `packages`.
2. **`vla_env/gym.py`**: `GymEnvironment` y `to_canonical_image`, tal como los describe el
   diseño. La validación de la acción usa `check_action_chunk` de `vla_common`.
3. **`vla_env/reach.py`**: `ReachEnvironment(GymEnvironment)` sobre un mini-simulador
   `_ReachSim` en numpy con API Gymnasium (`reset(seed) -> (obs, info)`,
   `step(a) -> (obs, r, term, trunc, info)` con `info["is_success"]`). Replica a propósito
   el layout del `FakeEnvironment`, para ser compatible sin cambios con las configs tiny
   del Learner, con `FakeRewardModel` y con `FakeWorldModel`:
   - cámara `front`;
   - `proprio = [pos, goal]` (4-D);
   - `action_dim = 2`, `tasks = ["reach the goal"]`;
   - mismos `BOUND` y `THRESHOLD`.

   Diferencia: `max_steps` cuenta acciones (por defecto 20). La imagen es negra, como la
   del fake: el backbone tiny del Learner aplana la imagen junto al `proprio`, así que
   dibujar la posición cambiaría el aprendizaje y la E2E de la fase 4 dejaría de ser
   comparable. Igual pasa por `to_canonical_image`: el simulador la entrega en HWC
   `uint8`. El docstring es la plantilla que se copia para agregar un simulador.
4. **`vla_env/server.py`**: `EnvironmentServer(env, address)` con `start`/`stop` y
   context manager: un loop `ROUTER` que llama a `handle_environment` con el sobre
   `ok`/`error`.
5. **`vla_env/factory.py`**: `build_environment(config)` con `reach`. Si un import
   perezoso falla, da un `ImportError` con el `uv sync --extra ...` que falta. La rama
   `libero` entra en la fase 5, junto con su módulo (antes, pyright no la resolvería).
6. **`vla_env/cli.py`**:
   - Modo servidor por defecto; `--address` pisa `server.address`; Ctrl-C hace
     `stop()` + `env.close()`.
   - `--smoke N`: imprime el `spec`, hace `reset` y N `step` con
     `environment.smoke_action` (o ceros) e informa forma, dtype, mín y máx por cámara y
     del `proprio`, tiempo medio por `step`, `terminated`/`truncated`/`success`. Cierra
     con `emergency_stop` + `close`.
   - `--save DIR`: guarda un `npz` con las observaciones del smoke.
   - La lógica vive en funciones testeables (`build_server(config, address)`,
     `smoke(env, n, action) -> SmokeReport`); `main` solo parsea argumentos.
7. **`configs/env/reach.yaml`**.

### Tests (`packages/vla_env/tests/`, todos `fast`)

Los archivos llevan el prefijo `test_env_`: pytest no distingue dos archivos de test con el
mismo nombre en paquetes distintos (`vla_core` ya tiene `test_server.py`, el Learner `test_arch.py`).


| ID | Archivo | Qué verifica |
|---|---|---|
| GYM-01 | `test_env_gym.py` | Corte a mitad de chunk: el resultado es el de la acción que terminó y las acciones siguientes no llegan al simulador (el sim de prueba cuenta llamadas) |
| GYM-02 | idem | `max_steps` en acciones: un chunk que cruza el límite se corta ahí y da `truncated=True`, `terminated=False` |
| GYM-03 | idem | `info["is_success"]` del simulador → `terminated=True` y `success=True`, aunque el sim no marque `terminated` |
| GYM-04 | idem | La recompensa del simulador se ignora: `reward is None` |
| GYM-05 | idem | `step` después de un final, sin `reset`, es error |
| GYM-06 | idem | Una acción mal dimensionada es error y el simulador no recibe ninguna acción del chunk |
| GYM-07 | idem | `reset(task=None)` usa `tasks[0]`; una tarea desconocida es error; `_make` recibe la tarea pedida |
| GYM-08 | idem | `to_canonical_image`: HWC `uint8` → CHW `float32` en `[0, 1]`; CHW `float` se respeta; una forma sin 3 canales es error |
| REACH-01 | `test_env_reach.py` | `ReachEnvironment` pasa la suite ENV (ENV-01…11) en memoria |
| ARCH-ENV-02/03 | `test_env_arch.py` | `vla_env/src` no importa `vla_core`, `vla_learner` ni `torch` |
| REACH-02 | idem | Con `chunk_size = 1` reproduce la trayectoria del `FakeEnvironment` para la misma semilla y acciones (mismo `proprio`). Es lo que garantiza que las fases 3 y 4 midan lo mismo que la E2E actual. Se implementa replicando la fórmula, sin importar `vla_core` |
| FACT-01/02 | `test_env_cli.py` | `reach` se construye desde el YAML; un tipo desconocido da error. FACT-03 (`libero` sin el extra) pasa a la fase 5 |
| SRVE-01 | `test_env_server.py` | `EnvironmentServer(ReachEnvironment)` + `EnvironmentClient` de `vla_core` pasan la suite ENV por la red |
| SRVE-02 | idem | Un error del adaptador (tarea desconocida) vuelve como `{ok: False}` y el cliente levanta error; el servidor sigue atendiendo |
| CLI-01 | `test_env_cli.py` | `smoke(env, 5, ...)` sobre `configs/env/reach.yaml` devuelve un reporte con el spec, las formas y los tiempos; `main(["--config", ..., "--smoke", "5"])` termina con código 0 |
| CLI-02 | idem | `--save` escribe un `npz` con una clave por cámara y el `proprio` |
| CLI-03 | idem | `build_server` respeta `--address` por sobre `server.address` |

Sobre SRVE-01: **`vla_env` no depende de `vla_core` en runtime.** En los tests sí lo
importa, igual que los tests de `vla_core` ya importan el `CoreClient` de `vla_learner`.
Así se prueba el cliente real contra el servidor real, sin escribir un cliente de mentira
en el test (ver "Tests" en `Environment Design.md`).

### Cierre

`fast` + ruff + pyright en verde. Se prueba a mano
`uv run vla-env --config configs/env/reach.yaml --smoke 10`. Commit:
`feat(env): paquete vla_env con GymEnvironment, reach, servidor y CLI`.

---

## Fase 3 — Core con ambientes por config

**Objetivo.** Que Core se conecte a procesos `vla-env` reales declarados en su YAML, con
los timeouts de la config, y que una caída real de un proceso quede aislada.

### Cambios

1. **`build_core`**:
   - Si el YAML trae `environments:`, crea un `EnvironmentRegistry` normal (sin autoregistro)
     con un `EnvironmentClient(address, timeout_s, retries)` por entrada.
   - Valida que haya `id` y `address`, y que los `id` no se repitan (`ValueError` claro al
     arrancar).
   - Lee `timeouts.environment_s` y `timeouts.retries` (por defecto 5 s y 1).
   - Sin el bloque, nada cambia.
2. **`make_fake_core(environments=...)`**: acepta un registro inyectado, igual que hoy
   acepta `dataset`.
3. **Configs**: `configs/core/env_reach.yaml` (dos ambientes en `5601`/`5602`) y su
   comentario con los tres comandos para levantarlo a mano (dos `vla-env` y `vla-core`).

### Tests

| ID | Archivo | Qué verifica | Marca |
|---|---|---|---|
| CFG-01 | `vla_core/tests/test_builder_environments.py` | Con `environments:`, `build_core` registra un cliente por entrada con el timeout de la config; un `id` repetido o sin `address` es error | `fast` |
| CFG-02 | idem | Un `env_id` que no está en la config da `UnknownEnvironment` (sin autoregistro) | `fast` |
| CFG-03 | idem | Round-trip `reset`/`step` por `CoreClient` contra el Core construido desde el YAML, con los ambientes servidos en el mismo test por `RouterServer(handle_environment)` sobre `FakeEnvironment` (direcciones del YAML reemplazadas por puertos efímeros) | `fast` |
| CFG-04 | `test_configs.py` | `env_reach.yaml` se construye (sin round-trip: sus ambientes son procesos aparte) | `fast` |
| DIST-01 | `vla_core/tests/test_env_processes.py` | Core + dos **subprocesos** `python -m vla_env.cli --config configs/env/reach.yaml --address tcp://127.0.0.1:<efímero>`: `reset`/`step` enrutan, respetan el orden y registran una transición por ambiente | `distributed` |
| DIST-02 | idem | Se mata uno de los procesos: ese ambiente da `DownstreamUnavailable` y queda marcado no disponible; el otro sigue respondiendo | `distributed` |
| DIST-03 | idem | `CoreServer.shutdown()` (lo que hace `vla-core` al salir) manda `close` a cada proceso: el ambiente queda cerrado (un `step` pide `reset`) y vuelve a abrirse con `reset`. **Desvío:** al implementarlo se vio que nadie llamaba a `Core.close()` (el `close` del Learner no cierra el backend), así que se agregó `CoreServer.shutdown` y la CLI lo usa | `distributed` |

Los tests `distributed` lanzan `vla-env` con `sys.executable -m vla_env.cli`: usan el venv
del workspace sin importar `vla_env` desde `vla_core`. Esperan a que cada proceso responda
un `spec` (con tope de tiempo) antes de empezar.

### Integración verificada

- **Core ↔ `vla_env`** por un proceso real: transporte, `spec` perezoso, timeouts,
  aislamiento de fallas.
- **Learner:** sin cambios; la suite CORE sigue usando el registro fake.

### Cierre

`fast` + `distributed` + ruff + pyright en verde. Commit:
`feat(core): ambientes declarados en la config contra procesos vla-env`.

---

## Fase 4 — Learner → Core → `vla-env`, de punta a punta

**Objetivo.** Demostrar que el Learner real entrena por Core contra ambientes que son
procesos `vla-env`. Es la misma prueba que la E2E actual, pero con el Environment fuera
de Core, y no cambia ni el Learner ni el protocolo.

### Cambios

1. **Config del Learner** `configs/ppo_env_reach.yaml`: copia de `ppo_tiny.yaml` con
   `core: {type: zmq, address: ..., timeout_s: ...}` y
   `data.env_ids: [reach-0, reach-1]`.
2. **Helper compartido** `lanzar_vla_env(n)` en `vla_core/tests/conftest.py` (ya existe desde
   la fase 3): levanta N procesos `vla-env` `reach` y espera a que respondan. Lo reusan
   DIST-01…03 y E2E-ENV.

### Tests (`vla_core/tests/test_e2e_env.py`, marca `slow`)

| ID | Qué verifica |
|---|---|
| E2E-ENV-01 | **RL en ambiente:** PPO tiny conducido por el `Learner` real, por `CoreClient`, contra Core + 2 procesos `vla-env reach`. Tras el entrenamiento, la tasa de éxito supera la línea base con los **mismos umbrales** que el flujo RL de `test_e2e.py` (REACH-02 garantiza que es la misma dinámica). |
| E2E-ENV-02 | **RL en world model con inicios reales:** los inicios salen de `vla-env` por Core (`reset` de los `env_ids` declarados) y el rollout del `FakeWorldModel` (que lee `proprio = [pos, goal]`). El entrenamiento avanza con los umbrales del flujo WM de `test_e2e.py`. **Desvío:** el plan decía `start_selector: reset`, pero ese selector del Learner inventa `env_id` nuevos (`wm-start-N`) y depende del autoregistro, que un Core con ambientes declarados no hace; se usan los inicios fijos de `env_ids`, que alcanzan los mismos umbrales. Ver "Decisiones abiertas" en `Environment Design.md`. |
| E2E-ENV-03 | **Episodios y Dataset:** después de E2E-ENV-01, `get_episode` devuelve transiciones `source="env"` con `info["success"]`, y `ingest_episode` reinyecta exactamente los episodios exitosos en el `FakeDataset`, que `sample_supervised` sirve después. |

### Integración verificada

| Componente | Cómo se verifica |
|---|---|
| Learner | Entrena sin cambios de código (E2E-ENV-01/02) |
| Reward Model | `FakeRewardModel` puntúa observaciones producidas por otro proceso (E2E-ENV-01) |
| WorldModel | `encode` de observaciones reales de `vla-env` (E2E-ENV-02) |
| EpisodeStore y Dataset | Registro, `success` y reinyección (E2E-ENV-03) |

### Cierre

`fast` + `slow` + `distributed` + ruff + pyright en verde. Commit:
`test(env): Learner entrena por Core contra procesos vla-env`.

---

## Fase 5 — Adaptador LIBERO

**Objetivo.** `LiberoEnvironment` cumple la suite ENV en el cluster y su observación
coincide con la del dataset con el que se entrenó π0.

### Cambios

1. **Dependencia**: `uv add --package vla-env --optional libero hf-libero` y en la raíz
   el extra `libero = ["vla-env[libero]"]`.
   - El proceso de ambiente corre en su propio venv (`UV_PROJECT_ENVIRONMENT=.venv-env`),
     igual que Core usa `.venv-core` en `slurm/pi0_bc.sbatch`, así que no necesita
     convivir instalado con `pi0`.
   - El **lock** sí debe resolver. Si `hf-libero` choca con `openvla` o `pi0`, se declaran
     extras en conflicto en `[tool.uv]`, igual que `openvla`/`pi0` hoy.
2. **`vla_env/libero.py`**, separado en dos partes:
   - **Funciones puras de mapeo, en numpy, importables sin `hf-libero`:**
     - `quat2axisangle`;
     - `libero_proprio(raw) -> float32[8]` (`eef_pos`, eje-ángulo, `gripper_qpos`);
     - `libero_images(raw, cameras, ...)` (rotación 180° → `to_canonical_image`);
     - `index_tasks(suites) -> {language: (suite, task_id)}`;
     - `max_steps_for(suite)` (spatial 280, object 280, goal 300, 10 520, 90 400).
   - **`LiberoEnvironment(GymEnvironment)`**, que importa `libero` de forma perezosa:
     - `_make(task)` crea el `OffScreenRenderEnv` y lo recrea solo si cambia la tarea;
     - `reset` hace `seed`, estado inicial `init_states[seed % N]`, 10 pasos no-op
       `[0,0,0,0,0,0,-1]` y control relativo;
     - `_success` usa `check_success()`;
     - `max_steps` es el de la suite de la tarea actual.
3. **`configs/env/libero.yaml`** (ejemplo del diseño) y **`configs/core/libero.yaml`**
   (`environments:` con 30 s de timeout).
4. **`scripts/check_env_vs_dataset.py`**: compara el `npz` de `vla-env --smoke --save`
   con `data/pi0_libero` para la misma tarea:
   - mismas claves de cámara, forma y dtype;
   - rango de imágenes en `[0, 1]`;
   - `proprio` de 8-D dentro del rango mín/máx del dataset (con margen);
   - **orientación**: la correlación del primer frame del ambiente con los primeros frames
     del dataset es mayor que con su versión rotada 180°.

   Imprime un reporte y termina con código ≠ 0 si algo falla.
5. **`slurm/libero_env_check.sbatch`**:
   - variables: `MUJOCO_GL=egl` (o `osmesa` si el nodo no tiene GPU) y
     `UV_PROJECT_ENVIRONMENT=.venv-env`;
   - preparación: crea la config de LIBERO **antes** del primer import (ver "Riesgos");
   - pasos: `vla-env --smoke 50 --save`, la suite ENV `heavy` y `check_env_vs_dataset.py`.

   Se documenta en `slurm/README.md`.

### Tests

| ID | Archivo | Qué verifica | Marca |
|---|---|---|---|
| LIB-01 | `vla_env/tests/test_env_libero.py` | `quat2axisangle` contra valores conocidos (identidad, 90° en cada eje, el caso `w≈1`) | `fast` |
| LIB-02 | idem | `libero_proprio` arma los 8 valores en el orden del dataset desde un `raw` sintético | `fast` |
| LIB-03 | idem | `libero_images` rota 180° (un píxel marcado arriba a la izquierda termina abajo a la derecha) y entrega CHW `float32` en `[0, 1]` | `fast` |
| LIB-04 | idem | `index_tasks` con un benchmark falso: idiomas → (suite, `task_id`) y error si una tarea se repite; límites por suite | `fast` |
| LIB-08 | `test_env_libero.py` | El envoltorio `_LiberoSim` contra un `OffScreenRenderEnv` falso: semilla → `init_states[seed % N]`, pasos de asentamiento con la no-op, control relativo, `is_success` desde `check_success()`; `seed=None` elige un estado válido | `fast` |
| LIB-09 | idem | `ensure_libero_config` escribe los valores por defecto de LIBERO sin importarlo y no pisa una config existente | `fast` |
| FACT-03 | idem | `libero` sin el extra da un `ImportError` que nombra `uv sync --extra libero` | `fast` |
| MAP | `test_env_map_check.py` | `check_env_vs_dataset.py` sobre datos sintéticos detecta cada desajuste (cámara, dimensión y rango del estado, rotación, instrucción) y `main` devuelve 1 | `fast` |
| LIB-05 | `vla_env/tests/test_env_from_config.py` | Suites ENV y SRVE contra `VLA_ENV_CONFIG=configs/env/libero.yaml` (genérico: sirve para cualquier adaptador futuro; verificado localmente con `reach.yaml`) | `heavy` |
| LIB-06 | idem | Recrear el simulador al cambiar de tarea mantiene el contrato: `reset(task=a)`, `reset(task=b)` y `reset(task=a)` dan instrucciones correctas y el mismo estado inicial que la primera vez con la misma semilla | `heavy` |
| LIB-07 | `check_env_vs_dataset.py` en sbatch | Consistencia de la observación con el dataset (punto 4) | cluster |

### Cierre

- **Local:** `fast` + ruff + pyright en verde.
- **Cluster:** `libero_env_check.sbatch` en verde (suite ENV, smoke y consistencia), con
  el número de job anotado en `CLAUDE.md` como en las fases de π0.

Commit: `feat(env): adaptador LIBERO`.

---

## Fase 6 — Tasa de éxito de π0 en LIBERO

**Objetivo.** Medir qué fracción de episodios resuelve un checkpoint de π0 en las tareas
de LIBERO, conduciendo los ambientes por Core. Es el paso 3 de "Próximos pasos" de
`Pi0 E2E Plan.md`.

### Cambios

1. **`scripts/eval_success.py`**:
   - Carga la política desde la config del Learner y un checkpoint:
     `build_components(config)` y `Learner.resume(path)`. Si resulta incómodo (arma
     optimizador y fuente de datos que no se usan), se agrega un `build_policy` público al
     builder del Learner, como cambio aditivo documentado en `Learner Design.md`.
   - Por cada tarea de la lista (por CLI o YAML; el Learner no ve el `spec`) y por cada
     semilla:
     - `reset(env_ids, task, seed)`;
     - `step` con `policy.act(..., deterministic=True)` y `with_reward=False` hasta que
       cada ambiente termine;
     - cuenta `info["success"]`.
   - Reparte los episodios entre los `env_ids` disponibles, así N procesos `vla-env`
     evalúan en paralelo.
   - Salida: tabla por tarea y total en consola, más un JSON en `--out`.
   - La lógica vive en `evaluate(policy, client, tasks, episodes_per_task, env_ids, seed)`,
     testeable sin la CLI.
2. **`slurm/pi0_libero_eval.sbatch`**:
   - procesos: N `vla-env` LIBERO (`.venv-env`), Core (`.venv-core`) y el script con
     `--extra pi0` en la GPU, todos por sockets `ipc://` del job (no puertos `tcp` fijos;
     la config de Core se genera con esas direcciones);
   - cada venv se sincroniza una vez y los procesos arrancan con `uv run --no-sync`, para
     que N procesos no compitan por instalar el mismo venv;
   - variables: `TASKS` (obligatoria), `CHECKPOINT`, `N_ENVS`, `EPISODES`, `SEED`.

### Tests

| ID | Archivo | Qué verifica | Marca |
|---|---|---|---|
| EVAL-01 | `vla_core/tests/test_eval_success.py` | `evaluate` con una política tiny sin entrenar contra un Core fake por la red: devuelve una tasa en `[0, 1]` por tarea, cuenta por `success` (no por `terminated`) y reparte los episodios entre los `env_ids` | `fast` |
| EVAL-02 | idem | Una tarea desconocida para el ambiente da un error claro, sin cuelgues | `fast` |
| EVAL-03 | idem | Una política tiny entrenada con BC unas iteraciones tiene una tasa de éxito mayor que la sin entrenar, contra Core + procesos `vla-env reach` (`core_con_procesos` del `conftest.py`) | `slow` |
| EVAL-05 | idem | La config de evaluación: `--core-address` pisa `core.address`, y con checkpoint `normalizer.stats: fit` pasa a `<checkpoint>/policy` (nunca se re-ajusta contra el dataset del Core de evaluación); error claro si faltan las estadísticas; sin checkpoint, el último `step_*` | `fast` |
| EVAL-04 | sbatch | π0 entrenado en LIBERO (`slurm/pi0_bc.sbatch`) evaluado en un subconjunto de tareas; se registra la tasa por tarea | cluster |

### Cierre

- **Local:** `fast` + `slow` + ruff + pyright en verde.
- **Cluster:** EVAL-04 corrido, con el resultado y el número de job en `Pi0 E2E Plan.md` y
  `CLAUDE.md`.

Commit: `feat(eval): tasa de éxito por Core contra ambientes reales`.

---

## Prerrequisitos para RL sobre LIBERO (fuera de este plan)

Se deciden antes de entrenar π0 con PPO/GRPO en LIBERO. No se implementan en este plan:

1. **Recompensa.** El Reward Model recibe `ScoreItem` sin `info`, así que no ve
   `info["success"]`. Opciones:
   - agregar `info` (o `success`) a `ScoreItem`: cambio interno de Core, sin tocar el
     protocolo del Learner;
   - un Reward Model que infiera el éxito de las imágenes.
2. **Tarea por ambiente en el Learner.** `EnvRolloutSource` no pasa `task` a
   `collect_env`, así que hoy todos los episodios usan `tasks[0]`. Hace falta un
   `data.tasks` (asignación por ambiente o rotación) en el builder del Learner: cambio
   chico y aditivo, que se documenta en `Learner Design.md`.
3. **Largo de episodio vs. chunk.** Con `chunk_size = 50`, un episodio de 280–520
   acciones son 6–11 pasos de RL. Hay que revisar `steps_per_batch` y el descuento.
4. **π0 no tiene `LOG_PROB`** (flow matching): no es compatible con PPO/GRPO tal como
   están. Haría falta otra política o un objetivo distinto.

## Riesgos y mitigaciones

| Riesgo | Fase | Mitigación |
|---|---|---|
| El corte a mitad de chunk del fake cambia los resultados de la E2E actual | 1 | Se corre `slow` en la fase 1; solo afecta `chunk_size > 1` y en dirección favorable |
| `hf-libero` no resuelve en el lock junto a `openvla`/`pi0` | 5 | Extras en conflicto en `[tool.uv]`; el proceso de ambiente ya usa su propio venv |
| LIBERO pregunta de forma interactiva por su config en el primer import (rutas de datasets/assets) | 5 | El sbatch crea la config de LIBERO antes del primer import; el adaptador falla con un mensaje claro si falta |
| Render sin pantalla en el cluster | 5 | `MUJOCO_GL=egl` (`osmesa` sin GPU), fijado en el sbatch |
| `reset` de LIBERO (crear el sim + 10 pasos de asentamiento) supera el timeout de 5 s | 3, 5 | `timeouts.environment_s: 30` en `configs/core/libero.yaml` |
| Imagen o estado distintos de los del dataset, que dejan a la política ciega sin que falle ningún test | 5 | `check_env_vs_dataset.py` es requisito de cierre (LIB-07) |
| Memoria: cada proceso LIBERO carga MuJoCo y su render | 5, 6 | Un proceso por `env_id`; empezar con 2–4 por nodo y medir |
| El determinismo de MuJoCo puede variar entre máquinas | 5 | ENV-05/09 comparan dentro de un mismo proceso; no se comparan observaciones entre máquinas |
