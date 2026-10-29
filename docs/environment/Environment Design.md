# Environment Design

Diseño del componente **Environment**: cómo se implementa un ambiente (simulador o robot
real), cómo se levanta como proceso y cómo lo consume Core. Deriva de
[`../Architecture.md`](../Architecture.md) (sección "Environment, WorldModel, Reward Model
y Dataset") y de [`../core/Core Design.md`](../core/Core%20Design.md) (sección "Ambientes").
Ante una duda sobre el protocolo hacia el Learner, mandan esos documentos.

**Objetivo de este diseño:** que agregar un ambiente nuevo sea escribir **un archivo
con dos o tres métodos y un YAML**, y que probar un robot o un simulador no exija
levantar el Learner ni Core.

## Principio rector

**Core no sabe qué ambiente hay detrás.** Core sostiene objetos que cumplen la ABC
`Environment` (en la práctica, `EnvironmentClient` hablando por ZeroMQ) y solo llama
`reset`, `step`, `close`, `emergency_stop` y `spec`. Todo lo específico de un simulador
o de un robot (su API, sus nombres de cámaras, su vector de estado, sus límites) vive en
un **adaptador** dentro de su propio proceso, que traduce al tipo canónico de
`vla_common`. Es el mismo principio que `Dataset Design.md` aplica a los corpus.

**`vla_env` y `vla_core` no dependen entre sí.** Se encuentran solo en `vla_common`, que
tiene el contrato (`Environment`, `EnvSpec`), el protocolo (`handle_environment`) y la
suite ENV. Es la misma división que ya existe entre el Learner y Core: el contrato y el
protocolo son puros, sin I/O, y cada proceso pone su propio transporte.

```
                         vla_common: Environment, EnvSpec, handle_environment, suite ENV
                          ▲                                        ▲
                          │                                        │
Learner ──> Core ──(EnvironmentClient, DEALER)──> EnvironmentServer (ROUTER), proceso vla-env
           vla_core                                        │
                                                 build_environment(config)
                                                           │
                                                  Adaptador (Environment)
                                               LIBERO · reach · robot real · …
```

## Decisiones

- **E1 — Paquete nuevo `packages/vla_env`, que depende solo de `vla_common` y `pyzmq`.**
  El Environment es un proceso aparte con dependencias pesadas (MuJoCo, robosuite, drivers
  de robot), a diferencia del Dataset, que vive dentro de Core. Separarlo deja a
  `vla_core` liviano y hace obvio qué código corre en qué proceso. Ni `vla_env` depende de
  `vla_core` ni al revés.
- **E2 — El contrato y el protocolo pasan a `vla_common`.** `Environment`, `EnvSpec` y
  `handle_environment` se mueven de `vla_core/environments.py` a
  `vla_common/environment.py`, y la suite ENV de `vla_core/testing/contracts/` a
  `vla_common/testing/contracts/`. Es lo que comparten los dos lados: Core lo consume y
  `vla_env` lo implementa. Ninguno de los tres hace I/O, así que `vla_common` sigue sin
  `zmq`. `vla_core` importa desde ahí directamente, sin reexportar, para que haya una
  sola ruta de import. `EnvironmentClient` y `EnvironmentRegistry` se quedan en `vla_core`
  porque son el lado de Core.
- **E2b — El servidor es parte del Environment.** `vla_env` tiene su propio
  `EnvironmentServer`: un loop `ROUTER` que llama a `handle_environment` con el sobre
  `ok`/`error`. Duplica unas 30 líneas del `RouterServer` de `vla_core.rpc`. Se acepta a
  cambio de no meter `pyzmq` en `vla_common`, como ya pasa con el `CoreClient` del
  Learner, que tiene su propio transporte. Los tests de red de Core siguen sirviendo el
  `FakeEnvironment` con su `RouterServer` y el mismo `handle_environment`, así que el
  protocolo es uno solo.
- **E3 — Una clase base para simuladores: `GymEnvironment`.** Casi todos los simuladores
  (LIBERO, MetaWorld, PushT, ALOHA) exponen una API tipo Gymnasium. La base implementa
  **una sola vez** las reglas del harness (chunk como un paso, corte a mitad de chunk,
  límite de pasos, éxito, semilla, validación de la acción) y el adaptador solo traduce.
  Es *duck typing*: no importa `gymnasium`, así que no agrega dependencias.
- **E4 — Un proceso por `env_id`.** Sin ambientes vectorizados en v1: N ambientes son N
  procesos `vla-env` con la misma config y distinta dirección. Es lo más simple, aísla
  fallas (Core ya marca un ambiente caído sin arrastrar al resto) y no toca el protocolo.
- **E5 — Factory `build_environment(config)` por `type`, con imports perezosos.** Igual que
  `build_dataset`. La dependencia de cada adaptador va en un grupo opcional de uv del
  paquete `vla_env` (`libero`, …), así que la instalación base no la trae.
- **E6 — LIBERO sobre `hf-libero` directo, no sobre `lerobot[libero]`.** El `LiberoEnv` de
  lerobot importa `torch` y `lerobot[libero]` exige `transformers>=4.57`, que choca con el
  `transformers` parcheado del extra `pi0`. El adaptador propio sobre `OffScreenRenderEnv`
  es corto (ver "LIBERO") y no necesita `torch`. El proceso de ambiente corre en su
  propio venv (`UV_PROJECT_ENVIRONMENT=.venv-env`), igual que Core ya usa `.venv-core` en
  `slurm/pi0_bc.sbatch`, así que no tiene que convivir instalado con `pi0`. Solo el lock
  debe resolver; si choca, se declaran extras en conflicto en `[tool.uv]`, como hoy
  `openvla`/`pi0`.
- **E7 — El éxito es explícito: `info["success"]`.** `terminated` significa "la tarea
  terminó" (éxito **o** fallo, según `Architecture.md`), así que no sirve para saber si
  hubo éxito. Todo ambiente devuelve `info["success"]: bool` en cada `StepResult`.
  `Core.ingest_episode` y la medición de tasa de éxito leen ese campo.
- **E8 — Evaluación de tasa de éxito a través de Core.** Medir si una política cumple la
  tarea usa los mismos Environment que el RL, conducidos por el Learner vía Core con
  `with_reward=False`. La evaluación formal contra benchmarks publicados sigue fuera de
  este harness, con vla-evaluation-harness, como dice `Architecture.md`. Esto ajusta la
  nota "La evaluación no usa Core ni sus ambientes" (ver "Cambios en otros documentos").

## Convenciones canónicas

Todo adaptador entrega exactamente esto. La suite ENV lo verifica.

| Campo | Convención |
|---|---|
| `Observation.images` | `dict[nombre canónico → array]`, `float32`, **CHW**, 3 canales, valores en `[0, 1]`. Es lo que ya usan `FakeEnvironment`, `LocalDataset` y los procesadores del Learner. |
| Nombres de cámara | Los canónicos de la config (`cameras: {canónico: clave del simulador}`), **los mismos** que usa el Dataset con el que se entrenó la política. |
| `Observation.proprio` | Vector `float32` 1-D, con el **mismo layout** que el estado del Dataset correspondiente. |
| `Observation.instruction` | Texto de la tarea; es exactamente la `task` pedida en `reset`. |
| `Action.values` | `float32`, forma `(spec.action_dim,)`, **unidades reales** (la política desnormaliza). |
| `StepResult.reward` | Siempre `None`; la recompensa la pone Core. |
| `StepResult.info` | `dict` con `"success": bool` obligatorio y, opcionalmente, otros valores escalares (`bool`, `int`, `float`, `str`) que msgpack serialice. Nunca arrays. |
| Límite de episodio | `max_steps` en **acciones de bajo nivel** (no en chunks), así no depende del `chunk_size` de la política. |

`FakeEnvironment` (el doble en memoria de los tests de Core) es la única excepción al
límite: cuenta chunks, para seguir siendo idéntico al `ToyReachEnv` del Learner, que es la
referencia de la suite CORE y de los tests de aprendizaje.

## La interfaz (en `vla_common/environment.py`)

```python
@dataclass
class EnvSpec:
    action_dim: int
    cameras: list[str]
    tasks: list[str]          # NUEVO: tareas que acepta reset(task=...)

class Environment(ABC):
    def reset(self, task: str | None, seed: int | None) -> Observation: ...
    def step(self, actions: ActionChunk) -> StepResult: ...   # reward=None, info["success"]
    def close(self) -> None: ...
    def emergency_stop(self) -> None: ...                     # no-op en simuladores
    @property
    def spec(self) -> EnvSpec: ...
```

Semántica obligatoria:

- `reset(task=None)` usa `spec.tasks[0]`; una `task` fuera de `spec.tasks` es un error
  (`ValueError`, que llega al Learner como `RuntimeError`).
- `reset(seed=s)` es **determinista**: la misma semilla da la misma observación, imágenes
  incluidas. GRPO lo necesita, porque reinicia todo un grupo con la misma semilla
  (`collector.py`). `seed=None` elige un estado inicial al azar.
- `step` ejecuta las acciones del chunk en orden y **se detiene en la primera** que
  termina el episodio (éxito, fallo o límite), devolviendo la observación en ese punto. El
  chunk cuenta como un solo paso.
- `step` con una acción de dimensión distinta de `spec.action_dim` es un error, **antes**
  de mover nada.
- `step` después de un final, sin `reset`, es un error. Core ya lo impide
  (`EpisodeAlreadyDone`); el ambiente lo repite para cuando se usa sin Core (smoke test,
  contrato).

Respecto de `Core Design.md` cambian tres cosas: dónde vive la interfaz (E2), el campo
`EnvSpec.tasks` y estas reglas, que antes eran implícitas.

En el mismo módulo hay dos reglas compartidas, para que Core y los ambientes no las dupliquen:

- `check_action_chunk(actions, action_dim)` levanta `ValueError` si el chunk está vacío o
  una acción no tiene forma `(action_dim,)`;
- `resolve_task(task, spec)` devuelve la tarea efectiva (`None` → `tasks[0]`) o levanta
  `ValueError` ("tarea desconocida").

También está `handle_environment(request, env) -> dict`, que traduce un
request del wire (`reset`, `step`, `close`, `emergency_stop`, `spec`) a una llamada sobre
`env` usando `vla_common.wire`. Es la única definición del protocolo del Environment: la
usan el `EnvironmentServer` de `vla_env` y los servidores de prueba de Core.

## La base para simuladores: `GymEnvironment` (en `vla_env`)

```python
class GymEnvironment(Environment):
    """Envuelve un simulador tipo Gymnasium y aplica las reglas del harness."""

    def __init__(self, spec: EnvSpec, max_steps: int): ...

    # --- Lo que implementa cada adaptador ---
    @abstractmethod
    def _make(self, task: str) -> Any:
        """Devuelve el simulador para ``task`` (puede reusar el actual si no cambió)."""

    @abstractmethod
    def _observation(self, raw: Any, task: str) -> Observation:
        """Traduce la observación cruda del simulador al tipo canónico."""

    def _success(self, raw_info: dict) -> bool:
        return bool(raw_info.get("is_success", False))   # convención de Gymnasium

    # --- Lo que resuelve la base (no se reimplementa) ---
    def reset(self, task, seed) -> Observation: ...
    def step(self, actions) -> StepResult: ...
    def close(self) -> None: ...
    def emergency_stop(self) -> None: ...   # no-op
```

Lo que hace la base, para que ningún adaptador lo repita:

1. `reset`: valida `task`, llama `_make(task)` y `sim.reset(seed=seed)`, reinicia el contador
   de acciones y devuelve `_observation(raw, task)`.
2. `step`: valida cada acción contra `action_dim`; por cada acción llama
   `sim.step(a) -> (raw, _, terminated, truncated, info)` e ignora la recompensa del
   simulador. Corta si `terminated`, `truncated`, `_success(info)` o si llegó a
   `max_steps`. Arma el `StepResult` con:
   - `terminated = terminated or success`;
   - `truncated = not terminated and (truncated or límite alcanzado)`;
   - `info = {"success": success}`.
3. Errores: un `step` después de un final, o una acción mal dimensionada, levanta
   `ValueError`/`RuntimeError` con un mensaje claro.
4. Helper `to_canonical_image(array)`: HWC o CHW, `uint8` o `float` → `float32` CHW en
   `[0, 1]`. Los adaptadores lo usan en `_observation`.

Un **robot real** no suele tener API Gymnasium: implementa `Environment` directamente
(ver "Robot real") y reusa `to_canonical_image`.

## El proceso: CLI `vla-env`

```bash
uv run vla-env --config configs/env/<archivo>.yaml                       # atiende a Core
uv run vla-env --config configs/env/<archivo>.yaml --address tcp://0.0.0.0:5601
uv run vla-env --config configs/env/<archivo>.yaml --smoke 50            # prueba local
```

- **Modo servidor (por defecto).** `build_environment(config["environment"])`, luego
  `EnvironmentServer(env, address).start()` hasta Ctrl-C, y al salir `env.close()`.
  `--address` pisa `server.address` para levantar N procesos con un solo YAML (E4).

```python
class EnvironmentServer:            # vla_env/server.py
    """Socket ROUTER que atiende el protocolo de un Environment (handle_environment)."""
    def __init__(self, env: Environment, address: str): ...
    def start(self) -> str: ...     # bindea, devuelve el endpoint real, atiende en un hilo
    def stop(self) -> None: ...
```
- **Modo `--smoke N` (para probar un robot o simulador con comodidad).** No abre sockets
  ni necesita Core. Construye el ambiente, imprime el `spec`, hace `reset` y N `step` con
  una acción neutra (`environment.smoke_action` del YAML, por defecto ceros), y reporta
  formas, dtypes y rangos de cada cámara y del `proprio`, el tiempo por `step`, y si
  terminó o tuvo éxito. Con `--save DIR` guarda las observaciones en un `npz` para
  inspeccionarlas (orientación de cámaras, escala del estado). Termina con
  `emergency_stop` y `close`.

YAML de un proceso de ambiente:

```yaml
# configs/env/libero.yaml
server: { address: "tcp://127.0.0.1:5601" }
environment:
  type: libero
  suites: [libero_spatial, libero_object, libero_goal, libero_10]
  cameras:                                   # nombre canónico -> cámara de LIBERO
    observation.images.base_0_rgb: agentview_image
    observation.images.left_wrist_0_rgb: robot0_eye_in_hand_image
  image_size: 256
  smoke_action: [0, 0, 0, 0, 0, 0, -1]       # no-op de LIBERO (gripper abierto)
```

## Factory: `build_environment(config)` (en `vla_env/factory.py`)

```python
def build_environment(config: dict) -> Environment:
    tipo = config.get("type", "reach")
    if tipo == "reach":
        from vla_env.reach import ReachEnvironment
        return ReachEnvironment(**{k: v for k, v in config.items() if k != "type"})
    if tipo == "libero":
        from vla_env.libero import LiberoEnvironment     # grupo opcional `libero`
        return LiberoEnvironment(**{k: v for k, v in config.items() if k != "type"})
    raise ValueError(f"tipo de ambiente desconocido: {tipo!r}")
```

Si falta el grupo opcional, el import perezoso falla con un `ImportError` que dice qué
extra instalar (`uv sync --extra libero`), igual que `build_dataset`.

**`ReachEnvironment` (`type: reach`) es el adaptador de referencia.** Es un simulador
"reach" 2D en numpy, sin dependencias, escrito como subclase de `GymEnvironment` sobre un
mini-simulador con API Gymnasium. Cumple tres funciones:

- es la **plantilla** que se copia para agregar un simulador;
- prueba la base `GymEnvironment` en `fast`;
- permite correr la tubería completa (procesos `vla-env` → Core → Learner) sin
  dependencias pesadas.

Cumple el mismo papel que `LocalDataset` en `Dataset Design.md`. No reemplaza al
`FakeEnvironment` de `vla_core/testing`, que sigue siendo el doble en memoria de los tests
de Core. `vla_env` no lo importa, así que los dos paquetes siguen sin depender entre sí.

## Cómo lo consume Core

Cambios chicos en `vla_core` (Core sigue siendo un gateway sin lógica de ambiente):

- **`build_core` arma el registro desde la config.** Si el YAML trae `environments:`, Core
  usa un `EnvironmentRegistry` normal con un `EnvironmentClient` por entrada (el resto de
  los componentes sigue como diga `backend`). Sin ese bloque, se queda el
  `FakeEnvironmentRegistry` actual, que crea ambientes al vuelo (lo usa la suite CORE).
  También se lee `timeouts:` (hoy documentado pero ignorado por el builder).

  ```yaml
  # configs/core/libero.yaml
  server: { address: "tcp://127.0.0.1:5555" }
  backend: { type: fake }
  environments:
    - { id: libero-0, address: "tcp://127.0.0.1:5601" }
    - { id: libero-1, address: "tcp://127.0.0.1:5602" }
  timeouts: { environment_s: 30, retries: 1 }   # un reset de LIBERO tarda más que 5 s
  ```

- **El spec se pide perezosamente** (primera vez que se usa, como hoy, y queda en caché).
  Así el orden de arranque de los procesos no importa. "Declararse al registrarse" queda
  cumplido por la entrada en la config más el `spec` remoto.
- **Core valida la dimensión de la acción** en el pre-chequeo de `step`, contra
  `spec.action_dim`, antes de tocar ningún ambiente. Es una barrera de seguridad barata:
  un chunk mal formado nunca llega a un robot. La suite CORE usa acciones de dimensión 2,
  la misma que el `spec` del fake, así que sigue verde.
- **`Core.ingest_episode` usa `info["success"]`** de la última transición en vez de
  `terminated` (E7).

## Cómo agregar un ambiente nuevo (simulador)

1. `packages/vla_env/src/vla_env/<nombre>.py`, partiendo de una copia de `reach.py`: una
   subclase de `GymEnvironment` con `_make`, `_observation` y, si el simulador no usa
   `info["is_success"]`, `_success`. En el docstring, el mapeo simulador → canónico
   (cámaras, estado, unidades de acción).
2. Si tiene dependencias: `uv add --package vla-env --optional <nombre> <paquete>`.
3. Una rama `type: <nombre>` en `build_environment`.
4. `configs/env/<nombre>.yaml`.
5. Probarlo: `uv run --extra <nombre> vla-env --config configs/env/<nombre>.yaml --smoke 20`,
   y después la suite ENV completa con
   `VLA_ENV_CONFIG=configs/env/<nombre>.yaml uv run pytest -m heavy packages/vla_env`.
   **No hace falta escribir tests nuevos**: `test_env_from_config.py` corre la suite ENV
   contra lo que construya ese YAML.
6. Conectarlo a Core: agregar sus procesos al bloque `environments:` de un YAML de Core.

## Robot real

Implementa `Environment` directamente (sin `GymEnvironment`) y sigue las mismas
convenciones. Además:

- **Límites de seguridad en el adaptador.** Una acción fuera de los límites físicos
  declarados en la config del robot se **rechaza** (error, el robot no se mueve); no se
  recorta en silencio, porque eso escondería errores de la política.
- **`emergency_stop` real**, que detiene el robot y no falla si ya está detenido.
- **Watchdog local obligatorio.** El servidor de ambiente atiende de a un pedido por vez:
  si un `step` se cuelga, la `emergency_stop` que manda Core queda en cola detrás de él.
  Por eso el robot no puede depender de la red para frenar: el adaptador lleva un
  watchdog (hilo o la función equivalente del controlador) que detiene el robot si no
  recibe un comando dentro de un plazo configurable. La `emergency_stop` por red es una
  segunda línea.
- **`reset` incluye el regreso a una pose segura** antes de devolver la observación.
- Primero se prueba con `--smoke` y una acción neutra; recién después, por Core.

## LIBERO (primer ambiente real)

`LiberoEnvironment(GymEnvironment)` en `vla_env/libero.py`, sobre `hf-libero` (extra
`libero` de `vla_env`). Replica en numpy, sin `torch`, lo que hacen el `LiberoEnv` y el
`LiberoProcessorStep` de lerobot 0.4.4, para que la observación coincida con la del
dataset `lerobot/libero` con el que se entrenó π0:

| Aspecto | Mapeo |
|---|---|
| Tareas | `spec.tasks` = el `language` de cada tarea de las `suites` configuradas (las 40 de spatial, object, goal y 10). `task` → (suite, `task_id`). |
| Simulador | `OffScreenRenderEnv(bddl_file_name=..., camera_heights/widths=image_size)`. `_make` lo recrea solo si cambia la tarea; conviene que el Learner agrupe los episodios por tarea. |
| `reset(seed)` | `sim.seed(seed)`, `sim.reset()`, `set_init_state(init_states[seed % N])` (un `seed=None` elige un índice al azar) y 10 pasos con la acción no-op `[0,0,0,0,0,0,-1]` para que los objetos se asienten. Control relativo (`use_delta=True`). |
| Imágenes | Cámaras de la config, **rotadas 180°** (se invierten H y W, como `LiberoProcessorStep`), luego `to_canonical_image`. |
| `proprio` | 8-D: `eef_pos` (3) + `quat2axisangle(eef_quat)` (3) + `gripper_qpos` (2). |
| Acción | `action_dim = 7`, en unidades reales. |
| Éxito | `sim.check_success()` → `_success`. |
| Límite | `max_steps` por suite: spatial 280, object 280, goal 300, 10 520, 90 400 (los de lerobot). |
| Render en cluster | `MUJOCO_GL=egl` en el sbatch. |

Hay que **verificar con `--smoke --save`** que la orientación de las imágenes y el layout
del estado coinciden con los del dataset convertido (`data/pi0_libero`) antes de evaluar.
Una discrepancia ahí no rompe ningún test, pero deja a la política ciega.

## Tests

| Archivo (en `packages/vla_env/tests/`) | Qué prueba | Marca |
|---|---|---|
| `test_env_gym.py` | Las reglas de la base, sobre un simulador de prueba que registra las acciones recibidas: corte a mitad de chunk, `max_steps` en acciones, éxito, error de dimensión, `step` después de un final, `task` desconocida, `to_canonical_image` | `fast` |
| `test_env_reach.py` | `ReachEnvironment` pasa la suite ENV en memoria y reproduce la dinámica del fake | `fast` |
| `test_env_server.py` | `EnvironmentServer` sirve un `ReachEnvironment` y el `EnvironmentClient` de Core pasa la suite ENV por la red | `fast` |
| `test_env_cli.py` | La factory (`reach`, tipo desconocido), `--smoke`, `--save` y `--address` | `fast` |
| `test_env_arch.py` | `vla_env` no importa `vla_core`, `vla_learner` ni `torch` (salvo perezoso) | `fast` |
| `test_env_from_config.py` | Suite ENV contra el ambiente de `VLA_ENV_CONFIG` (cualquier adaptador) | `heavy` |

**`vla_env` no depende de `vla_core` en runtime.** En los tests sí lo importa
(`test_env_server.py` usa el `EnvironmentClient` real), igual que los tests de `vla_core` ya
importan el `CoreClient` de `vla_learner`. Así se prueba el cliente real contra el
servidor real sin escribir uno de mentira. El cruce con procesos separados se prueba en
`vla_core` (tests `distributed`), que lanza `vla-env` como subproceso. El detalle con IDs
de test está en el plan de implementación.

La suite ENV (ahora en `vla_common/testing/contracts/environment_contract.py`) se extiende
con casos genéricos que valen para cualquier ambiente:

- **ENV-06** `reset(task=spec.tasks[0])` da `instruction == task`; una tarea desconocida es
  error.
- **ENV-07** un chunk con dimensión distinta de `action_dim` es error.
- **ENV-08** `info["success"]` es `bool` y todo valor de `info` es escalar.
- **ENV-09** imágenes `float32`, CHW, 3 canales, en `[0, 1]`; deterministas con la misma
  semilla.
- **ENV-04 (corregido)** repite `step` con una acción neutra hasta un final (con un tope)
  y recién ahí prueba `emergency_stop` y `close`, como pide `Test Core.md`.

El corte a mitad de chunk no se puede forzar de forma genérica: lo prueban
`test_env_gym.py` y un test propio de `FakeEnvironment` (FAKE-01).

## Plan por fases

El detalle de cada fase (archivos, tests con su ID, integración con los demás componentes,
criterio de cierre y riesgos) está en
[`Environment Implementation Plan.md`](./Environment%20Implementation%20Plan.md). En
resumen:

1. **Contrato en `vla_common` y convenciones** (`vla_common`, `vla_core`).
2. **Paquete `vla_env`** con `GymEnvironment`, `reach`, `EnvironmentServer` y la CLI.
3. **Core con ambientes por config**, contra procesos `vla-env` (`distributed`).
4. **Learner → Core → `vla-env` de punta a punta** (`slow`).
5. **Adaptador LIBERO** (`fast` + `heavy` en el cluster).
6. **Tasa de éxito de π0 en LIBERO**, por Core.

## Cambios en otros documentos

**Hechos** junto con este diseño. Lo que todavía no está en el código va marcado como
*pendiente* con la fase que lo trae.

- `Architecture.md`: nota de evaluación (E8): la tasa de éxito dentro del harness usa Core;
  los benchmarks formales siguen con vla-evaluation-harness. Agregar `info["success"]` y
  el layout canónico de imágenes a "Tipos de datos".
- `Core Design.md` y `Core Reference.md`: la interfaz del Environment vive en
  `vla_common`; `EnvSpec.tasks`, validación de `action_dim` en `step`, `ingest_episode` por
  `success`, bloque `environments:`/`timeouts:` del builder.
- `Test Core.md`: la suite ENV vive en `vla_common` y tiene la lista nueva; corregir WM
  (`rollout` devuelve `StepResult`, no `Observation`).
- `CLAUDE.md`: `packages/vla_env` en "Estructura del repositorio" y en "Estado del
  proyecto"; `vla_common` también lleva el contrato del Environment; la regla "las suites
  de contrato viven en `vla_core/testing/contracts/`" tiene la ENV como excepción; el
  Environment real sale de "Fuera".

## Decisiones abiertas

- **Recompensa por éxito del simulador para RL.** El Reward Model recibe un `ScoreItem`
  (instrucción, observación, acción, observación siguiente) y **no** ve `info`. Un
  "detector de éxito del simulador" como Reward Model (lo que `Architecture.md` menciona)
  necesita que `ScoreItem` lleve `info["success"]`, o un Reward Model que lo infiera de
  las imágenes. Se decide antes de hacer RL sobre LIBERO; la evaluación (fase 5) no lo
  necesita.
- **Ambientes vectorizados.** Solo si una medición muestra que N procesos de un ambiente
  no alcanzan (E4).
- **Restos del proceso Dataset borrado.** `configs/core/supervised.yaml`,
  `configs/core/dataset.yaml` y `scripts/run_supervised_e2e.py` apuntan a
  `vla-core-dataset`/`DatasetClient`, que se eliminaron en `10c7d68`. No es parte de este
  diseño, pero conviene limpiarlos.
