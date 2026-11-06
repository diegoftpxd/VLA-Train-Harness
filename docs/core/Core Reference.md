# Referencia de Core

Manual de uso del proceso **Core** ya implementado (`packages/vla_core`). Reúne en un solo lugar qué expone, cómo se levanta, qué consume, cómo se comporta y cómo se extiende. Es la vista de **uso**; la racionalidad del diseño está en [`Core Design.md`](./Core%20Design.md), las suites de contrato en [`Test Core.md`](./Test%20Core.md) y la visión transversal en [`../Architecture.md`](../Architecture.md).

> Core es un **gateway** en CPU: recibe pedidos del Learner, los reenvía al componente que corresponde, agrega la recompensa cuando se pide y registra cada transición. No tiene modelo ni GPU, no decide cuándo actuar (el loop lo conduce el Learner) y no importa `torch`.

## Índice

1. [Cómo se levanta](#cómo-se-levanta)
2. [La interfaz hacia el Learner](#la-interfaz-hacia-el-learner)
3. [Tipos de datos](#tipos-de-datos)
4. [Componentes que consume](#componentes-que-consume)
5. [Reglas de dominio](#reglas-de-dominio)
6. [Los tres flujos](#los-tres-flujos)
7. [Manejo de fallas y disponibilidad](#manejo-de-fallas-y-disponibilidad)
8. [Gancho de episodios exitosos](#gancho-de-episodios-exitosos)
9. [Configuración (YAML)](#configuración-yaml)
10. [Transporte y protocolo](#transporte-y-protocolo)
11. [Errores de dominio](#errores-de-dominio)
12. [Mapa de módulos](#mapa-de-módulos)
13. [Pruebas](#pruebas)
14. [Puntos de extensión](#puntos-de-extensión)
15. [Restricciones](#restricciones)

## Cómo se levanta

Core es un miembro del workspace de `uv`. Se arranca desde un YAML:

```bash
uv run vla-core --config configs/core/fake.yaml
```

La CLI carga el YAML, llama a `build_core(config)` para armar un `CoreServer`, lo bindea y lo deja atendiendo hasta `Ctrl-C`. Configs de ejemplo en `configs/core/`:

- **`fake.yaml`** — gateway real con todos los componentes fake en memoria (reach 2D, CPU). Es el Core que conduce la integración extremo a extremo con el Learner.
- **`echo.yaml`** — backend estático de prueba: responde datos canónicos, sin componentes. Sirve para validar solo el transporte.
- **`local.yaml`, `local_openvla.yaml`, `lerobot.yaml`, `openx.yaml`, `pi0_pusht.yaml`, `pi0_libero.yaml`** — gateway fake con un Dataset real (ver `docs/dataset/Dataset Design.md`).
- **`env_reach.yaml`** — gateway fake con dos ambientes reales: procesos `vla-env` del ambiente `reach` (sus comentarios traen los comandos).
- **`libero.yaml`** — gateway fake con dos procesos `vla-env` de LIBERO (evaluación de π0; ver `slurm/pi0_libero_eval.sbatch`).

En código, el servidor se usa directo:

```python
from vla_core.server import CoreServer
from vla_core.testing.fakes import make_fake_core

server = CoreServer(make_fake_core(), "tcp://127.0.0.1:0")
address = server.start()   # bindea y atiende en un hilo; devuelve el endpoint real
...
server.shutdown()          # stop() + cierra el backend (manda `close` a cada ambiente)
```

`stop()` solo libera el socket; `shutdown()` además cierra el backend, y es lo que hace `vla-core` al salir. El `close` que manda el Learner por el protocolo **no** cierra el backend.

El Learner se conecta con su `CoreClient` (lado cliente, en `vla_learner`); Core es el servidor.

## La interfaz hacia el Learner

Core implementa `CoreInterface` (definida en `vla_common.protocol`), el mismo contrato que consume el `CoreClient` del Learner. Seis llamadas; cada respuesta viaja en un sobre `{ok: True, ...}` o `{ok: False, error}`.

| Llamada | Entrada | Salida | Qué hace |
|---|---|---|---|
| `reset(env_ids, task=None, seed=None)` | ids de ambiente, tarea y semilla | `list[(episode_id, Observation)]` | Abre un episodio por ambiente y devuelve su observación inicial |
| `step(env_ids, actions, with_reward)` | ids, un `ActionChunk` por ambiente, flag de recompensa | `list[StepResult]` | Ejecuta cada chunk en su ambiente, puntúa el batch si `with_reward`, registra una transición por ambiente |
| `simulate(start, action_sequences, with_reward)` | inicio (`Observation` o `(episode_id, step)`) y secuencias de chunks | `list[list[StepResult]]` | Imagina las secuencias sobre el WorldModel y las puntúa con el mismo Reward Model |
| `sample_supervised(n, filters=None)` | cantidad y filtros opcionales | `list[(Observation, ActionChunk)]` | Sirve demos supervisadas del Dataset, en unidades reales |
| `get_episode(episode_id)` | id de episodio | `list[Transition]` | Devuelve las transiciones registradas del episodio |
| `close()` | — | — | Cierra los ambientes y los canales |

Notas de semántica:

- El **orden** de la salida respeta el orden de los `env_ids`/secuencias de entrada.
- `with_reward=False` deja `reward=None` en cada `StepResult` (mismo tipo de retorno).
- Las observaciones y acciones viajan **sin pérdida** (igualdad bit a bit a través del wire).
- `get_episode` sobre un `episode_id` desconocido es error; un episodio recién reseteado sin pasos aún no está en el almacén.

## Tipos de datos

Son los de `vla_common` (compartidos con el Learner; Core no los redefine):

- **`Observation`** — `images: dict[str, Array]` (p. ej. `{"front": CHW float32}`), `proprio: Array` (float32), `instruction: str`.
- **`Action`** — `values: Array`. **`ActionChunk` = `list[Action]`** (una secuencia de acciones que cuenta como un paso del loop).
- **`StepResult`** — `observation`, `reward: float | None` (la agrega Core), `terminated: bool`, `truncated: bool`, `info: dict`.
- **`Transition`** — `episode_id`, `step: int`, `source: "env" | "world_model" | "dataset"`, `observation`, `action: ActionChunk`, `result: StepResult`. Es lo que guarda el almacén y lo que el Learner arma para su batch.

## Componentes que consume

Cada dependencia es una interfaz abstracta (`ABC`) con **un fake CPU** para desarrollo y pruebas y **un cliente ZeroMQ** (`DEALER`) para hablar con un proceso real. Core nunca ejecuta física ni corre modelos: pide.

| Interfaz | Métodos | Cliente de red | Fake |
|---|---|---|---|
| `Environment` | `reset`, `step`, `close`, `emergency_stop`, `spec` | `EnvironmentClient` | `FakeEnvironment` (reach 2D) |
| `WorldModel` | `encode`, `rollout` | `WorldModelClient` | `FakeWorldModel` |
| `RewardModel` | `score` (batch) | `RewardModelClient` | `FakeRewardModel` |
| `Dataset` / `AppendableDataset` | `sample` / `add` | — (vive dentro de Core en v1) | `FakeDataset` |
| `EpisodeStore` | `append`, `get` | — | `InMemoryEpisodeStore` |

Detalles:

- **`Environment.step`** devuelve el `StepResult` con `reward=None`: el ambiente no conoce la instrucción ni el reward model. Declara su espacio con **`EnvSpec(action_dim, cameras, tasks)`** y marca el éxito en `info["success"]`. `emergency_stop` es la parada de seguridad del robot (no-op en simuladores). El contrato (`Environment`, `EnvSpec`, `handle_environment`) vive en `vla_common.environment` y los ambientes reales en el paquete `vla_env` (proceso `vla-env`); ver [`../environment/Environment Design.md`](../environment/Environment%20Design.md).
- **`WorldModel.rollout`** devuelve un `StepResult` por chunk de cada secuencia (con `terminated`/`truncated`, `reward=None`), cortando una secuencia si el paso predicho termina el episodio. Trabaja en su espacio latente (`LatentState`); Core no lo toca.
- **`RewardModel.score`** recibe una lista de `ScoreItem(instruction, observation, action, next_observation)` y devuelve una recompensa por item, en orden; acepta batch vacío.
- **`RewardScorer`** envuelve al reward model y aísla la regla "una llamada a `score` por paso del loop".
- **`EnvironmentRegistry`** mapea `env_id` → `Environment` y lleva su disponibilidad (`get`, `register`, `is_available`, `mark_unavailable`, `mark_available`, `available`, `registered`).
- **`EpisodeManager`** lleva el estado en vuelo (id, paso actual, observación anterior, si terminó); **`EpisodeStore`** guarda y recupera las transiciones cerradas.

## Reglas de dominio

Invariantes que Core hace cumplir en todos los flujos:

- **Una llamada a `score` por paso.** En `step` y en `simulate`, Core junta las transiciones de todos los ambientes/secuencias en un solo batch y llama a `score` una sola vez; luego reparte las recompensas.
- **Regla de episodio.** Un `step` sobre un episodio `terminated`/`truncated` es error (`EpisodeAlreadyDone`) hasta el próximo `reset`.
- **Acción bien formada.** Antes de tocar ningún ambiente, `step` verifica que cada acción tenga forma `(spec.action_dim,)`; si no, es error y nada se mueve. El `spec` se pide a través del mismo escalón de fallas que `reset`/`step`.
- **El chunk es un paso.** Si el episodio termina a mitad de un chunk, se corta y se devuelve la observación en ese punto; el chunk cuenta como **una** transición y **una** recompensa.
- **`source` lo fija quien crea la transición.** `step` registra `source="env"`; las transiciones imaginadas de `simulate` no se guardan (el Learner las marca `source="world_model"`); las demos del Dataset son `source="dataset"`.
- **Observación anterior.** Core guarda la observación previa de cada ambiente (la necesita el reward model) y la usa como `observation` de la transición.

## Los tres flujos

**Supervisado.** El Learner llama `sample_supervised(n)` → Core delega en `Dataset.sample` y devuelve pares `(Observation, ActionChunk)` en unidades reales → el Learner calcula la loss de imitación.

**RL en el ambiente.** `reset(env_ids)` → el Learner genera un chunk por ambiente con el modelo → `step(env_ids, actions, with_reward=True)`: Core enruta cada chunk a su Environment, arma un `ScoreItem` por ambiente, puntúa el batch en una llamada, adjunta la recompensa, registra la transición (`source="env"`) y avanza el episodio.

**RL sobre el world model.** El Learner llama `simulate(start, action_sequences, with_reward=True)`: Core resuelve el inicio (una `Observation`, o `(episode_id, step)` buscado en el almacén), hace `encode` + `rollout`, puntúa **todas** las transiciones imaginadas con el **mismo** reward model en una sola llamada (para que las escalas real e imaginada coincidan) y devuelve un `StepResult` por paso de cada secuencia. No guarda lo imaginado.

## Manejo de fallas y disponibilidad

Dos niveles, cada uno con su responsabilidad:

- **Transporte (`RpcChannel`).** Un request confiable: envía, espera `timeout_s`, y ante el silencio reconecta el socket `DEALER` y reintenta hasta `retries` veces. Si se agotan todos los intentos, levanta `DownstreamTimeout`. Deja el socket reconectado, así que una llamada posterior a un componente que **se recuperó** vuelve a funcionar. El `close` de los clientes es best-effort (no propaga el timeout de un proceso ya caído).
- **Gateway (`Core`).** Toda llamada a un Environment pasa por un escalón de disponibilidad: si el ambiente ya está marcado no disponible, levanta `DownstreamUnavailable` sin tocarlo; si no responde (`DownstreamTimeout`), Core lo **marca no disponible**, invoca su `emergency_stop` (best-effort) y levanta `DownstreamUnavailable`. **Una caída no arrastra al resto**: los demás ambientes siguen disponibles y operando.

Los timeouts de WorldModel y Reward Model se propagan como `DownstreamTimeout` (sin estado de disponibilidad en v1): abortan el `step`/`simulate` en curso y el Learner reintenta.

**Defaults por componente** (configurables; motivo en [`Core Design.md`](./Core%20Design.md#manejo-de-fallas-fase-6)):

| Componente | `timeout_s` | `retries` |
|---|---|---|
| Environment | 5 | 1 |
| WorldModel | 10 | 1 |
| Reward Model | 10 | 1 |

## Gancho de episodios exitosos

`Core.ingest_episode(episode_id) -> int` reinyecta en el Dataset los pares `(observación, acción)` de un episodio **exitoso** (su última transición tiene `info["success"]` verdadero; `terminated` solo no alcanza, porque un episodio también termina por fallo), para que `sample_supervised` los sirva después. Devuelve cuántos pares agregó (`0` si no fue exitoso). Requiere un `AppendableDataset` (`add`); con un Dataset de solo-lectura levanta `DatasetNotAppendable`. Solo reinyecta lo real del almacén (`source="env"`), nunca transiciones imaginadas. **No** es parte del protocolo que el Learner invoca por el wire: es un gancho interno de Core (es Core quien registró los episodios).

## Configuración (YAML)

`build_core(config)` arma el `CoreServer` desde el diccionario cargado del YAML:

```yaml
server:  { address: "tcp://127.0.0.1:5555" }   # dónde bindea el ROUTER
backend: { type: fake, demo_chunk_size: 1 }    # qué backend envuelve
```

| Campo | Valores | Default |
|---|---|---|
| `server.address` | endpoint ZeroMQ (`tcp://`, `ipc://`) | `tcp://127.0.0.1:5555` |
| `backend.type` | `fake` (gateway real + componentes fake) · `static` (eco de prueba) | `static` |
| `backend.demo_chunk_size` | tamaño del chunk de las demos del `FakeDataset` | `1` |

El `fake` ensambla el gateway real con `FakeEnvironment` (autoregistrado por `env_id` al vuelo), `FakeWorldModel`, `FakeRewardModel`, `FakeDataset` e `InMemoryEpisodeStore`. El bloque `dataset` (opcional) reemplaza al `FakeDataset` por el que construye `build_dataset`.

**Ambientes reales.** Un bloque `environments:` reemplaza el registro fake por un `EnvironmentClient` por entrada, cada uno apuntando a un proceso `vla-env`, sin autoregistro; `timeouts:` fija timeout y reintentos de esos clientes. Ejemplo listo para usar: `configs/core/env_reach.yaml` (sus comentarios traen los tres comandos para levantarlo). Al salir, `vla-core` hace `CoreServer.shutdown()`, que cierra cada ambiente:

```yaml
environments:
  - { id: libero-0, address: "tcp://127.0.0.1:5601" }
  - { id: libero-1, address: "tcp://127.0.0.1:5602" }
timeouts: { environment_s: 30, retries: 1 }
```

Cablear WorldModel y Reward Model reales desde el YAML llega cuando esos componentes existan (los clientes `DEALER` ya están implementados y probados).

## Transporte y protocolo

- Core expone un socket **`ROUTER`** al Learner; se conecta como **`DEALER`** a Environment, WorldModel y Reward Model. `ipc://` en la misma máquina, `tcp://` entre máquinas.
- Los mensajes se arman con `vla_common.wire` + `vla_common.serialization`: encabezado en **msgpack** (tipo de llamada, ids, `dtype`/`shape` de cada array) y los arrays como **buffers crudos** (`np.frombuffer` al recibir). **Nada de `pickle` entre procesos.**
- `dispatch_request` (en `vla_common`) traduce wire↔tipos y mapea cada llamada a su método del backend; lo usan el `CoreServer` real y el `FakeCoreServer` del Learner, así que Core pasa la **misma suite de contrato** que el `CoreClient` ya consumía.
- Se usa `DEALER`/`ROUTER` (no `REQ`/`REP`) justamente para poder poner timeouts y reintentos sin reiniciar el socket.

## Errores de dominio

`dispatch_request` captura cualquier excepción y la devuelve como `{ok: False, error}`; el `CoreClient` la vuelve a levantar como `RuntimeError` en el Learner.

| Error | Cuándo |
|---|---|
| `CoreError` | base de las fallas de dominio |
| `UnknownEnvironment` | se pidió un `env_id` que no está en el registro |
| `EpisodeAlreadyDone` | `step` sobre un episodio ya `terminated`/`truncated` |
| `DownstreamTimeout` | un componente no respondió tras timeout + reintentos |
| `DownstreamUnavailable` | un ambiente caído/marcado no disponible |
| `DatasetNotAppendable` | `ingest_episode` sobre un Dataset de solo-lectura |

## Mapa de módulos

```
packages/vla_core/src/vla_core/
  server.py         # CoreServer (socket ROUTER); stop / shutdown
  core.py           # Core (el gateway)
  environments.py   # EnvironmentClient, EnvironmentRegistry (el contrato está en vla_common.environment)
  world_model.py    # WorldModel (ABC), WorldModelClient, LatentState
  reward.py         # RewardModel (ABC), RewardModelClient, RewardScorer, ScoreItem
  episodes.py       # EpisodeManager, EpisodeStore (ABC), InMemoryEpisodeStore
  dataset/          # Dataset/AppendableDataset (ABC), build_dataset, adaptadores local/lerobot/openx
  rpc.py            # RpcChannel (DEALER), RouterServer, make_handler
  errors.py         # CoreError y subclases
  builder.py        # build_core(config), build_environments(entries, timeouts)
  cli.py            # entrypoint vla-core
  testing/
    fakes.py          # FakeEnvironment/WorldModel/RewardModel/Dataset, FlakyEnvironment, SilentRouter, make_fake_core
    static_backend.py # StaticCoreBackend (eco de transporte)
    contracts/        # suites de contrato reutilizables (la ENV vive en vla_common.testing.contracts)
```

## Pruebas

Marcas (en `pyproject.toml`, con `--strict-markers`):

| Marca | Qué cubre | Comando |
|---|---|---|
| `fast` | Contratos de Core y de sus componentes con fakes CPU; reglas de Core sobre los ambientes (CENV); configs de ejemplo (CFG); evaluación (EVAL) | `uv run pytest -m fast` |
| `distributed` | Varios Environment por la red atendidos por un Core, incluidos procesos `vla-env` reales (DIST) | `uv run pytest -m distributed` |
| `slow` | Integración extremo a extremo Learner↔Core, los tres flujos, con ambientes en memoria (E2E) y como procesos `vla-env` (E2E-ENV); EVAL-03 | `uv run pytest -m slow` |

`uv run pytest` sin marca excluye `heavy`, `gpu` y `distributed`. Calidad: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`.

## Puntos de extensión

Para reemplazar un fake por un componente real, se implementa la `ABC` correspondiente y se hereda su suite de contrato:

- **Environment real** → un adaptador en el paquete `vla_env` (subclase de `GymEnvironment` para simuladores, o de `Environment` para un robot), servido por `vla-env` y consumido por `EnvironmentClient`. Pasa la suite **ENV**, declara su `EnvSpec` y, si es un robot, tiene una `emergency_stop` de verdad y un watchdog local. Pasos en [`../environment/Environment Design.md`](../environment/Environment%20Design.md).
- **WorldModel real** → implementa `encode`/`rollout` (suite **WM**); se asume preentrenado y congelado.
- **Reward Model real** → implementa `score` por batch (suite **RM**); puede ser un detector de éxito, un modelo aprendido o un VLM juez.
- **Dataset real** (p. ej. Open-X / LeRobot) → implementa `Dataset.sample` (suite **DSET**), en unidades reales; `AppendableDataset` solo si admite reinyección.
- **Almacén en disco** → implementa `EpisodeStore` (suite **STORE**) sin cambiar nada más.

Estas suites viven en `vla_core/testing/contracts/` precisamente para que una implementación futura las importe. La ENV es la excepción: pasa a `vla_common/testing/contracts/`, porque la importan tanto `vla_core` como `vla_env`.

## Restricciones

- **CPU, sin modelo, sin `torch`.** Core solo enruta mensajes y arrays de numpy. Los componentes con GPU (WorldModel, Reward Model) son procesos aparte; en los tests son fakes CPU livianos.
- **No reescribe el protocolo.** Reutiliza `vla_common` y debe pasar la suite CORE del Learner; cualquier cambio de protocolo se hace en los dos lados y en su suite.
- **Core no conduce el loop.** Es un gateway: atiende, reenvía, agrega recompensa si se pidió y registra. El loop lo lleva el Learner.
