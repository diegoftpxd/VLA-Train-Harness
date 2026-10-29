# Diseño de Core

Especificación de las clases e interfaces del proceso **Core**, derivada de [`../Architecture.md`](../Architecture.md). Core es el *gateway* entre el Learner y el resto del sistema: recibe pedidos del Learner, los reenvía al componente que corresponde, agrega la recompensa cuando se pide y registra cada transición. No tiene modelo ni GPU.

Este documento es la fuente de verdad de la implementación de Core, al mismo nivel que [`../learner/Learner Design.md`](../learner/Learner%20Design.md) lo es para el Learner. Las suites de contrato están en [`Test Core.md`](./Test%20Core.md).

## Principios de diseño

1. **Una responsabilidad por clase.** Core no es una clase gigante: es un coordinador delgado (`Core`) que delega en colaboradores, cada uno dueño de una sola cosa (transporte, registro de ambientes, estado de episodios, almacén, puntuación, datos). La prueba es poder nombrar la responsabilidad de cada clase en una frase.
2. **El gateway coordina; los colaboradores hacen.** `Core` arma la respuesta de cada llamada orquestando colaboradores, pero no implementa sus detalles (ni sockets, ni almacenamiento, ni puntuación).
3. **Core no cambia el protocolo.** Implementa exactamente la interfaz que el `CoreClient` del Learner ya consume y reutiliza `vla_common` (tipos, wire, serialización). Si algo del protocolo cambia, cambia en los dos lados y en su suite.
4. **Core no depende de `torch` ni del Learner.** `vla_core` importa `vla_common` y `zmq`; nunca `vla_learner` ni `torch`. Los componentes con modelo (WorldModel, Reward Model) viven detrás de interfaces y, en los tests, de *fakes* en CPU.
5. **El transporte es genérico sobre el contrato.** El servidor ZeroMQ envuelve *cualquier* implementación de la interfaz de Core. Por eso el `FakeCoreServer` del Learner (que envuelve `FakeCoreClient`) y el Core real comparten el mismo transporte y la misma suite CORE.

## El contrato: `CoreInterface`

Es la interfaz que el Learner ve. Ya existe de hecho: la declara `CoreClientProtocol` y la cumplen el `CoreClient` (red) y el `FakeCoreClient` (memoria). El Core real es **otra implementación de la misma interfaz**, solo que descompuesta en colaboradores en vez de resolver todo en una clase.

```python
class CoreInterface(Protocol):
    def reset(self, env_ids: list[str], task: str | None = None,
              seed: int | None = None) -> list[tuple[str, Observation]]: ...
    def step(self, env_ids: list[str], actions: list[ActionChunk],
             with_reward: bool) -> list[StepResult]: ...
    def simulate(self, start: Observation | tuple[str, int],
                 action_sequences: list[list[ActionChunk]],
                 with_reward: bool) -> list[list[StepResult]]: ...
    def sample_supervised(self, n: int,
                          filters: dict | None = None) -> list[tuple[Observation, ActionChunk]]: ...
    def get_episode(self, episode_id: str) -> list[Transition]: ...
    def close(self) -> None: ...
```

**Plan:** mover esta `Protocol` de `vla_learner/data/core_client.py` a `vla_common` (`vla_common.protocol`), para que cliente, servidor y ambos backends referencien una sola definición. Es un cambio chico y seguro: `vla_learner` la reexporta desde ahí y su suite sigue verde. Así `vla_core` la importa sin depender del Learner.

## Mapa de componentes

| Capa | Clase | Responsabilidad (una frase) | Paquete |
|---|---|---|---|
| Transporte | `CoreServer` | Poseer el socket `ROUTER`, recibir frames, despachar y responder | `vla_core` |
| Transporte | `dispatch_request` (función) | Traducir wire↔tipos y llamar al backend | `vla_common` |
| Orquestación | `Core` | Coordinar cada llamada entre los colaboradores | `vla_core` |
| Ambientes | `EnvironmentRegistry` | Mapear `env_id` → cliente de Environment + su spec + disponibilidad | `vla_core` |
| Ambientes | `Environment` (ABC), `EnvSpec`, `handle_environment` | Contrato y protocolo del Environment, compartidos con `vla_env` | `vla_common` |
| Ambientes | `EnvironmentClient` | Hablar el protocolo de un Environment por `DEALER` | `vla_core` |
| WorldModel | `WorldModel` (ABC) + `WorldModelClient` | `encode` una observación y `rollout` secuencias | `vla_core` |
| Reward | `RewardModel` (ABC) + `RewardModelClient` | Puntuar un batch de transiciones (`score`) | `vla_core` |
| Reward | `RewardScorer` | Armar el batch de puntuación y adjuntar recompensas | `vla_core` |
| Episodios | `EpisodeManager` | Llevar el estado en vuelo de cada episodio | `vla_core` |
| Episodios | `EpisodeStore` (ABC) + `InMemoryEpisodeStore` | Guardar y recuperar transiciones (`append`/`get`) | `vla_core` |
| Datos | `Dataset` / `AppendableDataset` (ABC) + `FakeDataset` | Servir pares supervisados (`sample`) y reinyectar demos (`add`) | `vla_core` |
| Red | `RpcChannel` | Request/response confiable (DEALER + timeout + reintentos) a un proceso | `vla_core` |
| Errores | `CoreError` y subclases | Errores de dominio que el servidor traduce a `{ok: False, error}` | `vla_core` |

Los *fakes* (`FakeEnvironment`, `FakeWorldModel`, `FakeRewardModel`, `FakeDataset`, `FakeEpisodeStore`) viven en `vla_core/.../testing/` e implementan las mismas ABCs; son las que se usan para construir y probar Core sin componentes reales.

## Transporte

Tres responsabilidades separadas: mover bytes, traducir el protocolo, y la lógica.

### `CoreServer`

Dueño del socket `ROUTER`. Su loop recibe un mensaje multiparte, separa la identidad del remitente del resto de los frames, llama a `dispatch_request` y devuelve la respuesta a esa identidad. Envuelve cualquier backend que cumpla `CoreInterface`.

```python
class CoreServer:
    def __init__(self, backend: CoreInterface, address: str): ...
    def start(self) -> str: ...   # bindea, devuelve el endpoint real, atiende en un hilo
    def stop(self) -> None: ...
```

No contiene lógica de negocio ni conoce los tipos del dominio: solo frames y despacho. Es casi idéntico al `FakeCoreServer` actual; la diferencia es que recibe el backend por parámetro en vez de crear un `FakeCoreClient`.

### `dispatch_request` (en `vla_common`)

Función pura, sin `zmq`: recibe los frames decodificados y el backend, traduce el request a los tipos del dominio (con `vla_common.wire`), llama al método correspondiente del backend y vuelve a traducir el resultado a wire. Envuelve el resultado en `{ok: True, ...}` o, ante una excepción, en `{ok: False, error: "..."}`.

```python
def dispatch_request(request: dict, backend: CoreInterface) -> dict: ...
```

Vive en `vla_common` porque es genérica sobre el contrato y la usan dos servidores: el `CoreServer` real y el `FakeCoreServer` del Learner (que así deja de duplicar el despacho). Es también el único lugar donde se decide el mapeo llamada→método.

## El gateway: `Core`

Implementa `CoreInterface`. Es delgado: cada método es una secuencia corta de llamadas a colaboradores. No abre sockets, no guarda transiciones a mano, no sabe cómo se puntúa: pide esas cosas.

```python
class Core:
    def __init__(self, environments: EnvironmentRegistry, episodes: EpisodeManager,
                 store: EpisodeStore, scorer: RewardScorer,
                 world_model: WorldModel, dataset: Dataset): ...
```

Responsabilidad de cada método (el detalle de los flujos está más abajo):

- `reset`: por cada `env_id`, pide el cliente al registro, llama `reset`, abre un episodio en `EpisodeManager` y devuelve `(episode_id, observación)`.
- `step`: enruta cada chunk a su Environment, respeta las reglas de episodio, puntúa el batch con `RewardScorer` si `with_reward`, registra cada `Transition` y devuelve los `StepResult`.
- `simulate`: resuelve el punto de partida, usa el `WorldModel` para imaginar las secuencias, las puntúa con el mismo `RewardScorer` y devuelve un `StepResult` por paso de cada secuencia.
- `sample_supervised`: delega en `Dataset.sample(n, filters)` (acciones en unidades reales; el Learner las marca con `source="dataset"`).
- `ingest_episode` (gancho, fuera del protocolo): reinyecta un episodio exitoso del almacén en el Dataset (ver "Datos").
- `get_episode`: delega en `EpisodeStore.get`.
- `close`: cierra ambientes y canales.

## Colaboradores

### Ambientes: `EnvironmentRegistry`, `Environment`, `EnvironmentClient`

El contrato del Environment (`Environment`, `EnvSpec`) y su protocolo (`handle_environment`) viven en **`vla_common/environment.py`**, no en `vla_core`. Los comparten Core, que los consume, y el paquete `vla_env`, que los implementa y sirve. Así ninguno de los dos depende del otro. La racionalidad, las convenciones y las reglas de semántica están en [`../environment/Environment Design.md`](../environment/Environment%20Design.md); aquí va lo que Core necesita saber.

Además del contrato, `vla_common.environment` tiene dos reglas que comparten Core y los ambientes, para no duplicarlas: `check_action_chunk(actions, action_dim)` (la forma de cada acción) y `resolve_task(task, spec)` (`None` → `tasks[0]`; una tarea desconocida es error).

```python
# vla_common/environment.py
@dataclass
class EnvSpec:
    action_dim: int
    cameras: list[str]
    tasks: list[str]            # tareas que acepta reset(task=...); task=None usa tasks[0]

class Environment(ABC):
    @abstractmethod
    def reset(self, task: str | None, seed: int | None) -> Observation: ...
    @abstractmethod
    def step(self, actions: ActionChunk) -> StepResult: ...  # reward=None; info["success"]: bool
    @abstractmethod
    def close(self) -> None: ...
    @abstractmethod
    def emergency_stop(self) -> None: ...  # robot real; no-op en simuladores
    @property
    @abstractmethod
    def spec(self) -> EnvSpec: ...

def check_action_chunk(actions: ActionChunk, action_dim: int) -> None: ...   # ValueError si no cumple
def resolve_task(task: str | None, spec: EnvSpec) -> str: ...
def handle_environment(request: dict, env: Environment) -> dict: ...   # wire → llamada sobre env
```

```python
# vla_core/environments.py
class EnvironmentRegistry:
    """Mapa env_id -> (Environment, disponibilidad). Resuelve, no enruta lógica."""
    def get(self, env_id: str) -> Environment: ...        # UnknownEnvironment si no existe
    def register(self, env_id: str, env: Environment) -> None: ...
    def is_available(self, env_id: str) -> bool: ...      # registrado y no caído
    def mark_unavailable(self, env_id: str) -> None: ...  # Core lo marca al caerse
    def mark_available(self, env_id: str) -> None: ...    # al recuperarse
    def available(self) -> list[str]: ...                 # solo los disponibles
    def registered(self) -> list[str]: ...                # todos (p. ej. para close)
```

`EnvironmentClient(Environment)` implementa la interfaz hablando por `RpcChannel` a un proceso Environment (un `vla-env`, que expone un `ROUTER`). `FakeEnvironment(Environment)` la implementa en memoria para los tests de Core. El registro se puebla desde el bloque `environments:` de la configuración (ver "Construcción desde configuración"); sin ese bloque, el backend fake crea un `FakeEnvironment` al vuelo por `env_id`. `register` queda disponible para auto-registro dinámico más adelante. El `spec` se pide perezosamente la primera vez que se usa y queda en caché, así que el orden de arranque de los procesos no importa.

**Nota:** `Environment.step` devuelve el `StepResult` **sin** recompensa (`reward=None`); la recompensa la agrega Core con el `RewardScorer`, porque el Environment no conoce la instrucción ni el Reward Model. Sí devuelve `info["success"]: bool`, porque `terminated` no distingue entre éxito y fallo.

**Validación de la acción.** En el pre-chequeo de `step`, antes de tocar ningún ambiente, Core verifica que cada acción tenga forma `(spec.action_dim,)`. Un chunk mal formado es un error y nunca llega a un robot.

### WorldModel: `WorldModel`, `WorldModelClient`, `LatentState`

```python
@dataclass
class LatentState:
    values: Array        # estado latente del world model (contenido propio del modelo)

class WorldModel(ABC):
    @abstractmethod
    def encode(self, observation: Observation) -> LatentState: ...
    @abstractmethod
    def rollout(self, latent: LatentState,
                action_sequences: list[list[ActionChunk]]) -> list[list[StepResult]]: ...
```

`rollout` devuelve un `StepResult` **por chunk de cada secuencia**, con la observación ya decodificada y `reward=None` (la recompensa la agrega Core). Una secuencia se corta antes si el paso predicho termina el episodio. **Decisión (Fase 4, desvío del borrador que devolvía solo `Observation`):** la terminación (`terminated`/`truncated`) es parte de la dinámica que el WorldModel conoce y Core, *model-free*, no podría inferirla de una observación suelta; devolver `StepResult` mantiene esa información y hace que `simulate` produzca pasos coherentes con el ambiente real (mismos datos que el `ToyWorldModel` del Learner). Core sigue sin tocar el espacio latente. `WorldModelClient` habla por `RpcChannel`; `FakeWorldModel` reproduce la dinámica reach.

### Reward: `RewardModel`, `RewardModelClient`, `RewardScorer`

```python
@dataclass
class ScoreItem:
    instruction: str
    observation: Observation
    action: ActionChunk
    next_observation: Observation

class RewardModel(ABC):
    @abstractmethod
    def score(self, items: list[ScoreItem]) -> list[float]: ...   # acepta batches

class RewardScorer:
    """Arma el batch de puntuación y devuelve una recompensa por item, en una sola llamada."""
    def __init__(self, reward_model: RewardModel): ...
    def score(self, items: list[ScoreItem]) -> list[float]: ...
```

El `RewardScorer` existe para aislar la regla **"una llamada a `score` por paso del loop"**: junta las transiciones de todos los ambientes (o de todas las secuencias de `simulate`) en un solo batch y reparte las recompensas de vuelta. `Core.step`/`Core.simulate` le pasan los items y reciben la lista de recompensas; no hablan con el Reward Model directamente.

### Episodios: `EpisodeManager` y `EpisodeStore`

Dos responsabilidades distintas: el estado **en vuelo** y el **almacén**.

```python
class EpisodeManager:
    """Estado vivo de cada episodio: id, paso actual, observación anterior, si terminó."""
    def open(self, env_id: str, observation: Observation) -> str: ...   # crea y devuelve episode_id
    def episode_of(self, env_id: str) -> str: ...
    def last_observation(self, env_id: str) -> Observation: ...
    def advance(self, env_id: str, next_observation: Observation,
                terminated: bool, truncated: bool) -> int: ...          # devuelve el nuevo step
    def is_done(self, env_id: str) -> bool: ...                         # para la regla de episodio

class EpisodeStore(ABC):
    @abstractmethod
    def append(self, transition: Transition) -> None: ...
    @abstractmethod
    def get(self, episode_id: str) -> list[Transition]: ...
```

`EpisodeManager` tiene la observación anterior (la necesita el Reward Model) y hace cumplir que un `step` sobre un episodio `terminated`/`truncated` sea un error hasta el próximo `reset` (`EpisodeAlreadyDone`). `InMemoryEpisodeStore` guarda en memoria; una variante en disco se agrega sin cambiar la interfaz. El `source` de cada transición lo fija quien la crea (`"env"` en `step`, `"world_model"` en `simulate`).

### Datos: `Dataset`, `AppendableDataset`

```python
class Dataset(ABC):
    @abstractmethod
    def sample(self, n: int, filters: dict | None = None) -> list[tuple[Observation, ActionChunk]]: ...

class AppendableDataset(Dataset):
    @abstractmethod
    def add(self, pairs: list[tuple[Observation, ActionChunk]]) -> int: ...   # devuelve cuántos agregó
```

Vive dentro de Core. Entrega acciones en unidades reales, sin normalizar. `filters` es un diccionario opcional de criterios (`task`, `embodiment`, …); cada implementación respeta los que entiende e ignora el resto. `FakeDataset` genera pares deterministas (como hoy hace `FakeCoreClient.sample_supervised`) y respeta el filtro `task` (la instrucción de los pares).

**Adaptadores concretos.** El `Dataset`/`AppendableDataset` (ABC) es la única frontera que Core ve; los formatos concretos viven detrás, elegidos por la factory `build_dataset(config)` según `type` (ver [[Dataset Design]]). Disponibles: `fake` (en memoria), `local` (layout on-disk, sin dependencias), `lerobot` (corpus LeRobot, grupo opcional uv) y `openx` (Open-X Embodiment / RLDS, grupo opcional uv). Core queda así agnóstico del formato y de cómo se consulta cada corpus.

**Gancho de episodios exitosos (Fase 5).** `AppendableDataset` separa la capacidad de **reinyectar** demos de la de solo leerlas: un corpus real de solo-lectura implementa solo `Dataset`; un dataset que acepta demos nuevas implementa `AppendableDataset`. Sobre esa capacidad, `Core.ingest_episode(episode_id)` toma las transiciones que Core registró para ese episodio, aplica el filtro de éxito (la última transición tiene `info["success"]` verdadero; `terminated` solo no alcanza, porque un episodio también termina por fallo) y agrega sus pares `(observación, acción)` al Dataset; devuelve cuántos agregó (`0` si no fue exitoso) y levanta `DatasetNotAppendable` si el Dataset no es agregable. Solo reinyecta lo real del almacén (`source="env"`), nunca transiciones imaginadas. El gancho vive en Core (es Core quien registró los episodios), apoyado en `AppendableDataset`; no es parte del `CoreInterface` que el Learner invoca por el protocolo.

### Red: `RpcChannel`

```python
class RpcChannel:
    """Request/response confiable a un proceso downstream: DEALER + timeout + reintentos."""
    def __init__(self, address: str, timeout_s: float, retries: int): ...
    def call(self, request: dict) -> dict: ...   # DownstreamTimeout si se agotan los reintentos
```

Es el gemelo hacia abajo del `CoreClient`: encapsula el socket `DEALER`, el `poll` con timeout, los reintentos (con reconexión del socket entre intentos) y la traducción de fallas a errores de dominio. Lo usan `EnvironmentClient`, `WorldModelClient` y `RewardModelClient`, para no repetir la mecánica de ZeroMQ en cada uno. El detalle del comportamiento ante fallas y la escalada a `DownstreamUnavailable` están en "Manejo de fallas".

### Errores de dominio

```python
class CoreError(Exception): ...
class UnknownEnvironment(CoreError): ...
class EpisodeAlreadyDone(CoreError): ...
class DownstreamTimeout(CoreError): ...
class DownstreamUnavailable(CoreError): ...
class DatasetNotAppendable(CoreError): ...   # ingest_episode sobre un Dataset de solo-lectura
```

`dispatch_request` captura `CoreError` (y cualquier excepción) y la devuelve como `{ok: False, error}`; el `CoreClient` la vuelve a levantar como `RuntimeError` en el Learner. Nada se cae por un pedido mal formado.

## Manejo de fallas (Fase 6)

Dos niveles, cada uno con una responsabilidad:

**Transporte (`RpcChannel`).** Un solo request confiable: envía, espera `timeout_s`, y ante un silencio reconecta el socket `DEALER` (recrearlo evita que una respuesta tardía desalinee las siguientes) y reintenta hasta `retries` veces. Si **todos** los intentos (`retries + 1`) se agotan, levanta `DownstreamTimeout`. También deja el socket reconectado al salir, así que una llamada posterior a un componente que se recuperó vuelve a funcionar (reconexión). El `close` de los clientes (`EnvironmentClient`, …) es best-effort: no propaga el timeout si el proceso ya no responde.

**Gateway (`Core`).** Core escala el timeout del transporte a una política de disponibilidad, solo para los ambientes (el robot es lo que puede quedar en un estado inseguro). Toda llamada a un Environment pasa por `_usar_ambiente`, que:

1. resuelve el ambiente con `EnvironmentRegistry.get` (autoregistro en el fake; `UnknownEnvironment` si no existe);
2. si está marcado no disponible, levanta `DownstreamUnavailable` sin tocarlo;
3. ejecuta la operación y, ante `DownstreamTimeout`, **marca el ambiente no disponible**, invoca su `emergency_stop` (best-effort: no-op en simulador, frenado en el robot; un fallo de la parada no enmascara la caída) y levanta `DownstreamUnavailable`.

Así una caída de un ambiente no arrastra a los demás: el resto sigue disponible y operando. Los timeouts de WorldModel y Reward Model se propagan como `DownstreamTimeout` (no tienen estado de disponibilidad en v1; su caída aborta el `step`/`simulate` en curso y el Learner reintenta).

**Timeouts y reintentos por componente (decisión, antes abierta).** `Architecture.md` los dejaba a la implementación. Se fijan como defaults configurables en el YAML (`timeouts:`), con este criterio: un paso de ambiente es corto y una respuesta perdida suele ser transitoria, así que basta un timeout chico con un reintento; el WorldModel y el Reward Model hacen trabajo más pesado (rollout, un VLM juez), así que llevan un timeout mayor.

| Componente | `timeout_s` | `retries` | Motivo |
|---|---|---|---|
| Environment | 5 | 1 | `step`/`reset` cortos; una pérdida suele ser transitoria |
| WorldModel | 10 | 1 | `rollout` de varias secuencias es más pesado |
| Reward Model | 10 | 1 | batch de `score`, posiblemente un VLM |

## Flujos

**`step` con recompensa** (el caso central):

1. `Core.step` valida, antes de tocar ningún ambiente, que cada `env_id` esté disponible (`EnvironmentRegistry`), que su episodio no esté terminado (`EpisodeManager`; `EpisodeAlreadyDone` si lo está) y que cada acción tenga la dimensión de `spec.action_dim`.
2. Por cada `env_id`: `EnvironmentRegistry.get(env_id).step(chunk)` → `StepResult` sin recompensa; guarda la observación anterior (de `EpisodeManager`).
3. Si `with_reward`: arma un `ScoreItem` por ambiente y llama **una vez** a `RewardScorer.score(items)`; adjunta cada recompensa a su `StepResult`.
4. Por cada ambiente, crea la `Transition` (`source="env"`, `step` de `EpisodeManager`) y la manda a `EpisodeStore.append`; luego `EpisodeManager.advance`.
5. Devuelve la lista de `StepResult`.

**`reset`:** por cada `env_id`, `Environment.reset` → observación; `EpisodeManager.open` crea el `episode_id` y guarda la observación inicial; devuelve `(episode_id, observación)`.

**`simulate`:** resuelve `start` (una `Observation`, o `(episode_id, step)` buscado en `EpisodeStore`); `WorldModel.encode` → latente; `WorldModel.rollout` → `StepResult` por chunk de cada secuencia (con `terminated`/`truncated`, `reward=None`); arma los `ScoreItem` de todas las secuencias (la observación previa del paso j es el inicio o el paso j-1) y puntúa en **una sola** llamada; devuelve los `StepResult` con la recompensa adjunta. Las transiciones imaginadas **no** se guardan en el almacén por ahora (igual que el `FakeCoreClient` de referencia); el Learner las marca con `source="world_model"` al armar su batch.

**`sample_supervised` / `get_episode`:** delegación directa en `Dataset` y `EpisodeStore`.

## Construcción desde configuración

Igual que el Learner, Core se arma desde un YAML y se levanta con `uv run vla-core --config configs/core/<archivo>.yaml`. Un `build_core(config)` (en `vla_core/builder.py`) construye el registro de ambientes, los clientes (reales o fakes según el `type`), el `Dataset`, el `EpisodeStore`, el `RewardScorer`, el `Core` y el `CoreServer`. Para no acoplar `vla_core` al Learner, Core tiene su propio builder; si se quiere un `Registry` por nombre como el del Learner, se lo sube antes a `vla_common` y lo comparten los dos (decisión abierta).

Bloques del YAML:

- **`backend`** elige el gateway (`fake`: `Core` con componentes fake; `static`: eco de transporte).
- **`dataset`**, si está, reemplaza al `FakeDataset` por el de `build_dataset`.
- **`environments`**, si está, reemplaza el registro fake por un `EnvironmentRegistry` con un `EnvironmentClient` por entrada, cada uno apuntando a un proceso `vla-env`. Sin este bloque, el registro fake crea un `FakeEnvironment` por `env_id` al vuelo (lo usa la suite CORE).
- **`timeouts`** fija los timeouts y reintentos de los clientes (ver "Manejo de fallas").

Ejemplo con ambientes reales (`configs/core/libero.yaml`):

```yaml
server: { address: "tcp://127.0.0.1:5555" }
backend: { type: fake }
environments:
  - { id: libero-0, address: "tcp://127.0.0.1:5601" }
  - { id: libero-1, address: "tcp://127.0.0.1:5602" }
timeouts: { environment_s: 30, retries: 1 }   # un reset de LIBERO tarda más que 5 s
```

**Estado:** los bloques `environments` y `timeouts` son la fase 3 de `Environment Design.md`. Hasta entonces, `build_core` solo lee `backend` y `dataset`.

## Estructura del paquete

```
packages/vla_core/
  pyproject.toml            # [project.scripts] vla-core; deps: vla-common, pyzmq, msgpack, numpy
  src/vla_core/
    server.py               # CoreServer (ROUTER)
    core.py                 # Core (gateway)
    environments.py         # EnvironmentClient, EnvironmentRegistry (el contrato está en vla_common.environment)
    world_model.py          # WorldModel (ABC), WorldModelClient, LatentState
    reward.py               # RewardModel (ABC), RewardModelClient, RewardScorer, ScoreItem
    episodes.py             # EpisodeManager, EpisodeStore (ABC), InMemoryEpisodeStore
    dataset/                # Dataset/AppendableDataset (ABC), build_dataset (factory), LocalDataset, LeRobotDataset, OpenXDataset
    rpc.py                  # RpcChannel
    errors.py               # CoreError y subclases
    builder.py              # build_core(config)
    cli.py                  # vla-core
    testing/
      fakes.py              # FakeEnvironment, FakeWorldModel, FakeRewardModel, FakeDataset, ...
      contracts/            # suites de contrato (las importan implementaciones futuras)
  tests/
```

## Relación con las herramientas de prueba

- Cada ABC (`WorldModel`, `RewardModel`, `Dataset`, `EpisodeStore`) trae su suite de contrato en `testing/contracts/`, que su *fake* y cualquier implementación real deben pasar. La suite ENV es la excepción: vive en `vla_common/testing/contracts/` junto al contrato del Environment, porque la importan tanto `vla_core` (fake y cliente) como `vla_env` (adaptadores).
- La **suite CORE** que ya existe (hoy corre `CoreClient` contra `FakeCoreServer`) se reusa: ahora también corre `CoreClient` contra `CoreServer(Core(...))` con componentes fake. Es la prueba de que el Core real respeta el protocolo.
- El `FakeCoreClient` del Learner sigue siendo la referencia monolítica del comportamiento esperado; `Core` debe producir los mismos resultados, pero con las responsabilidades separadas.

## Decisiones abiertas

- **Dónde vive `CoreInterface`.** Recomendado: moverla a `vla_common.protocol` y reexportarla desde `vla_learner`. Alternativa: que `vla_core` declare una `Protocol` equivalente. (Se resuelve en la Fase 1.)
- **`Registry` compartido vs. builder propio de Core.** Por ahora, builder propio; subir `Registry` a `vla_common` solo si se quiere el mismo mecanismo por nombre en los dos procesos.
- ~~**Timeouts y reintentos por componente.**~~ **Resuelto (Fase 6):** defaults configurables por componente, con su motivo, en "Manejo de fallas".
- **Persistencia del `EpisodeStore`.** `InMemoryEpisodeStore` para v1; una variante en disco (para reusar episodios como datos supervisados y entrenar el WorldModel) cuando se pida.
- **Registro de ambientes dinámico.** v1 lo puebla desde config (bloque `environments`); el auto-registro (`register` por mensaje) queda definido pero sin implementar hasta que haga falta.
```
