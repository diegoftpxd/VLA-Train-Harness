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
| Ambientes | `Environment` (ABC) + `EnvironmentClient` | Hablar el protocolo de un Environment (`reset`/`step`/`close`) | `vla_core` |
| WorldModel | `WorldModel` (ABC) + `WorldModelClient` | `encode` una observación y `rollout` secuencias | `vla_core` |
| Reward | `RewardModel` (ABC) + `RewardModelClient` | Puntuar un batch de transiciones (`score`) | `vla_core` |
| Reward | `RewardScorer` | Armar el batch de puntuación y adjuntar recompensas | `vla_core` |
| Episodios | `EpisodeManager` | Llevar el estado en vuelo de cada episodio | `vla_core` |
| Episodios | `EpisodeStore` (ABC) + `InMemoryEpisodeStore` | Guardar y recuperar transiciones (`append`/`get`) | `vla_core` |
| Datos | `Dataset` (ABC) + `FakeDataset` | Servir pares supervisados (`sample`) | `vla_core` |
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
- `sample_supervised`: delega en `Dataset.sample`.
- `get_episode`: delega en `EpisodeStore.get`.
- `close`: cierra ambientes y canales.

## Colaboradores

### Ambientes: `EnvironmentRegistry`, `Environment`, `EnvironmentClient`

```python
@dataclass
class EnvSpec:
    action_dim: int
    cameras: list[str]

class Environment(ABC):
    @abstractmethod
    def reset(self, task: str | None, seed: int | None) -> Observation: ...
    @abstractmethod
    def step(self, actions: ActionChunk) -> StepResult: ...  # reward=None; lo agrega Core
    @abstractmethod
    def close(self) -> None: ...
    @abstractmethod
    def emergency_stop(self) -> None: ...  # robot real; no-op en simuladores
    @property
    @abstractmethod
    def spec(self) -> EnvSpec: ...

class EnvironmentRegistry:
    """Mapa env_id -> (Environment, disponibilidad). Resuelve, no enruta lógica."""
    def get(self, env_id: str) -> Environment: ...        # UnknownEnvironment si no existe
    def register(self, env_id: str, env: Environment) -> None: ...
    def available(self) -> list[str]: ...
```

`EnvironmentClient(Environment)` implementa la interfaz hablando por `RpcChannel` a un proceso Environment. `FakeEnvironment(Environment)` la implementa en memoria (p. ej. envolviendo `ToyReachEnv`) para los tests. El registro se puebla desde configuración en v1; `register` queda disponible para auto-registro dinámico más adelante.

**Nota:** `Environment.step` devuelve el `StepResult` **sin** recompensa (`reward=None`); la recompensa la agrega Core con el `RewardScorer`, porque el Environment no conoce la instrucción ni el Reward Model.

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
                action_sequences: list[list[ActionChunk]]) -> list[list[Observation]]: ...
```

`rollout` devuelve **observaciones ya decodificadas** (una lista por secuencia, una observación por chunk). Con eso Core arma las transiciones imaginadas y las puntúa sin tocar el espacio latente: Core sigue siendo *model-free*. (Esto resuelve la decisión abierta de `Architecture.md` sobre dónde se decodifica: lo hace el WorldModel dentro de `rollout`.) `WorldModelClient` habla por `RpcChannel`; `FakeWorldModel` envuelve algo como `ToyWorldModel`.

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

### Datos: `Dataset`

```python
class Dataset(ABC):
    @abstractmethod
    def sample(self, n: int, filters: dict | None = None) -> list[tuple[Observation, ActionChunk]]: ...
```

Vive dentro de Core. Entrega acciones en unidades reales, sin normalizar. `FakeDataset` genera pares deterministas (como hoy hace `FakeCoreClient.sample_supervised`). El gancho para reinyectar episodios exitosos filtrados se agrega sobre esta interfaz.

### Red: `RpcChannel`

```python
class RpcChannel:
    """Request/response confiable a un proceso downstream: DEALER + timeout + reintentos."""
    def __init__(self, address: str, timeout_s: float, retries: int): ...
    def call(self, request: dict) -> dict: ...   # DownstreamTimeout si se agotan los reintentos
```

Es el gemelo hacia abajo del `CoreClient`: encapsula el socket `DEALER`, el `poll` con timeout, los reintentos y la traducción de fallas a errores de dominio. Lo usan `EnvironmentClient`, `WorldModelClient` y `RewardModelClient`, para no repetir la mecánica de ZeroMQ en cada uno.

### Errores de dominio

```python
class CoreError(Exception): ...
class UnknownEnvironment(CoreError): ...
class EpisodeAlreadyDone(CoreError): ...
class DownstreamTimeout(CoreError): ...
class DownstreamUnavailable(CoreError): ...
```

`dispatch_request` captura `CoreError` (y cualquier excepción) y la devuelve como `{ok: False, error}`; el `CoreClient` la vuelve a levantar como `RuntimeError` en el Learner. Nada se cae por un pedido mal formado.

## Flujos

**`step` con recompensa** (el caso central):

1. `Core.step` valida con `EnvironmentManager`/`EpisodeManager` que ninguno de los `env_ids` esté terminado (`EpisodeAlreadyDone` si lo está).
2. Por cada `env_id`: `EnvironmentRegistry.get(env_id).step(chunk)` → `StepResult` sin recompensa; guarda la observación anterior (de `EpisodeManager`).
3. Si `with_reward`: arma un `ScoreItem` por ambiente y llama **una vez** a `RewardScorer.score(items)`; adjunta cada recompensa a su `StepResult`.
4. Por cada ambiente, crea la `Transition` (`source="env"`, `step` de `EpisodeManager`) y la manda a `EpisodeStore.append`; luego `EpisodeManager.advance`.
5. Devuelve la lista de `StepResult`.

**`reset`:** por cada `env_id`, `Environment.reset` → observación; `EpisodeManager.open` crea el `episode_id` y guarda la observación inicial; devuelve `(episode_id, observación)`.

**`simulate`:** resuelve `start` (una `Observation`, o `(episode_id, step)` buscado en `EpisodeStore`); `WorldModel.encode` → latente; `WorldModel.rollout` → observaciones predichas por secuencia; arma los `ScoreItem` de todas las secuencias y puntúa en una sola llamada; devuelve un `StepResult` por paso, con `source="world_model"`; registra las transiciones imaginadas (configurable).

**`sample_supervised` / `get_episode`:** delegación directa en `Dataset` y `EpisodeStore`.

## Construcción desde configuración

Igual que el Learner, Core se arma desde un YAML y se levanta con `uv run vla-core --config configs/core/<archivo>.yaml`. Un `build_core(config)` (en `vla_core/builder.py`) construye el registro de ambientes, los clientes (reales o fakes según el `type`), el `Dataset`, el `EpisodeStore`, el `RewardScorer`, el `Core` y el `CoreServer`. Para no acoplar `vla_core` al Learner, Core tiene su propio builder; si se quiere un `Registry` por nombre como el del Learner, se lo sube antes a `vla_common` y lo comparten los dos (decisión abierta).

Ejemplo de `configs/core/fake.yaml` (todo fake, para desarrollo local):

```yaml
seed: 0
server: { address: "tcp://127.0.0.1:5555" }
environments:
  - { id: sim-0, type: fake }
  - { id: sim-1, type: fake }
world_model: { type: fake }
reward_model: { type: fake }
dataset: { type: fake, demo_chunk_size: 1 }
episode_store: { type: memory }
timeouts: { environment_s: 5, world_model_s: 10, reward_model_s: 10, retries: 2 }
```

## Estructura del paquete

```
packages/vla_core/
  pyproject.toml            # [project.scripts] vla-core; deps: vla-common, pyzmq, msgpack, numpy
  src/vla_core/
    server.py               # CoreServer (ROUTER)
    core.py                 # Core (gateway)
    environments.py         # Environment (ABC), EnvironmentClient, EnvironmentRegistry, EnvSpec
    world_model.py          # WorldModel (ABC), WorldModelClient, LatentState
    reward.py               # RewardModel (ABC), RewardModelClient, RewardScorer, ScoreItem
    episodes.py             # EpisodeManager, EpisodeStore (ABC), InMemoryEpisodeStore
    dataset.py              # Dataset (ABC)
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

- Cada ABC (`Environment`, `WorldModel`, `RewardModel`, `Dataset`, `EpisodeStore`) trae su suite de contrato en `testing/contracts/`, que su *fake* y cualquier implementación real deben pasar.
- La **suite CORE** que ya existe (hoy corre `CoreClient` contra `FakeCoreServer`) se reusa: ahora también corre `CoreClient` contra `CoreServer(Core(...))` con componentes fake. Es la prueba de que el Core real respeta el protocolo.
- El `FakeCoreClient` del Learner sigue siendo la referencia monolítica del comportamiento esperado; `Core` debe producir los mismos resultados, pero con las responsabilidades separadas.

## Decisiones abiertas

- **Dónde vive `CoreInterface`.** Recomendado: moverla a `vla_common.protocol` y reexportarla desde `vla_learner`. Alternativa: que `vla_core` declare una `Protocol` equivalente. (Se resuelve en la Fase 1.)
- **`Registry` compartido vs. builder propio de Core.** Por ahora, builder propio; subir `Registry` a `vla_common` solo si se quiere el mismo mecanismo por nombre en los dos procesos.
- **Timeouts y reintentos por componente.** Valores por defecto en la config; `Architecture.md` los deja a la implementación. Fijar los defaults y su motivo en la Fase 7 (fallas).
- **Persistencia del `EpisodeStore`.** `InMemoryEpisodeStore` para v1; una variante en disco (para reusar episodios como datos supervisados y entrenar el WorldModel) cuando se pida.
- **Registro de ambientes dinámico.** v1 lo puebla desde config; el auto-registro (`register` por mensaje) queda definido pero sin implementar hasta que haga falta.
```
