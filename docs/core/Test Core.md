# Test Core

Suites de contrato de Core y de sus componentes, al estilo de [`../learner/Test Learners.md`](../learner/Test%20Learners.md). Toda implementación de una interfaz (Core, Environment, WorldModel, Reward Model, Dataset, EpisodeStore) hereda la suite correspondiente. Las suites reutilizables viven en `packages/vla_core/src/vla_core/testing/contracts/`, para que implementaciones futuras las importen.

Principios: todo corre en **CPU** con *fakes* livianos y deterministas; nada descarga pesos; el transporte usa `tcp://127.0.0.1` con puerto efímero. Las marcas son las del proyecto (`fast`, `slow`, `distributed`, `heavy`).

## Panorama de suites

| Suite | Qué garantiza | Fase | Marca |
|---|---|---|---|
| **SRV** | Transporte: round-trip de cada llamada, sobre `ok`/`error`, wire bit a bit | 1 | `fast` |
| **CORE** (reusada) | El Core real pasa el mismo contrato que `FakeCoreClient` y `CoreClient`+`FakeCoreServer` | 2–6 | `fast` |
| **ENV** | Contrato de `Environment` (`reset`/`step`/`close`, specs) | 2 | `fast` |
| **WM** | Contrato de `WorldModel` (`encode`/`rollout`) | 2 | `fast` |
| **RM** | Contrato de `RewardModel` (`score` por batch) | 2 | `fast` |
| **DSET** | Contrato de `Dataset` (`sample`, unidades reales, filtros) | 6 | `fast` |
| **STORE** | Contrato de `EpisodeStore` (`append`/`get`) | 3 | `fast` |
| **EP** | `EpisodeManager`: paso, observación anterior, regla de episodio terminado | 3 | `fast` |
| **STEP/RESET** | `Core.step`/`reset`: enrutado, recompensa batcheada, registro de transiciones | 4 | `fast` |
| **SIM** | `Core.simulate`: rollout + score, `source="world_model"` | 5 | `fast` |
| **FAIL** | Timeouts, caída/reconexión de un componente, reintentos, parada de seguridad | 7 | `fast` + `distributed` |
| **E2E** | Learner real conduciendo un Core real con fakes, los tres flujos | 8 | `slow` |

## SRV — transporte (Fase 1, implementada)

`CoreServer` envuelve un backend que cumple `CoreInterface`; estas pruebas usan el `CoreClient` **real** del Learner contra un `CoreServer(StaticCoreBackend)`.

- **SRV-01** `reset` hace round-trip: devuelve `(episode_id, Observation)` por ambiente, con `proprio` en `float32` y las cámaras presentes.
- **SRV-02** `step` + `get_episode` conservan la acción **bit a bit** a través del wire (mismo `dtype`, mismos bytes).
- **SRV-03** `simulate` devuelve un `StepResult` por chunk de cada secuencia, con las formas correctas.
- **SRV-04** `sample_supervised(n)` devuelve `n` pares `(Observation, ActionChunk)`.
- **SRV-05** `with_reward=False` ⇒ el campo `reward` llega en `None` (mismo tipo de retorno).
- **SRV-06** una excepción del backend vuelve como sobre `{ok: False, error}` y el cliente la levanta como `RuntimeError`.

**Tolerancias:** los arrays viajan sin pérdida (igualdad bit a bit, no `allclose`). El orden de la salida respeta el orden de los `env_ids`/secuencias de entrada.

## CORE reusada — el Core real respeta el contrato (Fases 2–6)

La suite CORE que hoy corre `CoreClient` contra `FakeCoreServer` se parametriza para correr también `CoreClient` contra `CoreServer(Core(...))` con componentes fake. Cierra cuando el gateway `Core` implementa `reset`/`step`/`simulate`/`sample_supervised`/`get_episode`:

- **CORE-01** serialización exacta (acciones `float32` bit a bit). — ya existe
- **CORE-02** el orden de los ambientes se respeta. — ya existe
- **CORE-03** `step` sobre un episodio terminado es error (`EpisodeAlreadyDone` → `RuntimeError`). — Fase 4
- **CORE-05** recompensa opcional. — ya existe

## Contratos de componentes (Fase 2 en adelante)

Cada ABC trae su suite; su *fake* y cualquier implementación real deben pasarla. Un fixture entrega la instancia bajo prueba (como hace `CoreClientContract` con `client`).

- **ENV**: `reset` devuelve una `Observation` con las cámaras de su `spec`; `step` devuelve `StepResult` con `reward=None` (la recompensa la agrega Core); tras `terminated`/`truncated`, `close` y `emergency_stop` no fallan.
- **WM**: `encode` acepta una `Observation` y `rollout` devuelve una lista de `Observation` por secuencia, con largo igual al de cada secuencia de acciones.
- **RM**: `score` acepta un batch de `ScoreItem` y devuelve una recompensa por item, en el mismo orden; acepta batch vacío.
- **STORE**: `append` luego `get(episode_id)` devuelve las transiciones en orden; `get` de un id desconocido es error.
- **EP**: `open` crea un `episode_id` único; `advance` incrementa el paso y actualiza la observación anterior; `is_done` refleja `terminated`/`truncated`.
- **DSET**: `sample(n)` devuelve `n` pares; las acciones están en unidades reales; los filtros se respetan.

## Flujos de Core (Fases 4–5)

- **RESET**: abre un episodio por ambiente y guarda la observación inicial.
- **STEP**: enruta cada chunk a su Environment; con `with_reward=True` hace **una sola** llamada a `score` para todo el batch; registra una `Transition` (`source="env"`) por ambiente; respeta la regla de episodio terminado.
- **SIM**: resuelve el punto de partida (observación o `(episode_id, step)`), corre el WorldModel, puntúa con el mismo Reward Model y marca `source="world_model"`.

## Fallas e integración (Fases 7–8)

- **FAIL**: un componente que no responde produce `DownstreamTimeout`; tras los reintentos, `DownstreamUnavailable`; varios Environment en paralelo se atienden en orden (test `distributed`); la parada de seguridad se invoca en el robot (fake) sin romper el resto.
- **E2E** (`slow`): el `Learner` real, con `CoreClient`, corre los tres flujos (supervisado, RL en ambiente, RL en world model) contra un `CoreServer(Core(...))` con fakes, y el entrenamiento avanza.
