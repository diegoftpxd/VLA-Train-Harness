# Test Core

Suites de contrato de Core y de sus componentes, al estilo de [`../learner/Test Learners.md`](../learner/Test%20Learners.md). Toda implementación de una interfaz (Core, Environment, WorldModel, Reward Model, Dataset, EpisodeStore) hereda la suite correspondiente. Las suites reutilizables viven en `packages/vla_core/src/vla_core/testing/contracts/`, para que implementaciones futuras las importen. La suite ENV es la excepción: vive en `packages/vla_common/src/vla_common/testing/contracts/`, junto al contrato del Environment, porque la importan `vla_core` y `vla_env`, que no dependen entre sí (ver [`../environment/Environment Design.md`](../environment/Environment%20Design.md)).

Principios: todo corre en **CPU** con *fakes* livianos y deterministas; nada descarga pesos; el transporte usa `tcp://127.0.0.1` con puerto efímero. Las marcas son las del proyecto (`fast`, `slow`, `distributed`, `heavy`).

## Panorama de suites

| Suite | Qué garantiza | Fase | Marca |
|---|---|---|---|
| **SRV** | Transporte: round-trip de cada llamada, sobre `ok`/`error`, wire bit a bit | 1 | `fast` |
| **CORE** (reusada) | El Core real pasa el mismo contrato que `FakeCoreClient` y `CoreClient`+`FakeCoreServer` | 2–6 | `fast` |
| **ENV** | Contrato de `Environment` (`reset`/`step`/`close`, specs) | 2 | `fast` |
| **WM** | Contrato de `WorldModel` (`encode`/`rollout`) | 2 | `fast` |
| **RM** | Contrato de `RewardModel` (`score` por batch) | 2 | `fast` |
| **DSET** | Contrato de `Dataset` (`sample`, unidades reales, filtros) y de `AppendableDataset` (`add`); `Core.sample_supervised` y el gancho `ingest_episode` | 5 | `fast` |
| **STORE** | Contrato de `EpisodeStore` (`append`/`get`) | 3 | `fast` |
| **EP** | `EpisodeManager`: paso, observación anterior, regla de episodio terminado | 3 | `fast` |
| **STEP/RESET** | `Core.step`/`reset`: enrutado, recompensa batcheada, registro de transiciones | 4 | `fast` |
| **SIM** | `Core.simulate`: rollout + score, `source="world_model"` | 5 | `fast` |
| **FAIL** | Timeouts, caída/reconexión de un componente, reintentos, parada de seguridad; multi-ambiente en paralelo | 6 | `fast` + `distributed` |
| **E2E** | Learner real conduciendo un Core real con fakes, los tres flujos | 7 | `slow` |

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

- **ENV** (la cumplen `FakeEnvironment`, `EnvironmentClient` y todo adaptador de `vla_env`). **La fuente de verdad es [`../environment/Test Environment.md`](../environment/Test%20Environment.md)**, que además define ENV-10/11 y las suites GYM, SRVE, MAP y ROBOT; aquí va el resumen:
  - **ENV-01** `reset` devuelve una `Observation` con `proprio` `float32` y las cámaras de su `spec`.
  - **ENV-02** `step` devuelve `StepResult` con `reward=None` (la recompensa la agrega Core) y `terminated`/`truncated` booleanos.
  - **ENV-03** el `spec` es válido: `action_dim > 0`, al menos una cámara y al menos una tarea.
  - **ENV-04** tras llegar a un final (`step` con una acción neutra hasta `terminated` o `truncated`, con un tope), `emergency_stop` y `close` no fallan.
  - **ENV-05** `reset` con la misma semilla es determinista.
  - **ENV-06** `reset(task=spec.tasks[0])` da `instruction == task`; una tarea desconocida es error.
  - **ENV-07** un chunk con dimensión distinta de `action_dim` es error.
  - **ENV-08** `info["success"]` es `bool` y todo valor de `info` es escalar.
  - **ENV-09** las imágenes son `float32`, CHW, de 3 canales y en `[0, 1]`, y son deterministas con la misma semilla.

  El corte a mitad de chunk no se puede forzar de forma genérica: lo prueban un test propio de `FakeEnvironment` y los tests de `GymEnvironment` en `vla_env`.
- **WM**: `encode` acepta una `Observation` y `rollout` devuelve una lista de `StepResult` por secuencia, uno por chunk, cortando la secuencia si un paso termina el episodio.
- **RM**: `score` acepta un batch de `ScoreItem` y devuelve una recompensa por item, en el mismo orden; acepta batch vacío.
- **STORE**: `append` luego `get(episode_id)` devuelve las transiciones en orden; `get` de un id desconocido es error.
- **EP**: `open` crea un `episode_id` único; `advance` incrementa el paso y actualiza la observación anterior; `is_done` refleja `terminated`/`truncated`.
- **DSET**: `sample(n)` devuelve `n` pares; las acciones están en unidades reales (ndarray `float32`, sin normalizar); los filtros se respetan (`sample(n, {"task": t})` ⇒ todos los pares tienen esa instrucción). Para `AppendableDataset`: `add(pairs)` devuelve cuántos agregó y lo reinyectado se vuelve a muestrear.

## Flujos de Core (Fases 4–5)

- **RESET**: abre un episodio por ambiente y guarda la observación inicial.
- **STEP**: enruta cada chunk a su Environment; con `with_reward=True` hace **una sola** llamada a `score` para todo el batch; registra una `Transition` (`source="env"`) por ambiente; respeta la regla de episodio terminado.
- **SIM**: resuelve el punto de partida (observación o `(episode_id, step)`), corre el WorldModel, puntúa con el mismo Reward Model y marca `source="world_model"`.
- **DSET (nivel Core)**: `sample_supervised(n, filters?)` delega en el Dataset y conserva los filtros (también por la red, `CoreClient`↔`CoreServer`); el gancho `ingest_episode` reinyecta un episodio exitoso (`info["success"]`; un episodio terminado por fallo no se reinyecta) y no uno truncado, y levanta `DatasetNotAppendable` sobre un Dataset de solo-lectura.

## Fallas e integración (Fases 7–8)

- **FAIL** (`fast`): a nivel de transporte, un componente colgado (`SilentRouter`, que recibe pero nunca responde) produce `DownstreamTimeout` tras agotar timeout y reintentos; un componente que se recupera en la misma dirección vuelve a responder (reconexión del `DEALER`). A nivel de gateway, un ambiente que no responde (`FlakyEnvironment`, que levanta `DownstreamTimeout` en memoria) se marca no disponible, se le invoca la parada de seguridad y el error se escala a `DownstreamUnavailable`; el resto de los ambientes sigue operando y reusar el caído da `DownstreamUnavailable` sin volver a tocarlo.
- **distributed**: varios Environment, cada uno su proceso ZeroMQ, atendidos por un solo Core; un `step` enruta cada chunk a su ambiente y devuelve los resultados en el orden de entrada, con una transición por ambiente (test marcado `distributed`).
- **E2E** (`slow`): el `Learner` real, con `CoreClient`, corre los tres flujos (supervisado, RL en ambiente, RL en world model) contra un `CoreServer(Core(...))` con fakes, por el wire ZeroMQ, y el entrenamiento avanza (éxito claramente por encima de la línea base en los tres modos). Los umbrales van con margen por debajo de los de LEARN: E2E prueba la integración de punta a punta, no recalibra la convergencia. Además, un test `fast` comprueba que los configs de ejemplo (`configs/core/*.yaml`) se construyen con `build_core` y responden un round-trip real.
