# CLAUDE.md

Harness para entrenar modelos VLA (vision-language-action) con aprendizaje supervisado y RL. El proceso **Learner ya está terminado** (primera fase del proyecto). El proceso **Core** —el gateway que conecta al Learner con los ambientes, el world model, el reward model y el dataset— también está **terminado** (las siete fases del plan, ver "Orden de implementación"). Lo que siga (componentes reales, adaptadores de datasets, evaluación) se aborda cuando se pida.

## Estado del proyecto

**En curso — Dataset y π0 (rama `Dataset`).** Adaptadores de dataset (`fake`, `local`, `lerobot`, `openx`; ver `docs/dataset/Dataset Design.md`), el adaptador `Pi0Policy` en el Learner y la prueba E2E de dos procesos de `docs/dataset/Pi0 E2E Plan.md`. Esa prueba **pasó en el cluster** (job 35206): Core sirve `pusht` convertido a `local` (`scripts/prepare_pi0_pusht.py`) y el Learner entrena π0 con BC, con `fm_mse`/`grad_norm` finitos en los 10 pasos. Se corre con `slurm/pi0_bc.sbatch` (ver `slurm/README.md`). Antes de un entrenamiento real faltan tres cosas, detalladas en "Puntos abiertos" del plan: normalizar acciones y estado, cargar el tokenizer de PaliGemma (hoy π0 no ve la instrucción) y auditar el warning `Could not remap state dict keys`.

**Terminado — Core (`packages/vla_core`).** Las siete fases del plan: diseño y transporte (`CoreServer`/`dispatch_request`), componentes y clientes `DEALER` (Environment, WorldModel, Reward Model, Dataset, EpisodeStore) con sus fakes, el gateway `Core` (`reset`/`step`/`simulate`/`sample_supervised`/`get_episode`), el gancho de episodios exitosos, el manejo de fallas (timeouts, reconexión, parada de seguridad, multi-ambiente), la CLI `vla-core` con configs de ejemplo y la integración extremo a extremo con el Learner real. Pasa las suites de contrato (`fast`), `distributed` y la E2E (`slow`).

**Terminado — Learner (`packages/vla_learner`).** Todas las clases de `Learner Design.md`: interfaces del modelo, `ComposedPolicy`, cabezas, normalizador, `BehaviorCloning`, `PPO`, `GRPO`, estimadores de ventaja, fuentes de datos, `CoreClient`, infraestructura, `Learner`, `Registry`, CLI, y el adaptador real `OpenVLAPolicy`. Pasa sus suites de contrato (`fast`) y los tests de aprendizaje (`slow`). **No se reescribe**; se toca solo si Core obliga a cambiar el protocolo.

**Terminado — `vla_common`.** Tipos compartidos (`Observation`, `Action`, `ActionChunk`, `StepResult`, `Transition`), serialización ZeroMQ sin pickle (`serialization.py`) y las conversiones de wire (`wire.py`). **Core reutiliza esto tal cual**; no se duplican tipos ni conversiones.

**Punto de partida de Core.** El `CoreClient` del Learner (`vla_learner/data/core_client.py`) y la herramienta de prueba `FakeCoreServer` (`vla_learner/testing/fake_core_server.py`) **ya fijan el protocolo** que Core debe hablar: socket `ROUTER`, mensajes con clave `"call"` (`reset`, `step`, `simulate`, `sample_supervised`, `get_episode`, `close`), respuesta con `ok`/`error`. `FakeCoreServer` es el patrón precursor: Core generaliza ese despacho, pero respaldado por componentes reales (vía fakes) en vez de `FakeCoreClient`.

## Documentos de diseño (fuente de verdad)

Léelos antes de empezar cualquier tarea. Ante cualquier duda, manda el documento.

- `docs/Architecture.md`: **la especificación de Core.** Procesos, responsabilidades de Core, interfaz hacia el Learner, interfaces que Core consume (Environment, WorldModel, Reward Model, Dataset, almacén de episodios), protocolo ZeroMQ y flujos principales.
- `docs/core/Core Design.md`: **a escribir en la Fase 1.** Clases e interfaces internas de Core, derivadas de `Architecture.md`, al nivel de detalle que tiene `Learner Design.md`. Es la especificación a implementar.
- `docs/core/Test Core.md`: **a escribir en la Fase 1.** Suites de contrato que toda implementación de Core y de sus componentes debe pasar, al estilo de `Test Learners.md`.
- `docs/learner/Learner Design.md` y `docs/learner/Test Learners.md`: referencia. Core debe honrar el `CoreClientProtocol` y el formato de wire que estos describen; no los contradigas.
- Cada componente tiene su carpeta en `docs/` (`docs/core/`, `docs/learner/`, `docs/environment/`, `docs/world-model/`, `docs/reward-model/`, `docs/dataset/`), con un `README.md` que lo resume y apunta a su sección de `Architecture.md`. `Architecture.md` queda en la raíz de `docs/` porque es transversal.

Si al implementar necesitas desviarte del diseño (una firma, un tipo, una clase nueva), actualiza el documento en el mismo cambio y explica el motivo en tu resumen. No dejes código y documento en desacuerdo. **Si el cambio toca el protocolo, actualiza también el `CoreClient` del Learner y su suite CORE**: el cliente y el servidor no pueden divergir.

## Alcance

**Dentro:**

- **El proceso Core** (`packages/vla_core`): servidor ZeroMQ `ROUTER` que expone la interfaz del Learner; despacho de llamadas; registro de ambientes (`env_id`, disponibilidad, espacio de acciones y cámaras declarados al registrarse); estado por episodio (paso actual, observación anterior); agregación de recompensa (`with_reward`); servicio de datos supervisados desde el Dataset; almacén de episodios (`append`/`get`); manejo de fallas (timeouts, caídas, reintentos, parada de seguridad).
- **Clientes hacia abajo de Core**: cómo Core llama por ZeroMQ (`DEALER`) a Environment (`reset`/`step`/`close`), WorldModel (`encode`/`rollout`), Reward Model (`score`) y cómo accede al Dataset y al almacén de episodios.
- **Herramientas de prueba de Core** (análogas a las del Learner): fakes en memoria y servidores ZeroMQ fake de Environment, WorldModel y Reward Model; un Dataset fake y un almacén de episodios de prueba; generadores de datos; y las suites de contrato en `testing/contracts/`.
- **CLI** `vla-core` y configs YAML de ejemplo.
- **Integración extremo a extremo**: el Learner real conduciendo un Core real con componentes fake, por los tres flujos (supervisado, RL en ambiente, RL en world model).

**Fuera (no implementar sin que se pida):**

- Memoria compartida para imágenes, migración a Ray o gRPC: solo si una medición lo justifica (ver `Architecture.md`).

## Hardware: Core es CPU; cuidado con los pesos

- **Core no tiene GPU ni modelo.** Todo su código corre en CPU y solo enruta mensajes y arrays de numpy. `vla_core` **no depende de `torch`** (igual que `vla_common`).
- Los componentes con GPU (WorldModel, Reward Model) son procesos aparte.
- La máquina de desarrollo (si estás en windows) tiene una **RTX 2050 con 4 GB de VRAM**. **Nunca descargues ni cargues pesos de modelos reales** (OpenVLA 7B, PaliGemma 3B, π0, world models, VLM jueces). No caben y no hacen falta para construir funcionalidades.
- Un test no debe usar más de ~2 GB de VRAM y el dispositivo por defecto es **CPU**. La GPU solo en tests marcados `gpu`.

## Reproducibilidad con uv

**Todo proceso del harness debe poder levantarse con `uv`.** No uses `pip`, `conda` ni entornos creados a mano.

- El repositorio es un **workspace de uv**. `uv.lock` y `.python-version` van en el control de versiones. `packages/vla_core` entra como nuevo miembro (el glob `packages/*` lo toma solo).
- Dependencias: siempre con `uv add` (o `uv add --dev`, o `uv add --optional <grupo>`). Nunca edites versiones a mano sin regenerar el lock. Pregunta antes de agregar una dependencia que no esté implícita en los documentos.
- Cada proceso tiene un punto de entrada en `[project.scripts]`. Core se levanta así:

  ```bash
  uv run vla-core --config configs/core/<archivo>.yaml
  ```

  El Learner sigue igual (`uv run vla-learner --config ...`).
- Dependencias pesadas o específicas de un componente van en grupos opcionales (por ejemplo `openvla`, `lerobot`, `eval`), para que la instalación base sea liviana. Core base debe ser liviano: ZeroMQ, msgpack y numpy; nada de torch.
- Cada corrida es reproducible: la configuración incluye `seed`, y Core registra el commit de git y el hash de `uv.lock` junto con los episodios que guarda.

## Estructura del repositorio

```
pyproject.toml              # raíz del workspace
uv.lock
.python-version
docs/                       # documentos de diseño, una carpeta por componente
  Architecture.md           # visión general, transversal a todos los componentes
  core/                     # Core Design.md, Test Core.md (Fase 1) + README
  learner/                  # Learner Design.md, Test Learners.md + README
  environment/              # README (componente fuera de alcance; se usa un fake)
  world-model/              # README (idem)
  reward-model/             # README (idem)
  dataset/                  # README (vive dentro de Core en v1)
configs/                    # YAML de ejemplo
  core/                     # configs de Core (nuevos)
packages/
  vla_common/               # tipos compartidos y protocolo (SIN torch, SIN zmq propio del Learner)
  vla_learner/              # proceso Learner (TERMINADO)
  vla_core/                 # proceso Core (FASE ACTUAL)
    src/vla_core/
      server.py             # servidor ROUTER: recibe del Learner, despacha, responde
      gateway.py            # lógica de Core: enruta, agrega recompensa, registra
      registry.py           # registro de ambientes (env_id, disponibilidad, specs)
      episodes.py           # estado por episodio y almacén de episodios (append/get)
      clients/              # clientes DEALER hacia Environment, WorldModel, Reward Model
      dataset/              # Dataset dentro de Core (fake por ahora)
      cli.py
      testing/              # fakes (Environment/WorldModel/RewardModel/Dataset), suites de contrato
    tests/
```

`vla_common` existe porque Core y el Learner usan los mismos tipos y el mismo protocolo. **Nada de `vla_common` puede depender de `torch`.** Las suites de contrato de Core viven en `vla_core/src/vla_core/testing/contracts/`, no en `tests/`, porque implementaciones futuras (un Environment real, un Dataset real) las importan.

## Protocolo: reglas que Core no puede romper

- **Reutiliza `vla_common`.** Los mensajes se arman con `vla_common.wire` + `vla_common.serialization`. No inventes un formato nuevo ni dupliques las conversiones.
- **Nada de `pickle` entre procesos.** El encabezado va en msgpack; los arrays van como buffers crudos (`dtype`/`shape` en el encabezado, `np.frombuffer` al recibir).
- **El esquema de request/response es el que ya consume `CoreClient`.** Core responde `{"ok": True, ...}` o `{"ok": False, "error": ...}`. Core debe pasar la **misma suite CORE** que hoy corre contra `FakeCoreServer`, ahora respaldada por el Core real con componentes fake.
- **Sockets.** Core expone `ROUTER` al Learner; se conecta como `DEALER` a Environment, WorldModel y Reward Model. `ipc://` en la misma máquina, `tcp://` entre máquinas. Timeouts y reintentos sin reiniciar el socket (por eso `DEALER`/`ROUTER` y no `REQ`/`REP`).
- **Core no decide cuándo actuar.** Es un gateway: atiende pedidos, reenvía, agrega recompensa si se pidió y registra. El loop lo conduce el Learner.
- **Una llamada a `score` por paso del loop.** Core junta las transiciones de todos los ambientes y las manda al Reward Model en un solo batch.
- **Reglas de episodio.** Un `step` sobre un ambiente con `terminated`/`truncated` es un error hasta el próximo `reset`. Si el episodio termina a mitad de un chunk, se corta y se devuelve la observación en ese punto. El chunk cuenta como un solo paso: una transición y una recompensa por chunk.

## Tests

Marcas registradas en `pyproject.toml` con `--strict-markers`:

| Marca | Qué incluye | ¿Correr en esta máquina? |
|---|---|---|
| `fast` | Contratos de Core y componentes con fakes CPU | Siempre, antes de terminar cualquier tarea |
| `slow` | Integración extremo a extremo Learner↔Core (los tres flujos) | Sí, al cerrar cada fase. Puede tardar |
| `gpu` | Solo si algún componente de prueba usa GPU (excepcional en Core) | Solo si aplica |
| `distributed` | Varios Environment en paralelo atendidos por un Core | Sí, cuando toques el enrutamiento multi-ambiente |
| `heavy` | Pesos reales o componentes reales con GPU grande | **No.** Se escriben, pero no se corren aquí |

Comandos:

```bash
uv run pytest -m fast                     # siempre
uv run pytest -m slow                     # al cerrar una fase
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

La configuración por defecto de pytest excluye `heavy`, `gpu` y `distributed` para que `uv run pytest` sea seguro de correr sin pensar.

Reglas:

- **No desactives, saltes ni relajes un test para que pase.** Si un test de contrato parece incorrecto, detente y explica por qué en vez de cambiarlo.
- Las tolerancias son las de `Test Core.md` (y `Test Learners.md` para lo del protocolo). No las cambies sin actualizar el documento.
- Todo código nuevo llega con sus tests. Toda implementación de una interfaz (Environment, WorldModel, Reward Model, Dataset) hereda su suite de contrato.
- **La suite CORE del Learner debe seguir verde**: es el contrato del protocolo y ahora el Core real es otra implementación que debe pasarla.

## Orden de implementación

Cada fase termina con `uv run pytest -m fast`, ruff y pyright en verde, y un commit. No empieces una fase con la anterior en rojo. El trabajo va en la rama `Core`.

1. ✅ **Diseño y base.** `docs/core/Core Design.md` y `docs/core/Test Core.md`. Paquete `vla_core` (CLI `vla-core`, sin torch). Servidor `ROUTER` que reutiliza `vla_common` y responde `ok`/`error`, con round-trips contra el `CoreClient` real. Suite SRV.
2. ✅ **Componentes y clientes hacia abajo.** Interfaces que Core consume (Environment, WorldModel, Reward Model, Dataset, almacén de episodios) y sus clientes `DEALER`. Fakes en memoria y como servidores ZeroMQ. Registro de ambientes. Incluyó `EpisodeManager`/`EpisodeStore` (adelantados) y las suites ENV/WM/RM/DSET/STORE/EP.
3. ✅ **Gateway Core: ciclo de episodios.** `Core.reset`, `Core.step` (enruta por ambiente, respeta la regla de episodio terminado, batchea `score` una vez, registra transiciones con `source="env"`) y `Core.get_episode`. Absorbió el `reset`/`step` que el plan tenía aparte. La suite CORE del Learner se reusa contra el Core real (incluye CORE-03, episodio terminado).
4. ✅ **`simulate` sobre el WorldModel.** `Core.simulate` resuelve el inicio (observación o `(episode_id, step)`), `WorldModel.encode` + `rollout` (devuelve `StepResult` por chunk, con terminación), puntúa con el mismo Reward Model en una sola llamada. `rollout` se ajustó para devolver `StepResult` (ver `Core Design.md`). Suite SIM (directa y por la red).
5. ✅ **Dataset supervisado.** `Core.sample_supervised(n, filters?)` delega en el Dataset (`source="dataset"`, acciones en unidades reales, filtro `task`). El gancho `Core.ingest_episode` reinyecta episodios exitosos (`terminated`) sobre `AppendableDataset` (`DatasetNotAppendable` si no lo es). Suite DSET a nivel de Core (incluye round-trip por la red).
6. ✅ **Fallas y robustez.** `RpcChannel` con timeout + reintentos + reconexión (`DownstreamTimeout`); `Core` escala la caída de un ambiente a `DownstreamUnavailable`, lo marca no disponible e invoca su parada de seguridad sin romper al resto; timeouts/reintentos por componente fijados y configurables (`Core Design.md`, "Manejo de fallas"). Suites FAIL (`fast`) y el test `distributed` (multi-ambiente por la red).
7. ✅ **Integración extremo a extremo.** `vla-core` con configs de ejemplo (`configs/core/*.yaml`, validados por un test `fast`) y el Learner real conduciendo un Core real con fakes por ZeroMQ en los tres flujos. Suite E2E (`slow`): el entrenamiento avanza en supervisado, RL en ambiente y RL en world model.

## Decisiones abiertas (de `Architecture.md`)

- ~~**Timeouts y política de reintentos de Core por componente.**~~ **Resuelto (Fase 6):** fijados con su motivo y configurables en `Core Design.md` ("Manejo de fallas"): Environment 5 s / 1 reintento; WorldModel y Reward Model 10 s / 1 reintento.
- **Recompensas sobre el WorldModel.** Para `simulate`, Core decodifica o recibe observaciones predichas y las pasa al mismo Reward Model, para que las escalas de recompensa real e imaginada coincidan. Define en el diseño dónde ocurre la decodificación.
- **Memoria de WorldModel y Reward Model en la misma GPU.** Fuera de alcance mientras sean fakes; dejar anotado para cuando entren los reales.

## Estilo de código

- Python con type hints completos. Dataclasses para los tipos de datos; `ABC` para las interfaces.
- Nombres de clases y métodos exactamente como en el documento de diseño.
- Nada de `pickle` entre procesos (ver protocolo en `Architecture.md`).
- `vla_common` y `vla_core` no importan `torch`. Dentro del Learner, solo `CoreClient` importa `zmq` y solo `OptimizerManager` llama a `backward()`/`step()` (lo verifican los tests ARCH).
- Docstrings y comentarios en español; identificadores en inglés.
