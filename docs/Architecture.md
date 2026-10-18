# Arquitectura del harness VLA

## Resumen

El harness entrena un modelo VLA con aprendizaje supervisado y con RL, sobre un ambiente real o simulado y sobre un world model. Está dividido en procesos independientes que se comunican solo por mensajes, para poder reiniciar, reemplazar o mover cada pieza sin tocar las demás.

En la versión 1 todo es síncrono y el loop lo conduce el Learner: pide estados, decide acciones con el modelo y entrena con lo que recibe. Esto evita sincronizar pesos entre procesos a costa de no poder escalar los rollouts por separado.

La idea es que cada proceso pueda ser reconstruible utilizando uv. Debe tener soporte para descargar pesos desde HF de los modelos.

## Componentes y decisiones de diseño

Hay cinco procesos principales. El Dataset no es un proceso propio en v1: vive dentro de Core.

|Componente|Proceso|Responsabilidad|Habla con|
|---|---|---|---|
|Learner|Propio (GPU)|Inferencia y entrenamiento del VLA, algoritmos de SL y RL|Core|
|Core|Propio (CPU)|Gateway: enruta pedidos, pide recompensas, registra episodios|Learner, Environment, WorldModel, Reward Model|
|WorldModel|Propio (GPU, compartida con Reward Model)|Simular transiciones a partir de un estado y acciones|Core|
|Environment|Propio (uno o varios)|Ambiente real o simulado: reset y step|Core|
|Reward Model|Propio (GPU, compartida con WorldModel)|Puntuar transiciones según la tarea|Core|
|Dataset|Dentro de Core (v1)|Servir ejemplos supervisados|Core|

**Decisiones**

1. **El Learner es el único con acceso al VLA y hace también la inferencia de rollouts.** No hay un proceso de política aparte, así que no hay que sincronizar pesos ni manejar desfase entre versiones. El costo es que actuar y entrenar se turnan en las mismas GPUs. Si más adelante los rollouts son el cuello de botella, se separa un proceso de política de solo inferencia.
2. **El Learner conduce el loop; Core es un gateway, no un orquestador.** Core no decide cuándo actuar: atiende pedidos del Learner, los reenvía al componente correcto, agrega la recompensa si se pidió y registra lo que pasa.
3. **Todo pasa por Core.** El Learner no conoce direcciones de Environment, WorldModel, Reward Model ni Dataset. Esto incluye los datos supervisados: aunque pasar demos por Core agrega una copia, se prefiere una sola puerta de entrada. Así se puede cambiar un ambiente simulado por el robot real, o un dataset por otro, sin tocar el Learner.
4. **Core registra todas las transiciones.** Aunque el Learner use los datos al momento, Core guarda cada episodio para depuración, para entrenar el WorldModel en otro momento y para volver a usarlos como datos supervisados.
5. **Todo síncrono en v1.** Cada llamada espera su respuesta antes de seguir. Es más lento, pero mucho más fácil de depurar.

**Fuera de alcance**

- **Entrenamiento del WorldModel.** Se hace por separado. Este harness asume un WorldModel ya entrenado y congelado.
- **Evaluación.** Es una etapa posterior al entrenamiento y la hace [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness) sobre los checkpoints guardados. Este harness solo se compromete a que esos checkpoints se puedan cargar en el model server que ese harness espera (ver sección Evaluación).

## Tipos de datos

Todos los procesos usan los mismos tipos. Definirlos antes que las interfaces evita que cada método invente su propio formato.

```python
@dataclass
class Observation:
    images: dict[str, Array]      # una entrada por cámara
    proprio: Array                # estado del robot (articulaciones, gripper)
    instruction: str              # tarea en lenguaje natural

@dataclass
class Action:
    values: Array                 # una acción de bajo nivel

ActionChunk = list[Action]        # acciones ejecutadas en secuencia

@dataclass
class StepResult:
    observation: Observation
    reward: float | None          # None si no se pidió
    terminated: bool              # la tarea terminó (éxito o fallo)
    truncated: bool               # se cortó por límite de pasos
    info: dict

@dataclass
class Transition:
    episode_id: str
    step: int
    source: Literal["env", "world_model", "dataset"]
    observation: Observation
    action: ActionChunk
    result: StepResult
```

Notas:

- **La instrucción viaja en la observación.** El Reward Model la necesita para saber si la tarea pedida se cumplió.
- **`terminated` y `truncated` separados.** RL los trata distinto al calcular retornos: un corte por tiempo no es un fracaso.
- **`source` en cada transición** permite ponderar distinto datos reales e imaginados.

## Learner

El Learner es el único proceso que tiene el modelo VLA: genera las acciones durante los rollouts y actualiza los pesos.

**Responsabilidades**

1. Mantener el modelo VLA, el optimizador y los checkpoints.
2. Ejecutar los algoritmos de aprendizaje como objetivos intercambiables (BC para SL; PPO, GRPO u otros para RL).
3. Conducir los rollouts: pedir estados a Core, elegir acciones y armar los batches de entrenamiento.
4. Si usa actor-critic, mantener la función de valor V(s), normalmente como una cabeza más del modelo.
5. Guardar checkpoints que un model server de vla-evaluation-harness pueda cargar.
6. Publicar métricas de entrenamiento.

**Interfaz hacia Core**

Todas las llamadas aceptan listas para atender varios ambientes en paralelo con un solo mensaje.

|Llamada|Entrada|Salida|Cambio respecto al borrador|
|---|---|---|---|
|`reset(env_ids, task?, seed?)`|Ambientes a reiniciar; tarea y semilla opcionales|`episode_id` y `Observation` por ambiente|Antes `reset_state`. Permite elegir tarea y reproducir episodios|
|`step(env_ids, actions, with_reward)`|Un `ActionChunk` por ambiente|`StepResult` por ambiente|Antes `execute_action`. Devuelve `terminated` y `truncated`|
|`simulate(start, action_sequences, with_reward)`|Observación o `episode_id` + paso de partida; varias secuencias de acciones|Una lista de `StepResult` por secuencia|Antes `simulate_actions`. Devuelve todos los pasos, no solo el final|
|`sample_supervised(n, filters?)`|Cantidad y filtros opcionales (tarea, embodiment)|Lista de pares (`Observation`, `ActionChunk`)|Antes `pull_supervised`. Corregido: la entrada es la cantidad, la salida son los pares|
|`get_episode(episode_id)`|Id de episodio|Lista de `Transition`|Nueva. Recuperar episodios registrados por Core|

**Notas**

- **Recompensa como campo opcional, no como bandera que cambia la firma.** Con `with_reward=False` el campo `reward` llega en `None`; el tipo de retorno es siempre el mismo.
- **Acciones como chunk.** Si el modelo emite varias acciones por paso, `step` las ejecuta en secuencia y devuelve solo la observación final. Para RL, el chunk completo cuenta como un único paso: una transición y una recompensa por chunk.
- **Episodio terminado.** Si un ambiente devuelve `terminated` o `truncated`, el siguiente `step` sobre él es un error hasta que se llame a `reset`. Si el episodio termina a mitad de un chunk, el Environment deja de ejecutar el resto de las acciones y devuelve la observación en ese punto.

## Core

Core recibe los pedidos del Learner, los reenvía al componente que corresponde, agrega la recompensa cuando se pide y registra cada transición.

**Responsabilidades**

1. Exponer al Learner la interfaz descrita arriba.
2. Mantener el registro de ambientes: qué procesos Environment existen, su `env_id` y si están disponibles.
3. Llevar el estado de cada episodio: `episode_id`, paso actual, observación anterior (la necesita el Reward Model).
4. Llamar al Reward Model cuando `with_reward=True`.
5. Servir los datos supervisados desde el Dataset.
6. Guardar cada `Transition` en el almacén de episodios.
7. Manejar fallas: timeouts, caídas de un Environment, reintentos y paradas de seguridad en el robot real.

**Interfaces que consume**

|Componente|Llamada|Entrada|Salida|
|---|---|---|---|
|Environment|`reset(task?, seed?)`|Tarea y semilla|`Observation`|
|Environment|`step(actions)`|`ActionChunk`|`Observation`, `terminated`, `truncated`, `info`|
|Environment|`close()`|—|—|
|WorldModel|`encode(observation)`|`Observation`|`LatentState`|
|WorldModel|`rollout(latent, action_sequences)`|Estado latente y secuencias de acciones|Observaciones predichas por secuencia|
|Reward Model|`score(transitions)`|Lista de (observación, acción, observación siguiente) con instrucción|Recompensa por transición|
|Dataset|`sample(n, filters?)`|Cantidad y filtros|Pares (`Observation`, `ActionChunk`)|
|Almacén de episodios|`append(transition)` / `get(episode_id)`|`Transition` / id|— / lista de `Transition`|

**Notas**

- **Core no tiene GPU ni modelo.** El Reward Model corre en su propio proceso, así que puede usar GPU o cambiarse por otro sin tocar Core.
- **Recompensas sobre el WorldModel.** Para `simulate`, Core decodifica o recibe observaciones predichas y las pasa al mismo Reward Model. Así las recompensas reales e imaginadas usan la misma escala.
- **Una llamada a `score` por paso del loop.** Core junta las transiciones de todos los ambientes y las manda al Reward Model en un solo batch, para pagar el salto entre procesos una vez y no una por ambiente.

## Environment, WorldModel, Reward Model y Dataset

**Environment.** Un proceso por ambiente o por grupo vectorizado, con una API tipo Gymnasium (`reset`, `step`, `close`). La misma interfaz cubre simuladores y robot real; el robot agrega límites de seguridad y una parada de emergencia que Core puede invocar. Cada proceso declara su espacio de acciones y sus cámaras al registrarse en Core.

**WorldModel.** Predice cómo cambian las observaciones ante una secuencia de acciones. Trabaja en su espacio latente: `encode` convierte una observación real en estado latente y `rollout` simula varias secuencias desde ahí. Se entrena fuera de este harness con transiciones reales; aquí se usa preentrenado y congelado.

**Reward Model.** Proceso propio; Core le habla solo por `score`. Recibe la instrucción y la transición (observación, acción, observación siguiente) y devuelve una recompensa. Puede ser un detector de éxito del simulador, un modelo aprendido o un VLM como juez. Debe aceptar batches.

**WorldModel y Reward Model en la misma GPU.** Son dos procesos distintos que comparten una GPU. Como el loop es síncrono, casi nunca trabajan al mismo tiempo: en `simulate` primero corre el WorldModel y después el Reward Model. Para que no compitan por memoria, cada proceso limita su porción de la GPU al arrancar (por ejemplo con `torch.cuda.set_per_process_memory_fraction`). Si no caben los dos, el Reward Model pasa a una GPU propia sin cambiar ninguna interfaz.

**Dataset.** Demos supervisadas en el mismo formato de `Transition`, con `source="dataset"`. Vive dentro de Core y solo se accede a través de él. Entrega las acciones en unidades reales, sin normalizar: cada política del Learner normaliza a su manera (ver [[Learner Design]]). Los episodios exitosos registrados por Core pueden agregarse aquí después de un filtro.

## Comunicación entre procesos

Los procesos se comunican con **ZeroMQ**, con los arrays enviados como bytes crudos y sin pasar por un formato intermedio.

**Cómo se arma un mensaje**

- Cada mensaje es un mensaje multiparte de ZeroMQ: la primera parte es un encabezado en msgpack (tipo de llamada, ids, `dtype` y `shape` de cada array) y las partes siguientes son los buffers de los arrays tal como están en memoria.
- Del lado que envía, el buffer de numpy se manda sin copiarlo. Del lado que recibe, se reconstruye con `np.frombuffer` usando el `dtype` y `shape` del encabezado.
- No se usa pickle: es inseguro entre procesos y se rompe entre versiones de Python o de librerías.

**Patrón de sockets**

- Core expone un socket `ROUTER`; Learner, Environment, WorldModel y Reward Model se conectan con `DEALER`.
- Aunque v1 es síncrono, se usa `DEALER`/`ROUTER` en vez de `REQ`/`REP`, porque `REQ`/`REP` queda bloqueado si una respuesta se pierde o hay timeout. Con `DEALER`/`ROUTER` se pueden poner timeouts y reintentos sin reiniciar el socket.
- Transporte `ipc://` si los procesos están en la misma máquina y `tcp://` si están en máquinas distintas. Cambiar de uno a otro es solo cambiar la dirección.

**Por qué no las otras opciones**

- **gRPC.** Funciona, pero serializar arrays grandes en protobuf agrega copias y codificación, tiene un límite de mensaje por defecto de 4 MB que hay que subir, y requiere definir y compilar archivos `.proto`. No aporta nada que se necesite aquí.
- **Ray.** Es buena opción para escalar a varias máquinas y trae un object store para pasar tensores sin copiar. Pero agrega un runtime pesado que v1 no necesita. Como las interfaces ya son mensajes, migrar a Ray después es posible.

**Optimización posterior.** Si al medir el paso de imágenes resulta ser el cuello de botella en una misma máquina, las imágenes se escriben en memoria compartida (`multiprocessing.shared_memory`) y el mensaje lleva solo la referencia. No se implementa hasta que una medición lo justifique.

## Evaluación

La evaluación es una etapa posterior: se corre después del entrenamiento, sobre los checkpoints que guardó el Learner, y ninguno de los procesos de este harness participa en ella.

Usa [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness), que no lee checkpoints directamente: se conecta por WebSocket a un **model server** que carga el modelo y responde acciones. El harness de evaluación maneja su propio loop de episodios y sus propios benchmarks.

**Contrato con el VLA**

El VLA se envuelve en una subclase de `PredictModelServer` que implementa:

- `predict(obs, ctx)`: recibe una observación del benchmark y devuelve la acción.
- `get_observation_spec()` y `get_action_spec()`: declaran qué cámaras, propiocepción y espacio de acciones usa el modelo.

Ese wrapper es el único punto de traducción entre los formatos de vla-evaluation-harness y los tipos `Observation` y `ActionChunk` de este documento.

**Cómo se corre**

1. El model server carga un checkpoint guardado por el Learner y se levanta con `vla-eval serve`.
2. El benchmark corre aparte con `vla-eval run` y se conecta a ese model server.

**Notas**

- **La evaluación no usa Core ni sus ambientes.** El harness de evaluación trae sus propios ambientes y habla directo con el model server. Por eso no contradice la decisión 3: ocurre fuera de este harness.
- **El modelo de evaluación no es el Learner.** El model server es un proceso nuevo que solo carga pesos; no necesita el optimizador ni nada del loop de entrenamiento.

## Flujos principales

Los tres modos de entrenamiento usan las mismas llamadas; cambia qué llama el Learner y qué objetivo aplica.

**Supervisado**

1. El Learner llama `sample_supervised(n)`.
2. Core pide los ejemplos al Dataset y los devuelve.
3. El Learner calcula la loss de imitación y actualiza pesos.

**RL en el ambiente**

1. El Learner llama `reset(env_ids)` y recibe las observaciones iniciales.
2. El Learner genera un `ActionChunk` por ambiente con el modelo, sin gradientes, y guarda la log-probabilidad.
3. El Learner llama `step(env_ids, actions, with_reward=True)`.
4. Core reenvía a cada Environment, pide la recompensa al Reward Model, registra la transición y devuelve los `StepResult`.
5. Se repiten 2 a 4 hasta que todos los episodios terminan o se junta el lote.
6. El Learner calcula ventajas, aplica el objetivo de RL y actualiza pesos.

**RL en el WorldModel**

1. El Learner elige un punto de partida: una observación real reciente o un episodio registrado.
2. Genera varias secuencias de acciones candidatas.
3. Llama `simulate(start, sequences, with_reward=True)`.
4. Core codifica la observación en el WorldModel, simula cada secuencia, puntúa con el Reward Model y devuelve los pasos.
5. El Learner entrena con esos datos, marcados con `source="world_model"` para ponderarlos menos que los reales.

## Pendientes

- [ ] Escribir el wrapper `PredictModelServer` del VLA y una función que cargue un checkpoint del Learner en él.
- [ ] Medir la memoria que usan WorldModel y Reward Model para fijar la porción de GPU de cada uno. Sin decidir por ahora.

**Se decide en la implementación**

- Timeouts y política de reintentos de Core para cada componente.