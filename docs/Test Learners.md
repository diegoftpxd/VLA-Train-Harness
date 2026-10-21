

# Test learner

Complementa a [[Learner Design]]. Define la batería de tests que toda implementación del Learner debe pasar para considerarse correcta.

## Objetivo

Cada interfaz del Learner (`VLAPolicy`, `ActionHead`, `Algorithm`, `DataSource`, etc.) tiene una **suite de contrato**: un conjunto de tests escritos una sola vez contra la interfaz, no contra una implementación. Cada implementación nueva hereda la suite, entrega un fixture que construye su instancia, y corre todos los tests sin escribir ninguno nuevo.

Así, si mañana se agrega π0 o un algoritmo nuevo, la pregunta "¿está bien implementado?" se responde corriendo su suite.

## Cómo se usan las suites

```python
class VLAPolicyContract:
    """Suite de contrato. No se ejecuta sola: la heredan las implementaciones."""

    @pytest.fixture
    def policy(self) -> VLAPolicy:
        raise NotImplementedError

    @pytest.fixture
    def observations(self, policy) -> list[Observation]:
        return make_observations(n=4, cameras=policy.processor.required_cameras, seed=0)

    def test_pol02_act_shapes(self, policy, observations):
        out = policy.act(policy.processor(observations))
        assert out.actions.shape == (4, policy.chunk_size, policy.action_dim)
        ...

# En el repositorio de cada implementación:
class TestOpenVLAPolicy(VLAPolicyContract):
    @pytest.fixture
    def policy(self):
        return Registry.build("policy", TINY_OPENVLA_CONFIG)
```

**Configuraciones pequeñas.** Los modelos reales son demasiado grandes para tests rápidos. Cada implementación entrega una configuración "tiny" (pocas capas, pesos aleatorios, imágenes de 32×32) que conserva la estructura del modelo. Los pesos reales solo se usan en los tests marcados `heavy`.

**Marcas de pytest.**

|Marca|Qué incluye|Dónde corre|
|---|---|---|
|`fast`|Contratos con configuraciones tiny|CPU, en cada commit; menos de 2 minutos por implementación|
|`slow`|Tests de aprendizaje en `ToyReachEnv` con modelos tiny|CPU, al cerrar una fase; tardan minutos|
|`gpu`|Movimiento entre dispositivos y precisión mixta, con modelos tiny|GPU pequeña (hasta ~2 GB de VRAM)|
|`distributed`|DDP con 2 procesos|CPU con backend gloo|
|`heavy`|Pesos reales, FSDP con varias GPUs o más de 2 GB de VRAM|Máquina con GPU grande; nunca en la máquina de desarrollo|

**Tolerancias.**

|Comparación|fp32|bf16|
|---|---|---|
|Tensores que deberían ser iguales|`atol=1e-5`|`atol=1e-2`|
|Cociente de probabilidades (debería ser 1)|`|ratio - 1| < 1e-4`|`|ratio - 1| < 1e-2`|

## Herramientas de prueba comunes

Viven en un paquete `learner.testing` que todas las suites comparten.

|Herramienta|Qué es|
|---|---|
|`make_observations(n, cameras, seed)`|Observaciones sintéticas reproducibles con imágenes, propiocepción e instrucciones de largo variable|
|`make_action_chunks(n, chunk_size, action_dim, seed)`|Acciones sintéticas en unidades reales, con escalas distintas por dimensión|
|`FakeCoreClient`|Implementación en memoria de `CoreClient`. Registra cada llamada recibida para que los tests la inspeccionen|
|`ToyReachEnv`|Ambiente mínimo dentro de `FakeCoreClient`: un punto en 2D debe llegar a un objetivo. La propiocepción de la observación es `[pos_x, pos_y, goal_x, goal_y]` (el objetivo viaja en el estado, para que una política tiny pueda aprender). Recompensa = −distancia; `terminated` al llegar; `truncated` a los 20 pasos|
|`ToyWorldModel`|Simulador perfecto de `ToyReachEnv`, para probar el camino de `simulate`|
|`TinyBackbone`|Backbone de pocos parámetros para probar cabezas de acción con `ComposedPolicy`|
|`StubPolicy`|Política mínima con capacidades configurables, para probar algoritmos sin depender de un modelo real. Es una Gaussiana diagonal con media y desviación aprendibles, independiente de la entrada (solo usa el tamaño de batch): sirve para la mecánica de los algoritmos, no para aprender tareas dependientes del estado|
|`success_rate`|Evalúa una política en `ToyReachEnv` en modo determinista y devuelve la fracción de episodios que llegan al objetivo (para los tests LEARN)|
|`ResetStartSelector`|`StartSelector` que reinicia ambientes frescos en cada llamada, para dar puntos de partida diversos al entrenar en el WorldModel|

## Suites de contrato

### `VLAPolicy` (POL)

La suite más importante: la debe pasar todo modelo, sea un adaptador o una `ComposedPolicy`.

|ID|Verifica|Cómo|
|---|---|---|
|POL-01|Las capacidades declaradas son honestas|Si falta `LOG_PROB`, `evaluate_actions` lanza `NotImplementedError`. Si está, devuelve tensores finitos. Igual para `VALUE` y `ENTROPY`|
|POL-02|Formas de salida de `act`|`actions` es `[B, chunk_size, action_dim]`, finito. `log_probs` es `[B]` si hay `LOG_PROB`, si no `None`. `values` es `[B]` solo si hay `VALUE`|
|POL-03|Funciona con distintos tamaños de batch|`act` con B=1 y B=4 no falla; en modo determinista, el resultado de un batch coincide con el de cada elemento por separado|
|POL-04|`deterministic=True` es reproducible|Dos llamadas con la misma entrada dan el mismo resultado. En modelos con ruido (diffusion, flow) esto exige fijar el ruido inicial en modo determinista|
|POL-05|El muestreo estocástico depende de la semilla|Con la misma semilla da lo mismo; con semillas distintas da resultados distintos (si el modelo es estocástico)|
|POL-06|`act` no tiene efectos secundarios|No crea grafo de gradientes, no cambia los pesos y no cambia el modo `train`/`eval`|
|POL-07|Las acciones salen en unidades reales|Con un normalizador cuyas estadísticas tienen escalas muy distintas por dimensión, las acciones de `act` quedan en esas escalas y no en [-1, 1]|
|POL-08|Consistencia entre `act` y `evaluate_actions`|Evaluar las acciones recién generadas, con sus `extras`, da la misma log-prob que entregó `act`. Es lo que garantiza que el cociente de PPO valga 1 al inicio|
|POL-09|Consistencia sin `extras`|Lo mismo que POL-08 pasando `extras=None`. Para cabezas con tokens se acepta la tolerancia de bf16, por el redondeo al re-tokenizar|
|POL-10|`evaluate_actions` produce gradientes|`log_probs.sum().backward()` deja gradientes finitos en todos los parámetros entrenables y en ninguno congelado|
|POL-11|`imitation_loss` es válida|Devuelve un escalar finito y métricas `float`; su backward deja gradientes solo en parámetros entrenables|
|POL-12|Congelado y LoRA se respetan|Tras un paso del optimizador, los parámetros fuera de `trainable_parameters()` no cambiaron|
|POL-13|Guardar y cargar conserva el modelo|Tras `save_pretrained` y `from_pretrained`, `act` determinista da el mismo resultado, incluidas las estadísticas de normalización|
|POL-14|`clone_frozen` es independiente|La copia da las mismas salidas al crearla, no tiene gradientes y no cambia si se entrena el original|
|POL-15|`extras` es seleccionable|Cada tensor en `extras` tiene como primera dimensión B, para que `Batch.select` pueda indexarlo|
|POL-16|Funciona con precisión mixta (`gpu`)|`act`, `evaluate_actions` e `imitation_loss` bajo autocast bf16 no producen NaN|
|POL-17|Se mueve de dispositivo (`gpu`)|Política y procesador funcionan tras `.to("cuda")`|
|POL-18|Puede imitar (`slow`; con pesos reales, `heavy`)|200 pasos de BC sobre 4 ejemplos fijos bajan `imitation_loss` al menos un 90%|

### `ObservationProcessor` (OBS)

|ID|Verifica|Cómo|
|---|---|---|
|OBS-01|Formas consistentes|`ModelInput.batch_size == len(observations)` y todos los tensores tienen B como primera dimensión|
|OBS-02|Cámaras faltantes y sobrantes|Si falta una cámara de `required_cameras`, lanza un error que la nombra. Las cámaras extra se ignoran|
|OBS-03|Determinista|La misma observación produce los mismos tensores|
|OBS-04|Instrucciones de largo variable|Un batch con instrucciones de 1 y de 50 palabras no falla|
|OBS-05|No modifica la entrada|Las observaciones originales quedan iguales tras procesarlas|

### `ActionNormalizer` (NRM)

Obligatoria para `ComposedPolicy` y para todo adaptador que lo use.

|ID|Verifica|Cómo|
|---|---|---|
|NRM-01|Ida y vuelta|`denormalize(normalize(x)) ≈ x` en los tres modos|
|NRM-02|Estadísticas por embodiment|Dos embodiments con estadísticas distintas no se mezclan; uno desconocido lanza error|
|NRM-03|`fit` calcula bien|Las estadísticas coinciden con una referencia en numpy|
|NRM-04|Dimensiones constantes|Una dimensión con varianza cero no produce NaN ni infinitos|
|NRM-05|Se guarda con los pesos|Las estadísticas sobreviven a `state_dict` / `load_state_dict`|

### `ActionHead` (HEAD)

Cada cabeza se prueba montada en `ComposedPolicy` con `TinyBackbone`, así que además pasa la suite POL completa. Estos tests son adicionales.

|ID|Verifica|Cómo|
|---|---|---|
|HEAD-01|Formas|`sample` devuelve acciones `[B, H, D]`, log-probs `[B]` o `None` según capacidades, y `extras` seleccionable|
|HEAD-02|Log-prob correcta (cabezas con `LOG_PROB`)|`GaussianActionHead` coincide con `torch.distributions.Normal`; `TokenActionHead` coincide con la suma de log-softmax de los tokens elegidos|
|HEAD-03|Log-prob de un chunk es la suma|La log-prob de un chunk de H acciones es la suma de las H log-probs individuales|
|HEAD-04|Discretización estable (`TokenActionHead`)|Acción → tokens → acción → tokens devuelve los mismos tokens|
|HEAD-05|`bc_loss` es mínima en el objetivo|Para cabezas con log-prob, la loss en la acción objetivo es menor que en una acción aleatoria tras entrenar unos pasos|

### `Batch` y `MinibatchIterator` (BAT, MB)

|ID|Verifica|Cómo|
|---|---|---|
|BAT-01|`select` es consistente|Seleccionar índices indexa todos los campos igual: tensores, `inputs.tensors`, `extras` y listas como `episode_ids`|
|BAT-02|Casos borde de `select`|Índices vacíos, todos los índices y una permutación devuelven el tipo correcto y el largo correcto|
|MB-01|Cubre todo una vez|En una pasada, cada fila aparece exactamente una vez|
|MB-02|Último minibatch parcial|Con 10 filas y tamaño 4 entrega minibatches de 4, 4 y 2|
|MB-03|`size=None`|Entrega el batch completo una sola vez|
|MB-04|Reproducible|Con la misma semilla, el orden es el mismo|

### `AdvantageEstimator` (ADV)

Tests numéricos contra valores calculados a mano en trayectorias pequeñas.

|ID|Verifica|Cómo|
|---|---|---|
|ADV-01|GAE en casos límite|Con λ=1 coincide con retornos Monte Carlo menos valores; con λ=0 coincide con el error TD de un paso|
|ADV-02|`terminated` vs `truncated`|`terminated` no usa el valor del estado siguiente; `truncated` sí lo usa|
|ADV-03|Varios episodios mezclados|Filas de episodios distintos y desordenadas se agrupan por `episode_ids` y se ordenan por `step_ids` antes de calcular|
|ADV-04|`GroupRelativeAdvantage`|Cada grupo queda con media 0 y desviación 1|
|ADV-05|Grupos degenerados|Un grupo con todas las recompensas iguales, o de tamaño 1, da ventajas 0 y no NaN|
|ADV-06|`MonteCarloReturn`|Coincide con la suma descontada calculada a mano|

### `Algorithm` (ALG)

Suite genérica que pasa todo algoritmo, más tests específicos de cada uno. Se usa `StubPolicy` para no depender de un modelo real.

|ID|Verifica|Cómo|
|---|---|---|
|ALG-01|Declaraciones|`required_capabilities` es un conjunto de `Capability`; `batch_type` es `SupervisedBatch` o `RolloutBatch`|
|ALG-02|`loss` válida|Devuelve un escalar finito y métricas `float`, con gradientes hacia la política|
|ALG-03|`prepare` no entrena|No crea gradientes, no cambia los pesos, y llamarlo dos veces da el mismo resultado|
|ALG-04|Dirección correcta (algoritmos de RL)|En un bandit de un paso, un paso de gradiente aumenta la probabilidad de la acción con ventaja positiva y baja la de ventaja negativa|

Tests específicos:

|ID|Verifica|Cómo|
|---|---|---|
|BC-01|Delega en el modelo|La loss es exactamente `policy.imitation_loss`|
|PPO-01|Sin cambio de pesos no hay recorte|Con los mismos pesos que en el rollout, el cociente es 1 y la fracción recortada es 0|
|PPO-02|El recorte funciona|Si el cociente supera 1+ε con ventaja positiva, esas filas no aportan gradiente al término de política|
|PPO-03|Loss de valor|Coincide con una referencia, con y sin `value_clip`|
|PPO-04|`target_kl` corta las épocas|Si el KL aproximado supera `target_kl`, no se hacen más épocas en esa iteración (ver Pendientes)|
|GRPO-01|Exige grupos|Sin `group_ids` en el batch, lanza un error claro|
|GRPO-02|Referencia coherente con `kl_coef`|Con `kl_coef > 0` y sin `ReferencePolicy`, falla al construirse|
|GRPO-03|KL nulo en el origen|Si la política es igual a la referencia, el término KL vale 0|

### `CoreClient` (CORE)

La misma suite corre contra `FakeCoreClient` y contra el `CoreClient` real conectado a un Core de prueba. Así se garantiza que el falso se comporta como el real.

|ID|Verifica|Cómo|
|---|---|---|
|CORE-01|Serialización exacta|`Observation`, `ActionChunk` y `StepResult` llegan con el mismo `dtype`, forma y valores, bit a bit|
|CORE-02|Orden de ambientes|Las respuestas a llamadas con varios `env_ids` vienen en el mismo orden que se pidieron|
|CORE-03|Episodio terminado|Un `step` sobre un ambiente terminado sin `reset` lanza error|
|CORE-04|Timeout|Si Core no responde, se lanza un error de timeout en el plazo configurado, sin bloquearse|
|CORE-05|Recompensa opcional|Con `with_reward=False`, `reward` llega en `None`|

### `DataSource` y `RolloutCollector` (DATA)

Todos corren contra `FakeCoreClient` con `ToyReachEnv`.

|ID|Verifica|Cómo|
|---|---|---|
|DATA-01|Tipo declarado|`next_batch` devuelve el tipo que dice `batch_type`|
|DATA-02|Supervisado en unidades reales|`SupervisedSource` entrega acciones iguales a las que devolvió Core, sin normalizar, y del tamaño pedido|
|DATA-03|Rollouts completos|`EnvRolloutSource` entrega `steps_per_batch` filas, con `old_log_probs` iguales a los que dio `act`, `extras` guardados y `terminated`/`truncated` propagados|
|DATA-04|Reinicio de ambientes|Un ambiente que termina se reinicia y el episodio siguiente tiene otro `episode_id`|
|DATA-05|Acciones enviadas sin alterar|Las acciones que recibe Core son exactamente las que devolvió `act`|
|DATA-06|Grupos en el ambiente|Con `group_size=4`, los `reset` de cada grupo usan la misma tarea y semilla, y `group_ids` los identifica|
|DATA-07|Grupos en el WorldModel|`WorldModelRolloutSource` arma un grupo por punto de partida, con `source="world_model"`|
|DATA-08|Mezcla de fuentes|`MixedSource` respeta los pesos en promedio sobre muchas llamadas; mezclar tipos de batch distintos falla al construirse|
|DATA-09|El colector no deja rastro|Tras recolectar, la política vuelve a modo `train` y no quedan gradientes acumulados|

### Infraestructura (OPT, DIST, CKPT, LOG)

|ID|Verifica|Cómo|
|---|---|---|
|OPT-01|Solo entrena lo entrenable|`step` cambia solo los parámetros entregados al construirlo|
|OPT-02|Clipping|La norma del gradiente aplicado no supera `grad_clip`|
|OPT-03|Acumulación|N pasos con `grad_accumulation=N` equivalen a un paso con un batch N veces más grande|
|OPT-04|Scheduler|El learning rate sigue la curva esperada (warmup y coseno)|
|OPT-05|fp16 con desborde (`gpu`)|Un gradiente infinito hace que se salte el paso sin corromper los pesos|
|DIST-01|`SingleDevice`|`is_main` es verdadero y `reduce_metrics` no cambia nada|
|DIST-02|DDP (`distributed`)|`reduce_metrics` promedia entre procesos y los pesos quedan iguales en todos tras un paso|
|DIST-03|FSDP (`distributed`, `heavy`)|`full_state_dict` devuelve los pesos completos y coincide con los de una corrida en una GPU|
|CKPT-01|Estructura|`save` crea `policy/` y `training_state.pt`; `keep_last` borra los más antiguos; `latest` devuelve el más reciente|
|CKPT-02|Retomar es exacto|Entrenar N pasos seguidos da los mismos pesos que entrenar K, guardar, cargar y entrenar N−K, con semillas fijas|
|LOG-01|Solo el proceso principal publica|En un proceso que no es principal, `log` no publica nada|

### `Registry` (REG)

|ID|Verifica|Cómo|
|---|---|---|
|REG-01|Todo lo registrado se construye|Cada nombre registrado se construye desde su configuración mínima de ejemplo|
|REG-02|Nombre desconocido|Lanza un error que lista los nombres disponibles|
|REG-03|Nombre duplicado|Registrar dos veces el mismo nombre lanza error|

### `Learner` (LRN)

Tests de integración: el Learner completo contra `FakeCoreClient`, con una política tiny.

|ID|Verifica|Cómo|
|---|---|---|
|LRN-01|Incompatibilidades fallan al construir|PPO con una política sin `VALUE`, tipos de batch distintos entre algoritmo y fuente, o cámaras faltantes lanzan error en el constructor, no durante `train`|
|LRN-02|Una iteración de cada modo|BC, PPO en ambiente y GRPO en WorldModel completan una iteración, cambian los pesos y publican métricas|
|LRN-03|Cantidad de pasos del optimizador|Por iteración hay `update_epochs × ceil(N / minibatch_size)` llamadas a `OptimizerManager.step`|
|LRN-04|Checkpoints periódicos|Con `checkpoint_every=2` y 5 iteraciones se guardan los checkpoints esperados|
|LRN-05|`resume` continúa donde quedó|Tras `resume`, la iteración inicial es la siguiente a la guardada|

### Reglas de arquitectura (ARCH)

Tests estáticos que revisan el código fuente y protegen los principios del diseño. No ejecutan nada del modelo.

|ID|Verifica|Cómo|
|---|---|---|
|ARCH-01|Solo `CoreClient` habla por la red|Ningún módulo fuera de `CoreClient` importa `zmq`|
|ARCH-02|Los algoritmos no tocan el optimizador|Ningún módulo de algoritmos llama a `.backward(` ni a `.step(`|
|ARCH-03|El Learner no conoce modelos concretos|`learner.py` no importa ninguna implementación de `VLAPolicy`, `Backbone` ni `ActionHead`|
|ARCH-04|Los algoritmos no conocen modelos concretos|Los módulos de algoritmos solo importan la interfaz `VLAPolicy`|

### Aprendizaje (LEARN)

Tests lentos que comprueban que el sistema completo aprende, no solo que no falla. Usan `ToyReachEnv`, donde una política aleatoria casi nunca llega al objetivo.

|ID|Verifica|Cómo|
|---|---|---|
|LEARN-01|BC aprende|Con demos de un controlador perfecto, la tasa de éxito en `ToyReachEnv` supera 90% tras el entrenamiento|
|LEARN-02|PPO mejora|Desde una política aleatoria, la tasa de éxito pasa de menos de 10% a más de 80%|
|LEARN-03|GRPO mejora|Igual que LEARN-02|
|LEARN-04|RL en el WorldModel transfiere|Entrenando solo con `ToyWorldModel`, la tasa de éxito mejora al evaluar en `ToyReachEnv`|
|LEARN-05|BC + RL|Partiendo de una política entrenada con BC, el RL no empeora la tasa de éxito|

Estos tests solo usan la política tiny y una cabeza gaussiana o de tokens. Su objetivo es validar algoritmos y datos, no un modelo concreto.

La tasa de éxito se mide con `success_rate` (en `testing/evaluation.py`): corre la política en modo **determinista** (usa la media, no muestrea) sobre episodios de estados distintos. Por eso la política tiny se inicializa con la media en cero (`GaussianActionHead(init_zero=True)`): arranca neutra, su éxito inicial es ~0 y cualquier mejora demuestra aprendizaje. Para que el entrenamiento en el WorldModel generalice (LEARN-04) se usan puntos de partida diversos (`ResetStartSelector`), no unos pocos fijos.

### Evaluación (EVAL)

`EvalModelServer` es un componente opcional, fuera del alcance base del harness (ver Alcance en `CLAUDE.md`): solo se implementa si se pide explícitamente. Esta suite solo corre cuando `EvalModelServer` existe; mientras no esté implementado, EVAL queda sin ejecutar y no bloquea ninguna fase.

|ID|Verifica|Cómo|
|---|---|---|
|EVAL-01|Se construye desde cualquier checkpoint|`EvalModelServer` carga un checkpoint de cada política registrada|
|EVAL-02|Misma acción que el entrenamiento|`predict` devuelve lo mismo que `act(deterministic=True)` de la política original, en unidades reales|
|EVAL-03|Especificaciones coherentes|`get_observation_spec` y `get_action_spec` coinciden con `required_cameras`, `chunk_size` y `action_dim`|

## Qué suites debe pasar cada implementación nueva

|Si se agrega...|Debe pasar|
|---|---|
|Un adaptador de modelo (`VLAPolicy`)|POL, OBS; NRM si usa `ActionNormalizer`; EVAL solo si `EvalModelServer` está implementado|
|Un backbone|POL montado en `ComposedPolicy` con una cabeza existente|
|Una cabeza de acción|HEAD y POL montada en `ComposedPolicy` con `TinyBackbone`|
|Un algoritmo|ALG, sus tests específicos y LEARN en `ToyReachEnv`|
|Un estimador de ventajas|ADV|
|Una fuente de datos|DATA|
|Un cambio en el protocolo hacia Core|CORE contra el cliente real y el falso|
|Cualquier cambio|ARCH y LRN|

## Decisiones resueltas

Las dos decisiones que este documento dejaba abiertas ya están resueltas e incorporadas a [[Learner Design]]:

- **Corte de épocas por KL (PPO-04).** `Algorithm.should_stop_epochs(metrics: dict[str, float]) -> bool`, con implementación por defecto que devuelve `False`. `PPO` devuelve `True` cuando `approx_kl > target_kl`. El `Learner` lo consulta después de cada época y corta las épocas restantes de esa iteración.
- **`deterministic=True` en modelos con ruido (POL-04).** El ruido inicial se deriva de una semilla constante mediante un generador propio de la política, sin tocar el estado aleatorio global.