# Test Environment

Complementa a [`Environment Design.md`](./Environment%20Design.md). Define los tests que
**todo ambiente** del harness debe pasar para considerarse correcto: simuladores,
adaptadores de benchmarks y robots reales. Es el equivalente de
[`../learner/Test Learners.md`](../learner/Test%20Learners.md) y
[`../core/Test Core.md`](../core/Test%20Core.md) para el componente Environment.

## Objetivo

La interfaz `Environment` tiene una **suite de contrato** (ENV) escrita una sola vez contra
la interfaz, no contra una implementación. Un ambiente nuevo entrega un fixture que lo
construye y hereda la suite, sin escribir tests nuevos. Si mañana se agrega MetaWorld o un
brazo real, la pregunta "¿está bien implementado?" se responde corriendo su suite.

Hay suites adicionales según el tipo de ambiente:
- **GYM:** reglas de la base `GymEnvironment`.
- **SRVE:** el ambiente servido por la red.
- **MAP:** consistencia con el dataset de entrenamiento.
- **ROBOT:** seguridad de un robot real.

Al final, una tabla dice qué suites debe pasar cada tipo de implementación.

## Cómo se usan las suites

```python
# vla_common/testing/contracts/environment_contract.py
class EnvironmentContract:
    """Suite de contrato. No se ejecuta sola: la heredan las implementaciones."""

    @pytest.fixture
    def environment(self) -> Environment:
        raise NotImplementedError

    def test_env01_reset_devuelve_observacion(self, environment): ...

# En los tests de cada implementación:
class TestReachEnvironment(EnvironmentContract):
    @pytest.fixture
    def environment(self):
        return ReachEnvironment()
```

Sin escribir ninguna clase, cualquier ambiente construible desde un YAML se prueba con:

```bash
VLA_ENV_CONFIG=configs/env/<nombre>.yaml uv run pytest -m heavy packages/vla_env/tests/test_env_from_config.py
```

Ese archivo corre ENV en memoria y SRVE por la red contra el ambiente que construye
`build_environment` desde el YAML.

**Fixture opcional `neutral_action`.** ENV-04 necesita una acción que no dañe nada mientras
lleva el episodio a un final. Por defecto son ceros. Un ambiente donde los ceros no son
neutros (LIBERO cierra el gripper con `-1`) la redefine; desde un YAML se toma de
`environment.smoke_action`.

**Marcas.**

| Marca | Qué incluye | Dónde corre |
|---|---|---|
| `fast` | ENV, GYM, SRVE y CLI sobre ambientes sin dependencias (`reach`, fakes); mapeos puros (sin simulador) | En cada commit, en CPU |
| `distributed` | Core contra procesos `vla-env` reales | Local, al tocar la integración con Core |
| `slow` | Learner → Core → `vla-env` de punta a punta | Local, al cerrar una fase |
| `heavy` | ENV/SRVE/MAP contra simuladores o robots reales | Cluster o el equipo del robot; **nunca** en la máquina de desarrollo |

ROBOT no se corre automáticamente: tiene partes que se verifican con el robot presente y
el operador a mano en la parada de emergencia física.

## Herramientas comunes

- **`EnvironmentContract`** (`vla_common/testing/contracts/`): la suite ENV.
- **`ReachEnvironment`** (`vla_env/reach.py`): ambiente de referencia sin dependencias,
  para los tests `fast` y la integración local.
- **`FakeEnvironment` / `FlakyEnvironment`** (`vla_core/testing/fakes.py`): dobles en
  memoria de Core; el segundo simula un proceso caído.
- **`EnvironmentServer`** (`vla_env/server.py`) y **`EnvironmentClient`**
  (`vla_core/environments.py`): el par servidor/cliente para SRVE.
- **`vla-env --smoke N --save DIR`** y **`scripts/check_env_vs_dataset.py`**: el chequeo
  rápido y la comparación con el dataset (MAP).

## Suites

### Contrato (ENV) — todo ambiente

| ID | Qué verifica |
|---|---|
| ENV-01 | `reset` devuelve una `Observation` con `proprio` `float32` 1-D y una entrada en `images` por cada cámara de `spec.cameras` |
| ENV-02 | `step` devuelve un `StepResult` con `reward is None` (la recompensa la pone Core) y `terminated`/`truncated` de tipo `bool` |
| ENV-03 | El `spec` es válido: `action_dim > 0`, al menos una cámara, al menos una tarea, sin nombres repetidos |
| ENV-04 | Con `neutral_action`, `step` llega a un final (`terminated` o `truncated`) antes de un tope; después, `emergency_stop` y `close` no fallan |
| ENV-05 | `reset` con la misma semilla es determinista en `proprio` |
| ENV-06 | `reset(task=spec.tasks[0])` da `instruction == task`; `reset(task=None)` usa `tasks[0]`; una tarea fuera de `spec.tasks` es error |
| ENV-07 | Un chunk con una acción de dimensión distinta de `action_dim` es error |
| ENV-08 | `info["success"]` existe y es `bool` en cada `step`; todo valor de `info` es escalar (`bool`, `int`, `float`, `str`) |
| ENV-09 | Las imágenes son `float32`, CHW, de 3 canales, con valores en `[0, 1]`, y son idénticas con la misma semilla |
| ENV-10 | `success=True` implica `terminated=True`, y `terminated` y `truncated` nunca son verdaderos a la vez |
| ENV-11 | `step` después de un final, sin `reset`, es error |

**Tolerancias.** El determinismo es igualdad exacta (`np.array_equal`), dentro de un
mismo proceso. No se comparan observaciones entre máquinas, porque MuJoCo puede variar.
El rango de imágenes es exacto: ningún valor fuera de `[0, 1]`.

### Base para simuladores (GYM) — `GymEnvironment`

Se corren una vez, sobre un simulador de prueba que cuenta las llamadas que recibe. Todo
adaptador que herede de `GymEnvironment` hereda estas garantías sin volver a probarlas.

| ID | Qué verifica |
|---|---|
| GYM-01 | Corte a mitad de chunk: el resultado es el de la acción que terminó y las siguientes no llegan al simulador |
| GYM-02 | `max_steps` cuenta acciones: un chunk que cruza el límite se corta ahí, con `truncated=True` y `terminated=False` |
| GYM-03 | `info["is_success"]` del simulador da `terminated=True` y `success=True`, aunque el simulador no marque `terminated` |
| GYM-04 | La recompensa del simulador se ignora (`reward is None`) |
| GYM-05 | `step` después de un final es error |
| GYM-06 | Una acción mal dimensionada es error y el simulador no recibe **ninguna** acción del chunk |
| GYM-07 | `reset(task=None)` usa `tasks[0]`; una tarea desconocida es error; `_make` recibe la tarea pedida |
| GYM-08 | `to_canonical_image`: HWC `uint8` → CHW `float32` en `[0, 1]`; CHW `float` se respeta; una forma sin 3 canales es error |
| GYM-09 | `reset` pasa la semilla al simulador y reinicia el contador de acciones |

### Por la red (SRVE) — todo ambiente servido por `vla-env`

| ID | Qué verifica |
|---|---|
| SRVE-01 | El ambiente servido por `EnvironmentServer` pasa ENV completo a través de `EnvironmentClient` |
| SRVE-02 | Las observaciones y acciones cruzan el wire sin pérdida (igualdad bit a bit con lo que devuelve el ambiente en memoria con la misma semilla) |
| SRVE-03 | Un error del ambiente vuelve como `{ok: False, error}`, el cliente levanta error y el servidor sigue atendiendo |
| SRVE-04 | `stop()` libera el puerto: otro servidor puede bindear la misma dirección |

### CLI (CLI) — `vla-env`

| ID | Qué verifica |
|---|---|
| CLI-01 | `--smoke N` imprime el spec, las formas, dtypes y rangos por cámara y del `proprio`, y el tiempo por `step`; termina con código 0 |
| CLI-02 | `--save DIR` escribe un `npz` con una clave por cámara y el `proprio` |
| CLI-03 | `--address` pisa `server.address` del YAML |
| CLI-04 | Un `type` desconocido o un extra sin instalar termina con un mensaje claro (para el extra, el comando `uv sync --extra ...`) |

### Consistencia con el dataset (MAP) — ambientes con dataset de entrenamiento

Una política entrenada con un dataset solo funciona en el ambiente si las observaciones se
parecen. Nada de esto rompe ENV, por eso es una suite aparte. La corre
`scripts/check_env_vs_dataset.py` sobre el `npz` de `--smoke --save`.

| ID | Qué verifica | Tolerancia |
|---|---|---|
| MAP-01 | Mismas claves de cámara que el dataset, misma forma y dtype | exacta |
| MAP-02 | `proprio` con la misma dimensión que el estado del dataset | exacta |
| MAP-03 | El `proprio` tras `reset` cae dentro del rango del dataset | mín/máx del dataset ± 10 % del rango, por componente |
| MAP-04 | Orientación: el primer frame del ambiente se parece más a los frames iniciales del dataset (misma tarea) que su versión rotada 180° | correlación mayor sin rotar |
| MAP-05 | La instrucción de cada tarea del ambiente existe tal cual entre las instrucciones del dataset | exacta |

### Robot real (ROBOT)

Además de ENV. Lo que se puede automatizar va como test `heavy` en el equipo del robot; el
resto es un checklist que se firma antes de dejar que una política mueva el robot.

| ID | Qué verifica | Cómo |
|---|---|---|
| ROBOT-01 | Una acción fuera de los límites de la config se rechaza y el robot no se mueve | `heavy` |
| ROBOT-02 | `emergency_stop` detiene el robot y es idempotente (llamarla dos veces, o con el robot quieto, no falla) | `heavy` |
| ROBOT-03 | El watchdog local detiene el robot si no recibe comandos dentro del plazo configurado, **sin** ayuda de la red (se corta el proceso de Core a mitad de un episodio) | `heavy` + operador |
| ROBOT-04 | `reset` deja al robot en la pose segura antes de devolver la observación | `heavy` |
| ROBOT-05 | Un `step` colgado (driver que no responde) no impide que el robot se detenga | operador |
| ROBOT-06 | `--smoke` con la acción neutra corre completo con el operador presente | operador |

## Integración con el resto del harness

Estas suites no prueban un ambiente sino su encaje con Core y el Learner. Viven en
`vla_core/tests` y se corren al tocar esa integración. El detalle está en el plan de
implementación.

| Suite | Qué verifica | Marca |
|---|---|---|
| CENV | Core valida la dimensión de la acción antes de mover nada, escala la falla del `spec`, y la reinyección usa `success` | `fast` |
| CFG | `build_core` arma el registro desde `environments:`/`timeouts:` | `fast` |
| DIST | Core contra procesos `vla-env` reales: enrutado, orden y aislamiento de la caída de un proceso | `distributed` |
| E2E-ENV | El Learner entrena por Core contra procesos `vla-env` (RL en ambiente, RL en world model con inicios reales, reinyección de episodios) | `slow` |
| EVAL | La medición de tasa de éxito cuenta por `success` y reparte episodios entre ambientes | `fast` + `slow` |

## Reglas de arquitectura (ARCH)

| ID | Qué verifica |
|---|---|
| ARCH-ENV-01 | Ningún módulo de `vla_common` (fuera de `testing/`) importa `zmq`, `torch`, `vla_core`, `vla_learner` ni `vla_env` |
| ARCH-ENV-02 | Ningún módulo de `vla_env/src` importa `vla_core` ni `vla_learner` (en los tests sí se permite) |
| ARCH-ENV-03 | La base de `vla_env` no importa `torch`; un adaptador puede hacerlo solo con import perezoso |

## Qué suites debe pasar cada implementación

| Implementación | ENV | GYM | SRVE | CLI | MAP | ROBOT |
|---|---|---|---|---|---|---|
| Simulador sobre `GymEnvironment` (LIBERO, MetaWorld, …) | sí | heredada | sí | — | sí, si hay dataset | — |
| Simulador que implementa `Environment` directo | sí | — (pasa ENV-10/11 por sí mismo) | sí | — | sí, si hay dataset | — |
| Robot real | sí | — | sí | — | sí | sí |
| `ReachEnvironment` (referencia) | sí | es la base de GYM | sí | sí | — | — |
| `FakeEnvironment` (doble de Core) | sí | — | sí (vía `EnvironmentClient`) | — | — | — |

Si un test de contrato parece incorrecto para un ambiente legítimo, no se relaja ni se
salta: se discute y, si corresponde, se cambia este documento y la suite en el mismo
commit.
