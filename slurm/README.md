# Scripts de Slurm

Scripts para levantar el harness en un cluster con Slurm. Todo se ejecuta con
`uv` (ver `CLAUDE.md`): no uses `pip` ni entornos a mano.

Recordá la división del harness:

- **Core** es CPU: solo enruta mensajes y arrays de numpy, no depende de `torch`.
- **Learner** es el que necesita GPU (OpenVLA-7B carga ~16 GB en bf16, aunque se
  entrene solo `lm_head`).

Se comunican por ZeroMQ: `tcp://` entre máquinas, `ipc://`/`tcp://localhost` en
la misma.

> **Antes de mandar nada:** ajustá en cada `.sbatch` el `--partition`, la
> `--account` (si tu cluster la exige), la GPU, memoria y tiempo a tu sitio. Los
> valores son un punto de partida razonable, no la configuración de tu cluster.
> Mandá los jobs **desde la raíz del repo** (o exportá `REPO_ROOT`).

## Scripts

| Script | Nodos | GPU | Qué hace |
|---|---|---|---|
| `smoke_cpu.sbatch` | 1 | no | Valida el entorno: contratos `fast` + BC tiny en CPU |
| `pi0_smoke.sbatch` | 1 | sí | Prueba de entrenamiento mínimo de π0: unos pasos de BC con solo el *action expert* entrenable (resto congelado) |
| `pi0_bc.sbatch` | 1 | sí | Core + Learner π0 (BC) en el mismo nodo, con un dataset real servido por Core (pusht o LIBERO) |
| `pi0_check_embed.sbatch` | 1 | sí | Verifica que el `embed_tokens` de π0 esté atado a `lm_head` y cargado (el warning `Could not remap state dict keys` es inofensivo); no entrena |
| `libero_env_check.sbatch` | 1 | sí (render EGL) | Chequeo del ambiente LIBERO (`vla-env`, venv `.venv-env`): smoke, suites ENV/SRVE contra el simulador real y comparación MAP con el dataset convertido |
| `train_openvla_bc.sbatch` | 1 | sí | Core + Learner en el mismo nodo (OpenVLA BC) |
| `core.sbatch` + `learner.sbatch` | 2 | sí (Learner) | Core en CPU y Learner en GPU, en nodos separados |

`common.sh` es un helper que sourcean todos (localiza el repo, deja `uv` en el
PATH, espera a que el Core abra su socket). No se manda con `sbatch`.

## Uso

### Smoke test en CPU (empezá por acá)

Comprueba que uv y el harness corren en un nodo de cómputo, sin GPU ni pesos:

```bash
sbatch slurm/smoke_cpu.sbatch
```

### Prueba de entrenamiento mínimo de π0 (GPU)

Carga el π0 completo (pesos reales vía LeRobot), **congela todo salvo una sección
pequeña** —por defecto el *action expert*, dejando PaliGemma congelado— y corre
unos pocos pasos de Behavior Cloning sobre datos sintéticos. Es autocontenido: no
levanta Core ni Dataset. Sirve para confirmar que los pesos cargan y caben en la
GPU, que el congelado parcial deja entrenable solo lo pedido, y que el bucle
`loss → backward → step` avanza (imprime `fm_mse` y `grad_norm` por paso).

```bash
sbatch slurm/pi0_smoke.sbatch
```

Parámetros opcionales por variable de entorno (con sus valores por defecto):

```bash
MODEL=lerobot/pi0_base \
FREEZE=action \
TRAIN_STEPS=5 \
BATCH_SIZE=2 \
sbatch slurm/pi0_smoke.sbatch
```

- `FREEZE` es el *substring* de los módulos que quedan entrenables: `freeze_except`
  congela todo lo demás. `action` entrena solo el *action expert*; probá otro
  nombre de módulo para fine-tunear otra sección.
- La corrida usa el grupo opcional de uv `pi0` (`uv run --extra pi0`, arrastra
  `lerobot` + `torch`): el script lo instala solo.
- **Primera corrida:** descarga ~7 GB de pesos de Hugging Face. Si tu `$HOME` es
  chico, exportá `HF_HOME` a un disco con espacio antes del `sbatch`.
- También carga el tokenizer de PaliGemma, que está en un repo con acceso
  restringido: hace falta el mismo login de HF que para `pi0_bc.sbatch` (abajo).
- Memoria: π0 ocupa ~7 GB en bf16; con solo el *action expert* entrenable, una
  GPU de ≥16 GB alcanza.

Revisá en el log (`slurm/logs/vla-pi0-smoke-<JOBID>.out`) que la loss `fm_mse`
cambie entre pasos y que `grad_norm` sea finito y > 0: eso confirma que la sección
entrenable recibe gradientes.

Para pasar de esta prueba al flujo supervisado con demos reales servidas por Core,
usá `pi0_bc.sbatch` (sección siguiente).

### π0 BC con dos procesos sobre un dataset real (GPU)

Prueba E2E de `docs/dataset/Pi0 E2E Plan.md`: el Core sirve el corpus `pusht`
convertido a formato `local` y el Learner entrena π0 con BC. Primero convertí el
corpus, una sola vez, en el nodo de login:

```bash
uv run --extra pi0 python -I scripts/prepare_lerobot_local.py   # escribe data/pi0_pusht/
sbatch slurm/pi0_bc.sbatch
```

- Core y Learner se hablan por un socket `ipc://` propio del job, no por
  `tcp://127.0.0.1:5555`: el nodo es compartido y ese puerto puede estar tomado.
  El script genera copias de `configs/core/pi0_pusht.yaml` y
  `configs/pi0_pusht_bc.yaml` con esa dirección; los YAML del repo no cambian.
- El Core corre en su propio venv, `.venv-core` (`UV_PROJECT_ENVIRONMENT`), para que
  el `uv run --extra pi0` del Learner no le reinstale numpy mientras corre.
- **Antes de la primera corrida:** el Learner carga el tokenizer de PaliGemma
  (`google/paligemma-3b-pt-224`), que está en un repo con acceso restringido.
  Aceptá la licencia en Hugging Face y guardá el token en el `HF_HOME` que usa el
  job (`.hf_cache`, ver `common.sh`):

  ```bash
  HF_HOME=.hf_cache uv run --extra pi0 hf auth login
  ```
- Al arrancar, el Learner calcula las estadísticas de normalización con 256 pares
  pedidos al Core (`normalizer.stats: fit` en `configs/pi0_pusht_bc.yaml`).
- Éxito: el log (`slurm/logs/pi0-bc-<JOBID>.out`) muestra los 10 pasos con
  `fm_mse` y `grad_norm` finitos, y el job termina en `COMPLETED`.
- **Otro dataset (LIBERO):** convertilo con el comando de la cabecera de
  `configs/core/pi0_libero.yaml` y elegí los configs por variable de entorno:

  ```bash
  CORE_CONFIG=configs/core/pi0_libero.yaml LEARNER_CONFIG=configs/pi0_libero_bc.yaml \
      sbatch slurm/pi0_bc.sbatch
  ```

  El job verifica que exista el `dataset.path` del config de Core y espera al Core hasta
  `CORE_WAIT_S` segundos (300 por defecto; con LIBERO tarda ~70 s en cargar).

### OpenVLA BC en un solo nodo (recomendado)

Core y Learner conviven en el nodo GPU y hablan por `tcp://127.0.0.1:5555`, que
es lo que ya esperan `configs/core/local_openvla.yaml` y `configs/openvla_bc.yaml`
sin tocarlos:

```bash
sbatch slurm/train_openvla_bc.sbatch
```

El script arranca el Core en segundo plano, espera a que atienda, corre el
Learner en primer plano y baja el Core al terminar.

> Ojo: el puerto 5555 es fijo. En un nodo compartido puede estar tomado por otro
> usuario; el Core falla con `Address already in use` y el chequeo de puerto da
> "listo" de todas formas. Si pasa, usá `ipc://` por job como `pi0_bc.sbatch`.

### Core y Learner en nodos separados (CPU + GPU)

Para que el Core use una partición CPU y el Learner una GPU. Se encadenan: el
Core publica su dirección en `slurm/rendezvous/`, y el Learner la lee, entrena y
cancela el Core al terminar.

```bash
CORE_JID=$(sbatch --parsable slurm/core.sbatch)
sbatch --dependency=after:${CORE_JID} slurm/learner.sbatch ${CORE_JID}
```

El `--dependency=after` hace que el Learner arranque una vez que el Core empezó
(no espera a que termine: el Core es un servidor de larga duración).

## Logs y artefactos

- Los logs van a `slurm/logs/%x-%j.{out,err}` (nombre de job y JOBID).
- `slurm/rendezvous/` guarda los archivos de dirección y los configs generados
  de la variante en dos nodos; se limpian solos al terminar los jobs.
- Ambos directorios están en `.gitignore`.
- Los checkpoints del entrenamiento salen donde diga el `checkpoint.directory`
  del config del Learner (por defecto `checkpoints/…`, también ignorado).
