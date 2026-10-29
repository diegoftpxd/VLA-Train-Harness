# Plan: π0 de dos procesos sobre un dataset real

Objetivo: correr el flujo supervisado completo con **dos procesos independientes**
—un `vla-core` (gateway) y un `vla-learner` (modelo π0)— comunicados por ZeroMQ,
entrenando π0 con Behavior Cloning sobre un **dataset real, descargado y simple**,
pero con la información que π0 necesita. Es la prueba de integración de
`examples/pi0_smoke.py` pero sacando los datos de un corpus real a través del
harness de verdad (Dataset → Core → Learner), no de datos sintéticos.

Este documento es un **plan de acción + runbook**; deriva de
[`Dataset Design.md`](Dataset%20Design.md), [`../Architecture.md`](../Architecture.md)
y de lo ya verificado del adaptador `Pi0Policy`.

## Qué ya existe (no hay que construirlo)

- **Adaptadores de dataset**: `local` (sin deps, offline) y `lerobot` (corpus HF)
  en `packages/vla_core/src/vla_core/dataset/`, con factory por `type`.
- **Core sirviendo demos**: `sample_supervised(n, filters)` ya implementado; configs
  de ejemplo `configs/core/local.yaml`, `configs/core/lerobot.yaml`.
- **Learner π0 BC**: `configs/pi0_bc.yaml` + el adaptador `Pi0Policy`
  (carga de pesos reales ya arreglada: se usa `PreTrainedConfig.from_pretrained`).
- **sbatch de dos procesos** (Core CPU + Learner GPU en un nodo):
  `slurm/train_openvla_bc.sbatch` como plantilla.
- **`SupervisedSource`** (`vla_learner/data/sources.py`): pide pares al Core y arma el
  batch con `policy.processor(observaciones)` y `torch.stack` de las acciones.

## Restricciones que impone π0 (verificadas en el código de LeRobot 0.4.4)

Dictan la forma del dataset. Todas salen de `lerobot/policies/pi0/modeling_pi0.py`:

1. **Cámaras.** `_preprocess_images` exige **al menos una** de las claves de
   `config.image_features` (`observation.images.base_0_rgb`,
   `observation.images.left_wrist_0_rgb`, `observation.images.right_wrist_0_rgb`).
   Las que **falten las rellena con -1 y las enmascara**. ⇒ basta **una sola cámara**,
   pero debe llamarse **exactamente** como una de esas tres (usamos `base_0_rgb`).
2. **Estado y acción.** `prepare_state`/`prepare_action` padean con ceros hasta
   `max_state_dim`/`max_action_dim` (**32**). ⇒ el estado y la acción del corpus
   pueden tener menos dims (p. ej. 2); π0 padea solo. El **dataset no normaliza**
   (entrega unidades reales); normaliza el adaptador `Pi0Policy` (ver "Puntos
   abiertos" y `Learner Design.md`, "Adaptador `Pi0Policy`").
3. **Largo del chunk.** `SupervisedSource` hace `torch.stack` de las acciones, así que
   **todas las chunks del batch deben tener el mismo largo H**, y π0 espera
   `H = n_action_steps = 50`. ⇒ el dataset tiene que entregar chunks de **largo fijo 50**.
   Con el adaptador `local`, esto se garantiza haciendo que cada episodio on-disk mida
   un múltiplo de 50 (lo más simple: **exactamente 50 pasos**), porque `LocalDataset`
   trocea en ventanas de `demo_chunk_size` y **corta la cola** del episodio.

## Decisión de arquitectura: Core liviano (`local`), no `lerobot` en vivo

Dos formas de que el Core sirva un corpus real:

- **Opción A — Core lee el corpus LeRobot en vivo** (`type: lerobot`). Requiere
  `uv run --extra lerobot vla-core ...`, lo que mete **torch** en el proceso Core
  (contra el espíritu de `CLAUDE.md`: "Core base liviano, nada de torch") y además
  exige arreglar un bug del adaptador (ver abajo).
- **Opción B — convertir una vez a formato `local` y servir con `type: local`**
  (**recomendada**). El Core corre con la base de `vla_core` (ZeroMQ + numpy, **sin
  torch, sin lerobot**), y la dependencia pesada queda solo en un script de
  preparación offline que se corre una vez. Esto hace a los dos procesos realmente
  **independientes en dependencias**: Core = base, Learner = `--extra pi0`. También
  deja control total sobre el troceo de 50 pasos (restricción 3).

> **Sobre "el bug de dependencias se arregla solo":** separar los procesos da
> *independencia de dependencias* (Core no necesita lo del Learner ni al revés),
> pero **no** arregla por sí mismo el bug del adaptador `lerobot`
> (`vla_core/dataset/lerobot.py:65` importa `lerobot.common.datasets.lerobot_dataset`,
> que **no existe en lerobot 0.4.4**; el path correcto es `lerobot.datasets.lerobot_dataset`).
> La Opción B lo **esquiva** porque el Core nunca usa ese adaptador. Si algún día se
> quiere la Opción A, hay que agregar el fallback de layout en ese import, igual que
> se hizo en el adaptador de π0.

## Dataset elegido

`lerobot/pusht`: corpus LeRobot real, chico (~un par de cientos de MB), **una cámara**
(`observation.image`, 96×96), estado 2-D (`observation.state`), acción 2-D (`action`).
Encaja con las restricciones de π0 vía padding/máscara. Es solo para probar el
cableado; la "tarea" en sí no es relevante para la prueba de humo.

Mapeo al formato que π0 necesita:

| π0 / canónico                         | pusht                 | Nota                              |
|---------------------------------------|-----------------------|-----------------------------------|
| imagen `observation.images.base_0_rgb`| `observation.image`   | única cámara; las otras 2 las masquea π0 |
| `proprio` (→ pad a 32)                | `observation.state` (2) | sin normalizar                  |
| `actions` (→ pad a 32)                | `action` (2)          | chunk de largo 50                 |
| `instruction`                         | task / fijo           | p. ej. `"push the T block"`       |

## Plan de acción

Cada paso cierra con `uv run pytest -m fast`, `ruff` y `pyright` en verde cuando toque
código, y un commit en la rama `Dataset`. **Estado: los cinco pasos están hechos.**
La prueba corrió en el cluster (job Slurm 35206, nodo `llaima`, A40) y cumplió el
criterio de éxito; ver el paso 5 y "Puntos abiertos".

1. ✅ **Script de preparación (offline, una vez).** `scripts/prepare_lerobot_local.py`:
   - carga pusht con `from lerobot.datasets.lerobot_dataset import LeRobotDataset`
     (**path correcto de 0.4.4**, no el del adaptador con bug), con respaldo al layout
     viejo `lerobot.common.datasets...`;
   - recorre cada episodio y lo corta en ventanas **de 50 pasos exactos** (descarta la
     cola < 50, para que todas las chunks midan 50 — restricción 3);
   - escribe cada ventana como un episodio en formato `local`:
     `data/pi0_pusht/<idx>/data.npz` con `observation.images.base_0_rgb` `(50,C,H,W)`,
     `proprio` `(50,2)`, `actions` `(50,2)`; y `meta.json` con `instruction`;
   - se corre con `uv run --extra pi0 python -I scripts/prepare_lerobot_local.py`.
     **Desviación respecto al plan original:** se usa `--extra pi0`, no `--extra lerobot`.
     El extra `lerobot` vive en `vla-core` y **no está expuesto** a nivel de workspace;
     el root sí expone `pi0` (`vla-learner[pi0]`), que fija `lerobot>=0.4.4`, justo la
     versión cuyo path de import usa el script. `data/` se agregó a `.gitignore` (dato
     derivado, no se versiona).
   - **Dato no confiable:** el corpus descargado va a su propio directorio y solo se
     lee por ruta (regla de `Dataset Design.md`); el script corre con `python -I`.
   - **Desviación: backend de vídeo `pyav`.** pusht guarda las cámaras como vídeo y
     lerobot las decodifica al indexar `dataset[i]`. Su backend por defecto,
     `torchcodec`, entra como dependencia transitiva de lerobot sin fijar, y uv
     resolvió la 0.10.0, que no es compatible a nivel binario con `torch 2.6.0`:
     `RuntimeError: Could not load libtorchcodec` (para el FFmpeg 6 del sistema,
     `undefined symbol: _ZN3c1013MessageLogger6streamB5cxx11Ev` de `libc10`). El
     script abre el dataset con `LeRobotDataset(repo_id, video_backend="pyav")`:
     PyAV trae su propio FFmpeg y decodifica vía torchvision 0.21, que sí casa con
     torch 2.6. No agrega ni cambia dependencias.
   - **Resultado:** 412 episodios de 50 pasos en `data/pi0_pusht` (287 MB), todos con
     `observation.images.base_0_rgb (50,3,96,96)`, `proprio (50,2)` y `actions (50,2)`
     en `float32`. El script decodifica todos los frames del corpus aunque se pase
     `--max-episodes` (el tope solo limita lo que se escribe) y los junta en RAM
     (~3 GB); tarda varios minutos.
   - Validación en `fast`: `packages/vla_core/tests/test_prepare_lerobot_local.py` ejercita
     los helpers puros del script (`ventanas`, `preparar_desde_episodios`) sobre arrays
     sintéticos, chequea el troceo a 50 y lee la salida con `LocalDataset`, sin descargar
     nada. `lerobot` se importa solo dentro de `main`, así que el test no lo necesita.

2. ✅ **Config de Core.** `configs/core/pi0_pusht.yaml`:
   ```yaml
   server: { address: "tcp://127.0.0.1:5555" }
   backend: { type: fake }          # Environment/WM/RM no se tocan en supervisado
   dataset:
     type: local
     path: data/pi0_pusht
     cameras: { observation.images.base_0_rgb: observation.images.base_0_rgb }
     demo_chunk_size: 50            # = n_action_steps de π0 (restricción 3)
   ```
   Validada en `test_prepare_lerobot_local.py` (`fast`), **no** en `test_configs.py`: ese
   suite construye y atiende cada config, y `path: data/pi0_pusht` no está en el repo
   (es el corpus convertido). En su lugar el test chequea los invariantes del YAML y
   sirve el bloque `dataset` por un Core real sobre un fixture sintético (mismo troceo
   de 50, cámara `base_0_rgb`), round-trip incluido por el `CoreClient`.

3. ✅ **Config de Learner.** `configs/pi0_pusht_bc.yaml` (clon de `pi0_bc.yaml` corregido):
   - `cameras: [observation.images.base_0_rgb]` — **clave:** debe coincidir con la
     cámara que sirve el dataset, no `primary` (restricción 1). El `_build_pi0` pasa
     `cameras` del YAML tal cual al procesador.
   - `available_cameras: [observation.images.base_0_rgb]`.
   - `policy: {type: pi0, pretrained: lerobot/pi0_base, torch_dtype: bfloat16,
     freeze: {trainable: [action]}}`.
   - `algorithm: {type: bc}`, `data: {type: supervised, batch_size: 2}`.
   - `core: {type: zmq, address: "tcp://localhost:5555", timeout_s: 60}`.
   - `num_iterations: 10` (prueba de humo).
   > Nota: `configs/pi0_bc.yaml` quedó con `cameras: [primary]` y "7 dims", que **no**
   > es compatible con `pi0_base` (3 cámaras, dims 32). Corregir o reemplazar por este.

4. ✅ **Orquestación de los dos procesos.** `slurm/pi0_bc.sbatch` (clon de
   `train_openvla_bc.sbatch`):
   - `CORE_CONFIG=configs/core/pi0_pusht.yaml`, `LEARNER_CONFIG=configs/pi0_pusht_bc.yaml`;
   - Core en background con `uv run vla-core ...` (**sin extra**: base liviana);
   - `wait_for_port 127.0.0.1 5555`;
   - Learner en foreground con `uv run --extra pi0 vla-learner ...` (GPU);
   - `trap cleanup EXIT` baja el Core al terminar;
   - recursos: GPU grande (π0 ~7 GB bf16 **no** cabe en la RTX 2050; sí en la 2080 Ti/llaima
     al borde), `--nodelist=llaima` como el smoke, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
   - Para correr a mano (dos terminales), mismos comandos.
   - **Desviaciones encontradas al correrlo en el cluster:**
     - **Socket `ipc://` por job, no `tcp://127.0.0.1:5555`.** `llaima` es compartido
       y otro usuario tenía tomado el 5555: el Core no pudo hacer bind y
       `wait_for_port` dio "listo" por el proceso ajeno (job 35205, cancelado). Como
       Core y Learner están en el mismo nodo, el sbatch genera copias de los dos YAML
       con `ipc:///tmp/vla-pi0-bc-<job>.XXXXXX/core.ipc` y espera a que exista el
       socket. Si el Core muere antes, el job termina con error. Los YAML del repo no
       cambian y las rutas relativas se siguen resolviendo contra la raíz del repo.
     - **Venv propio del Core (`.venv-core`).** El lock resuelve numpy 2.5.4 para la
       base y 2.2.6 para el extra `pi0`. Si los dos procesos compartieran `.venv`, el
       `uv run --extra pi0` del Learner reinstalaría numpy en disco con el Core ya
       corriendo. El Core corre con `UV_PROJECT_ENVIRONMENT=.venv-core`. Ese venv
       trae torch, porque la base del workspace instala `vla-learner`, pero el Core
       no lo importa.
     - **`#SBATCH --comment`.** El cluster rechaza los jobs que no lo traen.

5. ✅ **Criterio de éxito.** El Learner conecta al Core, `sample_supervised` devuelve
   pares reales, `imitation_loss` corre y `num_iterations` pasos avanzan imprimiendo
   `fm_mse` y `grad_norm` finitos (como el smoke, pero con datos reales). No se busca
   que converja: se busca que el **cableado de dos procesos con dataset real** funcione.

   **Resultado (job 35206, `COMPLETED`, 2 min 57 s):** los 10 pasos dieron valores
   finitos, con `fm_mse` entre 2465 y 7575 y `grad_norm` entre 533 y 1056 (paso 0:
   `fm_mse=4765.69`, `grad_norm=827.56`; paso 9: `fm_mse=7575.07`,
   `grad_norm=1055.67`). La pérdida oscila, lo esperable con 10 pasos y
   `batch_size: 2`. Antes de gastar GPU se verificó en CPU que el Core real con este
   config, por `ipc://`, entrega al `CoreClient` pares reales (cámara `base_0_rgb
   (3,96,96)`, `proprio (2,)`, chunk de 50 acciones).

## Fuera de alcance de esta prueba

- Que π0 **aprenda** la tarea, en pusht (acciones 2-D padeadas a 32) o en LIBERO: 10
  pasos son solo humo.
- El adaptador `lerobot` en vivo (Opción A) y su fix de import: solo si luego se pide
  servir corpus LeRobot sin convertir.
- Open-X / otros embodiments con 3 cámaras reales.

## Riesgos / puntos a vigilar

- **Largo de chunk no uniforme** → `torch.stack` falla. Mitigado por el troceo a 50
  exactos en el paso 1; si un corpus tiene episodios < 50, esas colas se descartan.
- **Nombre de cámara.** Si el YAML del Learner y el `cameras` del dataset no coinciden
  con `observation.images.base_0_rgb`, o el procesador exige una cámara que la
  `Observation` no trae, `Pi0Processor` lanza `ValueError`.
- **Memoria GPU.** Igual que el smoke: bf16, `batch_size` chico, `freeze: [action]`.
- **Warning de `state_dict`** observado al cargar `pi0_base` ("Could not remap state
  dict keys"): inofensivo. El `embed_tokens` que reporta faltante está atado a
  `lm_head` y queda cargado (job 35263; ver "Puntos abiertos").

## Segundo dataset real: LIBERO

Misma prueba de dos procesos, ahora sobre **LIBERO** (`lerobot/libero`): 1693 demos
de un Panda en 40 tareas de manipulación (suites Spatial, Object, Goal y 10), dos
cámaras de 256×256 (vista general y muñeca), estado de 8 dims y acción de 7. Son las
mismas demos que usa openpi para π0 (`physical-intelligence/libero`, que declara venir
de `openvla/modified_libero_rlds`, la versión de OpenVLA que descartó las demos cuya
reproducción fallaba): son **trayectorias exitosas**. Se usa `lerobot/libero` porque
está en el formato v3 que lee lerobot 0.4.4 y pesa 1,9 GB (vídeo AV1, que PyAV decodifica
con `libdav1d`).

Qué hubo que generalizar (pusht tenía una cámara, instrucción fija, imágenes de 96×96
y 25 mil frames; LIBERO, 2 cámaras, 40 instrucciones, 256×256 y 273 mil frames):

- **Conversor** `scripts/prepare_lerobot_local.py` (antes `prepare_pi0_pusht.py`):
  varias cámaras (`--camera clave_corpus=clave_canonica`), instrucción de cada episodio
  (`frame["task"]`; `--instruction` la fija), imágenes `uint8` comprimidas, un episodio
  en memoria a la vez (antes juntaba el corpus entero en RAM) y `--max-episodes N`, que
  elige N episodios repartidos por todo el corpus y baja y decodifica solo esos.
- **`LocalDataset`** guardaba vistas de los episodios completos en `float32`: con LIBERO
  entero serían ~430 GB de RAM en Core. Ahora guarda solo el frame de cada par (ver
  `Dataset Design.md`).
- **Configs** `configs/core/pi0_libero.yaml` y `configs/pi0_libero_bc.yaml`: vista general
  → `base_0_rgb`, muñeca → `left_wrist_0_rgb` (dos de las tres cámaras de π0).
- **`pi0_bc.sbatch`** toma `CORE_CONFIG`/`LEARNER_CONFIG` del entorno y espera hasta
  `CORE_WAIT_S` (300 s) al Core, que con LIBERO tarda ~70 s en cargar.

Runbook:

```bash
uv run --extra pi0 python -I scripts/prepare_lerobot_local.py \
    --repo-id lerobot/libero --out data/pi0_libero --max-episodes 200 \
    --camera observation.images.image=observation.images.base_0_rgb \
    --camera observation.images.image2=observation.images.left_wrist_0_rgb
CORE_CONFIG=configs/core/pi0_libero.yaml LEARNER_CONFIG=configs/pi0_libero_bc.yaml \
    sbatch slurm/pi0_bc.sbatch
```

**Resultado.** Conversión: 200 episodios (32.980 frames, las 40 tareas) → 557 ventanas de
50 pasos, 4,9 GB, en unos 15 minutos. Core: arranca en ~70 s y ocupa 0,9 GB de RAM.
Estadísticas (`fit`, 256 pares): acción con desvío 0,04–0,44 en posición/rotación y ~1 en
la pinza (±1). Entrenamiento (job 35231, `COMPLETED`, 3 min 25 s): los 10 pasos dieron
`fm_mse` entre 0.11 y 0.28 y `grad_norm` entre 1.0 y 4.1 (paso 0: `fm_mse=0.1228`,
`grad_norm=1.00`). Para LIBERO completo, sacar `--max-episodes` (~5000 ventanas, ~7 GB de
RAM en Core).

## Puntos abiertos (no bloquean la prueba de humo; sí un entrenamiento real)

- ✅ **Acciones sin normalizar (resuelto, verificado en el cluster: job 35229).** En
  el job 35206 `fm_mse` salió del orden de miles porque las acciones y el estado de
  pusht están en píxeles (~70–450) y nada los normalizaba: en lerobot 0.4.4 lo hace el
  pipeline de procesadores, que el adaptador no usa. Ahora normaliza `Pi0Policy`
  (acciones) y `Pi0Processor` (estado), en modo `mean_std`, con estadísticas que el
  builder calcula al arrancar (`normalizer: {stats: fit}` en `configs/pi0_pusht_bc.yaml`).
  Verificado en CPU contra el Core real con pusht: medias ~(234, 293) y desvíos ~(109,
  98) px; las acciones quedan normalizadas con media ~0,2 y desvío ~0,84.
- ✅ **π0 no veía la instrucción (resuelto, verificado en el cluster: job 35229).**
  `Pi0Policy` tomaba el tokenizer de `lerobot_policy.language_tokenizer`, que no existe
  en lerobot 0.4.4, y caía al placeholder de los tests. Ahora `_LeRobotPi0Model` carga
  el tokenizer de PaliGemma (`policy.tokenizer`) y `Pi0Processor` replica la
  tokenización de LeRobot (`"\n"` al final, relleno a la derecha, largo 48).
  **Requisito:** el repo `google/paligemma-3b-pt-224` es de acceso restringido. Hay que
  aceptar la licencia en Hugging Face y autenticarse con el mismo `HF_HOME` que usa el
  sbatch: `HF_HOME=.hf_cache uv run --extra pi0 hf auth login`. Sin eso, el Learner
  falla al arrancar con un error que lo explica.

  **Resultado (job 35229, `COMPLETED`, 3 min 41 s), con normalización y tokenizer:**
  los 10 pasos dieron `fm_mse` entre 0.057 y 0.126 y `grad_norm` entre 1.1 y 3.3
  (paso 0: `fm_mse=0.0610`, `grad_norm=1.14`), contra 2465–7575 y 533–1056 en el job
  35206. Es la escala esperable del flow matching con datos normalizados (el ruido
  tiene varianza 1), y el `grad_norm` ya no queda recortado a la fuerza por
  `grad_clip: 1.0`. La instrucción llega tokenizada como `'<bos>push the T block\n'`
  (48 tokens, relleno a la derecha). El `lr=0.0001` del log es el `5e-5` del config
  redondeado por el formato `.4f` del logger.
- ✅ **Warning de `state_dict` (resuelto, verificado en el cluster: job 35263): es
  inofensivo.** `scripts/check_pi0_embed_tie.py` (`slurm/pi0_check_embed.sbatch`) carga
  `pi0_base` por el mismo camino que el Learner y confirma que `embed_tokens.weight` y
  `lm_head.weight` son el mismo tensor (mismo `data_ptr`, `tie_word_embeddings=True`) y
  que tienen exactamente los valores de `paligemma_with_expert.paligemma.lm_head.weight`
  del checkpoint (desvío 0,184 en ambos). El aviso sale porque `state_dict()` lista los
  dos nombres del tensor atado y el checkpoint guarda solo uno. Análisis previo: el log
  completo dice qué falta: `Missing key(s): model.paligemma_with_expert.paligemma.model.
  language_model.embed_tokens.weight`, la tabla de embeddings del texto. LeRobot carga
  con `strict=True`: PyTorch copia todos los pesos que coinciden antes de fallar, y
  LeRobot atrapa el error y lo imprime como warning, así que lo único sin cargar es esa
  tabla. El checkpoint no la trae, pero sí `paligemma.lm_head.weight` (257152×2048, el
  mismo tamaño). En Gemma ambas matrices están atadas (`tie_word_embeddings`), así que
  el warning es inofensivo **si** el modelo de LeRobot las ata (lo hace: ver arriba).
- **`train_openvla_bc.sbatch` tiene el mismo riesgo de puerto** (`tcp://127.0.0.1:5555`
  fijo en un nodo compartido). No se cambió porque no se volvió a correr; si falla con
  `Address already in use`, se arregla igual que `pi0_bc.sbatch`.

## Próximos pasos (en orden)

La tubería de datos ya está probada con dos corpus reales: lo que falta es que π0
**aprenda** y poder **medirlo**. Cada paso depende del anterior.

1. ✅ **Confirmar el `embed_tokens`** (job 35263, 1 min). Están atadas y cargadas: el
   warning es inofensivo y no hace falta tocar `_LeRobotPi0Model`. Para re-verificar
   otro checkpoint: `MODEL=<ruta o id> sbatch slurm/pi0_check_embed.sbatch`.
2. **Entrenamiento de BC real sobre LIBERO.** Convertir el corpus completo (sin
   `--max-episodes`, ~5000 ventanas), subir `num_iterations` a miles, guardar
   checkpoints periódicos y registrar la curva de `fm_mse`. Éxito: la pérdida baja de
   forma sostenida y el checkpoint se recarga con sus estadísticas
   (`ACTION_STATS_FILE`/`STATE_STATS_FILE`). Conviene medir antes el rendimiento del
   Core con 5000 ventanas: si el muestreo frena al Learner, es la medición que pide
   `Architecture.md` antes de optimizar el transporte.
3. **Evaluación en simulación.** La pérdida no dice si el robot cumple la tarea. Hace
   falta un **Environment real de LIBERO** (proceso aparte con MuJoCo/robosuite que
   implemente `reset`/`step`/`close` y pase la suite ENV; diseño y fases en
   `docs/environment/Environment Design.md`) y correr el checkpoint contra
   las 40 tareas para medir la tasa de éxito. Es el primer componente real fuera de
   Core y el Learner, y abre el camino al RL en ambiente (PPO/GRPO) con π0.
   **Estado:** implementado y probado localmente (rama `Environment`, fases 5 y 6 del
   plan de `docs/environment/`). Falta correr en el cluster, en este orden:
   `sbatch slurm/libero_env_check.sbatch` (el ambiente pasa ENV/SRVE y se parece al
   dataset, suite MAP) y `TASKS="<tarea>|<tarea>" sbatch slurm/pi0_libero_eval.sbatch`
   (tasa de éxito por tarea). Para RL con π0 hay prerrequisitos abiertos: π0 no da
   log-probabilidades (ver "Prerrequisitos para RL sobre LIBERO" del plan).
