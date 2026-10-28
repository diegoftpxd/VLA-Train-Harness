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
   pueden tener menos dims (p. ej. 2); π0 padea solo. El adaptador **no** normaliza.
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
código, y un commit en la rama `Dataset`.

1. **Script de preparación (offline, una vez).** `scripts/prepare_pi0_pusht.py`:
   - carga pusht con `from lerobot.datasets.lerobot_dataset import LeRobotDataset`
     (**path correcto de 0.4.4**, no el del adaptador con bug);
   - recorre cada episodio y lo corta en ventanas **de 50 pasos exactos** (descarta la
     cola < 50, para que todas las chunks midan 50 — restricción 3);
   - escribe cada ventana como un episodio en formato `local`:
     `data/pi0_pusht/<idx>/data.npz` con `observation.images.base_0_rgb` `(50,C,H,W)`,
     `proprio` `(50,2)`, `actions` `(50,2)`; y `meta.json` con `instruction`;
   - se corre con `uv run --extra lerobot python -I scripts/prepare_pi0_pusht.py`.
   - **Dato no confiable:** el corpus descargado va a su propio directorio y solo se
     lee por ruta (regla de `Dataset Design.md`).
   - Validación en `fast`: un test que, sobre un fixture diminuto ya en el repo,
     chequee el troceo a 50 y las claves/forma de salida (sin descargar nada).

2. **Config de Core.** `configs/core/pi0_pusht.yaml`:
   ```yaml
   server: { address: "tcp://127.0.0.1:5555" }
   backend: { type: fake }          # Environment/WM/RM no se tocan en supervisado
   dataset:
     type: local
     path: data/pi0_pusht
     cameras: { observation.images.base_0_rgb: observation.images.base_0_rgb }
     demo_chunk_size: 50            # = n_action_steps de π0 (restricción 3)
   ```
   Validarla en `test_configs.py` (`fast`).

3. **Config de Learner.** `configs/pi0_pusht_bc.yaml` (clon de `pi0_bc.yaml` corregido):
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

4. **Orquestación de los dos procesos.** `slurm/pi0_bc.sbatch` (clon de
   `train_openvla_bc.sbatch`):
   - `CORE_CONFIG=configs/core/pi0_pusht.yaml`, `LEARNER_CONFIG=configs/pi0_pusht_bc.yaml`;
   - Core en background con `uv run vla-core ...` (**sin extra**: base liviana);
   - `wait_for_port 127.0.0.1 5555`;
   - Learner en foreground con `uv run --extra pi0 vla-learner ...` (GPU);
   - `trap cleanup EXIT` baja el Core al terminar;
   - recursos: GPU grande (π0 ~7 GB bf16 **no** cabe en la RTX 2050; sí en la 2080 Ti/llaima
     al borde), `--nodelist=llaima` como el smoke, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
   - Para correr a mano (dos terminales), mismos comandos.

5. **Criterio de éxito.** El Learner conecta al Core, `sample_supervised` devuelve pares
   reales, `imitation_loss` corre y `num_iterations` pasos avanzan imprimiendo `fm_mse`
   y `grad_norm` finitos (como el smoke, pero con datos reales). No se busca que
   converja: se busca que el **cableado de dos procesos con dataset real** funcione.

## Fuera de alcance de esta prueba

- Que π0 **aprenda** la tarea de pusht (acciones 2-D padeadas a 32; es solo humo).
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
  dict keys"): no es fatal (el modelo carga y entrena), pero conviene auditarlo aparte
  para descartar pesos sin mapear.
