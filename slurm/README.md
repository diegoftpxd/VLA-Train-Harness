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

### OpenVLA BC en un solo nodo (recomendado)

Core y Learner conviven en el nodo GPU y hablan por `tcp://127.0.0.1:5555`, que
es lo que ya esperan `configs/core/local_openvla.yaml` y `configs/openvla_bc.yaml`
sin tocarlos:

```bash
sbatch slurm/train_openvla_bc.sbatch
```

El script arranca el Core en segundo plano, espera a que atienda, corre el
Learner en primer plano y baja el Core al terminar.

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
