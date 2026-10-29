#!/usr/bin/env bash
# Prueba de entrenamiento MÍNIMO de π0 (pesos reales) con fine-tuning parcial.
#
# Qué hace: carga el π0 completo desde Hugging Face vía LeRobot con el adaptador
# `Pi0Policy`, CONGELA todo salvo una sección pequeña (por defecto el *action
# expert*; PaliGemma queda congelado) y corre unos POCOS pasos de Behavior
# Cloning sobre datos sintéticos, imprimiendo la loss (`fm_mse`) y el `grad_norm`
# paso a paso. Sirve para verificar en el cluster que:
#   - los pesos cargan y caben en la GPU,
#   - `freeze_except` deja entrenable solo la sección pedida,
#   - el bucle de entrenamiento (loss -> backward -> step) avanza.
#
# Es AUTOCONTENIDO: NO levanta Core ni Dataset. Es una prueba de humo del modelo
# y del bucle de entrenamiento, no del harness de 3 procesos. Para el flujo
# supervisado completo (Dataset -> Core -> Learner con demos reales), ver
# `configs/pi0_bc.yaml` y `train_openvla_bc.sbatch` como plantilla.
#
# Usa el grupo opcional de uv `pi0` (arrastra `lerobot` + `torch`); `uv run
# --extra pi0` lo instala en el venv antes de correr. Ver CLAUDE.md: todo con uv.
#
# Uso:
#     sbatch slurm/pi0_smoke.sbatch
#
# Parámetros (opcionales, por variable de entorno):
#     MODEL=lerobot/pi0_base   # id de HF o ruta local a un checkpoint de LeRobot
#     FREEZE=action            # substring de los módulos a dejar entrenables
#     TRAIN_STEPS=5            # cuántos pasos de BC correr
#     BATCH_SIZE=1             # subilo si te sobra VRAM (ver nota de memoria abajo)
# Ejemplo:
#     TRAIN_STEPS=10 FREEZE=action sbatch slurm/pi0_smoke.sbatch
#
# OJO (MEMORIA): π0 carga ~7-8 GB en bf16 (PaliGemma-3B + action expert) SOLO en
# pesos; esos pesos tienen que estar en VRAM para el forward aunque todo esté
# congelado (freeze baja el estado de gradientes/optimizador, NO el backbone). En
# una 2080 Ti de 11 GB entra MUY al borde: por eso el default es BATCH_SIZE=1 y se
# exporta PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (abajo) para evitar que
# la fragmentación tire un OutOfMemory por unos pocos MiB. Si igual no entra, usá
# una GPU de >=24 GB (3090/4090/A5000/A100). NO cabe en la RTX 2050 de desarrollo.
# La primera corrida DESCARGA los pesos de Hugging Face (~7 GB): exportá HF_HOME a
# un disco con espacio si tu $HOME es chico. Ajustá partición, cuenta, GPU, memoria
# y tiempo a tu sitio.

#SBATCH --job-name=pi0_smoke
#SBATCH --mail-type=END,FAIL
#SBATCH --comment="Prueba de entrenamiento supervisado"
#SBATCH --mail-user=diego.toledo@uc.cl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20gb
#SBATCH --partition=ialab-high-unlimit
#SBATCH --gres=gpu:1
#SBATCH --qos=debug
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=1:00:00
#SBATCH --nodelist=llaima

# Slurm copia este script a un spool temporal, así que "$0" NO apunta al repo.
# common.sh vive en <repo>/slurm; se localiza vía REPO_ROOT o el directorio desde
# donde se mandó el job (la raíz del repo, ver README), con el dir de "$0" como
# último recurso (corrida a mano, no bajo sbatch).
_COMMON=""
for _d in "${REPO_ROOT:-}/slurm" "${SLURM_SUBMIT_DIR:-}/slurm" "$(dirname "$0")"; do
    if [[ -f "${_d}/common.sh" ]]; then _COMMON="${_d}/common.sh"; break; fi
done
if [[ -z "${_COMMON}" ]]; then
    echo "ERROR: no encuentro slurm/common.sh (mandá el job desde la raíz del repo o exportá REPO_ROOT)." >&2
    exit 1
fi
source "${_COMMON}"

mkdir -p slurm/logs

MODEL="${MODEL:-lerobot/pi0_base}"
FREEZE="${FREEZE:-action}"
TRAIN_STEPS="${TRAIN_STEPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# expandable_segments evita que la fragmentación del allocator de PyTorch tire un
# CUDA OutOfMemory por unos pocos MiB cuando el modelo llena casi toda la VRAM
# (caso típico de π0 en una 2080 Ti de 11 GB). Respetá el valor si ya viene seteado.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo ">> π0 smoke: model=${MODEL} freeze=${FREEZE} train_steps=${TRAIN_STEPS} batch=${BATCH_SIZE}"

# `uv run --extra pi0` sincroniza el grupo opcional (lerobot + torch) y corre el
# ejemplo. `--freeze` congela todo salvo los módulos que contengan ese substring;
# `--train-steps` hace unos pocos pasos de BC sobre datos sintéticos.
# Salida COMPLETA (stdout+stderr) a un archivo propio, además de la de Slurm:
# `python -u` desactiva el buffering (no se pierde el traceback si falla) y el
# `tee` deja una copia cruda que el epílogo del cluster no toca. `set -o pipefail`
# (de common.sh) hace que el fallo de uv/python se propague pese al `tee`.
export PYTHONUNBUFFERED=1
DEBUG_LOG="slurm/logs/${SLURM_JOB_NAME:-pi0_smoke}-${SLURM_JOB_ID:-local}.debug.log"
echo ">> Volcando salida completa a ${DEBUG_LOG}"

uv run --extra pi0 python -u examples/pi0_smoke.py \
    --model "${MODEL}" \
    --freeze "${FREEZE}" \
    --train-steps "${TRAIN_STEPS}" \
    --batch-size "${BATCH_SIZE}" 2>&1 | tee "${DEBUG_LOG}"

echo ">> π0 smoke OK."
