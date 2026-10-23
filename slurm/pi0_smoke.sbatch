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
#     BATCH_SIZE=2
# Ejemplo:
#     TRAIN_STEPS=10 FREEZE=action sbatch slurm/pi0_smoke.sbatch
#
# OJO: π0 carga ~7 GB en bf16 (PaliGemma-3B + action expert); NO cabe en la RTX
# 2050 de desarrollo. Con solo el action expert entrenable la memoria de
# gradientes/optimizador es chica, así que una GPU de >=16 GB alcanza. La primera
# corrida DESCARGA los pesos de Hugging Face (~7 GB): exportá HF_HOME a un disco
# con espacio si tu $HOME es chico. Ajustá partición, cuenta, GPU, memoria y
# tiempo a tu sitio.

#SBATCH --job-name=vla-pi0-smoke
#SBATCH --partition=gpu              # <-- ajustá al nombre de tu partición GPU
##SBATCH --account=tu_cuenta         # <-- descomentá y ajustá si tu cluster lo exige
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00              # incluye la descarga de pesos la primera vez
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

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
BATCH_SIZE="${BATCH_SIZE:-2}"

echo ">> π0 smoke: model=${MODEL} freeze=${FREEZE} train_steps=${TRAIN_STEPS} batch=${BATCH_SIZE}"

# `uv run --extra pi0` sincroniza el grupo opcional (lerobot + torch) y corre el
# ejemplo. `--freeze` congela todo salvo los módulos que contengan ese substring;
# `--train-steps` hace unos pocos pasos de BC sobre datos sintéticos.
uv run --extra pi0 python examples/pi0_smoke.py \
    --model "${MODEL}" \
    --freeze "${FREEZE}" \
    --train-steps "${TRAIN_STEPS}" \
    --batch-size "${BATCH_SIZE}"

echo ">> π0 smoke OK."
