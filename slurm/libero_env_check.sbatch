#!/usr/bin/env bash
# Chequeo del ambiente LIBERO (fase 5 de docs/environment/Environment Implementation Plan.md).
#
# Corre, en este orden, lo que la máquina de desarrollo no puede (MuJoCo + render):
#   1. `vla-env --smoke` con la acción no-op: el ambiente arranca, informa formas,
#      rangos y tiempos, y guarda las observaciones en un npz;
#   2. la suite ENV + SRVE contra LIBERO real (test `heavy` genérico,
#      VLA_ENV_CONFIG=configs/env/libero.yaml), incluido el cambio de tarea;
#   3. la suite MAP: compara esas observaciones con el dataset con el que se
#      entrenó π0 (cámaras, estado, orientación, instrucciones).
# Si algo falla, el job termina con error y el log dice qué.
#
# Requisito para MAP: el dataset LIBERO convertido a `local` (ver la cabecera de
# configs/core/pi0_libero.yaml). Se cambia con DATA_DIR.
#
# Uso (desde la raíz del repo):
#     sbatch slurm/libero_env_check.sbatch
#     DATA_DIR=data/otro STEPS=50 sbatch slurm/libero_env_check.sbatch
#
#SBATCH --job-name=libero-env-check
#SBATCH --comment="VLA-Train-Harness: chequeo del Environment LIBERO (smoke, suite ENV/SRVE, MAP)"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --partition=ialab-high-unlimit   # <-- ajustá a tu sitio
#SBATCH --gres=gpu:1                     # render EGL de MuJoCo; sin GPU usá MUJOCO_GL=osmesa
#SBATCH --qos=debug
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=1:00:00

# Slurm copia este script a un spool temporal, así que "$0" NO apunta al repo.
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

ENV_CONFIG="${ENV_CONFIG:-configs/env/libero.yaml}"
DATA_DIR="${DATA_DIR:-data/pi0_libero}"
STEPS="${STEPS:-20}"
OUT="${OUT:-slurm/logs/libero-env-check-${SLURM_JOB_ID:-local}}"

# --- Entorno del proceso de ambiente ------------------------------------------
# Venv propio (.venv-env), como Core usa .venv-core: el extra `libero` no pisa
# las dependencias de los otros procesos.
export UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv-env"
# Render sin pantalla.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
# Config de LIBERO dentro del repo (gitignoreada). Si falta, el adaptador la
# escribe sola antes del primer import (LIBERO la pediría con input() y colgaría
# el job). Los assets de LIBERO se bajan de HF Hub en el primer uso, a HF_HOME.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.libero}"

echo ">> 1/3 Smoke: ${ENV_CONFIG} (${STEPS} pasos), observaciones en ${OUT}"
uv run --extra libero vla-env --config "${ENV_CONFIG}" --smoke "${STEPS}" --save "${OUT}"

echo ">> 2/3 Suites ENV + SRVE contra el ambiente real"
VLA_ENV_CONFIG="${ENV_CONFIG}" uv run --extra libero pytest -m heavy \
    packages/vla_env/tests/test_env_from_config.py

echo ">> 3/3 Suite MAP contra ${DATA_DIR}"
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: falta el dataset '${DATA_DIR}' para la suite MAP (ver configs/core/pi0_libero.yaml)." >&2
    exit 1
fi
uv run python scripts/check_env_vs_dataset.py --smoke "${OUT}/smoke.npz" --dataset "${DATA_DIR}"

echo ">> Chequeo de LIBERO OK."
