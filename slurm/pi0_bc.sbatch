#!/usr/bin/env bash
# Prueba E2E de DOS PROCESOS de π0 (BC) sobre un dataset real (pusht) en un nodo.
#
# Levanta el Core (CPU, ZeroMQ, base liviana SIN torch/lerobot) y el Learner π0
# (GPU) en el MISMO nodo, comunicados por un socket ipc:// propio del job (el
# script reescribe la dirección de los YAML; ver abajo), con los configs:
#     configs/core/pi0_pusht.yaml   (Core: dataset `local` = pusht convertido)
#     configs/pi0_pusht_bc.yaml     (Learner: pi0 BC)
#
# Es el paso 4 de docs/dataset/Pi0 E2E Plan.md. A diferencia de pi0_smoke.sbatch
# (un proceso, datos sintéticos), acá los datos salen de un corpus real a través
# del harness de verdad (Dataset -> Core -> Learner).
#
# ANTES de correr: convertí el corpus una sola vez (offline, necesita el extra
# `pi0` = lerobot 0.4.4) para poblar data/pi0_pusht/ en formato `local`:
#     uv run --extra pi0 python -I scripts/prepare_pi0_pusht.py
#
# Uso:
#     sbatch slurm/pi0_bc.sbatch
#
# OJO (MEMORIA): π0 carga ~7-8 GB en bf16 (PaliGemma-3B + action expert) solo en
# pesos; aunque se entrene solo el action expert, el backbone vive en VRAM. NO
# cabe en la RTX 2050 de desarrollo; en una 2080 Ti de 11 GB entra al borde (por
# eso batch_size=2 en el YAML y expandable_segments abajo). Ajustá partición,
# cuenta, GPU, memoria y tiempo a tu sitio.

#SBATCH --job-name=pi0-bc
#SBATCH --comment="VLA-Train-Harness: prueba E2E de dos procesos (Core + Learner pi0 BC) sobre el dataset real pusht"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.toledo@uc.cl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # Core (CPU) + dataloaders del Learner
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

CORE_CONFIG="configs/core/pi0_pusht.yaml"
LEARNER_CONFIG="configs/pi0_pusht_bc.yaml"

# Core y Learner corren en el MISMO nodo, así que se hablan por ipc:// (como pide
# Architecture.md) en un socket propio del job, NO por el tcp://...:5555 fijo de
# los YAML: el nodo es compartido y ese puerto puede estar tomado por otro
# usuario (pasó en el job 35205: el Core no pudo hacer bind y el chequeo de
# puerto dio "listo" por el proceso ajeno). Se generan copias de los YAML con la
# dirección del job; las rutas relativas siguen resolviéndose contra la raíz del
# repo (el cwd), no contra el directorio del YAML.
JOB_TMP="$(mktemp -d "/tmp/vla-pi0-bc-${SLURM_JOB_ID:-local}.XXXXXX")"
CORE_SOCKET="${JOB_TMP}/core.ipc"
CORE_ADDRESS="ipc://${CORE_SOCKET}"
sed -E "s#address: \"[^\"]+\"#address: \"${CORE_ADDRESS}\"#" "${CORE_CONFIG}" > "${JOB_TMP}/core.yaml"
sed -E "s#address: \"[^\"]+\"#address: \"${CORE_ADDRESS}\"#" "${LEARNER_CONFIG}" > "${JOB_TMP}/learner.yaml"
CORE_CONFIG="${JOB_TMP}/core.yaml"
LEARNER_CONFIG="${JOB_TMP}/learner.yaml"
echo ">> Core y Learner por ${CORE_ADDRESS}"

# expandable_segments evita que la fragmentación del allocator de PyTorch tire un
# CUDA OutOfMemory por unos pocos MiB cuando π0 llena casi toda la VRAM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

# El dataset `local` tiene que estar preparado antes (ver cabecera).
if [[ ! -d "data/pi0_pusht" ]]; then
    echo "ERROR: falta data/pi0_pusht. Corré primero:" >&2
    echo "    uv run --extra pi0 python -I scripts/prepare_pi0_pusht.py" >&2
    exit 1
fi

# --- Core en segundo plano (CPU, base liviana: SIN --extra) -----------------
# El Core usa su PROPIO venv (.venv-core). El lock resuelve numpy distinto para la
# base y para el extra `pi0`, así que si ambos procesos compartieran .venv, el
# `uv run --extra pi0` del Learner reinstalaría numpy en disco con el Core ya
# corriendo. Con venvs separados cada proceso queda con sus dependencias fijas.
echo ">> Arrancando Core: ${CORE_CONFIG}"
UV_PROJECT_ENVIRONMENT=.venv-core uv run vla-core --config "${CORE_CONFIG}" &
CORE_PID=$!

# Al terminar (éxito, error o cancelación), bajar el Core.
cleanup() {
    echo ">> Bajando Core (PID ${CORE_PID})..."
    kill "${CORE_PID}" 2>/dev/null || true
    wait "${CORE_PID}" 2>/dev/null || true
    rm -rf "${JOB_TMP}"
}
trap cleanup EXIT

# Esperar a que el Core cree su socket ipc antes de lanzar el Learner. Si el Core
# muere mientras tanto (bind fallido, config inválida), se corta acá con error en
# vez de lanzar el Learner contra nada.
echo "Esperando a que el Core abra ${CORE_SOCKET} (máx 120s)..."
for _ in $(seq 120); do
    [[ -S "${CORE_SOCKET}" ]] && break
    if ! kill -0 "${CORE_PID}" 2>/dev/null; then
        echo "ERROR: el Core terminó antes de abrir su socket (ver .err)." >&2
        exit 1
    fi
    sleep 1
done
if [[ ! -S "${CORE_SOCKET}" ]]; then
    echo "ERROR: el Core no abrió ${CORE_SOCKET} tras 120s." >&2
    exit 1
fi
echo ">> Core listo en ${CORE_ADDRESS}"

# --- Learner π0 en primer plano (GPU, extra `pi0`) --------------------------
echo ">> Arrancando Learner π0: ${LEARNER_CONFIG}"
DEBUG_LOG="slurm/logs/${SLURM_JOB_NAME:-pi0-bc}-${SLURM_JOB_ID:-local}.debug.log"
echo ">> Volcando salida completa a ${DEBUG_LOG}"
uv run --extra pi0 vla-learner --config "${LEARNER_CONFIG}" 2>&1 | tee "${DEBUG_LOG}"

echo ">> Learner terminó. El trap baja el Core."
