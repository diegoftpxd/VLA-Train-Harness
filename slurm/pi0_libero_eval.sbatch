#!/usr/bin/env bash
# Tasa de éxito de π0 en LIBERO, por Core (fase 6 de
# docs/environment/Environment Implementation Plan.md; paso 3 de "Próximos pasos"
# de docs/dataset/Pi0 E2E Plan.md).
#
# Levanta en el mismo nodo:
#   - N procesos vla-env con LIBERO (venv .venv-env, extra `libero`, render EGL),
#   - Core conectado a ellos (venv .venv-core, base liviana),
#   - scripts/eval_success.py con el checkpoint de π0 (venv del repo, extra `pi0`, GPU),
# todos hablando por sockets ipc:// propios del job (nunca puertos tcp fijos: el
# nodo es compartido, ver job 35205 en pi0_bc.sbatch).
#
# Variables:
#   TASKS        (obligatoria) instrucciones a evaluar, separadas por "|". La lista de
#                tareas la imprime `vla-env --config configs/env/libero.yaml --smoke 1`.
#   CHECKPOINT   step_* a evaluar (por defecto, el último de LEARNER_CONFIG).
#   LEARNER_CONFIG (configs/pi0_libero_bc.yaml), ENV_CONFIG (configs/env/libero.yaml),
#   N_ENVS (2), EPISODES (10 por tarea), SEED (0).
#
# Uso (desde la raíz del repo):
#     TASKS="put the bowl on the plate|open the top drawer" sbatch slurm/pi0_libero_eval.sbatch
#
#SBATCH --job-name=pi0-libero-eval
#SBATCH --comment="VLA-Train-Harness: tasa de exito de pi0 en LIBERO, por Core con ambientes vla-env"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --partition=ialab-high-unlimit   # <-- ajustá a tu sitio
#SBATCH --gres=gpu:1
#SBATCH --qos=debug
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=2:00:00

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

if [[ -z "${TASKS:-}" ]]; then
    echo "ERROR: definí TASKS con las instrucciones a evaluar, separadas por '|'." >&2
    exit 1
fi
IFS='|' read -r -a TASK_LIST <<< "${TASKS}"
LEARNER_CONFIG="${LEARNER_CONFIG:-configs/pi0_libero_bc.yaml}"
ENV_CONFIG="${ENV_CONFIG:-configs/env/libero.yaml}"
N_ENVS="${N_ENVS:-2}"
EPISODES="${EPISODES:-10}"
SEED="${SEED:-0}"
OUT="slurm/logs/${SLURM_JOB_NAME:-pi0-libero-eval}-${SLURM_JOB_ID:-local}.json"
WAIT_S="${WAIT_S:-900}"   # el primer reset de LIBERO baja assets y crea el simulador

JOB_TMP="$(mktemp -d "/tmp/vla-libero-eval-${SLURM_JOB_ID:-local}.XXXXXX")"
PIDS=()
cleanup() {
    echo ">> Bajando procesos: ${PIDS[*]:-ninguno}"
    for pid in "${PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
    wait 2>/dev/null || true
    rm -rf "${JOB_TMP}"
}
trap cleanup EXIT

wait_for_socket() {  # wait_for_socket RUTA PID NOMBRE
    for _ in $(seq "${WAIT_S}"); do
        [[ -S "$1" ]] && return 0
        if ! kill -0 "$2" 2>/dev/null; then
            echo "ERROR: ${3} terminó antes de abrir $1 (ver .err)." >&2
            exit 1
        fi
        sleep 1
    done
    echo "ERROR: ${3} no abrió $1 tras ${WAIT_S}s." >&2
    exit 1
}

# --- Venvs: se sincronizan UNA vez; los procesos arrancan con --no-sync --------
# (N `uv run` en paralelo sobre el mismo venv competirían por instalarlo.)
echo ">> Sincronizando .venv-env (extra libero) y .venv-core"
UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv-env" uv sync --extra libero
UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv-core" uv sync

# --- Procesos de ambiente LIBERO ---------------------------------------------
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.libero}"
export PYTHONUNBUFFERED=1
ENV_IDS=()
CORE_YAML="${JOB_TMP}/core.yaml"
{
    echo "server: { address: \"ipc://${JOB_TMP}/core.ipc\" }"
    echo "backend: { type: fake }"
    echo "timeouts: { environment_s: 60, retries: 1 }"
    echo "environments:"
} > "${CORE_YAML}"
for i in $(seq 0 $((N_ENVS - 1))); do
    socket="${JOB_TMP}/env-${i}.ipc"
    echo ">> Arrancando vla-env libero-${i} en ipc://${socket}"
    UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv-env" uv run --no-sync \
        vla-env --config "${ENV_CONFIG}" --address "ipc://${socket}" &
    PIDS+=($!)
    ENV_IDS+=("libero-${i}")
    echo "  - { id: libero-${i}, address: \"ipc://${socket}\" }" >> "${CORE_YAML}"
done

# --- Core ----------------------------------------------------------------------
echo ">> Arrancando Core con ${N_ENVS} ambientes"
UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv-core" uv run --no-sync vla-core --config "${CORE_YAML}" &
CORE_PID=$!
PIDS+=("${CORE_PID}")

for i in $(seq 0 $((N_ENVS - 1))); do
    wait_for_socket "${JOB_TMP}/env-${i}.ipc" "${PIDS[$i]}" "vla-env libero-${i}"
done
wait_for_socket "${JOB_TMP}/core.ipc" "${CORE_PID}" "Core"
echo ">> Ambientes y Core listos"

# --- Evaluación (GPU, extra pi0) ----------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CHECKPOINT_ARGS=()
[[ -n "${CHECKPOINT:-}" ]] && CHECKPOINT_ARGS=(--checkpoint "${CHECKPOINT}")
echo ">> Evaluando ${#TASK_LIST[@]} tareas x ${EPISODES} episodios; resultado en ${OUT}"
uv run --extra pi0 python scripts/eval_success.py \
    --learner-config "${LEARNER_CONFIG}" "${CHECKPOINT_ARGS[@]}" \
    --core-address "ipc://${JOB_TMP}/core.ipc" \
    --tasks "${TASK_LIST[@]}" --episodes "${EPISODES}" --seed "${SEED}" \
    --env-ids "${ENV_IDS[@]}" --out "${OUT}"

echo ">> Evaluación terminada. El trap baja Core y ambientes."
