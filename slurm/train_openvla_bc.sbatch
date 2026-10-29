#!/usr/bin/env bash
# Fine-tuning supervisado (Behavior Cloning) de OpenVLA-7B en un solo nodo con GPU.
#
# Levanta el Core (CPU, ZeroMQ) y el Learner (GPU) en el MISMO nodo y los
# comunica por tcp://127.0.0.1:5555, que es justo lo que esperan los configs sin
# tocarlos:
#     configs/core/local_openvla.yaml   (Core: dataset local compatible con OpenVLA)
#     configs/openvla_bc.yaml           (Learner: core.address = tcp://localhost:5555)
#
# Es la opción recomendada: Core y Learner conviven en el nodo, Core usa los
# núcleos de CPU y el Learner la GPU. Para repartirlos en nodos distintos (Core
# en una partición CPU), ver core.sbatch + learner.sbatch.
#
# Uso:
#     sbatch slurm/train_openvla_bc.sbatch
#
# OJO: OpenVLA-7B carga ~16 GB en bf16 aunque se entrene solo `lm_head`. Necesita
# una GPU grande (A100 40 GB o similar); NO corre en la RTX 2050 de desarrollo.
# Ajustá partición, cuenta, GPU, memoria y tiempo a tu sitio.

#SBATCH --job-name=vla-openvla-bc
#SBATCH --partition=gpu              # <-- ajustá al nombre de tu partición GPU
##SBATCH --account=tu_cuenta         # <-- descomentá y ajustá si tu cluster lo exige
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # Core (CPU) + dataloaders del Learner
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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

CORE_CONFIG="configs/core/local_openvla.yaml"
LEARNER_CONFIG="configs/openvla_bc.yaml"
CORE_HOST="127.0.0.1"
CORE_PORT="5555"

# --- Core en segundo plano (CPU) --------------------------------------------
echo ">> Arrancando Core: ${CORE_CONFIG}"
uv run vla-core --config "${CORE_CONFIG}" &
CORE_PID=$!

# Al terminar (éxito, error o cancelación), bajar el Core.
cleanup() {
    echo ">> Bajando Core (PID ${CORE_PID})..."
    kill "${CORE_PID}" 2>/dev/null || true
    wait "${CORE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Esperar a que el Core esté atendiendo antes de lanzar el Learner.
wait_for_port "${CORE_HOST}" "${CORE_PORT}"

# --- Learner en primer plano (GPU) ------------------------------------------
echo ">> Arrancando Learner: ${LEARNER_CONFIG}"
uv run vla-learner --config "${LEARNER_CONFIG}"

echo ">> Learner terminó. El trap baja el Core."
