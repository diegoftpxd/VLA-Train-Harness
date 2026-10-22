#!/usr/bin/env bash
# Learner (GPU) como job independiente, emparejado con core.sbatch (CPU) para
# correr Core y Learner en nodos distintos.
#
# Uso (ver slurm/README.md para el detalle):
#     CORE_JID=$(sbatch --parsable slurm/core.sbatch)
#     sbatch --dependency=after:${CORE_JID} slurm/learner.sbatch ${CORE_JID}
#
# El Learner recibe el JOBID del Core como argumento, lee su host:puerto del
# archivo de rendezvous que el Core publicó, genera un config apuntando a esa
# dirección y entrena. Al terminar, cancela el job del Core.
#
# OJO: carga pesos reales de OpenVLA-7B (~16 GB en bf16). Necesita GPU grande.
# Ajustá partición, cuenta, GPU, memoria y tiempo a tu sitio.

#SBATCH --job-name=vla-learner
#SBATCH --partition=gpu              # <-- ajustá al nombre de tu partición GPU
##SBATCH --account=tu_cuenta         # <-- descomentá y ajustá si tu cluster lo exige
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

source "$(dirname "$0")/common.sh"

mkdir -p slurm/logs

CORE_JID="${1:-}"
if [[ -z "${CORE_JID}" ]]; then
    echo "ERROR: falta el JOBID del Core como argumento." >&2
    echo "       sbatch --dependency=after:<CORE_JID> slurm/learner.sbatch <CORE_JID>" >&2
    exit 1
fi

RDV_FILE="slurm/rendezvous/${CORE_JID}.addr"

# Esperar a que el Core publique su dirección (puede tardar en arrancar).
echo ">> Esperando rendezvous del Core (${RDV_FILE})..."
waited=0
until [[ -s "${RDV_FILE}" ]]; do
    sleep 2
    waited=$((waited + 2))
    if (( waited >= 300 )); then
        echo "ERROR: el Core no publicó ${RDV_FILE} tras 300s." >&2
        exit 1
    fi
done
CORE_ADDR="$(cat "${RDV_FILE}")"              # host:puerto
CORE_HOST="${CORE_ADDR%%:*}"
CORE_PORT="${CORE_ADDR##*:}"
echo ">> Core en ${CORE_ADDR}"

GEN_LEARNER_CONFIG="slurm/rendezvous/${CORE_JID}.learner.yaml"

# Cancelar el Core y borrar el config generado cuando el Learner termine (fin
# normal, error o cancelación).
cleanup() {
    echo ">> Cancelando Core (job ${CORE_JID})..."
    scancel "${CORE_JID}" 2>/dev/null || true
    rm -f "${GEN_LEARNER_CONFIG}"
}
trap cleanup EXIT

wait_for_port "${CORE_HOST}" "${CORE_PORT}"

# Config del Learner apuntando a la dirección real del Core (el de referencia
# usa tcp://localhost). Se parcha una copia; el original no se toca.
BASE_LEARNER_CONFIG="configs/openvla_bc.yaml"
sed "s#tcp://localhost:5555#tcp://${CORE_HOST}:${CORE_PORT}#" \
    "${BASE_LEARNER_CONFIG}" > "${GEN_LEARNER_CONFIG}"

echo ">> Arrancando Learner: ${GEN_LEARNER_CONFIG}"
uv run vla-learner --config "${GEN_LEARNER_CONFIG}"

echo ">> Learner terminó. El trap cancela el Core."
