#!/usr/bin/env bash
# Core (CPU, ZeroMQ) como job independiente, para repartir Core y Learner en
# nodos distintos: Core en una partición CPU, Learner en una GPU. Es la variante
# "separada" de train_openvla_bc.sbatch (que los pone en el mismo nodo).
#
# Core NO tiene GPU ni modelo: solo enruta mensajes y arrays de numpy (ver
# CLAUDE.md). Acá se bindea a 0.0.0.0 para que el nodo del Learner lo alcance por
# tcp, y publica su host:puerto en un archivo de rendezvous en el filesystem
# compartido para que learner.sbatch sepa a dónde conectarse.
#
# No se lanza suelto: lo encadena learner.sbatch. Ver slurm/README.md.
#
# Ajustá partición, cuenta, memoria y tiempo a tu sitio.

#SBATCH --job-name=vla-core
#SBATCH --partition=cpu              # <-- ajustá al nombre de tu partición CPU
##SBATCH --account=tu_cuenta         # <-- descomentá y ajustá si tu cluster lo exige
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

mkdir -p slurm/logs slurm/rendezvous

CORE_PORT="${CORE_PORT:-5555}"
RDV_FILE="slurm/rendezvous/${SLURM_JOB_ID}.addr"

# Config de Core bindeando a todas las interfaces (0.0.0.0) en vez de localhost,
# para que sea alcanzable desde el nodo del Learner. Se genera una copia parchada
# del config de referencia; el original no se toca.
BASE_CORE_CONFIG="configs/core/local_openvla.yaml"
GEN_CORE_CONFIG="slurm/rendezvous/${SLURM_JOB_ID}.core.yaml"
sed "s#tcp://127.0.0.1:${CORE_PORT}#tcp://0.0.0.0:${CORE_PORT}#" \
    "${BASE_CORE_CONFIG}" > "${GEN_CORE_CONFIG}"

# Publicar host:puerto para el Learner y limpiar los archivos al salir.
echo "$(hostname):${CORE_PORT}" > "${RDV_FILE}"
echo ">> Rendezvous publicado en ${RDV_FILE}: $(cat "${RDV_FILE}")"

cleanup() {
    rm -f "${RDV_FILE}" "${GEN_CORE_CONFIG}"
}
trap cleanup EXIT

echo ">> Arrancando Core: ${GEN_CORE_CONFIG}"
exec uv run vla-core --config "${GEN_CORE_CONFIG}"
