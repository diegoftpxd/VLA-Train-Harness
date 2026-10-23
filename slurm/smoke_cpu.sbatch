#!/usr/bin/env bash
# Smoke test en CPU: valida que el entorno uv y el harness corren en un nodo de
# cómputo, sin GPU ni pesos reales. Útil para probar el cluster antes de pedir
# una GPU grande para train_openvla_bc.sbatch.
#
# Corre dos cosas rápidas y deterministas:
#   1. la suite de contratos `fast` (incluye los contratos de Core con fakes),
#   2. un entrenamiento tiny de Behavior Cloning (configs/bc_tiny.yaml), que usa
#      el FakeCoreClient en proceso (no levanta servidor): política compuesta
#      tiny de 2 dims contra demos fake, todo en CPU.
#
# Uso:
#     sbatch slurm/smoke_cpu.sbatch
#
# Ajustá partición, cuenta, memoria y tiempo a tu sitio.

#SBATCH --job-name=vla-smoke-cpu
#SBATCH --partition=cpu              # <-- ajustá al nombre de tu partición CPU
##SBATCH --account=tu_cuenta         # <-- descomentá y ajustá si tu cluster lo exige
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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

echo ">> 1/2 Contratos fast"
uv run pytest -m fast

echo ">> 2/2 Behavior Cloning tiny (CPU, FakeCoreClient en proceso)"
uv run vla-learner --config configs/bc_tiny.yaml

echo ">> Smoke OK."
