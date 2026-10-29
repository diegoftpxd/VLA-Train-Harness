#!/usr/bin/env bash
# Verifica que el `embed_tokens` de PaliGemma en π0 quede atado a `lm_head` y
# cargado (ver "Puntos abiertos" de docs/dataset/Pi0 E2E Plan.md). Carga el modelo
# con el mismo camino que el Learner y compara contra el checkpoint. No entrena.
#
# Uso:
#     sbatch slurm/pi0_check_embed.sbatch
#     MODEL=checkpoints/... sbatch slurm/pi0_check_embed.sbatch
#
# Éxito: el log termina con ">> OK" y el job en COMPLETED.

#SBATCH --job-name=pi0-check-embed
#SBATCH --comment="VLA-Train-Harness: verificar que embed_tokens de pi0 este atado a lm_head"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20gb
#SBATCH --partition=ialab-high-unlimit
#SBATCH --gres=gpu:1
#SBATCH --qos=debug
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=0:20:00
#SBATCH --nodelist=llaima

# Ver pi0_smoke.sbatch: Slurm copia el script a un spool, "$0" no apunta al repo.
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
export PYTHONUNBUFFERED=1

uv run --extra pi0 python -u scripts/check_pi0_embed_tie.py --model "${MODEL:-lerobot/pi0_base}"
