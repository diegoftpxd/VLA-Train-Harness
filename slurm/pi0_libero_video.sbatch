#!/usr/bin/env bash
# Prueba de integración: videos de π0 resolviendo tareas de LIBERO, por Core.
#
# Recorre el harness completo con pesos reales: N procesos vla-env de LIBERO
# (grabando, `vla-env --record`) ← Core ← scripts/eval_success.py con π0 en la GPU.
# Usa el π0 ya ajustado a LIBERO por LeRobot (lerobot/pi0_libero_finetuned_v044,
# configs/pi0_libero_finetuned_eval.yaml), así los videos muestran a la política
# resolviendo tareas, no moviéndose al azar.
#
# Pasos:
#   1. exporta las estadísticas de normalización del modelo al formato de
#      Pi0Policy (scripts/export_lerobot_stats.py), si faltan;
#   2. corre slurm/pi0_libero_eval.sbatch con RECORD_DIR (un MP4 por episodio,
#      un cuadro por acción, las dos cámaras lado a lado);
#   3. verifica los videos: uno por episodio, legibles, con más de un cuadro,
#      ninguno incompleto, y tantos `_success` como éxitos cuenta el JSON.
#
# Variables (además de las de pi0_libero_eval.sbatch: N_ENVS, SEED, CHECKPOINT):
#   TASKS     instrucciones separadas por "|" (por defecto, una tarea por suite).
#   EPISODES  episodios por tarea (2).
#   MODEL     id de HF del π0 ajustado (lerobot/pi0_libero_finetuned_v044).
#   STATS_DIR carpeta de sus estadísticas (checkpoints/pi0_libero_finetuned_stats).
#
# Uso (desde la raíz del repo):
#     sbatch slurm/pi0_libero_video.sbatch
#     TASKS="put the bowl on the plate" EPISODES=5 sbatch slurm/pi0_libero_video.sbatch
# Videos en slurm/logs/pi0-libero-video-<JOBID>/libero-<i>/*.mp4
#
#SBATCH --job-name=pi0-libero-video
#SBATCH --comment="VLA-Train-Harness: videos de pi0 en LIBERO (prueba de integracion por Core)"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --partition=ialab-low            # nodo hydra (TITAN RTX); ajustá a tu sitio
#SBATCH --gres=gpu:titan_rtx:1           # NVIDIA: torch es CUDA (antuco tiene AMD MI210: el job 35603 cayó a CPU)
#SBATCH --qos=regular                   # ialab-low no admite debug
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

DEFAULT_TASKS="pick up the black bowl between the plate and the ramekin and place it on the plate"
DEFAULT_TASKS+="|pick up the alphabet soup and place it in the basket"
DEFAULT_TASKS+="|open the middle drawer of the cabinet"
DEFAULT_TASKS+="|put both the alphabet soup and the tomato sauce in the basket"
export TASKS="${TASKS:-${DEFAULT_TASKS}}"
export EPISODES="${EPISODES:-2}"
export LEARNER_CONFIG="${LEARNER_CONFIG:-configs/pi0_libero_finetuned_eval.yaml}"
export ENV_CONFIG="${ENV_CONFIG:-configs/env/libero_pi0_finetuned.yaml}"
MODEL="${MODEL:-lerobot/pi0_libero_finetuned_v044}"
STATS_DIR="${STATS_DIR:-checkpoints/pi0_libero_finetuned_stats}"
NAME="${SLURM_JOB_NAME:-pi0-libero-video}-${SLURM_JOB_ID:-local}"
export RECORD_DIR="${REPO_ROOT}/slurm/logs/${NAME}"
REPORT="slurm/logs/${NAME}.json"   # lo escribe pi0_libero_eval.sbatch

echo ">> 1/3 Estadísticas de ${MODEL} en ${STATS_DIR}"
if [[ -f "${STATS_DIR}/action_stats.pt" && -f "${STATS_DIR}/state_stats.pt" ]]; then
    echo "   ya estaban; no se re-exportan."
else
    uv run --extra pi0 python scripts/export_lerobot_stats.py --source "${MODEL}" --out "${STATS_DIR}"
fi

echo ">> 2/3 Evaluación con grabación: videos en ${RECORD_DIR}"
bash "${REPO_ROOT}/slurm/pi0_libero_eval.sbatch"

echo ">> 3/3 Verificación de los videos"
IFS='|' read -r -a TASK_LIST <<< "${TASKS}"
EXPECTED=$(( ${#TASK_LIST[@]} * EPISODES ))
UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv-env" uv run --no-sync python - \
    "${RECORD_DIR}" "${EXPECTED}" "${REPORT}" <<'EOF'
import json
import sys
from pathlib import Path

import imageio.v3 as iio

directorio, esperados, reporte = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
videos = sorted(directorio.rglob("*.mp4"))
errores = []
for video in videos:
    cuadros = sum(1 for _ in iio.imiter(video))
    print(f"   {video.relative_to(directorio)}: {cuadros} cuadros")
    if cuadros < 2:
        errores.append(f"{video.name}: {cuadros} cuadros")
    if video.name.endswith(("_incomplete.mp4", ".partial.mp4")):
        errores.append(f"{video.name}: episodio sin terminar")
if len(videos) != esperados:
    errores.append(f"{len(videos)} videos, se esperaban {esperados} (uno por episodio)")
exitos_videos = sum(v.name.endswith("_success.mp4") for v in videos)
datos = json.loads(reporte.read_text(encoding="utf-8"))
exitos_json = sum(t["successes"] for t in datos["tasks"].values())
print(f"   éxitos: {exitos_videos} en videos, {exitos_json} en {reporte} (tasa {datos['rate']:.2f})")
if exitos_videos != exitos_json:
    errores.append(f"éxitos distintos: videos {exitos_videos} vs JSON {exitos_json}")
if errores:
    print("ERROR: " + "; ".join(errores), file=sys.stderr)
    sys.exit(1)
EOF

echo ">> Videos OK: ${RECORD_DIR}"
