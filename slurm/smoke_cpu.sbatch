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

source "$(dirname "$0")/common.sh"

mkdir -p slurm/logs

echo ">> 1/2 Contratos fast"
uv run pytest -m fast

echo ">> 2/2 Behavior Cloning tiny (CPU, FakeCoreClient en proceso)"
uv run vla-learner --config configs/bc_tiny.yaml

echo ">> Smoke OK."
