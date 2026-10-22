#!/usr/bin/env bash
# Helpers compartidos por los scripts de Slurm del harness.
#
# Se incluye con `source "$(dirname "$0")/common.sh"` desde cada .sbatch. No es
# ejecutable por sí mismo: solo define funciones y variables.
#
# Responsabilidades:
#   - localizar la raíz del workspace de uv (donde vive pyproject.toml),
#   - dejar `uv` en el PATH (via `module` si el cluster lo usa),
#   - utilidades para esperar a que el Core abra su socket y para limpiar
#     procesos en segundo plano al salir del job.

set -euo pipefail

# --- Raíz del repositorio ----------------------------------------------------
# Por defecto, el directorio desde donde se hizo `sbatch` (lo normal: la raíz
# del repo). Se puede forzar con REPO_ROOT. Se valida que exista pyproject.toml
# para fallar temprano con un mensaje claro en vez de a mitad del entrenamiento.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
    echo "ERROR: no encuentro pyproject.toml en REPO_ROOT='${REPO_ROOT}'." >&2
    echo "       Mandá el job desde la raíz del repo (sbatch slurm/<script>.sbatch)" >&2
    echo "       o exportá REPO_ROOT apuntando a la raíz del workspace." >&2
    exit 1
fi
cd "${REPO_ROOT}"

# --- uv en el PATH -----------------------------------------------------------
# Muchos clusters exponen uv como módulo; si no, se asume que ya está en el
# PATH (p. ej. instalado en ~/.local/bin). Ajustá el nombre del módulo a tu
# sitio si hace falta.
if command -v module >/dev/null 2>&1; then
    module load uv 2>/dev/null || true
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' no está en el PATH. Cargá el módulo correcto o instalá uv." >&2
    exit 1
fi

echo "== Entorno Slurm =="
echo "  job          : ${SLURM_JOB_ID:-<interactivo>}"
echo "  nodo(s)      : ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "  repo         : ${REPO_ROOT}"
echo "  uv           : $(command -v uv) ($(uv --version 2>/dev/null))"
echo "===================="

# --- Utilidades --------------------------------------------------------------

# wait_for_port HOST PORT [TIMEOUT_S]
# Espera a que un socket TCP acepte conexiones (el Core ya está atendiendo).
# Usa el /dev/tcp de bash: no necesita nc ni dependencias extra.
wait_for_port() {
    local host="$1" port="$2" timeout="${3:-120}"
    local waited=0
    echo "Esperando a que ${host}:${port} acepte conexiones (máx ${timeout}s)..."
    until (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null; do
        sleep 1
        waited=$((waited + 1))
        if (( waited >= timeout )); then
            echo "ERROR: ${host}:${port} no respondió tras ${timeout}s." >&2
            return 1
        fi
    done
    exec 3>&- 2>/dev/null || true
    echo "${host}:${port} listo tras ${waited}s."
}
