#!/bin/zsh
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIRECTORY="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd)"
cd "${PROJECT_DIRECTORY}"

SMOKE_OUTPUT="${PROJECT_DIRECTORY}/results/smoke"
MAIN_OUTPUT="${PROJECT_DIRECTORY}/results/main"
ANALYSIS_OUTPUT="${MAIN_OUTPUT}/summary"

if [[ -e "${SMOKE_OUTPUT}" || -e "${MAIN_OUTPUT}" ]]; then
  echo "Refusing to overwrite existing results/smoke or results/main."
  echo "Move the existing result directory elsewhere before running again."
  exit 2
fi

export MPLCONFIGDIR="${PROJECT_DIRECTORY}/results/.mplconfig"
export XDG_CACHE_HOME="${PROJECT_DIRECTORY}/results/.cache"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p "${XDG_CACHE_HOME}"

echo "[1/3] Running the four-model MPS smoke test"
conda run -n chem_ai python src/run_experiment.py \
  --config configs/main.json \
  --output-dir "${SMOKE_OUTPUT}" \
  --smoke-test \
  --device mps

echo "[2/3] Running all 140 formal MPS models"
conda run -n chem_ai python src/run_experiment.py \
  --config configs/main.json \
  --output-dir "${MAIN_OUTPUT}" \
  --device mps

echo "[3/3] Aggregating tables and figures"
conda run -n chem_ai python src/analyze_results.py \
  --input-dir "${MAIN_OUTPUT}" \
  --output-dir "${ANALYSIS_OUTPUT}"

echo "Complete: ${ANALYSIS_OUTPUT}/report.md"
