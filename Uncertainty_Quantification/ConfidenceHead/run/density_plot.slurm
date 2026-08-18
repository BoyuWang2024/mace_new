#!/bin/bash -l
#SBATCH --job-name=mace_density
#SBATCH --output=./logs/density-%j.out
#SBATCH --error=./logs/density-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00

set -euo pipefail
conda activate mace_new

ROOT=/home/bywang/code/UQ/mace_new
SCRIPT="$ROOT/Uncertainty_Quantification/ConfidenceHead/scripts/plot_density_suite.py"
CONFIG_DIR=${CONFIG_DIR:-$ROOT/Uncertainty_Quantification/ConfidenceHead/configs}
PLOT_ROOT=${PLOT_ROOT:-$ROOT/Uncertainty_Quantification/Plots/ConfidenceHead}

if [[ "${SOURCE:?SOURCE is required}" == "external" ]]; then
  python "$SCRIPT" --source external --config "${CONFIG:?CONFIG is required}"
elif [[ "$SOURCE" == "test" ]]; then
  python "$SCRIPT" --source test --config-dir "$CONFIG_DIR" --plot-root "$PLOT_ROOT" --dataset-name "${DATASET_NAME:-matpes_test}"
else
  echo "unsupported SOURCE=$SOURCE" >&2
  exit 2
fi
