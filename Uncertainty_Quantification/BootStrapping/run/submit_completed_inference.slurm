#!/usr/bin/env bash
#SBATCH --job-name=mace-bootstrap-uq
#SBATCH --output=mace-bootstrap-uq-%A_%a.out
#SBATCH --error=mace-bootstrap-uq-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10-24:00:00
#SBATCH --array=0-1%1

set -euo pipefail

: "${MACE_BOOTSTRAP_PYTHON:?set MACE_BOOTSTRAP_PYTHON}"
: "${MACE_BOOTSTRAP_CONFIG:?set MACE_BOOTSTRAP_CONFIG}"
: "${MACE_BOOTSTRAP_OUTPUT_ROOT:?set MACE_BOOTSTRAP_OUTPUT_ROOT}"
MACE_BOOTSTRAP_DEVICE="${MACE_BOOTSTRAP_DEVICE:-cpu}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

datasets=(mad_test matpes_train)
dataset="${datasets[${SLURM_ARRAY_TASK_ID}]}"

"${MACE_BOOTSTRAP_PYTHON}" -m +  Uncertainty_Quantification.BootStrapping.scripts.run_completed_inference +  --config "${MACE_BOOTSTRAP_CONFIG}" +  --dataset "${dataset}" +  --output-root "${MACE_BOOTSTRAP_OUTPUT_ROOT}"

"${MACE_BOOTSTRAP_PYTHON}" -m +  Uncertainty_Quantification.BootStrapping.scripts.plot_completed +  --config "${MACE_BOOTSTRAP_CONFIG}" +  --dataset "${dataset}" +  --inference-root "${MACE_BOOTSTRAP_OUTPUT_ROOT}"
