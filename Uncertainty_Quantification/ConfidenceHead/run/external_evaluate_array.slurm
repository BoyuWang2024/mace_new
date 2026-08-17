#!/bin/bash -l
#SBATCH --job-name=mace_ext_head
#SBATCH --output=./logs/external-head-%A_%a.out
#SBATCH --error=./logs/external-head-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00

set -euo pipefail
HEAD_KEYS=(force energy_order1 energy_order2 energy_order3 energy_order4 energy_order5 energy_order6 energy_order7 energy_order8)
conda activate mace_new
python /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/scripts/evaluate_external.py --config "$CONFIG" --head-key "${HEAD_KEYS[$SLURM_ARRAY_TASK_ID]}"
