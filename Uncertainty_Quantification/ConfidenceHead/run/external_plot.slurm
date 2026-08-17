#!/bin/bash -l
#SBATCH --job-name=mace_ext_plot
#SBATCH --output=./logs/external-plot-%j.out
#SBATCH --error=./logs/external-plot-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00

set -euo pipefail
conda activate mace_new
if [[ "$SOURCE" == "external" ]]; then
  python /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/scripts/plot_dataset_suite.py --source external --config "$CONFIG"
else
  python /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/scripts/plot_dataset_suite.py --source test --config-dir /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/configs --plot-root /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/Plots/ConfidenceHead
fi
