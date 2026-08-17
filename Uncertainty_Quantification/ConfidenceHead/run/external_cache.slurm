#!/bin/bash -l
#SBATCH --job-name=mace_ext_cache
#SBATCH --output=./logs/external-cache-%j.out
#SBATCH --error=./logs/external-cache-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10-24:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
conda activate mace_new
python /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/scripts/build_external_cache.py --config "$CONFIG"
