#!/bin/bash -l
#SBATCH --job-name=mace_C
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err
## 如需指定分区，去掉下一行开头的“##”并把 gpu 改成你的分区名
## #SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10-24:00:00
#SBATCH --gres=gpu:1


conda activate mace_new
#ulimit -n 65535
#export WANDB_BASE_URL="https://api.bandw.top"
python /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/scripts/build_cache.py --config /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_full_energy_only_order3.yaml &&
python /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/scripts/fit_bins.py --config /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_full_energy_only_order3.yaml &&
python /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/scripts/train.py --config /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_full_energy_only_order3.yaml &&
python /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/scripts/check_training.py --config /home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead/configs/mace_matpes_full_energy_only_order3.yaml

