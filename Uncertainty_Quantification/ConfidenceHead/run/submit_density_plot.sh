#!/bin/bash
set -euo pipefail

ROOT=/home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead
PLOT_ROOT=/home/bywang/code/UQ/mace_new/Uncertainty_Quantification/Plots/ConfidenceHead
mkdir -p "$ROOT/run/logs"
cd "$ROOT/run"

for dataset in mad_test matpes_train; do
  config="$ROOT/configs/external_inference/${dataset}.yaml"
  job=$(sbatch --parsable --export=ALL,SOURCE=external,CONFIG="$config",PLOT_ROOT="$PLOT_ROOT" density_plot.slurm)
  echo "$dataset job=$job output=$PLOT_ROOT/density/$dataset"
done

job=$(sbatch --parsable --export=ALL,SOURCE=test,CONFIG_DIR="$ROOT/configs",PLOT_ROOT="$PLOT_ROOT",DATASET_NAME=matpes_test density_plot.slurm)
echo "matpes_test job=$job output=$PLOT_ROOT/density/matpes_test"
