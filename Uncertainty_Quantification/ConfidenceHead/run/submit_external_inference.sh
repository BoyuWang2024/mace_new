#!/bin/bash
set -euo pipefail

ROOT=/home/bywang/code/UQ/mace_new/Uncertainty_Quantification/ConfidenceHead
mkdir -p "$ROOT/run/logs"
cd "$ROOT/run"

submit_external() {
  local dataset="$1"
  local config="$ROOT/configs/external_inference/${dataset}.yaml"
  local cache_job
  local head_job
  cache_job=$(sbatch --parsable --export=ALL,CONFIG="$config" external_cache.slurm)
  head_job=$(sbatch --parsable --dependency=afterok:"$cache_job" --array=0-8%3 --export=ALL,CONFIG="$config" external_evaluate_array.slurm)
  sbatch --parsable --dependency=afterok:"$head_job" --export=ALL,SOURCE=external,CONFIG="$config" external_plot.slurm
}

case "${1:-all}" in
  mad_test) submit_external mad_test ;;
  matpes_train) submit_external matpes_train ;;
  all)
    submit_external mad_test
    submit_external matpes_train
    sbatch --parsable --export=ALL,SOURCE=test,CONFIG=none external_plot.slurm
    ;;
  *) echo "usage: $0 {mad_test|matpes_train|all}" >&2; exit 2 ;;
esac
