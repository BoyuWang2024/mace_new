#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 {smoke|formal} {mad|matpes-train}" >&2
  exit 64
fi

profile="$1"
dataset="$2"
remote_root="/home/bywang/code/UQ/mace_new-plots"
config_root="$remote_root/Uncertainty_Quantification/LLPR/configs"
wrapper="$remote_root/Uncertainty_Quantification/LLPR/scripts/submit_remote_stage.slurm"
conda_exe="/home/shared/spack/opt/spack/linux-icelake/miniforge3-25.3.0-3-7criefbpaxjacshuyjahvrpo6ppkvsfr/bin/conda"
log_prefix="$remote_root/Uncertainty_Quantification/LLPR/llpr-stage-%x-%j"

case "$profile:$dataset" in
  smoke:mad)
    compute_config="$config_root/gpu_mad_shared_curvature_smoke.yaml"
    plot_config="$config_root/plot_carnet_mad_test_smoke.yaml"
    ;;
  smoke:matpes-train)
    compute_config="$config_root/gpu_matpes_train_shared_curvature_smoke.yaml"
    plot_config="$config_root/plot_carnet_matpes_train_smoke.yaml"
    ;;
  formal:mad)
    compute_config="$config_root/gpu_mad_shared_curvature.yaml"
    plot_config="$config_root/plot_carnet_mad_test.yaml"
    ;;
  formal:matpes-train)
    compute_config="$config_root/gpu_matpes_train_shared_curvature.yaml"
    plot_config="$config_root/plot_carnet_matpes_train.yaml"
    ;;
  *)
    echo "unsupported Task 7 chain: $profile $dataset" >&2
    exit 64
    ;;
esac

if [ "$profile" = "smoke" ]; then
  compute_cpus=4
  compute_mem=16G
  compute_time=00:30:00
else
  compute_cpus=8
  compute_mem=64G
  compute_time=14-00:00:00
fi

print_stage() {
  local stage="$1"
  local config="$2"
  local dependency="$3"
  local cpus="$4"
  local memory="$5"
  local duration="$6"
  local gpu_option="$7"
  local dependency_option=""

  if [ "$dependency" != "none" ]; then
    dependency_option=$(printf ' --dependency=afterok:${%s_job_id}' "$dependency")
  fi

  printf '%s_job_id=$(sbatch --parsable --job-name=llpr-%s-%s-%s --chdir=%s --partition=gpu%s --cpus-per-task=%s --mem=%s --time=%s --output=%s.out --error=%s.err --export=ALL,LLPR_CONDA_EXE=%s%s %s %s %s)\n' \
    "$stage" "$profile" "$dataset" "$stage" "$remote_root" "$gpu_option" \
    "$cpus" "$memory" "$duration" "$log_prefix" "$log_prefix" "$conda_exe" \
    "$dependency_option" "$wrapper" "$stage" "$config"
}

print_stage calibrate "$compute_config" none "$compute_cpus" "$compute_mem" "$compute_time" " --gres=gpu:1"
print_stage evaluate "$compute_config" calibrate "$compute_cpus" "$compute_mem" "$compute_time" " --gres=gpu:1"
print_stage validate "$compute_config" evaluate 4 16G 02:00:00 ""
print_stage plot "$plot_config" validate 4 16G 02:00:00 ""
