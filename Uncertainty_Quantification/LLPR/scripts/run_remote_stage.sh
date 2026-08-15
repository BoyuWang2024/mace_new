#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 {calibrate|evaluate|validate|plot} CONFIG" >&2
  exit 64
fi

stage="$1"
config="$2"

case "$stage" in
  calibrate|evaluate|validate|plot) ;;
  *)
    echo "unsupported stage: $stage" >&2
    exit 64
    ;;
esac

if [ -n "${LLPR_CONDA_EXE:-}" ]; then
  conda_exe="$LLPR_CONDA_EXE"
else
  conda_exe="$(command -v conda || true)"
  if [ -z "$conda_exe" ]; then
    echo "conda not found; set LLPR_CONDA_EXE to an executable conda path" >&2
    exit 69
  fi
fi

if [ ! -f "$conda_exe" ] || [ ! -x "$conda_exe" ]; then
  echo "LLPR_CONDA_EXE is not executable: $conda_exe" >&2
  exit 69
fi

export LLPR_CONDA_EXE="$conda_exe"
exec "$LLPR_CONDA_EXE" run -n mace_new python -m Uncertainty_Quantification.LLPR.llpr "$stage" --config "$config"
