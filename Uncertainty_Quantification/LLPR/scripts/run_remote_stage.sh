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

exec conda run -n mace_new python -m Uncertainty_Quantification.LLPR.llpr "$stage" --config "$config"
