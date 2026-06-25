#!/usr/bin/env bash
# DAAPFL-RA top-level runner. Examples:
#   bash run.sh prepare                      # build cache from raw xBD
#   bash run.sh central configs/daapfl_ra.yaml
#   bash run.sh fed     configs/daapfl_ra.yaml
#   bash run.sh lodo    configs/daapfl_ra.yaml
#   bash run.sh ablation A
set -euo pipefail
CMD="${1:-help}"; shift || true
case "$CMD" in
  prepare)  python scripts/prepare_data.py "$@" ;;
  central)  python -m training.train_centralized --config "${1:-configs/daapfl_ra.yaml}" ;;
  fed)      python -m training.train_federated   --config "${1:-configs/daapfl_ra.yaml}" ;;
  lodo)     python -m experiments.run_lodo       --config "${1:-configs/daapfl_ra.yaml}" ;;
  cold)     python -m experiments.run_coldstart  --config "${1:-configs/daapfl_ra.yaml}" ;;
  comm)     python -m experiments.run_comm_efficiency --config "${1:-configs/daapfl_ra.yaml}" ;;
  sweep)    python -m experiments.run_hparam_sweep    --config "${1:-configs/daapfl_ra.yaml}" ;;
  ablation) bash "scripts/ablation_${1}.sh" ;;
  *) echo "usage: run.sh {prepare|central|fed|lodo|cold|comm|sweep|ablation}"; exit 1 ;;
esac
