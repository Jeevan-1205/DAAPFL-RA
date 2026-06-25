#!/usr/bin/env bash
# Ablation D -> configs/ablation/D_no_reliability.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[ablation D] running D_no_reliability"
python -m training.train_federated --config configs/ablation/D_no_reliability.yaml
python -m experiments.run_lodo     --config configs/ablation/D_no_reliability.yaml --max-folds 3
