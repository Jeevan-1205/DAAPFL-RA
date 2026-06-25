#!/usr/bin/env bash
# Ablation A -> configs/ablation/A_full.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[ablation A] running A_full"
python -m training.train_federated --config configs/ablation/A_full.yaml
python -m experiments.run_lodo     --config configs/ablation/A_full.yaml --max-folds 3
