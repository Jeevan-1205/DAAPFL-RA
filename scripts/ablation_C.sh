#!/usr/bin/env bash
# Ablation C -> configs/ablation/C_cosine_only.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[ablation C] running C_cosine_only"
python -m training.train_federated --config configs/ablation/C_cosine_only.yaml
python -m experiments.run_lodo     --config configs/ablation/C_cosine_only.yaml --max-folds 3
