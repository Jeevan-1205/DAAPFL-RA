#!/usr/bin/env bash
# Ablation E -> configs/ablation/E_no_attention.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[ablation E] running E_no_attention"
python -m training.train_federated --config configs/ablation/E_no_attention.yaml
python -m experiments.run_lodo     --config configs/ablation/E_no_attention.yaml --max-folds 3
