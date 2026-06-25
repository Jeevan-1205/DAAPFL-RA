#!/usr/bin/env bash
# Ablation B -> configs/ablation/B_no_multiproto.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[ablation B] running B_no_multiproto"
python -m training.train_federated --config configs/ablation/B_no_multiproto.yaml
python -m experiments.run_lodo     --config configs/ablation/B_no_multiproto.yaml --max-folds 3
