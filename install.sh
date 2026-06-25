#!/usr/bin/env bash
# DAAPFL-RA installer (Linux / WSL2). Usage: bash install.sh [cpu|cu121]
set -euo pipefail
TARGET="${1:-cu121}"
INDEX="https://download.pytorch.org/whl/${TARGET}"
echo "[install] python: $(python --version)"
python -m pip install --upgrade pip setuptools wheel
echo "[install] installing torch stack from ${INDEX}"
python -m pip install -r requirements.txt --extra-index-url "${INDEX}"
echo "[install] verifying torch / cuda"
python - << 'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
echo "[install] done."
