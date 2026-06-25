# DAAPFL-RA
### Disaster-Aware Adaptive Personalized Federated Learning with Reliability-Aware Aggregation for Building Damage Assessment

A complete, runnable research codebase for **reliability-aware, personalized federated learning** on the **xBD / xView2** building-damage dataset. Each client is a *disaster event*; the global model is personalized per client through a six-component aggregation scheme that combines multi-prototype similarity, attention, and data-reliability weighting.

> **Status:** Full implementation across all 11 phases (data → preprocessing → model → losses → baselines → DAAPFL-RA aggregation → training → evaluation → ablations → experiments → visualization). Pinned, reproducible environment.

---

## 1. Key Idea

Standard FedAvg averages client updates uniformly. That is a poor fit for disaster response, where events differ wildly in scale, building density, and damage distribution. **DAAPFL-RA** replaces uniform averaging with a *per-client personalized* aggregation:

For target client *i*, the weight assigned to source client *j* is

```
W_ij = λ · A_ij + (1 − λ) · R_j
```

where

- **A_ij — attention weight**: `softmax_j( S_ij / τ )`, a row-stochastic attention over a three-term client-similarity matrix **S**.
- **R_j — reliability weight**: `(N_j · F1_j) / Σ_k (N_k · F1_k)`, rewarding clients with more data *and* higher validation damage-F1.
- **λ ∈ [0,1]** balances personalization (attention) against reliability.

The similarity **S** is itself a convex blend of three signals between client **multi-prototypes** (K cluster centroids of encoder bottleneck features):

```
S = α · cosine + β · (1 / (1 + L2)) + γ · (−KL)
```

Only the **shared encoder** is exchanged with the server. Each client keeps its **decoder, segmentation head, and prototypes private** — this is what makes the scheme *personalized* federated learning.

### The six components (mapped to ablations)

| # | Component | File | Ablation that removes it |
|---|-----------|------|--------------------------|
| 1 | Multi-prototype client signature (K=3) | `models/prototype/prototype.py` | **B** (K=1) |
| 2 | Cosine similarity term (α) | `models/aggregation/similarity.py` | **C** keeps only this |
| 3 | Inverse-L2 similarity term (β) | `models/aggregation/similarity.py` | **C** (β=0) |
| 4 | Negative-KL similarity term (γ) | `models/aggregation/similarity.py` | **C** (γ=0) |
| 5 | Attention over S (τ, λ) | `models/aggregation/attention.py` | **E** (λ=0 → reliability only) |
| 6 | Reliability weighting (R) | `models/aggregation/reliability.py` | **D** (λ=1 → attention only) |

---

## 2. Project Structure

```
DAAPFL-RA/
├── configs/                 # YAML configs with _base_ inheritance
│   ├── base.yaml            # composes data/model/federated + loss/train
│   ├── data.yaml model.yaml federated.yaml daapfl_ra.yaml
│   ├── baselines/           # fedavg, fedprox, scaffold, fedper, fedrep,
│   │                        #   ditto, pfedme, fedala
│   └── ablation/            # A_full … E_no_attention
├── datasets/                # xBD scan, event split, LODO, client partition
├── preprocessing/           # rasterize polygons, tiling, augment, cache
├── models/
│   ├── siamese_unet.py      # shared encoder + private decoder/head
│   ├── encoder/ decoder/    # resnet encoder, unet decoder
│   ├── prototype/           # multi-prototype (KMeans + EMA)
│   └── aggregation/         # similarity, attention, reliability, strategy
├── federated/
│   ├── client.py server.py strategy.py   # DAAPFL-RA Flower strategy
│   └── baselines/           # 8 FL baseline strategies
├── losses/                  # focal, dice, hybrid + factory
├── training/                # trainer, centralized + federated entrypoints
├── evaluation/              # metrics (loc/dam F1, Jain), Wilcoxon, tables
├── experiments/             # LODO, cold-start, comm-efficiency, hparam sweep
├── visualization/           # Grad-CAM, training/comm/fairness plots
├── scripts/                 # prepare_data.py + ablation_{A..E}.sh
├── utils/                   # config, seed, logging, checkpoint, param utils
├── requirements.txt environment.yml install.sh run.sh
└── README.md
```

---

## 3. Installation

### Option A — pip + venv (recommended)

```bash
# GPU (CUDA 12.1):
bash install.sh cu121

# CPU only:
bash install.sh cpu
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate daapfl-ra
```

Core pins: `torch==2.3.1`, `flwr[simulation]==1.9.0`, `ray==2.10.0`, `segmentation-models-pytorch==0.3.4`, `numpy==1.26.4`, `scikit-learn==1.5.1`. Full list in `requirements.txt`.

---

## 4. Data Setup

Download xBD (xView2) and arrange it in the standard layout:

```
data/xBD/
├── train/  {images/*_{pre,post}_disaster.png, labels/*.json}
├── tier3/  ...
├── test/   ...
└── hold/   ...
```

Set `raw_root` in `configs/data.yaml` if your path differs, then build the tile cache:

```bash
bash run.sh prepare
# = python scripts/prepare_data.py --config configs/base.yaml
```

This rasterizes the JSON polygons (`properties.subtype` → 5 classes: background, no-/minor-/major-damage, destroyed), tiles 1024→512, and writes a compressed `.npz` cache to `outputs/cache/`.

---

## 5. Running

`run.sh` is the single dispatcher:

```bash
bash run.sh prepare                 # build tile cache
bash run.sh central                 # centralized upper-bound baseline
bash run.sh fed                     # federated DAAPFL-RA (default config)
bash run.sh fed configs/baselines/fedprox.yaml   # any baseline
bash run.sh lodo                    # leave-one-disaster-out evaluation
bash run.sh cold                    # cold-start on an unseen disaster
bash run.sh comm                    # communication-efficiency study
bash run.sh sweep                   # λ / τ / K hyperparameter sweep
bash run.sh ablation                # run ablations A–E
```

Direct invocation also works, e.g.:

```bash
python training/train_federated.py --config configs/daapfl_ra.yaml
python training/train_federated.py --config configs/base.yaml --method fedavg
python experiments/run_lodo.py --config configs/daapfl_ra.yaml
```

---

## 6. Baselines

Eight federated baselines are implemented as drop-in Flower strategies (`federated/baselines/`) and selectable via config or `--method`:

`fedavg`, `fedprox`, `scaffold`, `fedper`, `fedrep`, `ditto`, `pfedme`, `fedala`.

Each baseline config under `configs/baselines/` inherits `base.yaml` and sets only its method-specific hyperparameters (e.g. `fedprox.mu=0.01`, `pfedme.lambda_reg=15`).

---

## 7. Ablations

| Tag | Config | What it isolates |
|-----|--------|------------------|
| **A** | `ablation/A_full.yaml`        | Full DAAPFL-RA |
| **B** | `ablation/B_no_multiproto.yaml` | K=1 (single prototype) |
| **C** | `ablation/C_cosine_only.yaml`   | Cosine-only similarity (β=γ=0) |
| **D** | `ablation/D_no_reliability.yaml`| Attention-only (λ=1) |
| **E** | `ablation/E_no_attention.yaml`  | Reliability-only (λ=0) |

```bash
bash scripts/ablation_A.sh   # … through ablation_E.sh
```

---

## 8. Evaluation & Metrics

- **Localization F1** — binary building vs. background.
- **Damage F1** — macro-F1 over the four damage classes (1–4).
- **Overall** — `0.3 · loc_F1 + 0.7 · dam_F1` (xView2 convention).
- **Fairness** — Jain's index across per-client damage-F1.
- **Significance** — paired **Wilcoxon** signed-rank vs. a reference method (`evaluation/statistical.py`), with CSV/Markdown result tables (`evaluation/tables.py`).

---

## 9. Visualization

- `visualization/gradcam.py` — Siamese Grad-CAM on the damage class, with side-by-side comparison figures across methods.
- `visualization/plots.py` — training curves, communication efficiency, per-client fairness bars, confusion matrices.

---

## 10. Reproducibility Notes

- Every entrypoint seeds Python/NumPy/Torch via `utils/seed.py`.
- All configs use `_base_` inheritance (`utils/config.py`) so experiments differ only in their explicit overrides.
- AMP + gradient clipping are on by default; toggle in `configs/federated.yaml`.
- The aggregation math (attention rows sum to 1, reliability normalized, final per-client weights normalized, weighted layer aggregation) is unit-checked in isolation.

---

## 11. Method Summary (one paragraph)

Each disaster-event client trains a Siamese U-Net whose encoder is shared and whose decoder/head are private. After local training, a client computes K=3 multi-prototypes from encoder bottleneck features and reports them with its validation damage-F1 and sample count. The server builds a three-term similarity matrix between clients, converts each row to an attention distribution, computes a reliability vector from `(N·F1)`, and blends the two with λ to produce a **personalized** set of aggregation weights *per client*. It caches a personalized encoder for each client (used next round) and a plain FedAvg encoder for cold-starting unseen disasters. The result targets both higher damage-F1 and improved cross-event fairness compared to uniform federated averaging.
