"""Hyperparameter sweep for DAAPFL-RA over (lambda, tau, K) and the similarity
weights (alpha,beta,gamma). Runs short federated jobs and tabulates final
F1-dam to pick the best configuration."""
from __future__ import annotations
import argparse, copy, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from utils import load_config, set_seed, get_logger, Config
from datasets.xbd_dataset import scan_xbd
from datasets.splitting import event_split
from datasets.partition import build_client_partitions
from federated.simulation import run_federated

log = get_logger("sweep")


def final_f1(history) -> float:
    md = history.metrics_distributed or {}
    series = md.get("val_f1_dam") or md.get("val_overall") or []
    return float(series[-1][1]) if series else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/daapfl_ra.yaml")
    ap.add_argument("--lams", default="0.3,0.6,0.9")
    ap.add_argument("--taus", default="0.25,0.5,1.0")
    ap.add_argument("--ks", default="1,3")
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()

    base = load_config(args.config)
    set_seed(int(base.seed))
    device = base.device if torch.cuda.is_available() else "cpu"

    samples = scan_xbd(base.data.raw_root, list(base.data.splits))
    tr, va = event_split(samples, float(base.data.val_fraction), int(base.seed))
    parts = build_client_partitions(tr, va, seed=int(base.seed))

    grid = list(itertools.product(
        [float(x) for x in args.lams.split(",")],
        [float(x) for x in args.taus.split(",")],
        [int(x) for x in args.ks.split(",")],
    ))
    rows = []
    for lam, tau, k in grid:
        cfg = Config(copy.deepcopy(dict(base)))
        cfg.daapfl_ra.lam = lam; cfg.daapfl_ra.temperature = tau
        cfg.daapfl_ra.num_prototypes = k
        cfg.federated.num_rounds = args.rounds
        hist, _ = run_federated("daapfl_ra", cfg, parts, device=device)
        score = final_f1(hist)
        rows.append({"lam": lam, "tau": tau, "K": k, "f1_dam": score})
        log.info("lam=%.2f tau=%.2f K=%d -> f1_dam=%.4f", lam, tau, k, score)

    rows.sort(key=lambda r: r["f1_dam"], reverse=True)
    out_dir = os.path.join(base.output_dir, "sweep")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sweep.json"), "w") as f:
        json.dump(rows, f, indent=2)
    log.info("best: %s", rows[0])


if __name__ == "__main__":
    main()
