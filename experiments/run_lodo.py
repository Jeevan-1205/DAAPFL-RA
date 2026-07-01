"""Leave-One-Disaster-Out experiment.

For each disaster d: train federated DAAPFL-RA (or a baseline) on all OTHER
disasters as clients, then evaluate the personalized/global model on the
held-out disaster (cold global client). Aggregates per-fold F1-loc/F1-dam,
fairness, and writes a results table.
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from utils import load_config, set_seed, get_logger
from datasets.xbd_dataset import scan_xbd
from datasets.splitting import (event_split, leave_one_disaster_out, list_disasters)
from datasets.partition import build_client_partitions
from datasets.dataloaders import make_dataloader
from federated.simulation import run_federated
from models import build_model
from utils.param_utils import filter_state_dict
from evaluation.metrics import evaluate_model, fairness_index
from evaluation.tables import results_table, save_table

log = get_logger("lodo")


def _load_global_into_model(cfg, aggregator):
    model = build_model(cfg)
    if getattr(aggregator, "_global", None) is not None:
        model.load_state_dict(aggregator._global, strict=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/daapfl_ra.yaml")
    ap.add_argument("--method", default=None)
    ap.add_argument("--max-folds", type=int, default=0, help="0 = all disasters")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    method = args.method or cfg.get("method", "daapfl_ra")
    device = cfg.device if torch.cuda.is_available() else "cpu"

    samples = scan_xbd(cfg.data.raw_root, list(cfg.data.splits))
    disasters = list_disasters(samples)
    if args.max_folds > 0:
        disasters = disasters[:args.max_folds]
    log.info("LODO over %d disasters (method=%s)", len(disasters), method)

    per_fold = {}
    for d in disasters:
        seen, held = leave_one_disaster_out(samples, d)
        tr, va = event_split(seen, float(cfg.data.val_fraction), int(cfg.seed))
        parts = build_client_partitions(tr, va, seed=int(cfg.seed))
        _, aggregator = run_federated(method, cfg, parts, device=device)
        model = _load_global_into_model(cfg, aggregator).to(device)
        held_loader = make_dataloader(held, cfg, train=False)
        m = evaluate_model(model, held_loader, device, int(cfg.model.num_classes))
        per_fold[d] = m
        log.info("fold %s | f1_loc=%.4f f1_dam=%.4f", d, m["f1_loc"], m["f1_dam"])

    df = results_table(per_fold)
    df.loc["MEAN"] = df.mean(numeric_only=True)
    out_dir = os.path.join(cfg.output_dir, "lodo", method)
    paths = save_table(df, out_dir, "lodo_results")
    jfi = fairness_index([per_fold[d]["f1_dam"] for d in per_fold])
    with open(os.path.join(out_dir, "fairness.txt"), "w") as f:
        f.write(f"jain_fairness_f1dam={jfi:.4f}\n")
    log.info("LODO done -> %s | fairness=%.4f", paths["md"], jfi)


if __name__ == "__main__":
    main()
