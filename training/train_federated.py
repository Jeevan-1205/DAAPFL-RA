"""Federated training entrypoint (DAAPFL-RA or any baseline).
Usage: python -m training.train_federated --config configs/daapfl_ra.yaml
       python -m training.train_federated --config configs/baselines/fedavg.yaml"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from utils import load_config, set_seed, get_logger
from datasets.xbd_dataset import scan_xbd
from datasets.splitting import event_split
from datasets.partition import build_client_partitions
from federated.simulation import run_federated

log = get_logger("fed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/daapfl_ra.yaml")
    ap.add_argument("--method", default=None,
                    help="override cfg.method (e.g. fedavg, daapfl_ra)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    method = args.method or cfg.get("method", "daapfl_ra")
    device = cfg.device

    samples = scan_xbd(cfg.data.raw_root, list(cfg.data.splits))
    train_s, val_s = event_split(samples, float(cfg.data.val_fraction), int(cfg.seed))
    parts = build_client_partitions(
        train_s, 
        val_s, 
        seed=int(cfg.seed),
        client_partition=cfg.federated.client_partition
    )
    log.info("method=%s clients=%d", method, len(parts))

    hist, _ = run_federated(method, cfg, parts, device=device)

    # All artifacts (CSV, JSON, plots) are now generated automatically
    # by MetricsLogger inside run_federated(). No manual export needed.
    log.info(
        "done | best_overall=%.4f | output -> %s",
        hist.best_overall(),
        os.path.join(cfg.output_dir, "federated", method),
    )


if __name__ == "__main__":
    main()
