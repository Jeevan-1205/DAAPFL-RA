"""Communication-efficiency experiment. Tracks rounds-to-target F1-dam and
per-round uplink bytes (shared params only) for each method, then plots total
MB to reach the target."""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from utils import load_config, set_seed, get_logger
from datasets.xbd_dataset import scan_xbd
from datasets.splitting import event_split
from datasets.partition import build_client_partitions
from models import build_model
from utils.param_utils import filter_state_dict, state_dict_to_ndarrays
from federated.simulation import run_federated
from visualization.plots import plot_comm_efficiency

log = get_logger("comm")


def shared_param_bytes(cfg) -> float:
    model = build_model(cfg)
    shared = filter_state_dict(model.state_dict(),
                               list(cfg.model.private_param_patterns), keep=False)
    return float(sum(a.nbytes for a in state_dict_to_ndarrays(shared)))


def rounds_to_target(history, target: float) -> int:
    md = history.metrics_distributed or {}
    series = md.get("val_f1_dam") or md.get("val_overall") or []
    for r, v in series:
        if float(v) >= target:
            return int(r)
    return len(series) or int(history) if isinstance(history, int) else 9999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/daapfl_ra.yaml")
    ap.add_argument("--methods", default="fedavg,fedprox,daapfl_ra")
    ap.add_argument("--target", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = cfg.device if torch.cuda.is_available() else "cpu"
    methods = args.methods.split(",")

    samples = scan_xbd(cfg.data.raw_root, list(cfg.data.splits))
    tr, va = event_split(samples, float(cfg.data.val_fraction), int(cfg.seed))
    parts = build_client_partitions(tr, va, seed=int(cfg.seed))
    per_round_bytes = shared_param_bytes(cfg) * int(cfg.federated.clients_per_round)

    r2t, bpr = {}, {}
    for m in methods:
        hist, _ = run_federated(m, cfg, parts, device=device)
        r2t[m] = rounds_to_target(hist, args.target)
        bpr[m] = per_round_bytes if m != "daapfl_ra" else per_round_bytes  # same uplink
        log.info("method=%s rounds_to_%.2f=%d", m, args.target, r2t[m])

    out_dir = os.path.join(cfg.output_dir, "comm")
    os.makedirs(out_dir, exist_ok=True)
    fig = plot_comm_efficiency(r2t, bpr, os.path.join(out_dir, "comm.png"))
    with open(os.path.join(out_dir, "comm.json"), "w") as f:
        json.dump({"rounds_to_target": r2t, "bytes_per_round": bpr}, f, indent=2)
    log.info("comm-efficiency figure -> %s", fig)


if __name__ == "__main__":
    main()
