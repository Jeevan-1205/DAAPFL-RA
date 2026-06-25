"""Cold-start experiment. A new (unseen) disaster client joins after federated
training. We measure F1-dam after k local fine-tuning epochs (k=0,1,5,10)
starting from (a) the global encoder and (b) a randomly-initialized encoder,
to quantify how fast the personalized representation adapts."""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from utils import load_config, set_seed, get_logger
from datasets.xbd_dataset import scan_xbd
from datasets.splitting import event_split, leave_one_disaster_out, list_disasters
from datasets.partition import build_client_partitions
from datasets.dataloaders import make_dataloader
from federated.server import run_federated
from models import build_model
from losses import build_loss
from training.trainer import build_optimizer, local_train
from utils.param_utils import ndarrays_to_state_dict, filter_state_dict
from flwr.common import parameters_to_ndarrays
from evaluation.metrics import evaluate_model

log = get_logger("coldstart")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/daapfl_ra.yaml")
    ap.add_argument("--method", default=None)
    ap.add_argument("--ft-epochs", default="0,1,5,10")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    method = args.method or cfg.get("method", "daapfl_ra")
    device = cfg.device if torch.cuda.is_available() else "cpu"
    ft_list = [int(x) for x in args.ft_epochs.split(",")]

    samples = scan_xbd(cfg.data.raw_root, list(cfg.data.splits))
    cold = cfg.data.get("lodo_holdout", list_disasters(samples)[-1])
    seen, held = leave_one_disaster_out(samples, cold)
    tr, va = event_split(seen, float(cfg.data.val_fraction), int(cfg.seed))
    parts = build_client_partitions(tr, va, seed=int(cfg.seed))

    _, strategy = run_federated(method, cfg, parts, device=device)
    shared_keys = list(filter_state_dict(build_model(cfg).state_dict(),
                       list(cfg.model.private_param_patterns), keep=False).keys())

    # cold client splits
    cold_tr, cold_va = event_split(held, 0.3, int(cfg.seed))
    cold_train = make_dataloader(cold_tr, cfg, train=True, batch_size=int(cfg.federated.batch_size))
    cold_val = make_dataloader(cold_va, cfg, train=False, batch_size=int(cfg.federated.batch_size))

    results = {"warm": {}, "random": {}}
    for init in ["warm", "random"]:
        model = build_model(cfg).to(device)
        if init == "warm" and getattr(strategy, "_global", None) is not None:
            nds = parameters_to_ndarrays(strategy._global)
            model.load_state_dict(ndarrays_to_state_dict(shared_keys, nds), strict=False)
        loss_fn = build_loss(cfg).to(device)
        for k in ft_list:
            if k > 0:
                opt = build_optimizer(model, cfg)
                local_train(model, cold_train, loss_fn, opt, device, epochs=k,
                            amp=bool(cfg.federated.amp))
            m = evaluate_model(model, cold_val, device, int(cfg.model.num_classes))
            results[init][k] = m["f1_dam"]
            log.info("init=%s ft=%d f1_dam=%.4f", init, k, m["f1_dam"])

    out_dir = os.path.join(cfg.output_dir, "coldstart", method)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "coldstart.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("coldstart done -> %s", out_dir)


if __name__ == "__main__":
    main()
