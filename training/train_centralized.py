"""Centralized (non-federated) training entrypoint.
Usage: python -m training.train_centralized --config configs/daapfl_ra.yaml"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from utils import load_config, set_seed, get_logger, WandbLogger
from datasets.xbd_dataset import scan_xbd
from datasets.splitting import event_split
from datasets.dataloaders import make_dataloader
from models import build_model
from losses import build_loss
from training.trainer import Trainer
import time

log = get_logger("central")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/daapfl_ra.yaml")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = cfg.device if torch.cuda.is_available() else "cpu"

    samples = scan_xbd(cfg.data.raw_root, list(cfg.data.splits))
    train_s, val_s = event_split(samples, float(cfg.data.val_fraction), int(cfg.seed))
    log.info("train=%d val=%d", len(train_s), len(val_s))

    train_loader = make_dataloader(train_s, cfg, train=True)
    val_loader = make_dataloader(val_s, cfg, train=False)
    print("\n================ DATASET INFO ================")
    print("Train tiles   :", len(train_loader.dataset))
    print("Val tiles     :", len(val_loader.dataset))
    print("Train batches :", len(train_loader))
    print("Val batches   :", len(val_loader))
    print("Batch size    :", train_loader.batch_size)
    print("=============================================\n")

    model = build_model(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    loss_fn = build_loss(cfg)
    wb = WandbLogger(bool(cfg.log.wandb), cfg.log.project, "centralized", dict(cfg))

    trainer = Trainer(model, loss_fn, cfg, device=device)
    trainer.maybe_resume(args.resume)
    best = trainer.fit(train_loader, val_loader, logger=wb)
    log.info("centralized best overall = %.4f", best)
    wb.finish()


if __name__ == "__main__":
    main()
