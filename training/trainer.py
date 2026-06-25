"""Reusable training utilities shared by centralized and federated paths.

Provides:
  - build_optimizer / build_scheduler (config-driven)
  - local_train(): a single client/local fit pass (used by FL clients)
  - Trainer: full centralized loop with AMP, grad-clip, checkpoint + resume
"""
from __future__ import annotations
import os
from typing import Callable, Dict, Optional
import torch
from torch.cuda.amp import autocast, GradScaler

from utils import save_checkpoint, load_checkpoint, get_logger
from evaluation.metrics import evaluate_model

log = get_logger("trainer")


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    name = cfg.train.optimizer.lower()
    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(cfg.train.lr); wd = float(cfg.train.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"unknown optimizer '{name}'")


def build_scheduler(optimizer, cfg, steps_per_epoch: int = 1):
    sch = cfg.train.get("scheduler", "none")
    if sch == "cosine":
        T = max(1, int(cfg.train.epochs) * steps_per_epoch)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
    if sch == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return None


def local_train(model, loader, loss_fn, optimizer, device,
                epochs: int = 1, amp: bool = True, grad_clip: float = 1.0,
                proximal: Optional[Callable] = None) -> Dict[str, float]:
    """One local fit pass. `proximal(model)->tensor` adds a regularizer
    (used by FedProx / Ditto / pFedMe). Returns avg loss + sample count."""
    model.train()
    scaler = GradScaler(enabled=amp and device.startswith("cuda"))
    total_loss, n_batches, n_samples = 0.0, 0, 0
    for _ in range(epochs):
        for batch in loader:
            pre = batch["pre"].to(device, non_blocking=True)
            post = batch["post"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp and device.startswith("cuda")):
                logits = model(pre, post)
                loss = loss_fn(logits, mask)
                if proximal is not None:
                    loss = loss + proximal(model)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
            total_loss += float(loss.item()); n_batches += 1
            n_samples += pre.size(0)
    return {"loss": total_loss / max(n_batches, 1), "num_samples": n_samples}


class Trainer:
    """Centralized trainer with checkpoint/resume."""
    def __init__(self, model, loss_fn, cfg, device: str = "cuda"):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device
        self.optimizer = build_optimizer(self.model, cfg)
        self.scaler = GradScaler(enabled=cfg.train.amp and device.startswith("cuda"))
        self.scheduler = None
        self.start_epoch = 0
        self.best = 0.0
        self.ckpt_dir = cfg.train.ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def maybe_resume(self, path: Optional[str]):
        if path and os.path.exists(path):
            info = load_checkpoint(path, self.model, self.optimizer, self.scaler,
                                   map_location=self.device)
            self.start_epoch = info["epoch"] + 1
            self.best = info["best_metric"]
            log.info("resumed from %s @ epoch %d (best=%.4f)",
                     path, self.start_epoch, self.best)

    def _train_epoch(self, loader) -> float:
        self.model.train()
        running, nb = 0.0, 0
        for batch in loader:
            pre = batch["pre"].to(self.device, non_blocking=True)
            post = batch["post"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.cfg.train.amp and self.device.startswith("cuda")):
                logits = self.model(pre, post)
                loss = self.loss_fn(logits, mask)
            self.scaler.scale(loss).backward()
            gc = float(self.cfg.train.get("grad_clip", 0) or 0)
            if gc > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gc)
            self.scaler.step(self.optimizer); self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            running += float(loss.item()); nb += 1
        return running / max(nb, 1)

    def fit(self, train_loader, val_loader=None, logger=None):
        self.scheduler = build_scheduler(self.optimizer, self.cfg, len(train_loader))
        for epoch in range(self.start_epoch, int(self.cfg.train.epochs)):
            tr_loss = self._train_epoch(train_loader)
            msg = {"epoch": epoch, "train_loss": tr_loss}
            if val_loader is not None and epoch % int(self.cfg.train.val_interval) == 0:
                metrics = evaluate_model(self.model, val_loader, self.device,
                                         int(self.cfg.model.num_classes))
                msg.update({f"val_{k}": v for k, v in metrics.items()})
                cur = metrics["overall"]
                save_checkpoint(os.path.join(self.ckpt_dir, "last.pt"),
                                self.model, self.optimizer, self.scaler,
                                epoch, self.best)
                if cur > self.best:
                    self.best = cur
                    save_checkpoint(os.path.join(self.ckpt_dir, "best.pt"),
                                    self.model, self.optimizer, self.scaler,
                                    epoch, self.best)
            log.info("epoch %d | %s", epoch,
                     " ".join(f"{k}={v:.4f}" for k, v in msg.items() if k != "epoch"))
            if logger is not None:
                logger.log(msg, step=epoch)
        return self.best
