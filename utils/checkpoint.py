"""Checkpoint save/load with optimizer, scaler, epoch and arbitrary extras."""
from __future__ import annotations
import os
from typing import Any, Dict, Optional
import torch


def save_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
                    epoch: int = 0, best_metric: float = 0.0,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
                    map_location: str = "cpu", strict: bool = True) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return {"epoch": ckpt.get("epoch", 0),
            "best_metric": ckpt.get("best_metric", 0.0),
            "extra": ckpt.get("extra", {})}
