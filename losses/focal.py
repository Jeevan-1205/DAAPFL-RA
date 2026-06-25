"""Multi-class Focal Loss for segmentation."""
from __future__ import annotations
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0,
                 alpha: Optional[Sequence[float]] = None,
                 ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.register_buffer(
            "alpha",
            torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits (B,C,H,W) ; target (B,H,W)
        ce = F.cross_entropy(logits, target, weight=self.alpha,
                             ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
