"""Soft multi-class Dice loss (macro over classes)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        valid = (target != self.ignore_index)
        tgt = target.clone()
        tgt[~valid] = 0
        onehot = F.one_hot(tgt, num_classes).permute(0, 3, 1, 2).float()
        mask = valid.unsqueeze(1).float()
        probs = probs * mask
        onehot = onehot * mask
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        card = probs.sum(dims) + onehot.sum(dims)
        dice = (2.0 * inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice.mean()
