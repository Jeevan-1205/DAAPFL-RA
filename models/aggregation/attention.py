"""Component 4 — Attention over similarity.

    A = softmax(S / tau)   row-wise.

Row i gives the attention client i pays to every client j (incl. itself).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def attention_weights(S: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    tau = max(float(temperature), 1e-6)
    return F.softmax(S / tau, dim=-1)                  # (M, M), rows sum to 1
