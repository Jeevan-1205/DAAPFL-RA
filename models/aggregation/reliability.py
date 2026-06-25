"""Component 5 — Reliability weighting.

    R_j = (N_j * F1_j) / sum_k (N_k * F1_k)

N_j = client j's number of (train) samples; F1_j = its validation F1
(damage macro-F1) reported this round. Produces a 1-D vector over clients
that sums to 1. Robust to all-zero inputs (falls back to uniform).
"""
from __future__ import annotations
from typing import Sequence
import torch


def reliability_weights(num_samples: Sequence[int],
                        f1_scores: Sequence[float],
                        eps: float = 1e-12) -> torch.Tensor:
    n = torch.as_tensor(num_samples, dtype=torch.float32)
    f1 = torch.as_tensor(f1_scores, dtype=torch.float32).clamp(0.0, 1.0)
    raw = n * f1
    total = raw.sum()
    if total <= eps:
        m = len(num_samples)
        return torch.full((m,), 1.0 / max(m, 1))
    return raw / total
