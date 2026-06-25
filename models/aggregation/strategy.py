"""Component 6 — Final aggregation math.

Per target client i, the aggregation weight over source clients j is:

    W_i = lambda * A_i + (1 - lambda) * R          (then renormalized)

where A_i is row i of the attention matrix (client-specific) and R is the
global reliability vector (shared). The personalized shared-encoder for
client i is sum_j W_ij * encoder_j.

`aggregate_with_weights` performs the weighted sum over a list of parameter
ndarray-lists (the shared/encoder parameters only).
"""
from __future__ import annotations
from typing import List, Sequence
import numpy as np
import torch


def final_weights(attention_row: torch.Tensor, reliability: torch.Tensor,
                  lam: float) -> torch.Tensor:
    w = lam * attention_row + (1.0 - lam) * reliability
    s = w.sum()
    if s <= 0:
        return torch.full_like(w, 1.0 / w.numel())
    return w / s


def aggregate_with_weights(client_params: List[List[np.ndarray]],
                           weights: Sequence[float]) -> List[np.ndarray]:
    """client_params[j] is the ndarray list of client j's shared params.
    Returns a single weighted-average ndarray list."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(w.sum(), 1e-12)
    n_layers = len(client_params[0])
    agg: List[np.ndarray] = []
    for li in range(n_layers):
        stacked = np.stack([client_params[j][li] for j in range(len(client_params))], axis=0)
        # broadcast weights over the leading client axis
        shape = [len(client_params)] + [1] * (stacked.ndim - 1)
        agg.append((stacked * w.reshape(shape)).sum(axis=0).astype(stacked.dtype))
    return agg
