"""Component 2 — Multi-prototype representation.

Each client summarizes its feature distribution with K class-agnostic
prototypes obtained by K-means over bottleneck features (K=3 default).
If a client has too few samples for K clusters, it falls back to K=1
(the global mean), preserving a valid prototype set of shape (K, D) by
replicating the single centroid.

Prototypes are EMA-updated across local rounds for stability.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
from sklearn.cluster import KMeans


def compute_client_prototypes(features: torch.Tensor, k: int = 3,
                              seed: int = 42) -> torch.Tensor:
    """features: (N, D) -> (K, D) prototypes. Fallback to K=1 when N<k."""
    if features.ndim != 2:
        raise ValueError("features must be (N, D)")
    x = features.detach().cpu().numpy().astype(np.float64)
    n, d = x.shape
    if n == 0:
        return torch.zeros(k, d)
    if n < k or k <= 1:
        centroid = x.mean(axis=0, keepdims=True)              # (1, D)
        return torch.from_numpy(np.repeat(centroid, k, axis=0)).float()
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(x)
    return torch.from_numpy(km.cluster_centers_).float()      # (K, D)


class MultiPrototype:
    """Holds and EMA-updates a client's (K, D) prototype matrix."""
    def __init__(self, k: int, dim: int, momentum: float = 0.9):
        self.k = k
        self.dim = dim
        self.momentum = momentum
        self.protos: Optional[torch.Tensor] = None

    def update(self, new_protos: torch.Tensor) -> torch.Tensor:
        new_protos = new_protos.detach().float()
        if self.protos is None:
            self.protos = new_protos.clone()
        else:
            m = self.momentum
            self.protos = m * self.protos + (1.0 - m) * new_protos
        return self.protos

    def get(self) -> torch.Tensor:
        if self.protos is None:
            return torch.zeros(self.k, self.dim)
        return self.protos

    def to_numpy(self) -> np.ndarray:
        return self.get().cpu().numpy()

    @staticmethod
    def from_numpy(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(arr)).float()
