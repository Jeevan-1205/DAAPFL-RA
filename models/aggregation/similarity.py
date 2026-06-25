"""Component 3 — Three-term similarity between client prototype sets.

For two clients i,j with prototype matrices Pi,Pj in R^{KxD} we first
reduce each to a representative vector by mean over the K prototypes, then:

    S_ij = alpha * CosSim(pi, pj)
         + beta  * InvL2(pi, pj)        with InvL2 = 1 / (1 + ||pi-pj||_2)
         + gamma * (-KL(softmax(pi) || softmax(pj)))   (negated: higher = closer)

All three terms are in a comparable [0,1]-ish range (cos in [-1,1],
invL2 in (0,1], -KL in (-inf,0]); the attention temperature/softmax that
consumes S handles scaling. Returns a symmetric (M,M) matrix.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _reduce(protos: torch.Tensor) -> torch.Tensor:
    # (M, K, D) -> (M, D) mean over prototypes
    return protos.mean(dim=1)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()                                   # (M, M)


def inv_l2(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # pairwise euclidean distance -> 1/(1+d)
    d = torch.cdist(a, b, p=2)                          # (M, M)
    return 1.0 / (1.0 + d + eps)


def neg_kl(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pa = F.softmax(a, dim=-1).clamp_min(eps)           # (M, D)
    pb = F.softmax(b, dim=-1).clamp_min(eps)
    # KL(pa_i || pb_j) for all pairs -> (M, M)
    log_pa = pa.log()                                  # (M, D)
    log_pb = pb.log()                                  # (M, D)
    # KL = sum pa_i * (log pa_i - log pb_j)
    term1 = (pa * log_pa).sum(dim=-1, keepdim=True)    # (M, 1)
    term2 = pa @ log_pb.t()                            # (M, M)
    kl = term1 - term2
    return -kl                                         # higher = more similar


def three_term_similarity(protos: torch.Tensor,
                          alpha: float, beta: float, gamma: float,
                          invl2_eps: float = 1e-6,
                          kl_eps: float = 1e-8) -> torch.Tensor:
    """protos: (M, K, D). Returns (M, M) similarity matrix S."""
    v = _reduce(protos)                                # (M, D)
    s = (alpha * cosine_sim(v, v)
         + beta * inv_l2(v, v, invl2_eps)
         + gamma * neg_kl(v, v, kl_eps))
    return 0.5 * (s + s.t())                           # enforce symmetry
