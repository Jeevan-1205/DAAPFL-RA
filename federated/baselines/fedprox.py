"""FedProx (Li et al., 2020). Server-side identical to FedAvg; the proximal
term mu/2 * ||w - w_global||^2 is applied client-side (see DamageClient with
method='fedprox')."""
from __future__ import annotations
from flwr.server.strategy import FedAvg


class FedProxStrategy(FedAvg):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.mu = float(cfg.get_path("fedprox.mu", 0.01))
