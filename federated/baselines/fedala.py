"""FedALA (Zhang et al., 2023). Adaptive Local Aggregation: each client
learns element-wise weights to blend the downloaded global model with its
own local model before training (on the higher layers, layer_idx onward).
The adaptive blend is client-side; the server is plain FedAvg."""
from __future__ import annotations
from flwr.server.strategy import FedAvg


class FedALAStrategy(FedAvg):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.rand_percent = float(cfg.get_path("fedala.rand_percent", 0.8))
        self.layer_idx = int(cfg.get_path("fedala.layer_idx", 2))
        self.ala_epochs = int(cfg.get_path("fedala.ala_epochs", 1))
