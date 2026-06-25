"""Ditto (Li et al., 2021). The server trains a global model exactly via
FedAvg. Each client additionally keeps a personalized model regularized
toward the global one with strength lambda_reg (applied client-side). Server
behavior here is standard FedAvg over the shared params."""
from __future__ import annotations
from flwr.server.strategy import FedAvg


class DittoStrategy(FedAvg):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.lambda_reg = float(cfg.get_path("ditto.lambda_reg", 0.1))
        self.personal_epochs = int(cfg.get_path("ditto.personal_epochs", 1))
