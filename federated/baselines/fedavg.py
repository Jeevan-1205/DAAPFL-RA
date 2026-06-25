"""FedAvg (McMahan et al., 2017). Sample-weighted average of shared params."""
from __future__ import annotations
from flwr.server.strategy import FedAvg


class FedAvgStrategy(FedAvg):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
