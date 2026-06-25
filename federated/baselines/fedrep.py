"""FedRep (Collins et al., 2021). Like FedPer but local training alternates:
first update the head (decoder) with the body frozen, then update the body.
Server aggregation of the shared representation (body) is FedAvg. The
alternating schedule is configured via fedrep.head_epochs / body_epochs and
consumed client-side."""
from __future__ import annotations
from flwr.server.strategy import FedAvg


class FedRepStrategy(FedAvg):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.head_epochs = int(cfg.get_path("fedrep.head_epochs", 1))
        self.body_epochs = int(cfg.get_path("fedrep.body_epochs", 1))
