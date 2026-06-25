"""FedPer (Arivazhagan et al., 2019). Shared body (encoder) is FedAvg'd; the
personalization head (decoder + seg_head) stays local. Because DamageClient
already exchanges only non-private params, the server is a plain FedAvg over
the body. The private_param_patterns in configs/baselines/fedper.yaml define
the personalized layers."""
from __future__ import annotations
from flwr.server.strategy import FedAvg


class FedPerStrategy(FedAvg):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.private_patterns = list(
            cfg.get_path("fedper.private_param_patterns",
                         cfg.model.private_param_patterns))
