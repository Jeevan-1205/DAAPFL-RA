"""pFedMe (Dinh et al., 2020). Moreau-envelope personalization: each client
solves an inner problem producing a personalized theta, then sends the
local model; the server does a (beta-damped) moving average. Inner k-steps
and lambda are client-side; server applies the beta interpolation."""
from __future__ import annotations
from typing import List, Optional
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg


class PFedMeStrategy(FedAvg):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.beta = float(cfg.get_path("pfedme.beta", 1.0))
        self._prev: Optional[List[np.ndarray]] = None

    def aggregate_fit(self, server_round, results, failures):
        agg_params, metrics = super().aggregate_fit(server_round, results, failures)
        if agg_params is None:
            return agg_params, metrics
        new = parameters_to_ndarrays(agg_params)
        if self._prev is not None and self.beta != 1.0:
            new = [(1 - self.beta) * self._prev[i] + self.beta * new[i]
                   for i in range(len(new))]
        self._prev = new
        return ndarrays_to_parameters(new), metrics
