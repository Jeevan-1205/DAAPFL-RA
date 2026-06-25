"""SCAFFOLD (Karimireddy et al., 2020).

Server maintains a global control variate c. Each round it aggregates the
client control-variate deltas (returned in fit metrics, base64-pickled) and
updates c <- c + (1/N) * sum(delta_c_i). Model params are aggregated as in
FedAvg with server learning rate.
"""
from __future__ import annotations
import base64, pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
import flwr as fl
from flwr.common import (Parameters, Scalar, FitRes,
                         ndarrays_to_parameters, parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def _dec(s):  # decode control variate delta
    return pickle.loads(base64.b64decode(s.encode("ascii")))


class ScaffoldStrategy(FedAvg):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.server_lr = float(cfg.get_path("scaffold.server_lr", 1.0))
        self.c_global: Optional[List[np.ndarray]] = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        counts = [r.num_examples for _, r in results]
        total = max(sum(counts), 1)
        params = [parameters_to_ndarrays(r.parameters) for _, r in results]
        n_layers = len(params[0])
        agg = []
        for li in range(n_layers):
            stk = np.stack([params[j][li] for j in range(len(params))], 0)
            w = np.array(counts, dtype=np.float64) / total
            sh = [len(params)] + [1] * (stk.ndim - 1)
            agg.append((stk * w.reshape(sh)).sum(0).astype(stk.dtype))

        # control variate update from client deltas (if provided)
        deltas = [(_dec(r.metrics["delta_c"]) if "delta_c" in r.metrics else None)
                  for _, r in results]
        if all(d is not None for d in deltas):
            if self.c_global is None:
                self.c_global = [np.zeros_like(a) for a in agg]
            for li in range(n_layers):
                mean_delta = np.mean([deltas[j][li] for j in range(len(deltas))], axis=0)
                self.c_global[li] = self.c_global[li] + mean_delta

        # apply server learning rate (interpolate toward aggregate)
        if self.server_lr != 1.0 and hasattr(self, "_prev") and self._prev is not None:
            agg = [self._prev[li] + self.server_lr * (agg[li] - self._prev[li])
                   for li in range(n_layers)]
        self._prev = agg
        return ndarrays_to_parameters(agg), {"server_lr": self.server_lr}
