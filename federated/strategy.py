"""DAAPFL-RA Flower Strategy (Component 6 server side).

Extends FedAvg. Each round it:
  1. collects shared (encoder) params, sample counts, val F1, and prototypes;
  2. builds the (M,K,D) prototype tensor -> three-term similarity S;
  3. attention A = softmax(S/tau);
  4. reliability R = (N*F1)/sum(N*F1);
  5. per-client final weights W_i = lam*A_i + (1-lam)*R;
  6. produces a PERSONALIZED encoder per client = sum_j W_ij * encoder_j,
     and also a global average (returned to fresh/cold-start clients).

Personalized encoders are cached and dispatched via `configure_fit` so that
each client receives its own aggregated encoder next round.
"""
from __future__ import annotations
import base64, pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import flwr as fl
from flwr.common import (FitIns, FitRes, Parameters, Scalar, EvaluateIns,
                         ndarrays_to_parameters, parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from models.aggregation.similarity import three_term_similarity
from models.aggregation.attention import attention_weights
from models.aggregation.reliability import reliability_weights
from models.aggregation.strategy import final_weights, aggregate_with_weights
from utils import get_logger

log = get_logger("daapfl-strategy")


def _decode(s: str) -> np.ndarray:
    return pickle.loads(base64.b64decode(s.encode("ascii")))


class DAAPFLRAStrategy(FedAvg):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        d = cfg.daapfl_ra
        self.alpha = float(d.sim_alpha); self.beta = float(d.sim_beta)
        self.gamma = float(d.sim_gamma); self.tau = float(d.temperature)
        self.lam = float(d.lam); self.k = int(d.num_prototypes)
        self.invl2_eps = float(d.invl2_eps); self.kl_eps = float(d.kl_eps)
        # cache: cid -> personalized Parameters
        self._personalized: Dict[int, Parameters] = {}
        self._global: Optional[Parameters] = None

    # ---- dispatch personalized encoder to each client ----
    def configure_fit(self, server_round, parameters, client_manager):
        self._global = parameters
        sample = super().configure_fit(server_round, parameters, client_manager)
        out: List[Tuple[ClientProxy, FitIns]] = []
        for client, fit_ins in sample:
            cid = int(client.cid)
            params = self._personalized.get(cid, parameters)
            out.append((client, FitIns(params, fit_ins.config)))
        return out

    def configure_evaluate(self, server_round, parameters, client_manager):
        sample = super().configure_evaluate(server_round, parameters, client_manager)
        out = []
        for client, eval_ins in sample:
            cid = int(client.cid)
            params = self._personalized.get(cid, parameters)
            out.append((client, EvaluateIns(params, eval_ins.config)))
        return out

    # ---- core: similarity -> attention -> reliability -> weights ----
    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        cids, params, counts, f1s, protos = [], [], [], [], []
        for client, res in results:
            cid = int(res.metrics.get("cid", client.cid))
            cids.append(cid)
            params.append(parameters_to_ndarrays(res.parameters))
            counts.append(int(res.num_examples))
            f1s.append(float(res.metrics.get("val_f1_dam", 0.0)))
            if "prototypes" in res.metrics:
                protos.append(_decode(res.metrics["prototypes"]))
            else:
                protos.append(None)

        M = len(cids)
        # global average (FedAvg) for cold-start clients
        global_avg = aggregate_with_weights(
            params, weights=[c / max(sum(counts), 1) for c in counts])
        self._global = ndarrays_to_parameters(global_avg)

        # reliability vector R
        R = reliability_weights(counts, f1s)

        # similarity + attention (fallback to identity attention if no protos)
        have_protos = all(p is not None for p in protos) and M > 1
        if have_protos:
            P = torch.from_numpy(np.stack(protos, axis=0)).float()  # (M,K,D)
            if P.ndim == 2:  # K==1 stored as (M,D)
                P = P.unsqueeze(1)
            S = three_term_similarity(P, self.alpha, self.beta, self.gamma,
                                      self.invl2_eps, self.kl_eps)
            A = attention_weights(S, self.tau)                       # (M,M)
        else:
            A = torch.eye(M)

        # personalized encoder per client
        new_personalized: Dict[int, Parameters] = {}
        for i, cid in enumerate(cids):
            W = final_weights(A[i], R, self.lam).cpu().numpy()       # (M,)
            agg_i = aggregate_with_weights(params, W)
            new_personalized[cid] = ndarrays_to_parameters(agg_i)
        self._personalized = new_personalized

        log.info("round %d | clients=%d protos=%s lam=%.2f tau=%.2f",
                 server_round, M, have_protos, self.lam, self.tau)
        metrics = {"reliability_max": float(R.max()),
                   "attention_diag_mean": float(torch.diag(A).mean())}
        return self._global, metrics
