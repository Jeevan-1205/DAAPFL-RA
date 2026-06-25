"""Flower server: strategy factory + simulation driver."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays

from models import build_model
from utils.param_utils import filter_state_dict, state_dict_to_ndarrays
from utils import get_logger

log = get_logger("fed-server")


def _weighted_avg(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    keys = [k for k in metrics[0][1].keys() if isinstance(metrics[0][1][k], (int, float))]
    total = sum(n for n, _ in metrics)
    out = {}
    for k in keys:
        out[k] = sum(n * float(m[k]) for n, m in metrics) / max(total, 1)
    return out


def _initial_shared_params(cfg):
    model = build_model(cfg)
    shared = filter_state_dict(model.state_dict(),
                               list(cfg.model.private_param_patterns), keep=False)
    return ndarrays_to_parameters(state_dict_to_ndarrays(shared))


def build_strategy(method: str, cfg, num_clients: int):
    fed = cfg.federated
    common = dict(
        fraction_fit=float(fed.fraction_fit),
        fraction_evaluate=float(fed.fraction_eval),
        min_fit_clients=int(fed.min_fit_clients),
        min_evaluate_clients=int(fed.min_fit_clients),
        min_available_clients=int(fed.min_available_clients),
        initial_parameters=_initial_shared_params(cfg),
        evaluate_metrics_aggregation_fn=_weighted_avg,
        fit_metrics_aggregation_fn=_weighted_avg,
    )
    if method == "daapfl_ra":
        from federated.strategy import DAAPFLRAStrategy
        return DAAPFLRAStrategy(cfg, **common)

    # baselines
    from federated.baselines import (FedAvgStrategy, FedProxStrategy,
        ScaffoldStrategy, FedPerStrategy, FedRepStrategy, DittoStrategy,
        PFedMeStrategy, FedALAStrategy)
    table = {
        "fedavg": FedAvgStrategy, "fedprox": FedProxStrategy,
        "scaffold": ScaffoldStrategy, "fedper": FedPerStrategy,
        "fedrep": FedRepStrategy, "ditto": DittoStrategy,
        "pfedme": PFedMeStrategy, "fedala": FedALAStrategy,
    }
    if method not in table:
        raise ValueError(f"unknown method '{method}'")
    return table[method](cfg, **common)


def run_federated(method: str, cfg, partitions, device: str = "cuda"):
    from federated.client import make_client_fn
    num_clients = len(partitions)
    strategy = build_strategy(method, cfg, num_clients)
    client_fn = make_client_fn(partitions, cfg, method, device=device)

    use_gpu = torch.cuda.is_available() and device.startswith("cuda")
    client_resources = {
        "num_cpus": int(cfg.federated.ray_num_cpus),
        "num_gpus": float(cfg.federated.ray_num_gpus) if use_gpu else 0.0,
    }
    log.info("starting simulation: method=%s clients=%d rounds=%d gpu=%s",
             method, num_clients, int(cfg.federated.num_rounds), use_gpu)
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=int(cfg.federated.num_rounds)),
        strategy=strategy,
        client_resources=client_resources,
    )
    return hist, strategy
