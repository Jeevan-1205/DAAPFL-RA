"""
Federated simulation driver.

Replaces Flower's ``fl.simulation.start_simulation`` with a
plain Python for-loop. No Ray, no gRPC, no Flower.

This is the top-level entry point that experiment scripts
and ``training/train_federated.py`` call.

Responsibilities
----------------
- Build clients from partitions.
- Build the aggregator from the method name.
- Initialize the global encoder.
- Run the round loop, calling ``server.run_round`` each iteration.
- Record history.
- Return ``(History, aggregator)`` for downstream use.
"""

from __future__ import annotations

from typing import List, Tuple

from datasets.partition import ClientPartition
from models import build_model
from utils import get_logger
from utils.param_utils import filter_state_dict

from federated.aggregators import build_aggregator
from federated.aggregators.base import BaseAggregator
from federated.clients.base import FedClient
from federated.history import History
from federated.server import run_round

log = get_logger("fed-sim")


def run_federated(
    method: str,
    cfg,
    partitions: List[ClientPartition],
    device: str = "cuda",
) -> Tuple[History, BaseAggregator]:
    """
    Full federated training simulation.

    Signature is intentionally compatible with the old Flower-based
    ``run_federated`` so that callers (``train_federated.py``,
    ``run_lodo.py``, ``run_coldstart.py``, etc.) need minimal changes.

    Parameters
    ----------
    method : str
        Aggregation method name (e.g. ``"fedavg"``, ``"daapfl_ra"``).
    cfg : Config
        Full project configuration.
    partitions : List[ClientPartition]
        One partition per client (disaster event).
    device : str
        PyTorch device string.

    Returns
    -------
    Tuple[History, BaseAggregator]
        Training history and the aggregator (which holds the
        final global encoder state for downstream experiments).
    """

    num_rounds = int(cfg.federated.num_rounds)

    # ---- build clients ----

    clients: List[FedClient] = []

    for i, partition in enumerate(partitions):
        client = FedClient(
            client_id=i,
            partition=partition,
            cfg=cfg,
            device=device,
        )
        clients.append(client)

    log.info(
        "created %d clients (method=%s, rounds=%d)",
        len(clients),
        method,
        num_rounds,
    )

    # ---- build aggregator ----

    aggregator = build_aggregator(method, cfg)

    # ---- initialize global encoder ----

    init_model = build_model(cfg)

    global_encoder = filter_state_dict(
        init_model.state_dict(),
        list(cfg.model.private_param_patterns),
        keep=False,
    )

    del init_model

    # ---- round loop ----

    history = History()

    for round_idx in range(num_rounds):

        result = run_round(
            round_idx=round_idx,
            clients=clients,
            aggregator=aggregator,
            global_encoder=global_encoder,
        )

        global_encoder = result["global_encoder"]

        history.add_round(
            round_idx=round_idx,
            train_loss=result["train_loss"],
            metrics=result["val_metrics"],
        )

    log.info(
        "training complete | best_overall=%.4f",
        history.best_overall(),
    )

    return history, aggregator
