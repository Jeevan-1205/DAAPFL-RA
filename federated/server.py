"""
Federated server: orchestrates one communication round.

This is the single-round building block. It does NOT own the round
loop — that belongs to ``simulation.py``. Keeping the two separate
makes it easy to add logging, checkpointing, or early stopping in
the simulation layer without touching aggregation logic.

One round
---------
1. Push encoder weights to each selected client.
2. Each client runs ``fit()`` → ``ClientUpdate``.
3. Aggregator merges the updates → new global encoder.
4. Each client runs ``evaluate()`` → validation metrics.
5. Return round-level metrics to the simulation loop.
"""

from __future__ import annotations

import time
from typing import Dict, List

import torch

from federated.aggregators.base import BaseAggregator
from federated.clients.base import FedClient
from federated.update import ClientUpdate
from utils import get_logger

log = get_logger("fed-server")


def run_round(
    round_idx: int,
    clients: List[FedClient],
    aggregator: BaseAggregator,
    global_encoder: Dict[str, torch.Tensor],
) -> Dict:
    """
    Execute one federated communication round.

    Parameters
    ----------
    round_idx : int
        Current round number (0-indexed).
    clients : List[FedClient]
        The clients selected for this round.
    aggregator : BaseAggregator
        Aggregation algorithm.
    global_encoder : Dict[str, torch.Tensor]
        Current global encoder state dict.

    Returns
    -------
    dict
        Round results with keys:

        - ``"global_encoder"`` : updated global encoder state dict
        - ``"updates"``        : list of ClientUpdate from this round
        - ``"train_loss"``     : weighted average training loss
        - ``"val_metrics"``    : weighted average validation metrics
        - ``"client_metrics"`` : per-client metric dicts
        - ``"round_time"``     : wall-clock seconds for this round
    """

    t0 = time.time()

    aggregator.before_round(round_idx)

    # ---- 1. distribute encoder + local training ----

    updates: List[ClientUpdate] = []

    log.info("round %d | starting %d clients", round_idx, len(clients))

    for client in clients:

        encoder_for_client = aggregator.get_params_for_client(
            client.client_id,
            global_encoder,
        )

        client.load_encoder(encoder_for_client)

        log.debug(
            "round %d | client %d: encoder loaded",
            round_idx, client.client_id,
        )

        update = client.fit()

        log.debug(
            "round %d | client %d: training done (loss=%.4f)",
            round_idx, client.client_id, update.train_loss,
        )

        updates.append(update)

    # ---- 2. aggregate ----

    log.debug("round %d | aggregating %d updates", round_idx, len(updates))

    new_global_encoder = aggregator.aggregate(updates)

    aggregator.after_round(round_idx, updates)

    # ---- 3. evaluate (with appropriate encoder per client) ----

    total_samples = 0
    weighted_val: Dict[str, float] = {}
    weighted_train_loss = 0.0
    client_metrics: List[Dict[str, float]] = []

    for client, update in zip(clients, updates):

        # Evaluate each client with its own encoder.
        eval_encoder = aggregator.get_params_for_client(
            client.client_id, new_global_encoder,
        )
        client.load_encoder(eval_encoder)

        val = client.evaluate()

        # Store per-client metrics for downstream fairness/analysis
        client_metrics.append({
            "client_id": client.client_id,
            "train_loss": update.train_loss,
            "num_samples": update.num_samples,
            **val,
        })

        n = update.num_samples
        total_samples += n

        weighted_train_loss += update.train_loss * n

        for key, value in val.items():
            weighted_val[key] = weighted_val.get(key, 0.0) + value * n

    # ---- 4. compute weighted averages ----

    if total_samples > 0:
        weighted_train_loss /= total_samples

        for key in weighted_val:
            weighted_val[key] /= total_samples

    round_time = time.time() - t0

    log.info(
        "round %d | clients=%d | train_loss=%.4f | overall=%.4f | "
        "f1_dam=%.4f | miou=%.4f | %.1fs",
        round_idx,
        len(clients),
        weighted_train_loss,
        weighted_val.get("overall", 0.0),
        weighted_val.get("f1_dam", 0.0),
        weighted_val.get("miou", 0.0),
        round_time,
    )

    return {
        "global_encoder": new_global_encoder,
        "updates": updates,
        "train_loss": weighted_train_loss,
        "val_metrics": weighted_val,
        "client_metrics": client_metrics,
        "round_time": round_time,
    }
