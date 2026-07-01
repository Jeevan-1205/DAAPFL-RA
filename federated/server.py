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
    """

    aggregator.before_round(round_idx)

    # ---- 1. distribute encoder + local training ----

    updates: List[ClientUpdate] = []

    print(f"\n========== Round {round_idx+1} ==========")

    for client in clients:
        print(f"[Round {round_idx+1}] Starting Client {client.client_id}")

        encoder_for_client = aggregator.get_params_for_client(
            client.client_id,
            global_encoder,
        )

        client.load_encoder(encoder_for_client)

        print(f"[Round {round_idx+1}] Client {client.client_id}: loaded encoder")

        update = client.fit()

        print(f"[Round {round_idx+1}] Client {client.client_id}: finished training")

        updates.append(update)

    print("[Server] Aggregating updates...")
    # ---- 2. aggregate ----

    new_global_encoder = aggregator.aggregate(updates)

    aggregator.after_round(round_idx, updates)

    # ---- 3. evaluate (with appropriate encoder per client) ----

    total_samples = 0
    weighted_val: Dict[str, float] = {}
    weighted_train_loss = 0.0

    for client, update in zip(clients, updates):

        # Evaluate each client with its own encoder.
        eval_encoder = aggregator.get_params_for_client(
            client.client_id, new_global_encoder,
        )
        client.load_encoder(eval_encoder)

        val = client.evaluate()

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

    log.info(
        "round %d | clients=%d | train_loss=%.4f | overall=%.4f",
        round_idx,
        len(clients),
        weighted_train_loss,
        weighted_val.get("overall", 0.0),
    )

    return {
        "global_encoder": new_global_encoder,
        "updates": updates,
        "train_loss": weighted_train_loss,
        "val_metrics": weighted_val,
    }
