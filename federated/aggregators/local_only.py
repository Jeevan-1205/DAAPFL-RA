"""
Local Only aggregation (no-op baseline).

Each client trains exclusively on its own disaster partition
with zero cross-client communication.  The "aggregation" step
simply caches each client's encoder and returns it unchanged
in the next round — no averaging, no mixing.

This is the lower-bound baseline in federated learning
experiments: it measures what each client achieves without
any collaboration.

Design note
-----------
The ``get_params_for_client`` override is the key mechanism.
``run_round()`` in ``server.py`` calls this hook before both
training and evaluation.  By returning each client's own
cached encoder, we guarantee zero information leakage across
clients while reusing the entire simulation pipeline unchanged.

On round 0, no cache exists, so every client starts from the
same random initialization (matching the FL setup exactly).
"""

from __future__ import annotations

from typing import Dict, List

import torch

from federated.aggregators.base import BaseAggregator
from federated.update import ClientUpdate


class LocalOnlyAggregator(BaseAggregator):
    """
    No aggregation — each client keeps its own encoder.

    Implements the "Local Only" baseline where each disaster
    partition trains independently with no cross-client
    weight exchange.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # Cache: client_id → their own encoder state dict
        self._client_encoders: Dict[int, Dict[str, torch.Tensor]] = {}

    def aggregate(
        self,
        updates: List[ClientUpdate],
    ) -> Dict[str, torch.Tensor]:
        """
        No-op aggregation: cache each client's encoder unchanged.

        Returns the first client's encoder as the nominal "global"
        state.  This value is only used as a fallback by
        ``get_params_for_client`` for clients not yet in the cache
        (which should not happen after round 0).
        """

        for update in updates:
            self._client_encoders[update.client_id] = {
                k: v.clone().detach()
                for k, v in update.encoder_state.items()
        }

        return updates[0].encoder_state

    def get_params_for_client(
        self,
        client_id: int,
        global_encoder: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Return the client's OWN encoder, not the global one.

        On round 0, when no cache entry exists yet, falls back
        to ``global_encoder`` (the shared random initialization).
        """

        return self._client_encoders.get(client_id, global_encoder)
