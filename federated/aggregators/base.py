"""
Base interface for all federated aggregation algorithms.

Every FL method in this repository implements this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from federated.update import ClientUpdate


class BaseAggregator(ABC):
    """
    Abstract base class for all FL aggregation algorithms.
    """

    def __init__(self, cfg):

        self.cfg = cfg

    @abstractmethod
    def aggregate(
        self,
        updates: List[ClientUpdate],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate encoder weights from all selected clients.

        Returns
        -------
        Dict[str, Tensor]

            Updated global encoder.
        """
        pass

    def before_round(
        self,
        round_idx: int,
    ):
        """
        Optional hook before local training.
        """
        return

    def get_params_for_client(
        self,
        client_id: int,
        global_encoder: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Return the encoder state dict that should be sent to a
        specific client before local training.

        Default: return the global encoder (correct for FedAvg and
        most baselines). Personalized methods (e.g. DAAPFL-RA)
        override this to return per-client encoders.

        Parameters
        ----------
        client_id : int
            Target client identifier.
        global_encoder : Dict[str, torch.Tensor]
            Current global (FedAvg) encoder state.

        Returns
        -------
        Dict[str, torch.Tensor]
            Encoder state dict for this client.
        """

        return global_encoder

    def after_round(
        self,
        round_idx: int,
        updates: List[ClientUpdate],
    ):
        """
        Optional hook after aggregation.
        """
        return