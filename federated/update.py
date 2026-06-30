"""
Common data structure exchanged between FL clients and the server.

Every federated algorithm (FedAvg, FedPer, FedProx, FedRep,
SCAFFOLD, Ditto, pFedMe, FedALA, DAAPFL-RA)
uses exactly this interface.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class ClientUpdate:
    """
    Output produced by one client after local training.
    """

    # Client identifier
    client_id: int

    # Shared encoder weights (only these are aggregated)
    encoder_state: Dict[str, torch.Tensor]

    # Number of local training samples
    num_samples: int

    # Average training loss
    train_loss: float

    # Validation metrics
    val_metrics: Dict[str, float] = field(default_factory=dict)

    # DAAPFL-RA
    prototypes: Optional[torch.Tensor] = None

    class_distribution: Optional[torch.Tensor] = None