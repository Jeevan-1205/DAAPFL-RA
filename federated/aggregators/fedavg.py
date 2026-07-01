"""
FedAvg aggregation.

McMahan et al., 2017
Communication-Efficient Learning of Deep Networks
from Decentralized Data.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from federated.update import ClientUpdate
from federated.aggregators.base import BaseAggregator


class FedAvgAggregator(BaseAggregator):
    """
    Standard Federated Averaging.

    Aggregates encoder weights according to the
    number of local training samples.
    """

    def aggregate(self, updates):

        if len(updates) == 0:
            raise ValueError("No client updates received.")

        total_samples = sum(
            u.num_samples
            for u in updates
        )

        global_state = OrderedDict()

        first_state = updates[0].encoder_state

        for key in first_state.keys():

            global_state[key] = torch.zeros_like(
                first_state[key]
            )

            for key in first_state.keys():

                tensor = first_state[key]

        # Floating-point tensors → weighted average
                if tensor.is_floating_point():

                    global_state[key] = torch.zeros_like(tensor)

                    for update in updates:
                        weight = update.num_samples / total_samples
                        global_state[key] += update.encoder_state[key] * weight

                # Integer/bool tensors → copy from first client
                else:
                    global_state[key] = tensor.clone()

        return global_state