"""
FedPer aggregation (Arivazhagan et al., 2019).

Federated Learning with Personalization Layers.

In this framework, FedPer is computationally equivalent to FedAvg
because encoder-only aggregation is already the default behaviour:
``private_param_patterns`` ensures that the decoder and segmentation
head are never exchanged — only the encoder (base layers) is
FedAvg'd across clients.

This class exists as a named subclass for:
  - Experiment traceability (logs/checkpoints say "FedPerAggregator")
  - Paper reproducibility (reviewers can grep for "FedPer")
  - Future extensibility (configurable split depth, head freezing, etc.)
"""

from __future__ import annotations

from federated.aggregators.fedavg import FedAvgAggregator


class FedPerAggregator(FedAvgAggregator):
    """
    FedPer: shared encoder averaged, private decoder/head kept local.

    Identical to FedAvgAggregator in this framework because the
    encoder-only exchange boundary is enforced by the client and
    ``private_param_patterns`` config — not by the aggregator.
    """

    pass
