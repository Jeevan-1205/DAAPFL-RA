"""
Aggregator registry.

Maps method name strings to aggregator classes so that
server.py and simulation.py can instantiate them by config.
"""

from federated.aggregators.base import BaseAggregator
from federated.aggregators.fedavg import FedAvgAggregator
from federated.aggregators.fedper import FedPerAggregator
from federated.aggregators.local_only import LocalOnlyAggregator
from federated.aggregators.fedprox import FedProxAggregator

# ------------------------------------------------------------------ #
# Registry: method name (str) -> aggregator class                     #
# ------------------------------------------------------------------ #
# New algorithms are registered here as they are implemented.
# Every entry must be a subclass of BaseAggregator.

AGGREGATOR_REGISTRY = {
    "fedavg": FedAvgAggregator,
    "fedper": FedPerAggregator,
    "fedprox": FedProxAggregator,
    "local_only": LocalOnlyAggregator,
}


def build_aggregator(method: str, cfg) -> BaseAggregator:
    """
    Factory: build an aggregator instance from a method name.

    Parameters
    ----------
    method : str
        Algorithm name (must be a key in AGGREGATOR_REGISTRY).
    cfg : Config
        Full project configuration.

    Returns
    -------
    BaseAggregator
        Ready-to-use aggregator.
    """

    cls = AGGREGATOR_REGISTRY.get(method)

    if cls is None:
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            f"Available: {list(AGGREGATOR_REGISTRY.keys())}"
        )

    return cls(cfg)
