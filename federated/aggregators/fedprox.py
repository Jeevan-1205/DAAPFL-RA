"""
FedProx aggregation.

FedProx uses the exact same server-side aggregation as FedAvg.

The only algorithmic difference from FedAvg is the client-side
proximal regularization term, so this class simply inherits
FedAvgAggregator for experiment clarity.
"""

from federated.aggregators.fedavg import FedAvgAggregator


class FedProxAggregator(FedAvgAggregator):
    """
    FedProx server aggregation.

    Identical to FedAvg.

    The proximal objective is implemented entirely on the client.
    """
    pass