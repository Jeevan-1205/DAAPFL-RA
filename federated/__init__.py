"""
Federated learning package — pure PyTorch implementation.

Public API
----------
- ``run_federated``  : full simulation loop (simulation.py)
- ``run_round``      : single-round orchestration (server.py)
- ``FedClient``      : client wrapper (clients/base.py)
- ``ClientUpdate``   : client → server data structure (update.py)
- ``History``        : round-level metric recording (history.py)
- ``build_aggregator`` : aggregator factory (aggregators/__init__.py)
"""

from federated.simulation import run_federated
from federated.server import run_round
from federated.clients import FedClient
from federated.update import ClientUpdate
from federated.history import History
from federated.aggregators import build_aggregator
