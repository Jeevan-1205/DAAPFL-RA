"""
Training history for federated learning.

Designed to replace Flower's History object while remaining simple
and compatible with experiment scripts.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class History:
    """
    Stores metrics for every communication round.
    """

    # round numbers
    rounds: List[int] = field(default_factory=list)

    # training statistics
    train_loss: List[float] = field(default_factory=list)

    # validation statistics
    val_loss: List[float] = field(default_factory=list)

    # xView2 metrics
    f1_loc: List[float] = field(default_factory=list)

    f1_dam: List[float] = field(default_factory=list)

    overall: List[float] = field(default_factory=list)

    # arbitrary extra metrics
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def add_round(
        self,
        round_idx: int,
        train_loss: float,
        metrics: Dict[str, float],
    ) -> None:

        self.rounds.append(round_idx)

        self.train_loss.append(train_loss)

        self.val_loss.append(metrics.get("loss", 0.0))

        self.f1_loc.append(metrics.get("f1_loc", 0.0))

        self.f1_dam.append(metrics.get("f1_dam", 0.0))

        self.overall.append(metrics.get("overall", 0.0))

        for key, value in metrics.items():

            if key not in self.metrics:
                self.metrics[key] = []

            self.metrics[key].append(value)

    def latest(self):

        if not self.rounds:
            return None

        return {
            "round": self.rounds[-1],
            "train_loss": self.train_loss[-1],
            "overall": self.overall[-1],
        }

    def best_overall(self):

        if not self.overall:
            return 0.0

        return max(self.overall)