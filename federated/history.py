"""
Training history for federated learning.

Stores round-level aggregated metrics AND per-client metrics
for post-hoc analysis (fairness, std-dev, per-client plots).

Named attributes (train_loss, overall, f1_loc, f1_dam, miou,
precision, recall, accuracy) provide convenient access for
reporting and plotting.  All other metrics are stored in the
generic ``metrics`` dict.

Backward compatible with Flower History format properties.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class History:
    """
    Stores metrics for every communication round.

    Attributes
    ----------
    rounds : List[int]
        Round indices.
    train_loss : List[float]
        Weighted-average training loss per round.
    val_loss : List[float]
        Weighted-average validation loss per round.
    f1_loc, f1_dam, overall, miou : List[float]
        Weighted-average key metrics per round.
    precision, recall, accuracy : List[float]
        Macro-averaged precision, recall, and accuracy per round.
    metrics : Dict[str, List[float]]
        Arbitrary extra metrics (weighted-average).
    client_metrics : List[List[Dict[str, float]]]
        Per-round, per-client metric dicts.  ``client_metrics[r]`` is
        a list of dicts, one per client that participated in round r.
    round_times : List[float]
        Wall-clock seconds per round.
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

    miou: List[float] = field(default_factory=list)

    # macro-averaged precision / recall / accuracy
    precision: List[float] = field(default_factory=list)

    recall: List[float] = field(default_factory=list)

    accuracy: List[float] = field(default_factory=list)

    # arbitrary extra metrics
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    # per-client metrics (for fairness / std-dev analysis)
    client_metrics: List[List[Dict[str, float]]] = field(default_factory=list)

    # round timing
    round_times: List[float] = field(default_factory=list)

    def add_round(
        self,
        round_idx: int,
        train_loss: float,
        metrics: Dict[str, float],
        client_metrics: Optional[List[Dict[str, float]]] = None,
        round_time: Optional[float] = None,
    ) -> None:

        self.rounds.append(round_idx)

        self.train_loss.append(train_loss)

        self.val_loss.append(metrics.get("loss", 0.0))

        self.f1_loc.append(metrics.get("f1_loc", 0.0))

        self.f1_dam.append(metrics.get("f1_dam", 0.0))

        self.overall.append(metrics.get("overall", 0.0))

        self.miou.append(metrics.get("miou", 0.0))

        # Macro-averaged precision / recall / accuracy.
        # Computed from per-class values already present in the metrics dict.
        num_classes = 5
        prec_vals = [metrics.get(f"precision_class_{c}", 0.0)
                     for c in range(num_classes)]
        rec_vals = [metrics.get(f"recall_class_{c}", 0.0)
                    for c in range(num_classes)]
        self.precision.append(sum(prec_vals) / max(len(prec_vals), 1))
        self.recall.append(sum(rec_vals) / max(len(rec_vals), 1))

        # Accuracy: micro-average (TP_total / total_pixels).  When not
        # available separately, approximate as macro-average of recalls.
        self.accuracy.append(metrics.get("accuracy",
                             sum(rec_vals) / max(len(rec_vals), 1)))

        for key, value in metrics.items():

            if key not in self.metrics:
                self.metrics[key] = []

            self.metrics[key].append(value)

        # per-client data (optional — present when MetricsLogger is active)
        self.client_metrics.append(client_metrics or [])

        # timing (optional)
        self.round_times.append(round_time if round_time is not None else 0.0)

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

    def best_f1_loc(self):
        return max(self.f1_loc) if self.f1_loc else 0.0

    def best_f1_dam(self):
        return max(self.f1_dam) if self.f1_dam else 0.0

    def best_miou(self):
        return max(self.miou) if self.miou else 0.0

    def best_round(self, metric_list: List[float]) -> int:
        """Return the 0-indexed round where *metric_list* peaks."""
        if not metric_list:
            return 0
        return int(max(range(len(metric_list)),
                       key=lambda i: metric_list[i]))

    # ------------------------------------------------------------------ #
    # DataFrame / JSON export                                              #
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        """Return round-level metrics as a pandas DataFrame."""
        data = {
            "round": self.rounds,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "f1_loc": self.f1_loc,
            "f1_dam": self.f1_dam,
            "overall": self.overall,
            "miou": self.miou,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "round_time_s": self.round_times,
        }
        return pd.DataFrame(data)

    def client_metrics_dataframe(self) -> pd.DataFrame:
        """Return per-client, per-round metrics as a tidy DataFrame."""
        rows = []
        for round_idx, cm_list in zip(self.rounds, self.client_metrics):
            for cm in cm_list:
                row = {"round": round_idx}
                row.update(cm)
                rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of the full history."""
        return {
            "rounds": self.rounds,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "f1_loc": self.f1_loc,
            "f1_dam": self.f1_dam,
            "overall": self.overall,
            "miou": self.miou,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "round_times": self.round_times,
            "metrics": {k: v for k, v in self.metrics.items()},
            "best_overall": self.best_overall(),
            "best_f1_loc": self.best_f1_loc(),
            "best_f1_dam": self.best_f1_dam(),
            "best_miou": self.best_miou(),
        }

    # ------------------------------------------------------------------ #
    # Backward compatibility with Flower History format                    #
    # ------------------------------------------------------------------ #
    # Experiment scripts (run_comm_efficiency, run_hparam_sweep,
    # train_federated) expect hist.losses_distributed and
    # hist.metrics_distributed in the Flower format:
    #   losses_distributed: List[Tuple[int, float]]
    #   metrics_distributed: Dict[str, List[Tuple[int, float]]]
    # These properties reconstruct that format from the new internals.

    @property
    def losses_distributed(self):
        """List of (round, loss) tuples — matches Flower History."""
        return list(zip(self.rounds, self.val_loss))

    @property
    def metrics_distributed(self):
        """Dict[str, List[(round, value)]] — matches Flower History."""
        out = {}
        for key, values in self.metrics.items():
            out[key] = list(zip(self.rounds, values))
        return out