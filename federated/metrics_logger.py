"""
Unified metrics logger for federated training.

Orchestrates all logging concerns so that ``simulation.py`` only
needs ``logger.log_round(...)`` and ``logger.on_training_end()``.
Algorithms never touch this module.

Produces
--------
- Structured console output (replacing bare print statements)
- Per-round CSV (``round_metrics.csv``)
- Per-client CSV (``client_metrics.csv``)
- JSON experiment summary (``experiment_summary.json``)
- Automatic training plots (convergence curves, per-class F1, fairness)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation.metrics import fairness_index
from federated.history import History
from utils import get_logger

log = get_logger("metrics")


class MetricsLogger:
    """
    Central logging orchestrator for federated training.

    Parameters
    ----------
    method : str
        Algorithm name (e.g. ``"fedavg"``, ``"fedper"``).
    cfg : Config
        Full project configuration.
    out_dir : str, optional
        Override output directory.  Defaults to
        ``{cfg.output_dir}/federated/{method}``.
    """

    def __init__(
        self,
        method: str,
        cfg,
        out_dir: Optional[str] = None,
    ):
        self.method = method
        self.cfg = cfg
        self.out_dir = out_dir or os.path.join(
            cfg.output_dir, "federated", method,
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.history = History()
        self._start_time = time.time()

        # CSV files (written incrementally)
        self._round_csv = os.path.join(self.out_dir, "round_metrics.csv")
        self._client_csv = os.path.join(self.out_dir, "client_metrics.csv")
        self._round_csv_header_written = False
        self._client_csv_header_written = False

    # ------------------------------------------------------------------ #
    # Per-round logging                                                    #
    # ------------------------------------------------------------------ #

    def log_round(self, round_idx: int, result: Dict[str, Any]) -> None:
        """
        Record one round of training results.

        Parameters
        ----------
        round_idx : int
            Current round index.
        result : dict
            The dict returned by ``server.run_round()``.  Expected keys:
            ``train_loss``, ``val_metrics``, ``client_metrics``,
            ``round_time``.
        """
        train_loss = result["train_loss"]
        val_metrics = result["val_metrics"]
        client_metrics = result.get("client_metrics", [])
        round_time = result.get("round_time", 0.0)

        # ---- 1. Store in History ----
        self.history.add_round(
            round_idx=round_idx,
            train_loss=train_loss,
            metrics=val_metrics,
            client_metrics=client_metrics,
            round_time=round_time,
        )

        # ---- 2. Compute derived metrics ----
        fairness_jfi = 0.0
        std_overall = 0.0
        if len(client_metrics) >= 2:
            client_overalls = [
                cm.get("overall", 0.0) for cm in client_metrics
            ]
            fairness_jfi = fairness_index(client_overalls)
            std_overall = float(np.std(client_overalls))

        # ---- 3. Console output ----
        self._log_console(round_idx, train_loss, val_metrics,
                          fairness_jfi, std_overall, round_time,
                          len(client_metrics))

        # ---- 4. Append to round CSV ----
        self._append_round_csv(round_idx, train_loss, val_metrics,
                               fairness_jfi, std_overall, round_time)

        # ---- 5. Append to client CSV ----
        self._append_client_csv(round_idx, client_metrics)

    # ------------------------------------------------------------------ #
    # End-of-training                                                      #
    # ------------------------------------------------------------------ #

    def on_training_end(self) -> Dict[str, str]:
        """
        Finalize logging after all rounds complete.

        Returns
        -------
        Dict[str, str]
            Paths to all generated artifacts.
        """
        total_time = time.time() - self._start_time
        artifacts = {}

        # ---- JSON summary ----
        summary_path = os.path.join(self.out_dir, "experiment_summary.json")
        summary = self._build_summary(total_time)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=float)
        artifacts["json_summary"] = summary_path

        # ---- Final CSVs already written incrementally ----
        artifacts["round_csv"] = self._round_csv
        artifacts["client_csv"] = self._client_csv

        # ---- Automatic plots ----
        try:
            from visualization.plots import plot_training_summary
            plot_paths = plot_training_summary(self.history, self.out_dir)
            artifacts.update(plot_paths)
        except Exception as e:
            log.warning("plot generation failed: %s", e)

        log.info(
            "training complete | method=%s | best_overall=%.4f | "
            "total_time=%.1fs | artifacts -> %s",
            self.method,
            self.history.best_overall(),
            total_time,
            self.out_dir,
        )

        return artifacts

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _log_console(
        self,
        round_idx: int,
        train_loss: float,
        val_metrics: Dict[str, float],
        fairness_jfi: float,
        std_overall: float,
        round_time: float,
        num_clients: int,
    ) -> None:
        """Structured console output for one round."""
        overall = val_metrics.get("overall", 0.0)
        f1_dam = val_metrics.get("f1_dam", 0.0)
        f1_loc = val_metrics.get("f1_loc", 0.0)
        miou = val_metrics.get("miou", 0.0)

        print(f"\n{'='*60}")
        print(f"  Round {round_idx + 1} | {self.method.upper()} | "
              f"{num_clients} clients | {round_time:.1f}s")
        print(f"{'='*60}")
        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Overall    : {overall:.4f}    "
              f"F1-loc: {f1_loc:.4f}    F1-dam: {f1_dam:.4f}")
        print(f"  mIoU       : {miou:.4f}    "
              f"Fairness: {fairness_jfi:.4f}    "
              f"Std(overall): {std_overall:.4f}")
        print(f"{'='*60}")

    def _append_round_csv(
        self,
        round_idx: int,
        train_loss: float,
        val_metrics: Dict[str, float],
        fairness_jfi: float,
        std_overall: float,
        round_time: float,
    ) -> None:
        """Append one row to the round-level CSV."""
        row = {
            "round": round_idx,
            "train_loss": train_loss,
            "f1_loc": val_metrics.get("f1_loc", 0.0),
            "f1_dam": val_metrics.get("f1_dam", 0.0),
            "overall": val_metrics.get("overall", 0.0),
            "miou": val_metrics.get("miou", 0.0),
            "fairness_jfi": fairness_jfi,
            "std_overall": std_overall,
            "round_time_s": round_time,
        }
        # Add per-class metrics if present
        for key in sorted(val_metrics.keys()):
            if key.startswith(("f1_class_", "precision_class_",
                               "recall_class_", "iou_class_")):
                row[key] = val_metrics[key]

        df = pd.DataFrame([row])
        df.to_csv(
            self._round_csv,
            mode="a",
            header=not self._round_csv_header_written,
            index=False,
        )
        self._round_csv_header_written = True

    def _append_client_csv(
        self,
        round_idx: int,
        client_metrics: List[Dict[str, float]],
    ) -> None:
        """Append per-client rows to the client-level CSV."""
        if not client_metrics:
            return

        rows = []
        for cm in client_metrics:
            row = {"round": round_idx}
            row.update(cm)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(
            self._client_csv,
            mode="a",
            header=not self._client_csv_header_written,
            index=False,
        )
        self._client_csv_header_written = True

    def _build_summary(self, total_time: float) -> Dict[str, Any]:
        """Build the JSON experiment summary."""
        # Extract serializable config
        cfg_dict = {}
        try:
            cfg_dict = dict(self.cfg)
        except Exception:
            cfg_dict = {"note": "config not serializable"}

        return {
            "method": self.method,
            "seed": cfg_dict.get("seed", None),
            "num_rounds": len(self.history.rounds),
            "total_time_s": total_time,
            "config": {
                "federated": cfg_dict.get("federated", {}),
                "train": cfg_dict.get("train", {}),
                "model": cfg_dict.get("model", {}),
                "loss": cfg_dict.get("loss", {}),
            },
            "final_metrics": {
                "train_loss": self.history.train_loss[-1]
                if self.history.train_loss else None,
                "overall": self.history.overall[-1]
                if self.history.overall else None,
                "f1_loc": self.history.f1_loc[-1]
                if self.history.f1_loc else None,
                "f1_dam": self.history.f1_dam[-1]
                if self.history.f1_dam else None,
                "miou": self.history.miou[-1]
                if self.history.miou else None,
            },
            "best_metrics": {
                "overall": self.history.best_overall(),
                "f1_dam": max(self.history.f1_dam)
                if self.history.f1_dam else 0.0,
            },
            "history": self.history.to_json(),
        }
