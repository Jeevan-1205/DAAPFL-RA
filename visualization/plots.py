"""Standard result plots: training curves, communication efficiency,
fairness bars, confusion matrix, and auto-generated training summaries."""
from __future__ import annotations
import os
from typing import Dict, List, Sequence
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt


def _ensure(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_training_curves(history: Dict[str, List[float]],
                         out_path: str = "outputs/figures/curves.png"):
    _ensure(out_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, v in history.items():
        ax.plot(range(1, len(v) + 1), v, label=k, marker="o", ms=3)
    ax.set_xlabel("round"); ax.set_ylabel("metric"); ax.legend(); ax.grid(alpha=.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out_path


def plot_comm_efficiency(rounds_to_target: Dict[str, int],
                         bytes_per_round: Dict[str, float],
                         out_path: str = "outputs/figures/comm.png"):
    _ensure(out_path)
    methods = list(rounds_to_target.keys())
    total_mb = [rounds_to_target[m] * bytes_per_round.get(m, 0) / 1e6 for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(methods, total_mb)
    ax.set_ylabel("Total MB to reach target F1-dam")
    ax.set_xticklabels(methods, rotation=30, ha="right"); ax.grid(axis="y", alpha=.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out_path


def plot_fairness_bars(per_client_scores: Dict[str, Sequence[float]],
                       out_path: str = "outputs/figures/fairness.png"):
    _ensure(out_path)
    from evaluation.metrics import fairness_index
    methods = list(per_client_scores.keys())
    jfi = [fairness_index(per_client_scores[m]) for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(methods, jfi); ax.set_ylim(0, 1.0)
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_xticklabels(methods, rotation=30, ha="right"); ax.grid(axis="y", alpha=.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out_path


def plot_confusion(cm: np.ndarray, class_names: Sequence[str],
                   out_path: str = "outputs/figures/confusion.png"):
    _ensure(out_path)
    cmn = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im); ax.set_xlabel("pred"); ax.set_ylabel("true")
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out_path


# ------------------------------------------------------------------ #
# Automatic training summary plots                                     #
# ------------------------------------------------------------------ #

def plot_training_summary(history, out_dir: str) -> Dict[str, str]:
    """
    Generate all standard training plots from a History object.

    Called automatically by ``MetricsLogger.on_training_end()``.

    Parameters
    ----------
    history : History
        The training history with round-level and per-client data.
    out_dir : str
        Directory to save plot files.

    Returns
    -------
    Dict[str, str]
        Mapping from plot name to file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    rounds = [r + 1 for r in history.rounds]  # 1-indexed for display

    # ---- 1. Loss convergence ----
    if history.train_loss:
        path = os.path.join(out_dir, "curves_loss.png")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rounds, history.train_loss, "o-", color="#e74c3c",
                label="Train Loss", ms=4, linewidth=2)
        if any(v > 0 for v in history.val_loss):
            ax.plot(rounds, history.val_loss, "s-", color="#3498db",
                    label="Val Loss", ms=4, linewidth=2)
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training Loss Convergence", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["curves_loss"] = path

    # ---- 2. Score convergence (overall, F1-dam, F1-loc, mIoU) ----
    if history.overall:
        path = os.path.join(out_dir, "curves_score.png")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rounds, history.overall, "o-", color="#2ecc71",
                label="Overall", ms=4, linewidth=2)
        ax.plot(rounds, history.f1_dam, "s-", color="#e67e22",
                label="F1-dam", ms=4, linewidth=2)
        ax.plot(rounds, history.f1_loc, "^-", color="#9b59b6",
                label="F1-loc", ms=4, linewidth=2)
        if history.miou and any(v > 0 for v in history.miou):
            ax.plot(rounds, history.miou, "D-", color="#1abc9c",
                    label="mIoU", ms=4, linewidth=2)
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Metric Convergence", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.0)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["curves_score"] = path

    # ---- 3. Per-class F1 bar chart (final round) ----
    if history.metrics:
        f1_keys = sorted(
            k for k in history.metrics if k.startswith("f1_class_")
        )
        if f1_keys:
            path = os.path.join(out_dir, "per_class_f1.png")
            class_names = [
                k.replace("f1_class_", "Class ") for k in f1_keys
            ]
            final_f1 = [history.metrics[k][-1] for k in f1_keys]

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c", "#9b59b6"]
            bars = ax.bar(class_names, final_f1,
                          color=colors[:len(f1_keys)], edgecolor="white",
                          linewidth=1.5)
            for bar, val in zip(bars, final_f1):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                        fontweight="bold")
            ax.set_ylabel("F1 Score", fontsize=12)
            ax.set_title("Per-Class F1 (Final Round)", fontsize=14,
                         fontweight="bold")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths["per_class_f1"] = path

    # ---- 4. Fairness over rounds ----
    if history.client_metrics and any(len(cm) >= 2 for cm in history.client_metrics):
        from evaluation.metrics import fairness_index as jfi_fn
        path = os.path.join(out_dir, "fairness.png")
        jfi_per_round = []
        for cm_list in history.client_metrics:
            if len(cm_list) >= 2:
                scores = [cm.get("overall", 0.0) for cm in cm_list]
                jfi_per_round.append(jfi_fn(scores))
            else:
                jfi_per_round.append(1.0)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rounds, jfi_per_round, "o-", color="#2c3e50",
                ms=4, linewidth=2)
        ax.fill_between(rounds, jfi_per_round, alpha=0.15, color="#2c3e50")
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Jain's Fairness Index", fontsize=12)
        ax.set_title("Client Fairness Over Rounds", fontsize=14,
                     fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["fairness"] = path

    return paths
