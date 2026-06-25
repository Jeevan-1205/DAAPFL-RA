"""Standard result plots: training curves, communication efficiency,
fairness bars, confusion matrix."""
from __future__ import annotations
import os
from typing import Dict, List, Sequence
import numpy as np
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
