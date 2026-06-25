"""Wilcoxon signed-rank significance testing between methods."""
from __future__ import annotations
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def wilcoxon_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.allclose(a, b):
        return {"statistic": 0.0, "pvalue": 1.0, "median_diff": 0.0}
    try:
        stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p = float("nan"), 1.0
    return {"statistic": float(stat), "pvalue": float(p),
            "median_diff": float(np.median(a - b))}


def paired_comparison_table(scores: Dict[str, List[float]],
                            reference: str) -> pd.DataFrame:
    """scores: method -> per-fold metric list (aligned). Compares each method
    against `reference` with a Wilcoxon test."""
    if reference not in scores:
        raise ValueError(f"reference '{reference}' not in scores")
    rows = []
    ref = scores[reference]
    for method, vals in scores.items():
        if method == reference:
            continue
        r = wilcoxon_test(ref, vals)
        rows.append({"method": method, "reference": reference,
                     "mean": float(np.mean(vals)),
                     "mean_ref": float(np.mean(ref)),
                     "median_diff": r["median_diff"],
                     "pvalue": r["pvalue"],
                     "significant_0.05": r["pvalue"] < 0.05})
    return pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)
