"""Auto-generate result tables (markdown + csv)."""
from __future__ import annotations
import os
from typing import Dict, List
import pandas as pd


def results_table(per_method: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """per_method: method -> metric dict. Returns a tidy dataframe."""
    df = pd.DataFrame(per_method).T
    df.index.name = "method"
    cols = [c for c in ["f1_loc", "f1_dam", "overall"] if c in df.columns]
    other = [c for c in df.columns if c not in cols]
    return df[cols + other].round(4)


def save_table(df: pd.DataFrame, out_dir: str, name: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    md_path = os.path.join(out_dir, f"{name}.md")
    df.to_csv(csv_path)
    with open(md_path, "w") as f:
        f.write(df.to_markdown())
    return {"csv": csv_path, "md": md_path}
