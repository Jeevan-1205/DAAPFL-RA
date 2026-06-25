"""Event-based splitting and Leave-One-Disaster-Out (LODO)."""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple
import random
from .xbd_dataset import XBDSample


def list_disasters(samples: List[XBDSample]) -> List[str]:
    return sorted({s.disaster for s in samples})


def group_by_disaster(samples: List[XBDSample]) -> Dict[str, List[XBDSample]]:
    g: Dict[str, List[XBDSample]] = defaultdict(list)
    for s in samples:
        g[s.disaster].append(s)
    return g


def event_split(samples: List[XBDSample], val_fraction: float, seed: int
                ) -> Tuple[List[XBDSample], List[XBDSample]]:
    """Per-disaster train/val split so every event is represented in both."""
    rng = random.Random(seed)
    train, val = [], []
    for dis, items in group_by_disaster(samples).items():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def leave_one_disaster_out(samples: List[XBDSample], holdout: str
                           ) -> Tuple[List[XBDSample], List[XBDSample]]:
    """Return (seen, heldout) where heldout == one disaster's samples."""
    seen = [s for s in samples if s.disaster != holdout]
    held = [s for s in samples if s.disaster == holdout]
    if not held:
        raise ValueError(f"holdout disaster '{holdout}' not found in dataset")
    return seen, held
