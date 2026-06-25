"""Deterministic tiling of large images into fixed crops."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


def tile_coords(h: int, w: int, tile: int, stride: int) -> List[Tuple[int, int]]:
    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if (h - tile) % stride != 0:
        ys.append(h - tile)
    if (w - tile) % stride != 0:
        xs.append(w - tile)
    ys = sorted(set(max(0, y) for y in ys))
    xs = sorted(set(max(0, x) for x in xs))
    return [(y, x) for y in ys for x in xs]


def tile_image(arr: np.ndarray, y: int, x: int, tile: int) -> np.ndarray:
    return arr[y:y + tile, x:x + tile]
