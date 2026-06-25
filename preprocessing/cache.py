"""On-disk tile cache. Each sample stored as a compressed .npz with keys
pre,post,mask. An index parquet maps sample_id -> (path, disaster, split)."""
from __future__ import annotations
import hashlib, os
from typing import Dict
import numpy as np


def cache_key(disaster: str, image_id: str, y: int, x: int, tile: int) -> str:
    raw = f"{disaster}|{image_id}|{y}|{x}|{tile}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def is_cached(cache_root: str, key: str) -> bool:
    return os.path.exists(os.path.join(cache_root, "tiles", f"{key}.npz"))


class CacheWriter:
    def __init__(self, cache_root: str):
        self.root = cache_root
        os.makedirs(os.path.join(cache_root, "tiles"), exist_ok=True)

    def write(self, key: str, pre: np.ndarray, post: np.ndarray,
              mask: np.ndarray) -> str:
        path = os.path.join(self.root, "tiles", f"{key}.npz")
        np.savez_compressed(path, pre=pre, post=post, mask=mask)
        return path

    @staticmethod
    def read(path: str) -> Dict[str, np.ndarray]:
        with np.load(path) as z:
            return {"pre": z["pre"], "post": z["post"], "mask": z["mask"]}
