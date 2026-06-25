"""Albumentations pipelines. Pre/post images share the SAME geometric
transform (additional_targets), masks transformed jointly with pre image."""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import albumentations as A


class Normalizer:
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = np.asarray(mean, np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, np.float32).reshape(1, 1, 3)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return (img.astype(np.float32) / 255.0 - self.mean) / self.std


def build_train_aug(tile: int) -> A.Compose:
    return A.Compose(
        [
            A.RandomCrop(tile, tile) if False else A.NoOp(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
        ],
        additional_targets={"post": "image"},
    )


def build_eval_aug() -> A.Compose:
    return A.Compose([A.NoOp()], additional_targets={"post": "image"})


def apply_aug(aug: A.Compose, pre: np.ndarray, post: np.ndarray,
              mask: np.ndarray) -> Dict[str, np.ndarray]:
    out = aug(image=pre, post=post, mask=mask)
    return {"pre": out["image"], "post": out["post"], "mask": out["mask"]}
