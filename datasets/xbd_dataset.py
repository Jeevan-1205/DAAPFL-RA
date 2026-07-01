"""xBD dataset: scanning, indexing, and a torch Dataset that reads from the
tile cache (built by scripts/prepare_data.py) or rasterizes on the fly."""
from __future__ import annotations
import glob, os, re
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import cv2
import tifffile as tiff
import torch
from torch.utils.data import Dataset

from preprocessing.rasterize import rasterize_label
from preprocessing.tiling import tile_coords, tile_image
from preprocessing.cache import cache_key, CacheWriter
from preprocessing.augment import (build_train_aug, build_eval_aug,
                                   apply_aug, Normalizer)

_FNAME = re.compile(r"^(?P<dis>.+)_(?P<id>\d+)_(?P<phase>pre|post)_disaster\.(png|tif)$")


@dataclass
class XBDSample:
    disaster: str
    image_id: str
    split: str
    pre_img: str
    post_img: str
    pre_lbl: str
    post_lbl: str


def _parse(fname: str):
    m = _FNAME.match(os.path.basename(fname))
    return (m.group("dis"), m.group("id"), m.group("phase")) if m else None


def scan_xbd(raw_root: str, splits: List[str]) -> List[XBDSample]:
    """Pair pre/post images+labels per (disaster,id)."""

    samples: List[XBDSample] = []

    for split in splits:

        img_dir = os.path.join(
            raw_root,
            split,
            "images"
        )

        lbl_dir = os.path.join(
            raw_root,
            split,
            "labels"
        )

        if not os.path.isdir(img_dir):
            continue

        pre_imgs = (

            glob.glob(
                os.path.join(
                    img_dir,
                    "*_pre_disaster.tif"
                )
            )

            +

            glob.glob(
                os.path.join(
                    img_dir,
                    "*_pre_disaster.png"
                )
            )

        )

        for pre_img in sorted(pre_imgs):

            p = _parse(pre_img)

            if not p:
                continue

            dis, iid, _ = p

            if pre_img.endswith(".tif"):

                post_img = pre_img.replace(
                    "_pre_disaster.tif",
                    "_post_disaster.tif"
                )

            else:

                post_img = pre_img.replace(
                    "_pre_disaster.png",
                    "_post_disaster.png"
                )

            pre_lbl = os.path.join(
                lbl_dir,
                f"{dis}_{iid}_pre_disaster.json"
            )

            post_lbl = os.path.join(
                lbl_dir,
                f"{dis}_{iid}_post_disaster.json"
            )

            if not (
                os.path.exists(post_img)
                and
                os.path.exists(post_lbl)
            ):
                continue

            samples.append(

                XBDSample(

                    dis,
                    iid,
                    split,

                    pre_img,
                    post_img,

                    pre_lbl,
                    post_lbl

                )

            )

    return samples
"""Pair pre/post images+labels per (disaster,id)."""


class XBDDataset(Dataset):
    """Tile-level dataset. If a tile cache exists it is read directly;
    otherwise the source image is read+rasterized+tiled lazily."""

    def __init__(self, samples: List[XBDSample], cfg, train: bool = True,
                 cache_root: Optional[str] = None):
        self.samples = samples
        self.cfg = cfg
        self.train = train
        self.tile = int(cfg.data.tile_size)
        self.stride = int(cfg.data.stride)
        self.cache_root = cache_root or cfg.data.cache_root
        self.norm = Normalizer(cfg.data.mean, cfg.data.std)
        self.aug = build_train_aug(self.tile) if train else build_eval_aug()
        self.index = self._build_tile_index()

    def _build_tile_index(self):
        idx = []
        ts = int(self.cfg.data.source_size)

        total_tiles = 0
        cached_tiles = 0

        for si, s in enumerate(self.samples):
            for (y, x) in tile_coords(ts, ts, self.tile, self.stride):

                total_tiles += 1

                key = cache_key(
                    s.disaster,
                    s.image_id,
                    y,
                    x,
                    self.tile
                )

                cache_path = os.path.join(
                    self.cache_root,
                    "tiles",
                    f"{key}.npz"
                )

                # ONLY keep tiles that actually exist
                if not os.path.exists(cache_path):
                    continue

                cached_tiles += 1

                idx.append({
                    "si": si,
                    "y": y,
                    "x": x,
                    "key": key
                })

        
        return idx

    def __len__(self) -> int:
        return len(self.index)

    def _load_raw(self, s: XBDSample, y: int, x: int):
        pre = tiff.imread(s.pre_img)
        post = tiff.imread(s.post_img)

#        convert int16 -> uint8

        pre = pre.astype(np.float32)
        post = post.astype(np.float32)

        pre = ((pre - pre.min()) /
       (pre.max() - pre.min() + 1e-6) * 255).astype(np.uint8)

        post = ((post - post.min()) /
        (post.max() - post.min() + 1e-6) * 255).astype(np.uint8)

        mask = rasterize_label(
            s.post_lbl,
            pre.shape[0],
            pre.shape[1]
    )
        return (tile_image(pre, y, x, self.tile),
                tile_image(post, y, x, self.tile),
                tile_image(mask, y, x, self.tile))

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        rec = self.index[i]
        s = self.samples[rec["si"]]
        cache_path = os.path.join(self.cache_root, "tiles", f"{rec['key']}.npz")
        if os.path.exists(cache_path):
            try:
                d = CacheWriter.read(cache_path)
            except Exception:
                os.remove(cache_path)

                pre, post, mask = self._load_raw(
                    s,
                    rec["y"],
                    rec["x"]
                )

                CacheWriter(self.cache_root).write(
                    rec["key"],
                    pre,
                    post,
                    mask
                )

                d = CacheWriter.read(cache_path)
            pre, post, mask = d["pre"], d["post"], d["mask"]
        else:
            pre, post, mask = self._load_raw(s, rec["y"], rec["x"])


        out = apply_aug(self.aug, pre, post, mask)
        pre_t = torch.from_numpy(self.norm(out["pre"])).permute(2, 0, 1).float()
        post_t = torch.from_numpy(self.norm(out["post"])).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(out["mask"].astype(np.int64))
        return {"pre": pre_t, "post": post_t, "mask": mask_t,
                "disaster": s.disaster}
