"""Build the tile cache from raw xBD so training is IO-light.
Usage: python scripts/prepare_data.py --config configs/daapfl_ra.yaml"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import tifffile as tiff

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

from tqdm import tqdm

from utils import load_config, get_logger
from datasets.xbd_dataset import scan_xbd
from preprocessing.rasterize import rasterize_label
from preprocessing.tiling import tile_coords, tile_image
from preprocessing.cache import (
    CacheWriter,
    cache_key,
    is_cached,
)

log = get_logger("prepare")


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/daapfl_ra.yaml"
    )

    args = ap.parse_args()

    cfg = load_config(
        args.config
    )

    samples = scan_xbd(
        cfg.data.raw_root,
        list(cfg.data.splits)
    )

    log.info(
        "scanned %d pre/post pairs",
        len(samples)
    )
    log.info("Preparing cache for ALL %d image pairs", len(samples))
   # samples = samples[:1]

    writer = CacheWriter(
        cfg.data.cache_root
    )

    tile = int(
        cfg.data.tile_size
    )

    stride = int(
        cfg.data.stride
    )

    min_frac = float(
        cfg.data.min_building_frac
    )

    n_written = 0
    n_skipped = 0

    for sample_idx, s in enumerate(
        tqdm(samples, desc="Caching images"),
        start=1,
    ):
        if sample_idx % 100 == 0:
            log.info(
                "Processed %d/%d image pairs | cache tiles written=%d",
                sample_idx,
                len(samples),
                n_written,
        )
        

        pre = tiff.imread(s.pre_img)
        post = tiff.imread(s.post_img)

        # Convert int16 -> uint8 if necessary
        if pre.dtype != np.uint8:
            pre = pre.astype(np.float32)
            pre = ((pre - pre.min()) /
                (pre.max() - pre.min() + 1e-6) * 255).astype(np.uint8)

        if post.dtype != np.uint8:
            post = post.astype(np.float32)
            post = ((post - post.min()) /
                    (post.max() - post.min() + 1e-6) * 255).astype(np.uint8)
        mask = rasterize_label(
            s.post_lbl,
            pre.shape[0],
            pre.shape[1]
        )

        for (y, x) in tile_coords(
            pre.shape[0],
            pre.shape[1],
            tile,
            stride
        ):

            mask_tile = tile_image(
                mask,
                y,
                x,
                tile
            )

            building_frac = (
                mask_tile > 0
            ).mean()

            key = cache_key(
                s.disaster,
                s.image_id,
                y,
                x,
                tile
            )

            if building_frac < min_frac:
                n_skipped += 1
                continue

            

            if is_cached(
                cfg.data.cache_root,
                key
            ):
                continue

            pre_tile = tile_image(pre, y, x, tile)
            post_tile = tile_image(post, y, x, tile)


            writer.write(
                key,
                pre_tile,
                post_tile,
                mask_tile
            )
            n_written += 1


            

    log.info(
        "done. wrote %d tiles | skipped %d | cache=%s",
        n_written,
        n_skipped,
        cfg.data.cache_root
    )


if __name__ == "__main__":
    main()