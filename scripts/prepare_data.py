"""Build the tile cache from raw xBD so training is IO-light.
Usage: python scripts/prepare_data.py --config configs/daapfl_ra.yaml"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from tqdm import tqdm
from utils import load_config, get_logger
from datasets.xbd_dataset import scan_xbd
from preprocessing.rasterize import rasterize_label
from preprocessing.tiling import tile_coords, tile_image
from preprocessing.cache import CacheWriter, cache_key, is_cached

log = get_logger("prepare")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/daapfl_ra.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    samples = scan_xbd(cfg.data.raw_root, list(cfg.data.splits))
    log.info("scanned %d pre/post pairs", len(samples))
    writer = CacheWriter(cfg.data.cache_root)
    tile, stride = int(cfg.data.tile_size), int(cfg.data.stride)
    n_written = 0
    for s in tqdm(samples, desc="caching"):
        pre = cv2.cvtColor(cv2.imread(s.pre_img), cv2.COLOR_BGR2RGB)
        post = cv2.cvtColor(cv2.imread(s.post_img), cv2.COLOR_BGR2RGB)
        mask = rasterize_label(s.post_lbl, pre.shape[0], pre.shape[1])
        for (y, x) in tile_coords(pre.shape[0], pre.shape[1], tile, stride):
            key = cache_key(s.disaster, s.image_id, y, x, tile)
            if is_cached(cfg.data.cache_root, key):
                continue
            writer.write(key, tile_image(pre, y, x, tile),
                         tile_image(post, y, x, tile),
                         tile_image(mask, y, x, tile))
            n_written += 1
    log.info("done. wrote %d tiles to %s", n_written, cfg.data.cache_root)


if __name__ == "__main__":
    main()
