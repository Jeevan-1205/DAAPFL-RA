"""xBD polygon JSON -> dense class mask.

Classes: 0=background 1=no-damage 2=minor-damage 3=major-damage 4=destroyed.
Pre-disaster labels carry no damage subtype, so they rasterize to a binary
building mask (class 1) which is used only for the localization branch.
"""
from __future__ import annotations
import json
from typing import Dict, List, Tuple
import numpy as np
import cv2
from shapely import wkt as shapely_wkt

SUBTYPE_TO_CLASS: Dict[str, int] = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1,   # treat unknown as building/no-damage for loc
}


def _polygons_from_json(label_json: dict) -> List[Tuple[np.ndarray, int]]:
    out: List[Tuple[np.ndarray, int]] = []
    feats = label_json.get("features", {}).get("xy", [])
    for feat in feats:
        props = feat.get("properties", {})
        subtype = props.get("subtype", "no-damage")
        cls = SUBTYPE_TO_CLASS.get(subtype, 1)
        geom = shapely_wkt.loads(feat["wkt"])
        if geom.is_empty:
            continue
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            coords = np.asarray(poly.exterior.coords, dtype=np.int32)
            if coords.shape[0] >= 3:
                out.append((coords, cls))
    return out


def rasterize_label(label_path: str, height: int, width: int) -> np.ndarray:
    """Return uint8 mask (H,W). Larger-damage polygons drawn last to win overlaps."""
    with open(label_path, "r") as f:
        data = json.load(f)
    mask = np.zeros((height, width), dtype=np.uint8)
    polys = sorted(_polygons_from_json(data), key=lambda pc: pc[1])  # ascending severity
    for coords, cls in polys:
        cv2.fillPoly(mask, [coords], int(cls))
    return mask
