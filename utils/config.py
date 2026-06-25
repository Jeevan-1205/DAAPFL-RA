"""YAML config loader with attribute access and deep merge."""
from __future__ import annotations
import copy, os
from typing import Any, Dict
import yaml


class Config(dict):
    """dict with dot-access; nested dicts auto-wrapped."""
    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def _deep_merge(base: Dict, override: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> Config:
    """Load a yaml config; supports a top-level `_base_` list of parent yamls."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    merged: Dict = {}
    for base_rel in raw.pop("_base_", []) or []:
        base_path = base_rel if os.path.isabs(base_rel) else os.path.join(os.path.dirname(path), base_rel)
        merged = _deep_merge(merged, load_config(base_path))
    merged = _deep_merge(merged, raw)
    return Config(merged)
