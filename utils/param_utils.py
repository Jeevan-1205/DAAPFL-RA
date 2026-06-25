"""Conversions between torch state_dicts and Flower ndarray lists, plus
layer-name filtering used by personalized FL methods (FedPer/FedRep/DAAPFL-RA).

Ordering contract: we ALWAYS iterate state_dict in insertion order
(`model.state_dict()` is ordered) and keep a parallel key list so that
ndarray <-> state_dict round-trips are exact.
"""
from __future__ import annotations
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch


def get_param_keys(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    return list(state_dict.keys())


def state_dict_to_ndarrays(state_dict: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    return [t.detach().cpu().numpy() for t in state_dict.values()]


def ndarrays_to_state_dict(keys: Sequence[str],
                           arrays: Sequence[np.ndarray]) -> "OrderedDict[str, torch.Tensor]":
    if len(keys) != len(arrays):
        raise ValueError(f"key/array length mismatch: {len(keys)} vs {len(arrays)}")
    return OrderedDict((k, torch.as_tensor(a)) for k, a in zip(keys, arrays))


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(p in name for p in patterns)


def split_params_by_keys(
    state_dict: Dict[str, torch.Tensor],
    private_patterns: Sequence[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return (shared, private). A param is private if its name contains any
    of `private_patterns` (e.g. ['decoder', 'seg_head', 'prototype'])."""
    shared, private = OrderedDict(), OrderedDict()
    for k, v in state_dict.items():
        (private if _matches(k, private_patterns) else shared)[k] = v
    return shared, private


def filter_state_dict(state_dict: Dict[str, torch.Tensor],
                      patterns: Sequence[str], keep: bool = True
                      ) -> "OrderedDict[str, torch.Tensor]":
    """keep=True -> only params whose name matches; keep=False -> the complement."""
    return OrderedDict(
        (k, v) for k, v in state_dict.items() if _matches(k, patterns) == keep
    )
