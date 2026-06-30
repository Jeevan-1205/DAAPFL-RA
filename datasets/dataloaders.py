"""DataLoader factories for centralized and federated training."""
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from .xbd_dataset import XBDDataset, XBDSample
from .partition import ClientPartition


def make_dataloader(samples: List[XBDSample], cfg, train: bool,
                    batch_size: int = None, num_workers: int = None) -> DataLoader:
    ds = XBDDataset(samples, cfg, train=train)

    workers = (
    num_workers
    if num_workers is not None
    else cfg.train.num_workers
)

    return DataLoader(
    ds,
    batch_size=batch_size or cfg.train.batch_size,
    shuffle=train,
    num_workers=workers,

    pin_memory=True,

    persistent_workers=(workers > 0),

    prefetch_factor=4 if workers > 0 else None,

    drop_last=train,
)


def make_client_loaders(part: ClientPartition, cfg
                        ) -> Tuple[DataLoader, DataLoader]:
    bs = int(cfg.federated.batch_size)
    train_loader = make_dataloader(part.train, cfg, train=True, batch_size=bs)
    val_loader = (make_dataloader(part.val, cfg, train=False, batch_size=bs)
                  if part.val else None)
    return train_loader, val_loader
