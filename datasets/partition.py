"""Client partition creation: one client per disaster event (default), with
optional further sharding of large events into multiple clients."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import random
from .xbd_dataset import XBDSample
from .splitting import group_by_disaster, group_by_disaster_type


@dataclass
class ClientPartition:
    client_id: int
    disaster: str
    train: List[XBDSample] = field(default_factory=list)
    val: List[XBDSample] = field(default_factory=list)


def build_client_partitions(
    train_samples: List[XBDSample],
    val_samples: List[XBDSample],
    max_shards_per_event: int = 1,
    seed: int = 42,
    client_partition: str = "type",
) -> List[ClientPartition]:
    rng = random.Random(seed)
    
    if client_partition == "type":
        train_by = group_by_disaster_type(train_samples)
        val_by = group_by_disaster_type(val_samples)
    else:
        train_by = group_by_disaster(train_samples)
        val_by = group_by_disaster(val_samples)
        
    parts: List[ClientPartition] = []
    cid = 0
    for dis in sorted(train_by.keys()):
        tr = train_by[dis][:]
        va = val_by.get(dis, [])[:]
        rng.shuffle(tr)
        shards = max(1, min(max_shards_per_event, len(tr)))
        tr_chunks = [tr[k::shards] for k in range(shards)]
        va_chunks = [va[k::shards] for k in range(shards)]
        for k in range(shards):
            parts.append(ClientPartition(cid, dis, tr_chunks[k], va_chunks[k]))
            cid += 1
            
    print("\nClient Summary:")
    for p in parts:
        print(f"Client {p.client_id} -> {p.disaster}")
    print(f"\nTotal clients: {len(parts)}\n")
            
    return parts
