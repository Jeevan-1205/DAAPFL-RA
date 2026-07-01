import sys
from utils import load_config
from datasets.xbd_dataset import scan_xbd
from datasets.splitting import event_split
from datasets.partition import build_client_partitions

cfg = load_config("configs/daapfl_ra.yaml")
samples = scan_xbd(cfg.data.raw_root, list(cfg.data.splits))
train_s, val_s = event_split(samples, float(cfg.data.val_fraction), int(cfg.seed))
parts = build_client_partitions(train_s, val_s, seed=int(cfg.seed))
