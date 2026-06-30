from utils import load_config
from datasets.xbd_dataset import scan_xbd
from datasets.splitting import event_split
from datasets.dataloaders import make_dataloader

cfg = load_config("configs/daapfl_ra.yaml")

cfg.train.batch_size = 1
cfg.train.num_workers = 0

samples = scan_xbd(
    cfg.data.raw_root,
    list(cfg.data.splits)
)

train, _ = event_split(
    samples,
    float(cfg.data.val_fraction),
    int(cfg.seed)
)

loader = make_dataloader(train, cfg, train=True)

batch = next(iter(loader))

pre = batch["pre"]
post = batch["post"]

print("Mean absolute difference:", (pre - post).abs().mean().item())
print("Fraction of identical values:", (pre == post).float().mean().item())