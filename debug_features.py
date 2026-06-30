import torch
from utils import load_config
from models import build_model
from datasets.xbd_dataset import scan_xbd
from datasets.splitting import event_split
from datasets.dataloaders import make_dataloader

cfg = load_config("configs/daapfl_ra.yaml")

cfg.train.batch_size = 1
cfg.train.num_workers = 0

samples = scan_xbd(cfg.data.raw_root, list(cfg.data.splits))
train, _ = event_split(samples, float(cfg.data.val_fraction), int(cfg.seed))

loader = make_dataloader(train, cfg, train=True)

batch = next(iter(loader))

model = build_model(cfg).cuda()

ckpt = torch.load("outputs/checkpoints/last.pt")

model.load_state_dict(ckpt["model"])

model.eval()

with torch.no_grad():

    fa = model.encoder(batch["pre"].cuda())
    fb = model.encoder(batch["post"].cuda())

print()

for i, f in enumerate(fa):

    print(
        "Encoder",
        i,
        f.shape,
        "mean",
        f.mean().item(),
        "std",
        f.std().item()
    )