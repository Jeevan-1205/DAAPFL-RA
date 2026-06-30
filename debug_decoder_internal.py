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

model = build_model(cfg).cuda().eval()

pre = batch["pre"].cuda()
post = batch["post"].cuda()

fa = model.encoder(pre)
fb = model.encoder(post)
fused = model._fuse(fa, fb)

decoder = model.decoder.decoder

# Patch every decoder block
for idx, block in enumerate(decoder.blocks):
    original = block.forward

    def make_forward(i, orig):
        def wrapped(x, skip=None):
            y = orig(x, skip)
            print(
                f"Block {i}:",
                "mean =", y.mean().item(),
                "std =", y.std().item(),
                "max =", y.max().item(),
                "min =", y.min().item(),
            )
            return y
        return wrapped

    block.forward = make_forward(idx, original)

with torch.no_grad():
    out = decoder(*fused)

print("Final:", out.mean().item(), out.std().item())