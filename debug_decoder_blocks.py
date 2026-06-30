import torch

from utils import load_config
from datasets.xbd_dataset import scan_xbd
from datasets.splitting import event_split
from datasets.dataloaders import make_dataloader
from models import build_model

cfg = load_config("configs/daapfl_ra.yaml")
cfg.train.batch_size = 2
cfg.train.num_workers = 0

samples = scan_xbd(cfg.data.raw_root, list(cfg.data.splits))
train, _ = event_split(samples, float(cfg.data.val_fraction), int(cfg.seed))

loader = make_dataloader(train, cfg, train=True)
batch = next(iter(loader))

model = build_model(cfg).cuda().eval()

pre = batch["pre"].cuda()
post = batch["post"].cuda()

with torch.no_grad():

    fa = model.encoder(pre)
    fb = model.encoder(post)

    fused = [b - a for a, b in zip(fa, fb)]

    print("===== INPUT FEATURES =====")
    for i, f in enumerate(fused):
        print(i, f.mean().item(), f.std().item())

    # Pass through decoder block by block
    features = fused[::-1]

    head = features[0]
    skips = features[1:]

    x = head

    print("\n===== DECODER BLOCKS =====")

    for i, block in enumerate(model.decoder.decoder.blocks):

        skip = skips[i] if i < len(skips) else None

        x = block(x, skip)

        print(
            i,
            "mean =", x.mean().item(),
            "std =", x.std().item(),
            "max =", x.max().item()
        )