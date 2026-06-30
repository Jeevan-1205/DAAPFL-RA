import torch

from utils import load_config
from models import build_model
from datasets.xbd_dataset import scan_xbd
from datasets.splitting import event_split
from datasets.dataloaders import make_dataloader

# -----------------------
# Config
# -----------------------
cfg = load_config("configs/daapfl_ra.yaml")

cfg.train.batch_size = 1
cfg.train.num_workers = 0

# -----------------------
# Dataset
# -----------------------
samples = scan_xbd(
    cfg.data.raw_root,
    list(cfg.data.splits)
)

train, _ = event_split(
    samples,
    float(cfg.data.val_fraction),
    int(cfg.seed)
)

loader = make_dataloader(
    train,
    cfg,
    train=True
)

batch = next(iter(loader))

# -----------------------
# Model
# -----------------------
model = build_model(cfg).cuda()

#ckpt = torch.load(
#    "outputs/checkpoints/last.pt",
#    map_location="cuda"
#)

#model.load_state_dict(ckpt["model"])
#model.eval()

# -----------------------
# Forward inspection
# -----------------------
with torch.no_grad():

    pre = batch["pre"].cuda()
    post = batch["post"].cuda()

    fa = model.encoder(pre)
    fb = model.encoder(post)

    fused = model._fuse(fa, fb)

    print("=" * 60)
    print("FUSED FEATURES")
    print("=" * 60)

    for i, f in enumerate(fused):
        print(
            f"Fused {i}",
            "shape =", tuple(f.shape),
            "mean =", f.mean().item(),
            "std =", f.std().item(),
            "min =", f.min().item(),
            "max =", f.max().item(),
        )

    print("\n" + "=" * 60)
    print("DECODER OUTPUT")
    print("=" * 60)

    dec = model.decoder(fused)

    print(
        "shape =", tuple(dec.shape),
        "mean =", dec.mean().item(),
        "std =", dec.std().item(),
        "min =", dec.min().item(),
        "max =", dec.max().item(),
    )

    print("\n" + "=" * 60)
    print("SEGMENTATION LOGITS")
    print("=" * 60)

    logits = model.segmentation_head(dec)

    print(
        "shape =", tuple(logits.shape),
        "mean =", logits.mean().item(),
        "std =", logits.std().item(),
        "min =", logits.min().item(),
        "max =", logits.max().item(),
    )

    print("\nPer-class statistics:")

    for c in range(5):
        x = logits[:, c]
        print(
            f"Class {c}:",
            "mean =", x.mean().item(),
            "std =", x.std().item(),
            "min =", x.min().item(),
            "max =", x.max().item(),
        )