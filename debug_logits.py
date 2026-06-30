from utils import load_config
from datasets.xbd_dataset import scan_xbd
from datasets.splitting import event_split
from datasets.dataloaders import make_dataloader
from models import build_model
import torch

cfg = load_config("configs/daapfl_ra.yaml")
cfg.train.batch_size = 2
cfg.train.num_workers = 0

samples = scan_xbd(cfg.data.raw_root, list(cfg.data.splits))
train, val = event_split(
    samples,
    float(cfg.data.val_fraction),
    int(cfg.seed)
)

loader = make_dataloader(val, cfg, train=False)

batch = next(iter(loader))

model = build_model(cfg).cuda()

ckpt = torch.load(
    "outputs/checkpoints/last.pt",
    map_location="cuda"
)

model.load_state_dict(ckpt["model"])

model.eval()

with torch.no_grad():

    logits = model(
        batch["pre"].cuda(),
        batch["post"].cuda()
    )

print("Shape:", logits.shape)

for c in range(5):
    x = logits[:, c]
    print(
        f"class {c}",
        "mean=", x.mean().item(),
        "std=", x.std().item(),
        "max=", x.max().item(),
        "min=", x.min().item(),
    )