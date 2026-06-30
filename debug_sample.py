import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import load_config
from datasets.xbd_dataset import scan_xbd, XBDDataset
from preprocessing.cache import CacheWriter

# ---------------------------------------------------------
# Load config
# ---------------------------------------------------------

cfg = load_config("configs/base.yaml")

# ---------------------------------------------------------
# Build dataset
# ---------------------------------------------------------

samples = scan_xbd(
    cfg.data.raw_root,
    list(cfg.data.splits)
)

print(f"Total samples : {len(samples)}")

dataset = XBDDataset(
    samples,
    cfg,
    train=False
)

print(f"Total cached tiles : {len(dataset)}")

# ---------------------------------------------------------
# Read FIRST cached tile directly
# ---------------------------------------------------------

rec = dataset.index[0]

cache_path = os.path.join(
    cfg.data.cache_root,
    "tiles",
    f"{rec['key']}.npz"
)

print("\nCache file:")
print(cache_path)

data = CacheWriter.read(cache_path)

pre = data["pre"]
post = data["post"]
mask = data["mask"]

# ---------------------------------------------------------
# Print statistics
# ---------------------------------------------------------

print("\n========== PRE ==========")
print("Shape :", pre.shape)
print("Type  :", pre.dtype)
print("Min   :", pre.min())
print("Max   :", pre.max())
print("Mean  :", pre.mean())

print("\n========== POST ==========")
print("Shape :", post.shape)
print("Type  :", post.dtype)
print("Min   :", post.min())
print("Max   :", post.max())
print("Mean  :", post.mean())

print("\n========== DIFFERENCE ==========")
print("Mean Absolute Difference :",
      np.abs(pre.astype(np.float32) -
             post.astype(np.float32)).mean())

print("\n========== MASK ==========")
print("Shape :", mask.shape)
print("Type  :", mask.dtype)
print("Classes :", np.unique(mask))


print("PRE")
print("dtype :", pre.dtype)
print("shape :", pre.shape)
print("min   :", pre.min())
print("max   :", pre.max())

print("\nPOST")
print("dtype :", post.dtype)
print("shape :", post.shape)
print("min   :", post.min())
print("max   :", post.max())

# ---------------------------------------------------------
# Overlay
# ---------------------------------------------------------

overlay = pre.copy()

if overlay.dtype != np.uint8:
    overlay = overlay.astype(np.uint8)

overlay = overlay.copy()

colors = {
    1: (0,255,0),      # green
    2: (255,255,0),    # cyan
    3: (255,128,0),    # orange
    4: (255,0,0),      # red
}

for cls, color in colors.items():
    overlay[mask == cls] = color

# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

fig, ax = plt.subplots(1,4, figsize=(20,5))

ax[0].imshow(pre)
ax[0].set_title("Pre")

ax[1].imshow(post)
ax[1].set_title("Post")

ax[2].imshow(mask)
ax[2].set_title("Mask")

ax[3].imshow(overlay)
ax[3].set_title("Overlay")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()

