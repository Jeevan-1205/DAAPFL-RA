"""Grad-CAM for the Siamese U-Net. We hook the last encoder stage applied to
the POST image and attribute w.r.t. the damage-class score, producing a
heatmap that explains where the model 'looks' for destruction evidence.

Generates side-by-side comparison figures across methods/checkpoints.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SiameseGradCAM:
    def __init__(self, model, target_layer: Optional[torch.nn.Module] = None,
                 device: str = "cuda"):
        self.model = model.eval().to(device)
        self.device = device
        self.activations = None
        self.gradients = None
        layer = target_layer or self._default_layer()
        layer.register_forward_hook(self._fwd_hook)
        layer.register_full_backward_hook(self._bwd_hook)

    def _default_layer(self):
        # deepest conv block of the shared resnet encoder
        enc = self.model.encoder.encoder
        for name in ["layer4", "stages", "blocks"]:
            if hasattr(enc, name):
                return getattr(enc, name)[-1] if hasattr(getattr(enc, name), "__getitem__") \
                    else getattr(enc, name)
        # fallback: last module with parameters
        last = None
        for m in enc.modules():
            if isinstance(m, torch.nn.Conv2d):
                last = m
        return last

    def _fwd_hook(self, m, i, o):
        self.activations = o.detach()

    def _bwd_hook(self, m, gi, go):
        self.gradients = go[0].detach()

    def __call__(self, pre: torch.Tensor, post: torch.Tensor,
                 target_class: int = 4) -> np.ndarray:
        pre = pre.to(self.device); post = post.to(self.device)
        self.model.zero_grad()
        logits = self.model(pre, post)                       # (1,C,H,W)
        score = logits[:, target_class].mean()
        score.backward()
        # GAP over gradients -> channel weights
        w = self.gradients.mean(dim=(2, 3), keepdim=True)    # (1,K,1,1)
        cam = F.relu((w * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=pre.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def _denorm(img: torch.Tensor, mean, std) -> np.ndarray:
    x = img.cpu().numpy().transpose(1, 2, 0)
    x = x * np.asarray(std) + np.asarray(mean)
    return np.clip(x, 0, 1)


def gradcam_comparison_figure(models: Dict[str, torch.nn.Module],
                              pre: torch.Tensor, post: torch.Tensor,
                              mean, std, target_class: int = 4,
                              out_path: str = "outputs/figures/gradcam.png",
                              device: str = "cuda"):
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(models) + 2
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    axes[0].imshow(_denorm(pre[0], mean, std)); axes[0].set_title("pre"); axes[0].axis("off")
    axes[1].imshow(_denorm(post[0], mean, std)); axes[1].set_title("post"); axes[1].axis("off")
    base = _denorm(post[0], mean, std)
    for ax, (name, model) in zip(axes[2:], models.items()):
        cam = SiameseGradCAM(model, device=device)(pre[:1], post[:1], target_class)
        ax.imshow(base); ax.imshow(cam, cmap="jet", alpha=0.5)
        ax.set_title(f"{name}"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
