"""ResNet encoder wrapper (via segmentation_models_pytorch) returning the
multi-scale feature pyramid and the deepest bottleneck feature used for
prototypes. Shared (aggregated) component in personalized FL."""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ResNetEncoder(nn.Module):
    def __init__(self, name: str = "resnet50", weights: str = "imagenet",
                 in_channels: int = 3, depth: int = 5):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            name, in_channels=in_channels, depth=depth, weights=weights
        )
        self.out_channels: List[int] = list(self.encoder.out_channels)

    @property
    def bottleneck_dim(self) -> int:
        return self.out_channels[-1]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.encoder(x)  # list of feature maps, shallow -> deep
