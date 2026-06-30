"""U-Net decoder built on smp's decoder block, taking a fused feature pyramid.
This is the PRIVATE / personalized component in DAAPFL-RA & FedPer/FedRep."""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class UNetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=list(decoder_channels),
            n_blocks=len(decoder_channels),
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:

        x = self.decoder(*features)
        x = self.dropout(x)

        return x

class SegmentationHead(nn.Module):
    """
    Original single-task segmentation head.
    Kept for backward compatibility and ablation studies.
    """

    def __init__(self, in_channels: int, num_classes: int, upsample: int = 1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=3,
            padding=1
        )

        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=upsample)
            if upsample > 1 else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        return self.up(self.conv(x))


class LocalizationHead(nn.Module):
    """
    Binary building localization.

    Output:
        Channel 0 -> Background
        Channel 1 -> Building
    """
    def __init__(self, in_channels: int, upsample: int = 1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            2,
            kernel_size=3,
            padding=1
        )

        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=upsample)
            if upsample > 1 else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        return self.up(self.conv(x))


class DamageHead(nn.Module):
    """
    Damage classification.

    Output channels

    0 -> No Damage
    1 -> Minor Damage
    2 -> Major Damage
    3 -> Destroyed
    """

    def __init__(self, in_channels: int, upsample: int = 1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            4,
            kernel_size=3,
            padding=1
        )

        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=upsample)
            if upsample > 1 else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        return self.up(self.conv(x))