"""Siamese U-Net for building damage assessment.

Pipeline:
    pre  --shared encoder-->  feats_pre
    post --shared encoder-->  feats_post
    fused = fuse(feats_pre, feats_post)        # diff | absdiff | concat
    fused --private decoder--> dec
    dec   --seg head-->        logits (B, 5, H, W)

Exposed helpers required by the project:
    - shared_encoder_state()  : encoder params (aggregated across clients)
    - private_decoder_state() : decoder+head params (kept local)
    - extract_bottleneck()    : deepest post-pre feature for prototypes
"""
from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder.resnet_encoder import ResNetEncoder
from models.decoder.unet_decoder import UNetDecoder, SegmentationHead


class SiameseUNet(nn.Module):
    def __init__(self, encoder: str = "resnet50", encoder_weights: str = "imagenet",
                 in_channels: int = 3, num_classes: int = 5,
                 decoder_channels=(256, 128, 64, 32, 16),
                 fusion: str = "diff", dropout: float = 0.1):
        super().__init__()
        self.fusion = fusion
        self.num_classes = num_classes

        # ---- shared encoder (aggregated) ----
        self.encoder = ResNetEncoder(encoder, encoder_weights, in_channels)
        enc_ch = self.encoder.out_channels

        # concat fusion doubles channels per level
        if fusion == "concat":
            fused_ch = [c * 2 for c in enc_ch]
        else:
            fused_ch = list(enc_ch)

        # ---- private decoder + head (personalized) ----
        self.decoder = UNetDecoder(fused_ch, decoder_channels, dropout)
        self.segmentation_head = SegmentationHead(decoder_channels[-1], num_classes)

        self._bottleneck_dim = enc_ch[-1] if fusion != "concat" else enc_ch[-1] * 2

    # ---------------- fusion ----------------
    def _fuse(self, fa: List[torch.Tensor], fb: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.fusion == "diff":
            return [b - a for a, b in zip(fa, fb)]
        if self.fusion == "absdiff":
            return [torch.abs(b - a) for a, b in zip(fa, fb)]
        if self.fusion == "concat":
            return [torch.cat([a, b], dim=1) for a, b in zip(fa, fb)]
        raise ValueError(f"unknown fusion '{self.fusion}'")

    # ---------------- forward ----------------
    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        fa = self.encoder(pre)
        fb = self.encoder(post)
        fused = self._fuse(fa, fb)
        dec = self.decoder(fused)
        logits = self.segmentation_head(dec)
        if logits.shape[-2:] != pre.shape[-2:]:
            logits = F.interpolate(logits, size=pre.shape[-2:],
                                   mode="bilinear", align_corners=False)
        return logits

    # ------------- feature extraction for prototypes -------------
    @torch.no_grad()
    def extract_bottleneck(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Return (B, C) globally-pooled deepest fused feature."""
        fa = self.encoder(pre)
        fb = self.encoder(post)
        fused = self._fuse(fa, fb)[-1]
        return F.adaptive_avg_pool2d(fused, 1).flatten(1)

    @property
    def bottleneck_dim(self) -> int:
        return self._bottleneck_dim

    # ------------- shared / private separation -------------
    def shared_encoder_state(self) -> Dict[str, torch.Tensor]:
        return {f"encoder.{k}": v for k, v in self.encoder.state_dict().items()}

    def private_decoder_state(self) -> Dict[str, torch.Tensor]:
        out = {f"decoder.{k}": v for k, v in self.decoder.state_dict().items()}
        out.update({f"segmentation_head.{k}": v
                    for k, v in self.segmentation_head.state_dict().items()})
        return out

    def load_shared_encoder(self, state: Dict[str, torch.Tensor]) -> None:
        clean = {k.replace("encoder.", "", 1): v for k, v in state.items()}
        self.encoder.load_state_dict(clean, strict=True)


def build_model(cfg) -> SiameseUNet:
    m = cfg.model
    return SiameseUNet(
        encoder=m.encoder,
        encoder_weights=m.encoder_weights,
        in_channels=int(m.in_channels),
        num_classes=int(m.num_classes),
        decoder_channels=tuple(m.decoder_channels),
        fusion=m.fusion,
        dropout=float(m.dropout),
    )
