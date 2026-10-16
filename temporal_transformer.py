"""Adaptive multiscale temporal stream corresponding to Eqs. (11)--(21)."""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Project implementations of window attention and the temporal encoder are omitted.
from rp_stds_dtr.layers.temporal import TemporalEncoder, WindowSelfAttention


SCALES = (2, 4, 6, 8, 10, 12, 20, 24, 30, 40)
DEFAULT_SEQUENCE_LENGTH = 10080


class AdaptiveMultiscaleTemporal(nn.Module):
    def __init__(
        self, classes: int = 3, width: int = 64,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ) -> None:
        super().__init__()
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.scales = SCALES
        self.embedding = nn.Linear(1, width)
        self.local_attention = WindowSelfAttention(width)
        self.scale_logits = nn.Parameter(torch.randn(classes, len(SCALES)))
        self.position = nn.Parameter(torch.zeros(1, sequence_length, width))
        self.encoder = TemporalEncoder(width)

    def forward(
        self, radiation: Tensor, retained: Optional[Sequence[int]] = None
    ) -> tuple[Tensor, Tensor]:
        if radiation.ndim != 2 or radiation.shape[1] != self.position.shape[1]:
            raise ValueError("radiation must have shape (B, sequence_length)")
        x = self.embedding(radiation.unsqueeze(-1))                # Eq. (11)
        indices = tuple(range(len(self.scales))) if retained is None else tuple(retained)
        if not indices or any(i < 0 or i >= len(self.scales) for i in indices):
            raise ValueError("retained must contain valid scale indices")
        if any(radiation.shape[1] % self.scales[i] for i in indices):
            raise ValueError("sequence_length must be divisible by each retained scale")

        # Each branch uses non-overlapping windows: stride equals window length.
        branches = torch.stack([
            self.local_attention(x, window=self.scales[i], stride=self.scales[i])
            for i in indices
        ], dim=1)                                                  # Eqs. (12)--(15)

        weights = F.softmax(self.scale_logits[:, list(indices)], dim=-1)
        class_features = torch.einsum("ck,bktd->bctd", weights, branches)
        encoded = self.encoder(class_features, self.position)      # Eqs. (20)--(21)
        return encoded, weights


# Omitted: padding masks, variable-length batching and attention diagnostics.
