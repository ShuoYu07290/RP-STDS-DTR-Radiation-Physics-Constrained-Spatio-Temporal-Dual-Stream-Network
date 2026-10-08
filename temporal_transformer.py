"""Selected adaptive multiscale temporal-stream excerpt, Eqs. (16)--(26)."""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Project implementations of window attention and the temporal encoder are omitted.
from rp_stds_dtr.layers.temporal import TemporalEncoder, WindowSelfAttention


SCALES = (2, 4, 6, 8, 10, 12, 20, 24, 30, 40)


class AdaptiveMultiscaleTemporal(nn.Module):
    def __init__(self, classes: int = 3, width: int = 64) -> None:
        super().__init__()
        self.scales = SCALES
        self.embedding = nn.Linear(1, width)
        self.local_attention = WindowSelfAttention(width)
        self.scale_logits = nn.Parameter(torch.randn(classes, len(SCALES)))
        self.position = nn.Parameter(torch.zeros(1, 100, width))
        self.encoder = TemporalEncoder(width)

    def forward(
        self, radiation: Tensor, retained: Optional[Sequence[int]] = None
    ) -> tuple[Tensor, Tensor]:
        x = self.embedding(radiation.unsqueeze(-1))                # Eq. (16)
        indices = tuple(range(len(self.scales))) if retained is None else retained

        # Each branch uses non-overlapping windows: stride equals window length.
        branches = torch.stack([
            self.local_attention(x, window=self.scales[i], stride=self.scales[i])
            for i in indices
        ], dim=1)                                                  # Eqs. (17)--(20)

        weights = F.softmax(self.scale_logits[:, list(indices)], dim=-1)
        class_features = torch.einsum("ck,bktd->bctd", weights, branches)
        encoded = self.encoder(class_features, self.position)      # Eqs. (25)--(26)
        return encoded, weights


# Omitted: padding masks, variable-length batching and attention diagnostics.
