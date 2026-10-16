"""Core dual-stream fusion corresponding to Eqs. (25)--(30)."""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Public excerpt modules in this folder.
from dilated_resnet import DilatedResNet18
from temporal_transformer import AdaptiveMultiscaleTemporal


class RPSTDSTR(nn.Module):
    def __init__(
        self, classes: int = 3, width: int = 256,
        sequence_length: int = 10080,
    ) -> None:
        super().__init__()
        self.spatial = DilatedResNet18(in_channels=1)
        self.temporal = AdaptiveMultiscaleTemporal(
            classes=classes, sequence_length=sequence_length
        )
        self.project_s = nn.Conv2d(512, width, 1, bias=False)
        self.project_t = nn.Linear(64, width, bias=False)
        self.channel_gate = nn.Linear(width, width)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.fusion = nn.Linear(2 * width, width)
        self.decision = nn.Parameter(torch.empty(classes, width))
        self.decision_bias = nn.Parameter(torch.zeros(classes))
        self.response_head = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.decision)

    def forward(
        self, radiation: Tensor, image: Tensor,
        retained_scales: Optional[Sequence[int]] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        spatial = self.project_s(self.spatial(image))
        temporal, scale_weights = self.temporal(radiation, retained_scales)
        temporal_mean = temporal.mean(dim=2)                       # B x C x D
        temporal_projected = self.project_t(temporal_mean)

        gate = torch.sigmoid(self.channel_gate(temporal_projected))
        spatial = spatial[:, None] * gate[..., None, None]         # Eq. (26)
        spatial = spatial.mean(dim=(-1, -2))

        joint = torch.cat((self.alpha * spatial,
                           self.beta * temporal_projected), dim=-1)
        fused = torch.sigmoid(self.fusion(joint))                  # Eqs. (27)--(28)
        logits = (torch.einsum("bcd,cd->bc", fused, self.decision)
                  + self.decision_bias)                            # Eq. (29)

        posterior = F.softmax(logits, dim=-1)
        response = torch.einsum("bc,bctd->btd", posterior, temporal)
        response = self.response_head(response).squeeze(-1)
        return logits, scale_weights, response


# Omitted: deployment wrappers and additional diagnostics.
