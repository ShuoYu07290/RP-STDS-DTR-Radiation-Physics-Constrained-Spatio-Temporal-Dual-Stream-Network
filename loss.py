"""Physics-guided objective excerpt for Eqs. (24) and (37)--(45)."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Label mapping and numerical-stability helpers belong to the internal project.
from rp_stds_dtr.data.labels import DEBRIS, DECOY, TARGET


class PhysicsGuidedObjective(nn.Module):
    def __init__(
        self,
        gamma: float = 0.15,
        delta: float = 0.45,
        lambda_time: float = 0.4,
        lambda_frequency: float = 0.4,
        lambda_entropy: float = 0.2,
    ) -> None:
        super().__init__()
        self.gamma, self.delta = gamma, delta
        self.lambda_time = lambda_time
        self.lambda_frequency = lambda_frequency
        self.lambda_entropy = lambda_entropy

    def forward(
        self, logits: Tensor, labels: Tensor,
        response: Tensor, scale_weights: Tensor,
    ) -> Tensor:
        difference = response[:, 1:] - response[:, :-1]
        variance = difference.var(dim=1, unbiased=False)           # Eqs. (37)--(39)
        time_penalty = torch.where(
            labels == TARGET, variance,
            torch.where(labels == DECOY,
                        (variance - self.gamma).square(),
                        F.relu(self.gamma - variance).square()),
        )                                                         # Eq. (40)

        spectrum = torch.fft.rfft(response, dim=1).abs()
        concentration = spectrum.amax(1) / spectrum.sum(1).clamp_min(1e-8)
        frequency_penalty = torch.where(
            labels == TARGET, F.relu(self.delta - concentration).square(),
            torch.where(labels == DECOY,
                        (concentration - self.delta).square(),
                        concentration.square()),
        )                                                         # Eqs. (41)--(42)

        entropy = (scale_weights.clamp_min(1e-8) *
                   scale_weights.clamp_min(1e-8).log()).sum(1).mean()
        auxiliary = (self.lambda_time * time_penalty.mean()
                     + self.lambda_frequency * frequency_penalty.mean()
                     + self.lambda_entropy * entropy)             # Eqs. (43)--(44)
        return F.cross_entropy(logits, labels) + auxiliary         # Eq. (45)


# Omitted: loss logging, distributed reduction and ablation switches.
