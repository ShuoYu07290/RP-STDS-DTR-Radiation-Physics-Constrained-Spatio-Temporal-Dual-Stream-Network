"""Selected spatial-stream excerpt from the RP-STDS-DTR project."""

from torch import Tensor, nn

# Internal residual blocks, initialization and checkpoint adapters are omitted.
from rp_stds_dtr.backbones.residual import ResidualStem, make_residual_stage


class DilatedResNet18(nn.Module):
    """ResNet-18 with a non-downsampling dilated fourth stage (Eq. 29)."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.stem = ResidualStem(in_channels, channels=64)
        self.layer1 = make_residual_stage(64, 64, blocks=2, stride=1)
        self.layer2 = make_residual_stage(64, 128, blocks=2, stride=2)
        self.layer3 = make_residual_stage(128, 256, blocks=2, stride=2)

        # Preserve small-target detail: remove the final downsampling operation.
        self.layer4 = make_residual_stage(
            256, 512, blocks=2, stride=1, dilation=2, padding=2
        )

    def forward(self, image: Tensor) -> Tensor:
        feature = self.stem(image)
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            feature = stage(feature)
        return feature


# Omitted from the public excerpt:
# pretrained-weight conversion, initialization utilities and feature diagnostics.
