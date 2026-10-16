"""Spatial stream corresponding to Eqs. (22)--(24)."""

from torch import Tensor, nn

# Internal residual blocks, initialization and checkpoint adapters are omitted.
from rp_stds_dtr.backbones.residual import ResidualStem, make_residual_stage


class DilatedResNet18(nn.Module):
    """Modified ResNet-18 with a stride-1, dilation-2 fourth stage."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.stem = ResidualStem(in_channels, channels=64)
        self.layer1 = make_residual_stage(64, 64, blocks=2, stride=1)
        self.layer2 = make_residual_stage(64, 128, blocks=2, stride=2)
        self.layer3 = make_residual_stage(128, 256, blocks=2, stride=2)

        # Preserve small-target detail: remove the final downsampling operation.
        self.layer4 = make_residual_stage(
            256, 512, blocks=2, stride=1, dilation=2, padding=2
        )

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != self.in_channels:
            raise ValueError("image must have shape (B, in_channels, H, W)")
        feature = self.stem(image)
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            feature = stage(feature)
        if image.shape[-2:] == (256, 256) and feature.shape[1:] != (512, 16, 16):
            raise RuntimeError("a 256 x 256 frame must yield a 512 x 16 x 16 feature")
        return feature


# Omitted from the public excerpt:
# pretrained-weight conversion, initialization utilities and feature diagnostics.
