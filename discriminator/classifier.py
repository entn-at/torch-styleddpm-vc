from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from starddpm.resblock import ResidualBlock


class Classifier(nn.Module):
    """Classify the speakers and compute pitch frequencies.
    """
    def __init__(self,
                 domains: int,
                 channels: int,
                 kernels: int,
                 stages: int,
                 blocks: int):
        """Initializer.
        Args:
            domains: the number of the output domains.
            channels: size of the convolutional channels.
            kernels: size of the convolutional kernels.
            stages: the number of the residual blocks.
            blocks: the number of the convolution blocks before residual connection.
        """
        super().__init__()
        self.domains = domains
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, kernels, blocks)
            for _ in range(stages)])

        self.neck = ResidualBlock(channels, kernels, blocks)

        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, domains))

        self.pitchext = nn.Sequential(
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the style code.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensor, spectrogram.
            code: [torch.long; [B]], domain code.
        Returns:
            [torch.float32; [B, domains]], logits.
            [torch.float32; [B]], pitch frequencies.
        """
        x = inputs
        for block in self.blocks:
            # [B, C, T // 2 ** i]
            x = F.interpolate(block(x), scale_factor=0.5, mode='nearest')
        # [B, C]
        x = self.neck(x).mean(dim=-1)
        # [B, domains], [B]
        return self.proj(x), self.pitchext(x).squeeze(dim=1)
