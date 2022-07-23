from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resblock import ResidualBlock


class StyleEncoder(nn.Module):
    """Encode the style token from inputs.
    """
    def __init__(self,
                 mel: int,
                 styles: int,
                 channels: int,
                 kernels: int,
                 stages: int,
                 blocks: int):
        """Initializer.
        Args:
            mel: size of the mel filter channels.
            styles: size of the style tokens.
            channels: size of the convolutional channels.
            kernels: size of the convolutional kernels.
            stages: the number of the residual blocks.
            blocks: the number of the convolution blocks before residual connection.
        """
        super().__init__()
        self.proj_inputs = nn.Sequential(
            nn.Conv1d(mel, channels, 1),
            nn.ReLU())

        self.blocks = nn.ModuleList([
            ResidualBlock(channels, kernels, blocks)
            for _ in range(stages)])

        self.neck = ResidualBlock(channels, kernels, blocks)

        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(channels, styles + 1))

    def forward(self, inputs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the style code.
        Args:
            inputs: [torch.float32; [B, mel, T]], input spectrogram.
        Returns:
            [torch.float32; [B]], average pitch level.
            [torch.float32; [B, styles]], style code.
        """
        x = self.proj_inputs(inputs)
        for block in self.blocks:
            # [B, C, T // 2 ** i]
            x = F.interpolate(block(x), scale_factor=0.5, mode='nearest')
        # [B, styles + 1], global average pool
        x = self.proj(self.neck(x).mean(dim=-1))
        # [B], [B, styles]
        return x[:, 0], F.normalize(x[:, 1:], p=2, dim=-1)
