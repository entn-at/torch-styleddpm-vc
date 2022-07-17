from typing import Tuple

import torch
import torch.nn as nn

from .config import Config
from .classifier import Classifier


class Discriminator(nn.Module):
    """Discriminating the inputs.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: configurations.
        """
        super().__init__()
        self.proj_inputs = nn.Sequential(
            nn.Conv1d(config.mel, config.channels, 1),
            nn.ReLU())

        self.classifier = Classifier(
            config.domains,
            config.channels,
            config.kernels,
            config.stages,
            config.blocks)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classifying the inputs.
        Args:
            input: [torch.float32; [B, mel, T]], input signal.
        Returns:
            [torch.float32; [B, domains]], speaker domains.
            [torch.float32; [B]], expected mean pitches.
        """
        # [B, C, T]
        x = self.proj_inputs(inputs)
        # [B, domains], [B]
        return self.classifier(x)
