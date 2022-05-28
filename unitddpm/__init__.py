import torch
import torch.nn as nn

from .config import Config
from .embedder import Embedder
from .unet import UNet


class UnitDDPMVC(nn.Module):
    """UNIT-DDPM for Voice conversion.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: configurations.
        """
        super().__init__()
        self.proj_inputs = nn.Sequential(
            nn.Conv1d(config.mel * 2, config.channels, 1),
            nn.ReLU())

        self.embedder = Embedder(
            config.pe, config.embeddings, config.steps, config.mappings)

        self.unet = UNet(
            config.channels, config.kernels, config.embeddings,
            config.stages, config.blocks)

        self.proj_outputs = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(config.channels, config.mel, 1))

    def denoise(self,
                signal: torch.Tensor,
                outpart: torch.Tensor,
                steps: torch.Tensor) -> torch.Tensor:
        """Denoise the signal w.r.t. outpart signal.
        Args:
            signal: [torch.float32; [B, mel, T]], input signal.
            outpart: [torch.float32; [B, mel, T]], counter part signal.
            steps: [torch.long; [B]], diffusion steps, zero-based.
        Returns:
            [torch.float32; [B, mel, T]], denoised signal.
        """
        # [B, C, T]
        x = self.proj_inputs(torch.cat([signal, outpart], dim=1))
        # [B, E]
        embed = self.embedder(steps)
        # [B, C, T]
        x = self.unet(x, embed)
        # [B, mel, T]
        return self.proj_outputs(x)
