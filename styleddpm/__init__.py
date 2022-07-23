from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .embedder import Embedder
from .encoder import StyleEncoder
from .scheduler import Scheduler
from .unet import UNet


class StyleDDPMVC(nn.Module):
    """Style-based DDPM for Voice conversion.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: configurations.
        """
        super().__init__()
        self.steps = config.steps
        self.proj_inputs = nn.Sequential(
            nn.Conv1d(config.mel * 2, config.channels, 1),
            nn.ReLU())

        self.embedder = Embedder(
            config.pe, config.embeddings, config.steps, config.mappings)

        self.encoder = StyleEncoder(
            config.mel, config.styles, config.channels, config.kernels,
            config.style_stages, config.style_blocks)

        self.unet = UNet(
            config.channels, config.kernels, config.embeddings, config.styles,
            config.stages, config.blocks)

        self.proj_outputs = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(config.channels, config.mel, 1))

        self.scheduler = Scheduler(
            config.steps, config.internals, config.logit_min, config.logit_max)
    
    def forward(self,
                context: torch.Tensor,
                styles: torch.Tensor,
                signal: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Generated waveform conditioned on mel-spectrogram.
        Args:
            context: [torch.float32; [B, mel, T]], context mel-spectrogram.
            styles: [torch.float32; [B, mel, T] or [B, styles]], style vectors.
            signal: [torch.float32; [B, mel, T]], initial noise.
        Returns:
            [torch.float32; [B, mel, T]], denoised result.
            S x [np.float32; [B, mel, T]], internal representations.
        """
        # [B, mel, T]
        signal = signal or torch.randn_like(context)
        if styles.dim() == 3:
            # [B, styles]
            _, styles = self.encoder(styles)
        # S x [B, mel, T]
        ir = [signal.cpu().detach().numpy()]
        # zero-based step
        for step in range(self.steps - 1, -1, -1):
            # [1]
            step = torch.tensor([step], device=signal.device)
            # [B, mel, T], [B]
            mean, std = self.inverse(signal, context, styles, step)
            # [B, mel, T]
            signal = mean + torch.randn_like(mean) * std[:, None, None]
            ir.append(signal.cpu().detach().numpy())
        # [B, mel, T]
        return signal, ir

    def diffusion(self,
                  signal: torch.Tensor,
                  steps: torch.Tensor,
                  next_: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffusion process.
        Args:
            signal: [torch.float32; [B, mel, T]], input signal.
            steps: [torch.long; [B]], t, target diffusion steps, zero-based.
            next_: whether move single steps or multiple steps.
                if next_, signal is z_{t - 1}, otherwise signal is z_0.
        Returns:
            [torch.float32; [B, mel, T]], z_{t}, diffused mean.
            [torch.float32; [B]], standard deviation.
        """
        # [S + 1]
        logsnr, betas = self.scheduler()
        if next_:
            # [B], one-based sample
            beta = betas[steps + 1]
            # [B, mel, T], [B]
            return (1. - beta[:, None, None]).sqrt() * signal, beta.sqrt()
        # [S + 1]
        alphas_bar = torch.sigmoid(logsnr)
        # [B], one-based sample
        alpha_bar = alphas_bar[steps + 1]
        # [B, mel, T], [B]
        return alpha_bar[:, None, None].sqrt() * signal, (1 - alpha_bar).sqrt()

    def inverse(self,
                signal: torch.Tensor,
                context: torch.Tensor,
                styles: torch.Tensor,
                steps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse process, single step denoise.
        Args:
            signal: [torch.float32; [B, mel, T]], input signal, z_{t}.
            context: [torch.float32; [B, mel, T]], context spectrogram.
            styles: [torch.float32; [B, styles]], style vector.
            steps: [torch.long; [B]], t, diffusion steps, zero-based.
        Returns:
            [torch.float32; [B, mel, T]], waveform mean, z_{t - 1}
            [torch.float32; [B]], waveform std.
        """
        # [S + 1]
        logsnr, betas = self.scheduler()
        # [S + 1]
        alphas, alphas_bar = 1. - betas, torch.sigmoid(logsnr)
        # [B, mel, T]
        denoised = self.denoise(signal, context, styles, steps)
        # [B], make one-based
        prev, steps = steps, steps + 1
        # [B, mel, T]
        mean = alphas_bar[prev, None, None].sqrt() * betas[steps, None, None] / (
                1 - alphas_bar[steps, None, None]) * denoised + \
            alphas[steps, None, None].sqrt() * (1. - alphas_bar[prev, None, None]) / (
                1 - alphas_bar[steps, None, None]) * signal
        # [B]
        var = (1 - alphas_bar[prev]) / (1 - alphas_bar[steps]) * betas[steps]
        return mean, var.sqrt()

    def denoise(self,
                signal: torch.Tensor,
                context: torch.Tensor,
                styles: torch.Tensor,
                steps: torch.Tensor) -> torch.Tensor:
        """Denoise the signal w.r.t. outpart signal.
        Args:
            signal: [torch.float32; [B, mel, T]], input signal.
            context: [torch.float32; [B, mel, T]], context spectrogram.
            styles: [torch.float32; [B, styles]], style vector.
            steps: [torch.long; [B]], diffusion steps, zero-based.
        Returns:
            [torch.float32; [B, mel, T]], denoised signal.
        """
        # [B, C, T]
        x = self.proj_inputs(torch.cat([signal, context], dim=1))
        # [B, E]
        embed = self.embedder(steps)
        # [B, C, T]
        x = self.unet(x, embed, styles)
        # [B, mel, T]
        return self.proj_outputs(x)
