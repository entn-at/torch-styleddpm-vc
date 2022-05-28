from typing import List

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Convolutional residual block with auxiliary contexts.
    """
    def __init__(self, channels: int, kernels: int, aux: int):
        """Initializer.
        Args:
            channels: size of the convolutional channels.
            kernels: size of the convolutional kernels.
            aux: size of the auxiliary contexts.
        """
        super().__init__()
        self.preblock = nn.Sequential(
            nn.Conv1d(channels, channels, kernels, padding=kernels // 2),
            nn.ReLU(),
            nn.BatchNorm1d(channels))

        self.proj = nn.Linear(aux, channels, bias=False)

        self.postblock = nn.Sequential(
            nn.Conv1d(channels, channels, kernels, padding=kernels // 2),
            nn.ReLU(),
            nn.BatchNorm1d(channels))

    def forward(self, inputs: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input 1D feature map.
            aux: [torch.float32; [B, E]], auxiliary embedding.
        Returns:
            [torch.float32; [B, C, T]], residually connected.
        """
        return inputs + self.postblock(self.preblock(inputs) + self.proj(aux)[..., None])


class AuxSequential(nn.Module):
    """Sequential wrapper for auxiliary input passing.
    """
    def __init__(self, lists: List[nn.Module]):
        """Initializer.
        Args:
            lists: module lists.
        """
        super().__init__()
        self.lists = nn.ModuleList(lists)

    def forward(self, inputs: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Chaining outputs with auxiliary inputs.
        """
        x = inputs
        for module in self.lists:
            x = module(x, aux)
        return x


class UNet(nn.Module):
    """Spectrogram U-Net for noise estimator.
    """
    def __init__(self, channels: int, kernels: int, aux: int, stages: int, blocks: int):
        """Initializer.
        Args:
            channels: size of the hidden channels.
            kernels: size of the convolutional kernels.
            aux: size of the auxiliary channels.
            stages: the number of the resolution scales.
            blocks: the number of the residual blocks in each stages.
        """
        super().__init__()
        self.dblocks = nn.ModuleList([
            AuxSequential([
                ResidualBlock(channels * 2 ** i, kernels, aux)
                for _ in range(blocks)])
            for i in range(stages - 1)])

        self.downsamples = nn.ModuleList([
            # half resolution
            nn.Conv1d(channels * 2 ** i, channels * 2 ** (i + 1), kernels, 2, padding=kernels // 2)
            for i in range(stages - 1)])

        self.neck = ResidualBlock(channels * 2 ** (stages - 1), kernels, aux)

        self.upsamples = nn.ModuleList([
            # double resolution
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels * 2 ** i, channels * 2 ** (i - 1), kernels, padding=kernels // 2))
            for i in range(stages - 1, 0, -1)])

        self.ublocks = nn.ModuleList([
            AuxSequential([
                ResidualBlock(channels * 2 ** i, kernels, aux)
                for _ in range(blocks)])
            for i in range(stages - 2, -1, -1)])

    def forward(self, inputs: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Spectrogram U-net.
        Args:
            inputs: [torch.float32; [B, C(=channels), T]], input tensor, spectrogram.
            aux: [torch.float32; [B, A(=aux)]], auxiliary informations, times.
        Returns:
            [torch.float32; [B, C, T]], transformed.
        """
        # [B, C, T]
        x = inputs
        # (stages - 1) x [B, C x 2^i, T / 2^i]
        internals = []
        for dblock, downsample in zip(self.dblocks, self.downsamples):
            # [B, C x 2^i, T / 2^i]
            x = dblock(x, aux)
            internals.append(x)
            # [B, C x 2^(i + 1), T / 2^(i + 1)]
            x = downsample(x)
        # [B, C x 2^stages, T / 2^stages]
        x = self.neck(x, aux)
        for i, ublock, upsample in zip(reversed(internals), self.ublocks, self.upsamples):
            # [B, C x 2^i, T / 2^i]
            x = ublock(upsample(x) + i, aux)
        # [B, C, T]
        return x


if __name__ == '__main__':
    # Test for U-net
    def test():
        unet = UNet(16, 3, 8, 5, 2)
        inputs = torch.randn(2, 16, 64)
        aux = torch.randn(2, 8)
        assert unet(inputs, aux).shape == inputs.shape

    test()
