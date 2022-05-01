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


class UNet(nn.Module):
    """Spectrogram U-Net for noise estimator.
    """
    def __init__(self, channels: int, stages: int):
        super().__init__()
