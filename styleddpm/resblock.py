import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Convolutional residual block.
    """
    def __init__(self, channels: int, kernels: int, blocks: int):
        """Initializer.
        Args:
            channels: size of the convolutional channels.
            kernels: size of the convolutional kernels.
            blocks: the number of the convolution blocks before residual connection.
        """
        super().__init__()
        self.block = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(channels, channels, kernels, padding=kernels // 2),
                nn.ReLU(),
                nn.BatchNorm1d(channels))
            for _ in range(blocks)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input 1d feature map.
        Returns:
            [torch.float32; [B, C, T]], residually connected.
        """
        return inputs + self.block(inputs)


class AuxResidualBlock(nn.Module):
    """Convolutional residual block with auxiliary contexts.
    """
    def __init__(self, channels: int, kernels: int, aux: int, context: int):
        """Initializer.
        Args:
            channels: size of the convolutional channels.
            kernels: size of the convolutional kernels.
            aux: size of the auxiliary contexts.
            context: size of the context vector.
        """
        super().__init__()
        self.proj = nn.Linear(aux, channels, bias=False)
        self.preblock = nn.Sequential(
            nn.Conv1d(channels, channels, kernels, padding=kernels // 2),
            nn.ReLU(),
            nn.BatchNorm1d(channels))
        
        self.proj_context = nn.Conv1d(context, channels, 1, bias=False)

        self.postblock = nn.Sequential(
            nn.Conv1d(channels, channels, kernels, padding=kernels // 2),
            nn.ReLU(),
            nn.BatchNorm1d(channels))

    def forward(self, inputs: torch.Tensor, aux: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input 1D feature map.
            aux: [torch.float32; [B, E]], auxiliary embedding.
            context: [torch.float32; [B, mel, T]], context vectors.
        Returns:
            [torch.float32; [B, C, T]], residually connected.
        """
        # [B, C, T]
        ir = self.preblock(inputs + self.proj(aux)[..., None])
        # [B, C, T]
        context = self.proj_context(
            F.interpolate(context, size=ir.shape[-1], mode='nearest'))
        return inputs + self.postblock(ir + context)


class ModulatedConv1d(nn.Module):
    """Conv1d with weight modulation.
    """
    def __init__(self, channels: int, kernels: int, styles: int):
        """Initializer.
        Args:
            channels: size of the convolutional channels.
            kernels: size of the convolutional kernels.
            styles: size of the style vector.
        """
        super().__init__()
        self.kernels = kernels
        self.style_proj = nn.Linear(styles, channels)
        self.weights = nn.Parameter(torch.randn(1, channels, channels, kernels))

    def forward(self, inputs: torch.Tensor, styles: torch.Tensor) -> torch.Tensor:
        """Convolve the inputs with modulated weights.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensor.
            styles: [torch.float32; [B, styles]], style vector.
        Returns:
            [torch.float32; [B, C, T]], convolved.
        """
        # B, C, T
        bsize, channels, timesteps = inputs.shape
        # [B, C]
        styles = self.style_proj(styles)
        # [B, C(=output), C(=input), K], modulation
        w = self.weights * styles[:, None, :, None]
        # [B, C, C, K], demodulation
        w = w * (w.square().sum(dim=[-1, -2], keepdim=True) + 1e-8).rsqrt()
        # [B, C, T]
        return F.conv1d(
            # [1, B x C, T]
            inputs.view(1, -1, timesteps),
            # [B x C, C, K]
            w.view(-1, channels, self.kernels),
            padding=self.kernels // 2,
            # dynamic convolution with batch-axis grouping
            groups=bsize).view(bsize, channels, timesteps)


class ModulatedAuxResidualBlock(nn.Module):
    """Convolutional residual block with auxiliary contexts and weight modulation.
    """
    def __init__(self, channels: int, kernels: int, aux: int, styles: int):
        """Initializer.
        Args:
            channels: size of the convolutional channels.
            kernels: size of the convolutional kernels.
            aux: size of the auxiliary contexts.
            styles: size of the style vectors.
        """
        super().__init__()
        self.preconv = ModulatedConv1d(channels, kernels, styles)
        self.preblock = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(channels))

        self.proj = nn.Linear(aux, channels, bias=False)

        self.postconv = ModulatedConv1d(channels, kernels, styles)
        self.postblock = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(channels))

    def forward(self, inputs: torch.Tensor, aux: torch.Tensor, styles: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input 1D feature map.
            aux: [torch.float32; [B, E]], auxiliary embedding.
            styles: [torch.float32; [B, styles]], style vector.
        Returns:
            [torch.float32; [B, C, T]], residually connected.
        """
        # [B, C, T]
        x = self.preblock(self.preconv(inputs, styles)) + self.proj(aux)[..., None]
        # [B, C, T]
        return inputs + self.postblock(self.postconv(x, styles))
