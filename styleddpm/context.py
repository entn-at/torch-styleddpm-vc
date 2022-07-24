from typing import Optional, Tuple

import torch
import torch.nn as nn

from .embedder import Embedder


class ContextEncoder(nn.Module):
    """Context encoder with masked spectrogram models.
    """
    def __init__(self,
                 mel: int,
                 patch: int,
                 pe: int,
                 channels: int,
                 heads: int,
                 ffns: int,
                 dropout: float,
                 layers: int):
        """Initializer.
        Args:
            mel: size of the mel filterbank.
            patch: size of the patch, equal on frequency axis and temporal axis.
            pe: size of the positional encodings.
            channels: size of the hidden channels.
            haes: the number of the attention heads.
            ffns: size of the feed-forward network hidden channels.
            dropout: dropout rates.
            layers: the number of the transformer encoder layers.
        """
        super().__init__()
        self.patch = patch
        # 12 seconds max
        self.register_buffer('pe', Embedder.sinusoidal(1000, pe))
        self.register_buffer('mask_token', torch.randn(channels))

        self.patch_embed = nn.Linear(patch * patch, channels, bias=False)
        
        self.proj_pefreq = nn.Linear(pe, channels, bias=False)
        self.proj_petemp = nn.Linear(pe, channels)

        self.proj_out = nn.Conv1d(
            int(mel * channels * patch ** -2), channels, 1)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=heads,
                dim_feedforward=ffns,
                dropout=dropout,
                batch_first=True),
            layers)

    def forward(self, spec: torch.Tensor, ratio: Optional[float] = None) -> torch.Tensor:
        """Encode the contextual features.
        Args:
            spec: [torch.float32; [B, mel, T]], spectrogram.
            ratio: masking ratio,
                mask the spectrogram before feed to encoder if ratio provided.
        Returns:
            [torch.float32; [B, C, T]], encoded.
        """
        # [B, F, S, C]
        patches = self.embed(spec)
        bsize, freqs, temps, channels = patches.shape
        # [B, F, S, C]
        patches = patches + \
            self.proj_pefreq(self.pe[None, :freqs, None]) + \
            self.proj_petemp(self.pe[None, None, :temps])
        # [B, N(=F x S), C]
        flat = patches.view(bsize, freqs * temps, channels)
        if ratio is not None:
            # [B, N'(=N x (1 - ratio)), C], [B, N', C]
            flat, selects = self.random_mask(flat, ratio)
        # [B, N, C]
        flat = self.encoder(flat)
        if ratio is not None:
            # [B, N, C]
            flat = self.scatter(flat, selects)
        # [B, C, T]
        return self.reorder(flat, freqs, temps)

    def embed(self, spec: torch.Tensor) -> torch.Tensor:
        """Flatten the spectrogram and apply patch embedding.
        Args:
            spec: [torch.float32; [B, mel, T]], spectrogram.
        Returns:
            [torch.float32; [B, mel // patch, T // patch, channels]], patch embeddings.
        """
        # B, mel, T
        bsize, bins, timesteps = spec.shape
        # F(=mel // P), S(=T // P)
        freqs, temps = bins // self.patch, timesteps // self.patch
        # [B, F, P1, S, P2]
        # > [B, F, S, P1, P2]
        # > [B, F, S, P1 x P2]
        # > [B, F, S, C]
        return self.patch_embed(
            spec
            .view(bsize, freqs, self.patch, temps, self.patch)
            .permute(0, 1, 3, 2, 4)
            .view(bsize, freqs, temps, -1))

    def reorder(self, flat: torch.Tensor, freqs: int, temps: int) -> torch.Tensor:
        """Reorder the flattened embeddings to original temporal axis.
        Args:
            flat: [torch.float32; [B, N(=freqs x temps), C]], flat embeddings.
            freqs: F, mel // patch.
            temps: S, T // patch.
        Returns:
            [torch.float32; [B, C, T(=temps x patch)]], recovered.
        """
        # B, _, C
        bsize, _, channels = flat.shape
        # [B, F, S, C // P, P2]
        # > [B, F, C // P, S, P2]
        # > [B, F(=mel // P) x C // P, T(=S x P)]
        # > [B, C, T]
        return self.proj_out(flat \
            .view(bsize, freqs, temps, channels // self.patch, self.patch) \
            .permute(0, 1, 3, 2, 4) \
            .view(bsize, -1, temps * self.patch))

    def random_mask(self, flat: torch.Tensor, ratio: float) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample the patches.
        Args:
            flat: [torch.float32; [B, N, C]], flat patches.
            ratio: drop ratio.
        Returns:
            [torch.float32; [B, N x (1 - ratio), C]], randomly dropped.
            [torch.float32; [B, S, C]], selects.
        """
        # B, N, C
        bsize, patches, channels = flat.shape
        # S(=N x (1 - ratio))
        samples = int(patches * (1 - ratio))
        # [B, S, C]
        selects = torch.stack([
            torch.randperm(patches, device=flat.device)[:samples]
            for _ in range(bsize)], dim=0)[..., None].repeat(1, 1, channels)
        return flat.gather(1, selects), selects

    def scatter(self, sampled: torch.Tensor, selects: torch.Tensor, n: int) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Scatter to given indices, fill blanks with `mask_token`.
        Args:
            sampled: [torch.float32; [B, S, C]], randomly dropped.
            selects: [torch.float32; [B, S, C]], indices.
            n: N, original sizes.
        Returns:
            [torch.float32; [B, N, C]], recovered.
        """
        # [B, N, C]
        base = self.mask_token[None, None].repeat(sampled.shape[0], n, 1)
        # [B, N, C]
        return base.scatter(1, selects, sampled)
