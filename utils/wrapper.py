import sched
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from disc import Discriminator
from starddpm import StarDDPMVC


class TrainingWrapper:
    """Training wrapper.
    """
    def __init__(self,
                 model: StarDDPMVC,
                 disc: Discriminator,
                 config: Config,
                 device: torch.device):
        """Initializer.
        Args:
            model: StarDDPM model.
            disc: discriminator.
            config: training configurations.
            device: torch device.
        """
        self.model = model
        self.disc = disc
        self.config = config
        self.device = device

    def wrap(self, bunch: List[np.ndarray]) -> List[torch.Tensor]:
        """Wrap the array to torch tensor.
        Args:
            bunch: input tensors.
        Returns:
            wrapped.
        """
        return [torch.tensor(array, device=self.device) for array in bunch]

    def random_segment(self, bunch: List[np.ndarray]) -> List[np.ndarray]:
        """Segment the spectrogram and audio into fixed sized array.
        Args:
            bunch: input tensors.
                ids: [np.long; [B]], auxiliary ids.
                pitch: [np.float32; [B, T]], pitch sequence.
                mel: [np.float32; [B, T, mel]], mel spectrogram.
                lengths: [np.long; [B]], spectrogram lengths.
        Returns:
            randomly segmented spectrogram and audios.
        """
        # [B], [B, T], [B, T, mel], [B]
        ids, pitch, mel, lengths = bunch
        # [B]
        start = np.random.randint(lengths - self.config.train.seglen)
        # [B, S]
        pitch = np.array(
            [p[s:s + self.config.train.seglen] for p, s in zip(pitch, start)])
        # [B, S, mel]
        mel = np.array(
            [m[s:s + self.config.train.seglen] for m, s in zip(mel, start)])
        return ids, pitch, mel

    def compute_loss(self, bunch: List[np.ndarray]) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss.
        Args:
            bunch: list of inputs.
        Returns:
            loss and dictionaries.
        """
        # [1 + S]
        logsnr, _ = self.model.scheduler()
        # [1 + S]
        alphas_bar = torch.sigmoid(logsnr)
        # [], prior loss
        schedule_loss = torch.log(
            torch.clamp_min(1 - alphas_bar[-1], 1e-7)) + torch.log(
                torch.clamp_min(alphas_bar[0], 1e-7))

        # [B], [B, T], [B, T, mel]
        ids, pitch, mel = self.wrap(self.random_segment(bunch))
        # [B, mel, T]
        mel = mel.transpose(1, 2)
        # [B], zero-based
        steps = torch.randint(
            self.config.model.steps, (mel.shape[0],), device=mel.device)
        # [B, mel, T]
        mean, std = self.model.diffusion(mel, steps)
        # [B, mel, T]
        base = mean + torch.randn_like(mean) * std[:, None, None]
        # [B, mel, T]
        denoised = self.model.denoise(base, mel, steps)
        # []
        noise_estim = (denoised - mel).square().mean()
        # total loss
        loss = schedule_loss + noise_estim
        return loss, {'sched': schedule_loss.item(), 'estim': noise_estim.item()}
