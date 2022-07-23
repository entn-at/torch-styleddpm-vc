from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from starddpm import StarDDPMVC


class TrainingWrapper:
    """Training wrapper.
    """
    def __init__(self,
                 model: StarDDPMVC,
                 config: Config,
                 device: torch.device):
        """Initializer.
        Args:
            model: StarDDPM model.
            config: training configurations.
            device: torch device.
        """
        self.model = model
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
                text: [np.long; [B, S]], text token sequence.
                pitch: [np.float32; [B, T]], pitch sequence.
                mel: [np.float32; [B, T, mel]], mel spectrogram.
                textlen: [np.long; [B]], text lengths.
                mellen: [np.long; [B]], spectrogram lengths.
        Returns:
            randomly segmented spectrogram and audios.
        """
        # [B], [B, T], [B, T, mel], [B]
        ids, _, pitch, mel, _, lengths = bunch
        # [B]
        start = np.random.randint(lengths - self.config.train.seglen)
        # [B, S]
        pitch = np.array(
            [p[s:s + self.config.train.seglen] for p, s in zip(pitch, start)])
        # [B, S, mel]
        mel = np.array(
            [m[s:s + self.config.train.seglen] for m, s in zip(mel, start)])
        return ids, pitch, mel.transpose(0, 2, 1)

    def compute_loss(self, bunch: List[np.ndarray]) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss.
        Args:
            bunch: list of inputs.
        Returns:
            loss and dictionaries.
        """
        ## 1. Dynamic scheduler loss
        # [1 + S]
        logsnr, _ = self.model.scheduler()
        # [1 + S]
        alphas_bar = torch.sigmoid(logsnr)
        # [], prior loss
        schedule_loss = torch.log(
            torch.clamp_min(1 - alphas_bar[-1], 1e-7)) + torch.log(
                torch.clamp_min(alphas_bar[0], 1e-7))

        # [B], [B, T], [B, mel, T]
        ids, pitch, mel = self.wrap(self.random_segment(bunch))
        # B
        bsize = ids.shape[0]
        # [], range [1, B - 1]
        start = np.random.randint(bsize - 1) + 1
        # [B], for shuffling
        indices = (np.arange(bsize) + start) % bsize
        # [B, styles]
        style = self.model.encoder(mel, ids)

        ## 2. Cyclic noise estimation
        # [B], zero-based
        steps = torch.randint(
            self.config.model.steps, (mel.shape[0],), device=mel.device)
        # [B, mel, T]
        mean, std = self.model.diffusion(mel, steps)
        # [B, mel, T]
        base = mean + torch.randn_like(mean) * std[:, None, None]
        # [B, mel, T]
        converted = self.model.denoise(base, mel, style[indices], steps)



        # []
        noise_estim = (denoised - mel).square().mean()

        ## 3. Cycle consistency


        ## 3. Average pitch estimation and speaker classification
        # [B, domains], [B]
        logits, avgpit = self.disc(denoised)
        # []
        spkclf = F.cross_entropy(logits, ids)

        ## 4. Cycle consistency

        ## 5. ASR-guiding phonetic structure preservation

        ## 6. Style reconstruction

        ## 7. Norm consistency

        # total loss
        loss = schedule_loss + noise_estim + spkclf
        return loss, {'sched': schedule_loss.item(), 'estim': noise_estim.item()}
