from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from styleddpm import StyleDDPMVC


class TrainingWrapper:
    """Training wrapper.
    """
    def __init__(self,
                 model: StyleDDPMVC,
                 config: Config,
                 device: torch.device):
        """Initializer.
        Args:
            model: StyleDDPM-VC model.
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
        ids, pitch, mel, lengths = bunch
        # [B]
        start = np.random.randint(lengths - self.config.train.seglen)
        # [B, seglen]
        pitch = np.array(
            [p[s:s + self.config.train.seglen] for p, s in zip(pitch, start)])
        # [B, seglen, mel]
        mel = np.array(
            [m[s:s + self.config.train.seglen] for m, s in zip(mel, start)])
        # [B], [B, seglen], [B, mel, seglen]
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
        avgpit, style = self.model.encoder(mel)

        ## 1. Reconstruction
        # [B], zero-based
        steps = torch.randint(
            self.config.model.steps, (mel.shape[0],), device=mel.device)
        # [B, mel, T]
        mean, std = self.model.diffusion(mel, steps)
        # [B, mel, T]
        mel_t = mean + torch.randn_like(mean) * std[:, None, None]
        # [B, mel, T], contextualized
        mel_c = self.model.masked_encoder(mel, ratio=self.config.train.mask_ratio)
        # [B, mel, T]
        mel_0 = self.model.denoise(mel_t, mel_c, style, steps)
        # []
        noise_estim = F.mse_loss(mel, mel_0)

        ## 2. Cycle consistency
        # [B, mel, T], unpaired generation, unmask the context vector for baseline.
        unit_0 = self.model.denoise(
            torch.randn_like(mel),
            self.model.masked_encoder(mel),
            style[indices],
            torch.tensor([self.config.model.steps] * bsize, device=mel.device))
        # [B, mel, T]
        mean, std = self.model.diffusion(unit_0, steps)
        # [B, mel, T]
        unit_t = mean + torch.randn_like(mean) * std[:, None, None]
        # [B, mel, T]
        unit_c = self.model.masked_encoder(unit_0, ratio=self.config.train.mask_ratio)
        # []
        unit_estim = F.mse_loss(
            unit_0, self.model.denoise(unit_t, unit_c, style[indices], steps))
        # []
        cycle_estim = \
            F.mse_loss(mel, self.model.denoise(mel_t, unit_c, style, steps)) + \
            F.mse_loss(unit_0, self.model.denoise(unit_t, mel_c, style[indices], steps))

        ## 3. Style reconstruction
        def contrast(a, b, i):
            # [B, B]
            confusion = torch.matmul(
                F.normalize(a, p=2, dim=-1),
                F.normalize(b, p=2, dim=-1).T)
            # [B]
            arange = torch.arange(confusion.shape[0])
            # [B], diagonal selection and negative case contrast
            cont = confusion[arange, arange] - torch.logsumexp(
                confusion.masked_fill(i[:, None] == i, -np.inf), dim=-1)
            # []
            return -cont.mean()
        # [B], [B, styles]
        avgpit_re, style_re = self.model.encoder(mel_0)
        # [B], [B, styles]
        avgpit_unit, style_unit = self.model.encoder(unit_0)
        # []
        style_cont = \
            contrast(style, style_re, ids) + \
            contrast(style[indices], style_unit, ids[indices])

        ## 3. Average pitch estimation
        # [B], non-zero average
        avgpit_gt = pitch.sum(axis=-1) / (pitch > 0).sum(axis=-1).clamp_min(1)
        # [B]
        pitch_estim = \
            F.mse_loss(avgpit_gt, avgpit) + \
            F.mse_loss(avgpit_gt, avgpit_re) + \
            F.mse_loss(avgpit_gt[indices], avgpit_unit)

        ## 4. Masked autoencoder reconstruction
        # []
        mae_rctor = F.mse_loss(mel, self.model.masked_encoder.decoder(mel_c))

        # total loss
        loss = schedule_loss + \
            noise_estim + unit_estim + \
            cycle_estim + \
            style_cont + pitch_estim + \
            mae_rctor
        return loss, {
            'sched': schedule_loss.item(),
            'mae-rctor': mae_rctor.item(),
            'noise-estim': noise_estim.item(),
            'unit-estim': unit_estim.item(),
            'cycle-estim': cycle_estim.item(),
            'style-cont': style_cont.item(),
            'pitch-estim': pitch_estim.item()}
