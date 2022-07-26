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
                pitch: [np.float32; [B, T]], pitch.
                mel: [np.float32; [B, T, mel]], mel spectrogram.
                lengths: [np.long; [B]], spectrogram lengths.
        Returns:
            randomly segmented spectrogram and audios.
        """
        # prefilter
        prefilter = bunch[-1] >= self.config.train.seglen
        ids, pitch, mel, lengths = [i[prefilter] for i in bunch]
        # [B]
        start = np.random.randint(lengths - self.config.train.seglen + 1)
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
        unit = self.model.denoise(
            torch.randn_like(mel),
            self.model.masked_encoder(mel),
            style[indices],
            torch.tensor([self.config.model.steps - 1] * bsize, device=mel.device))
        # [B, mel, T]
        mean, std = self.model.diffusion(unit, steps)
        # [B, mel, T]
        unit_t = mean + torch.randn_like(mean) * std[:, None, None]
        # [B, mel, T]
        unit_c = self.model.masked_encoder(unit, ratio=self.config.train.mask_ratio)
        # [B, mel, T]
        unit_0 = self.model.denoise(unit_t, unit_c, style[indices], steps)
        # []
        unit_estim = F.mse_loss(unit, unit_0)
        # [B, mel, T]
        mel_0_unit_c = self.model.denoise(mel_t, unit_c, style, steps)
        unit_0_mel_c = self.model.denoise(unit_t, mel_c, style[indices], steps)
        # []
        cycle_estim = F.mse_loss(mel, mel_0_unit_c) + F.mse_loss(unit, unit_0_mel_c)

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
        avgpit_unit, style_unit = self.model.encoder(unit)
        # []
        style_cont = \
            contrast(style, style_re, ids) + \
            contrast(style[indices], style_unit, ids[indices])

        ## 3. Average pitch estimation
        def log_mse(a, b): return F.mse_loss(torch.log(a + 1e-5), torch.log(b + 1e-5))
        # [B], non-zero average
        avgpit_gt = pitch.sum(axis=-1) / (pitch > 0).sum(axis=-1).clamp_min(1)
        # [B]
        pitch_estim = \
            log_mse(avgpit_gt, avgpit) + \
            log_mse(avgpit_gt, avgpit_re) + \
            log_mse(avgpit_gt[indices], avgpit_unit)

        ## 4. Masked autoencoder reconstruction
        # []
        mae_rctor = F.mse_loss(mel, self.model.masked_encoder.decoder(mel_c))

        # total loss
        loss = schedule_loss + \
            noise_estim + unit_estim + \
            cycle_estim + \
            style_cont + pitch_estim + \
            mae_rctor
        losses = {
            'sched': schedule_loss.item(),
            'mae-rctor': mae_rctor.item(),
            'noise-estim': noise_estim.item(),
            'unit-estim': unit_estim.item(),
            'cycle-estim': cycle_estim.item(),
            'style-cont': style_cont.item(),
            'pitch-estim': pitch_estim.item()}
        return loss, losses, {
            'alphas_bar': alphas_bar.detach().cpu().numpy(),
            'mel': mel.cpu().numpy(),
            'mel_t': mel_t.detach().cpu().numpy(),
            'mel_0': mel_0.detach().cpu().numpy(),
            'unit': unit.detach().cpu().numpy(),
            'unit_t': unit_t.detach().cpu().numpy(),
            'unit_0': unit_0.detach().cpu().numpy(),
            'mel_0_unit_c': mel_0_unit_c.detach().cpu().numpy(),
            'unit_0_mel_c': unit_0_mel_c.detach().cpu().numpy()}
