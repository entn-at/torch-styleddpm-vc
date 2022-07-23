import argparse
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.functional as F
import torchaudio.functional as AF

import speechset


class VCDataset(speechset.AcousticDataset):
    """ID, Pitch-wrapper for voice conversion dataset support.
    """
    def __init__(self,
                 rawset: speechset.datasets.DataReader,
                 config: speechset.Config,
                 report_level: Optional[int] = None,
                 device: Union[str, torch.device] = 'cpu'):
        """Initializer.
        Args:
            rawset: file-format datum reader.
            config: configuration.
            report_level: text normalizing error report level.
            device: computing device for pitch extraction support.
        """
        super().__init__(rawset, config, report_level)
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def normalize(self, ids: int, text: str, speech: np.ndarray) \
            -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Normalize datum with auxiliary ids.
        Args:
            ids: auxiliary ids.
            text: transcription.
            speech: [np.float32; [T]], speech in range (-1, 1).
        Returns:
            normalized datum.
                ids: int, auxiliary ids.
                labels: [np.long; [S]], labeled text sequence.
                pitch: [np.float32; [T // hop]], pitch sequence.
                mel: [np.float32; [T // hop, mel]], mel spectrogram.
        """
        # [S], [T // hop, mel]
        labels, mel = super().normalize(text, speech)
        with torch.no_grad():
            # [_]
            pitch = AF.detect_pitch_frequency(
                torch.tensor(speech, device=self.device), self.reader.SR)
            # [1, 1, T // hop]
            pitch = F.interpolate(pitch[None, None], size=mel.shape[0], mode='linear')
            # [T // hop], squeezing
            pitch = pitch[0, 0].cpu().numpy()
        return ids, labels, pitch, mel

    def collate(self, bunch: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collate bunch of datum to the batch data.
        Args:
            bunch: B x [...], list of normalized inputs.
                ids: int, auxiliary ids.
                labels: [np.long; [S]], text token sequence.
                pitch: [np.float32; [T // hop]], pitch sequences.
                mel: [np.float32; [T // hop, mel]], mel spectrogram.
        Returns:
            bunch data.
                ids: [np.long; [B]], auxiliary ids.
                text: [np.long; [B, S]], labeled text sequence.
                pitch: [np.float32; [B, T // hop]], pitch sequence.
                mel: [np.float32; [B, T // hop, mel]], mel spectrogram.
                textlen: [np.long; [B]], text lengths.
                mellen: [np.long; [B]], spectrogram lengths.
        """
        # [B]
        ids = np.array([ids for ids, _, _, _ in bunch], dtype=np.long)
        # [B, S], [B, T // hop, mel], [B], [B]
        text, mel, textlen, mellen = super().collate([
            (labels, mel) for _, labels, _, mel in bunch])
        # [B, T // hop]
        pitch = np.stack([
            np.pad(pitch, [0, mellen.max() - len(pitch)]) for _, _, pitch, _ in bunch])
        return ids, text, pitch, mel, textlen, mellen


def dump(data_dir: str,
         out_dir: str,
         num_proc: int,
         chunksize: int = 1,
         device: Union[str, torch.device] = 'cpu',
         config: Optional[speechset.Config] = None) -> int:
    """Dump preprocessed LibriTTS datasets.
    Args:
        data_dir: dataset directory.
        out_dir: path to the dumped dataset.
        num_proc: the number of the processor.
        chunksize: multiprocessor chunk size.
        device: torch computing device.
        config: dataset configuration, if provided.
    Returns:
        dataset lengths.
    """
    config = config or speechset.Config()
    libri = speechset.datasets.LibriTTS(data_dir)
    # construct voice conversion dataset
    vcdata = VCDataset(libri, config, device=device)
    # dump
    return speechset.utils.mp_dump(vcdata, out_dir, num_proc, chunksize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--num-proc', defulat=4, type=int)
    parser.add_argument('--chunksize', default=1, typ=int)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    dump(args.data_dir, args.out_dir, args.num_proc, args.chunksize, args.device)
