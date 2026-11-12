"""Log-Mel et SpecAugment du notebook (cellules 3 et 15), repris à l'identique.

Ces deux étapes tournent sur le GPU, batch par batch. La normalisation utilise une moyenne et un
écart-type globaux (CMVN), estimés sur la parole propre d'entraînement et conservés dans le modèle.
Un masque SpecAugment remplace donc les valeurs par la moyenne, soit 0.
"""

import torch
import torch.nn as nn
import torchaudio


def make_pad_mask(lengths, max_length):
    """True sur les positions de padding : (B, max_length)."""
    return torch.arange(max_length, device=lengths.device)[None] >= lengths[:, None]


class LogMel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hop_length = cfg.hop_length
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
        )
        self.register_buffer("mean", torch.zeros(cfg.n_mels))
        self.register_buffer("std", torch.ones(cfg.n_mels))

    def forward(self, wavs, wav_lens):
        feats = self.melspec(wavs).clamp_min(1e-10).log().transpose(1, 2)
        feats = (feats - self.mean) / self.std
        feat_lens = wav_lens // self.hop_length + 1
        return feats.masked_fill(make_pad_mask(feat_lens, feats.size(1))[:, :, None], 0.0), feat_lens


class SpecAugment(nn.Module):
    """Positions et largeurs tirées indépendamment pour chaque énoncé, seulement à l'entraînement."""

    def __init__(self, cfg):
        super().__init__()
        self.freq_masks = cfg.freq_masks
        self.freq_mask_width = cfg.freq_mask_width
        self.time_masks = cfg.time_masks
        self.time_mask_width = cfg.time_mask_width

    @staticmethod
    def _bands(n_masks, max_width, available, size):
        """Masque (B, size) : union de n_masks bandes de largeur ≤ max_width dans [0, available[."""
        draw = lambda: torch.rand(available.size(0), n_masks, device=available.device)
        widths = (draw() * (torch.minimum(available, max_width)[:, None] + 1)).long()
        starts = (draw() * (available[:, None] - widths + 1)).long()
        positions = torch.arange(size, device=available.device)[None, None]
        inside = (positions >= starts[:, :, None]) & (positions < (starts + widths)[:, :, None])
        return inside.any(dim=1)

    def forward(self, feats, feat_lens):
        if not self.training:
            return feats
        batch_size, n_frames, n_bins = feats.shape
        all_bins = feat_lens.new_full((batch_size,), n_bins)
        freq = self._bands(self.freq_masks, feat_lens.new_tensor(self.freq_mask_width), all_bins, n_bins)
        time = self._bands(self.time_masks, feat_lens.new_tensor(self.time_mask_width), feat_lens, n_frames)
        return feats.masked_fill(freq[:, None, :] | time[:, :, None], 0.0)
