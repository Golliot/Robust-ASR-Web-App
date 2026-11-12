"""Conversion d'un audio décodé au format de travail du modèle : 16 kHz, mono, float32.

C'est exactement ce que lit le notebook (`load_audio`) sur LibriSpeech. L'entraînement n'applique
aucune normalisation de gain, de moyenne ou de pré-accentuation : aucune ne l'est ici non plus.

Mono : moyenne des canaux. Pour un fichier mono, ou un stéréo aux canaux identiques comme celui d'un
micro de navigateur, le résultat est exactement le premier canal, celui que lit `load_audio`.

Rééchantillonnage : un vrai filtrage par `torchaudio.functional.resample`, avec les réglages par
défaut que download_noise.py applique déjà aux corpus publiés en 48 kHz. Il est omis à 16 kHz : un
fichier LibriSpeech arrive donc au modèle sans aucune modification.
"""

import numpy as np
import torch
import torchaudio

from src.audio.decoding import DecodedAudio
from src.audio.errors import AudioTooLongError, AudioTooShortError, InvalidAudioError

TARGET_SAMPLE_RATE = 16_000


def check_duration(duration_s: float, *, min_duration_s: float, max_duration_s: float) -> None:
    """Refuse un audio hors de [min_duration_s, max_duration_s]."""
    if duration_s < min_duration_s:
        raise AudioTooShortError(duration_s, min_duration_s)
    if duration_s > max_duration_s:
        raise AudioTooLongError(duration_s, max_duration_s)


def to_mono(samples: np.ndarray) -> torch.Tensor:
    """(trames, canaux) -> (trames,) : moyenne des canaux. Un signal mono est renvoyé tel quel."""
    audio = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
    if audio.size(1) == 1:
        return audio[:, 0].clone()
    return audio.mean(dim=1)


def resample(waveform: torch.Tensor, orig_freq: int, new_freq: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """Rééchantillonne réellement le signal ; il est renvoyé tel quel s'il est déjà à new_freq."""
    if orig_freq == new_freq:
        return waveform
    return torchaudio.functional.resample(waveform, orig_freq, new_freq)


def prepare_waveform(
    audio: DecodedAudio,
    *,
    min_duration_s: float,
    max_duration_s: float,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> torch.Tensor:
    """Vérifie l'audio décodé et le convertit en signal mono float32 à target_sample_rate : (n,)."""
    if not np.isfinite(audio.samples).all():
        raise InvalidAudioError("L'audio contient des valeurs non numériques (NaN ou infinies).")
    check_duration(audio.duration_s, min_duration_s=min_duration_s, max_duration_s=max_duration_s)
    waveform = resample(to_mono(audio.samples), audio.sample_rate, target_sample_rate)
    return waveform.contiguous()
