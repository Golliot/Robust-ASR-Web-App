"""Tests de la conversion au format du modèle : mono, 16 kHz, float32, sans autre transformation."""

import math

import numpy as np
import pytest
import torch
import torchaudio

from src.audio.decoding import DecodedAudio, decode_audio
from src.audio.errors import AudioTooLongError, AudioTooShortError, InvalidAudioError
from src.audio.preprocessing import TARGET_SAMPLE_RATE, prepare_waveform, resample, to_mono
from tests.support.audio import dominant_frequency, encoded, soundfile_bytes, tone

LIMITS = {"min_duration_s": 0.1, "max_duration_s": 30.0}


def decoded(samples: np.ndarray, sample_rate: int) -> DecodedAudio:
    frames = samples[:, None] if samples.ndim == 1 else samples
    return DecodedAudio(frames.astype(np.float32), sample_rate, "PCM", "", "test")


def test_mono_16k_reaches_the_model_unchanged() -> None:
    """Un fichier LibriSpeech (16 kHz, mono) n'est ni rééchantillonné ni normalisé."""
    signal = tone(seconds=1.0) * 0.01  # signal faible : aucun gain ne doit être appliqué
    waveform = prepare_waveform(decoded(signal, 16_000), **LIMITS)
    assert waveform.dtype == torch.float32
    assert waveform.is_contiguous()
    assert torch.equal(waveform, torch.from_numpy(signal))


def test_stereo_with_identical_channels_equals_the_first_channel() -> None:
    """Cas du micro de navigateur : la moyenne redonne exactement le canal lu par load_audio."""
    signal = tone(seconds=0.5)
    waveform = prepare_waveform(decoded(np.stack([signal, signal], axis=1), 16_000), **LIMITS)
    assert torch.equal(waveform, torch.from_numpy(signal))


def test_stereo_is_averaged() -> None:
    left, right = tone(440.0, seconds=0.5), tone(880.0, seconds=0.5)
    waveform = to_mono(np.stack([left, right], axis=1))
    assert torch.allclose(waveform, torch.from_numpy((left + right) / 2), atol=1e-7)


@pytest.mark.parametrize("sample_rate", [8_000, 22_050, 44_100, 48_000])
def test_resampling_is_a_real_resampling(sample_rate: int) -> None:
    """La durée et la fréquence sont conservées : ce n'est pas un simple changement de métadonnées."""
    signal = tone(1_000.0, seconds=1.0, sample_rate=sample_rate)
    waveform = prepare_waveform(decoded(signal, sample_rate), **LIMITS)
    assert waveform.size(0) == math.ceil(signal.size * TARGET_SAMPLE_RATE / sample_rate)
    assert dominant_frequency(waveform.numpy(), TARGET_SAMPLE_RATE) == pytest.approx(1_000.0, abs=2.0)
    # Hors des bords, le signal rééchantillonné coïncide avec le même son généré à 16 kHz.
    reference = tone(1_000.0, seconds=1.0, sample_rate=TARGET_SAMPLE_RATE)
    interior = slice(400, 15_600)
    assert np.abs(waveform.numpy()[interior] - reference[interior]).max() < 0.01


def test_resampling_uses_the_project_convention() -> None:
    """Mêmes réglages que download_noise.convert_to_16k_mono : torchaudio.functional.resample par défaut."""
    signal = torch.from_numpy(tone(seconds=0.5, sample_rate=48_000))
    assert torch.equal(resample(signal, 48_000), torchaudio.functional.resample(signal, 48_000, 16_000))
    assert resample(signal, 16_000, 16_000) is signal


def test_browser_recording_reaches_16k_mono() -> None:
    """Chaîne complète d'un enregistrement Chrome : WebM/Opus, 48 kHz, stéréo."""
    signal = tone(440.0, seconds=2.0, sample_rate=48_000)
    audio = decode_audio(encoded("webm_opus", np.stack([signal, signal], axis=1), 48_000))
    waveform = prepare_waveform(audio, **LIMITS)
    assert waveform.ndim == 1
    assert waveform.size(0) / TARGET_SAMPLE_RATE == pytest.approx(2.0, abs=0.1)
    assert dominant_frequency(waveform.numpy(), TARGET_SAMPLE_RATE) == pytest.approx(440.0, abs=5.0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_samples_are_refused(value: float) -> None:
    signal = tone(seconds=0.5)
    signal[100] = value
    with pytest.raises(InvalidAudioError, match="non numériques"):
        prepare_waveform(decoded(signal, 16_000), **LIMITS)


def test_float_wav_with_nan_is_refused_end_to_end() -> None:
    signal = tone(seconds=0.5)
    signal[10] = np.nan
    with pytest.raises(InvalidAudioError):
        prepare_waveform(decode_audio(soundfile_bytes(signal, 16_000, subtype="FLOAT")), **LIMITS)


def test_too_short_audio_is_refused() -> None:
    """En dessous de 257 échantillons, le spectrogramme du modèle échouerait."""
    with pytest.raises(AudioTooShortError, match="au moins 0,1 s") as error:
        prepare_waveform(decoded(tone(seconds=0.05), 16_000), **LIMITS)
    assert error.value.code == "AUDIO_TOO_SHORT"


def test_too_long_audio_is_refused() -> None:
    with pytest.raises(AudioTooLongError, match="limite est de 30 s"):
        prepare_waveform(decoded(np.zeros(int(30.5 * 16_000), dtype=np.float32), 16_000), **LIMITS)


def test_limits_are_inclusive() -> None:
    assert prepare_waveform(decoded(np.zeros(30 * 16_000, dtype=np.float32), 16_000), **LIMITS).size(0) == 480_000
    assert prepare_waveform(decoded(np.zeros(1_600, dtype=np.float32), 16_000), **LIMITS).size(0) == 1_600
