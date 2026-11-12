"""Tests du décodage audio : formats réellement produits par les navigateurs, fichiers invalides, limites."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio import decoding
from src.audio.decoding import decode_audio
from src.audio.errors import AudioTooLongError, EmptyAudioError, InvalidAudioError
from tests.support.audio import COMPRESSED_FORMATS, dominant_frequency, encoded, soundfile_bytes, tone


def test_wav_is_read_exactly_like_the_notebook(tmp_path: Path) -> None:
    """Même lecteur que load_audio : sf.read en float32, toujours en 2D."""
    path = tmp_path / "speech.wav"
    sf.write(path, tone(seconds=0.5), 16_000, subtype="PCM_16")
    decoded = decode_audio(path)
    expected, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    assert decoded.decoder == "soundfile"
    assert decoded.sample_rate == sample_rate == 16_000
    assert decoded.samples.dtype == np.float32
    assert np.array_equal(decoded.samples, expected)


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
def test_wav_subtypes_keep_their_samples(subtype: str) -> None:
    signal = tone(seconds=0.25)
    decoded = decode_audio(soundfile_bytes(signal, 16_000, "WAV", subtype))
    tolerance = 1e-4 if subtype == "PCM_16" else 1e-6
    assert np.allclose(decoded.samples[:, 0], signal, atol=tolerance)
    assert decoded.description == f"WAV ({subtype})"


def test_flac_is_lossless() -> None:
    signal = np.round(tone(seconds=0.5) * 32767) / 32768  # valeurs exactement représentables en 16 bits
    decoded = decode_audio(soundfile_bytes(signal, 16_000, "FLAC", "PCM_16"))
    assert np.array_equal(decoded.samples[:, 0], signal.astype(np.float32))


@pytest.mark.parametrize(
    ("kind", "sample_rate", "decoder"),
    [
        ("webm_opus", 48_000, "pyav"),
        ("ogg_opus", 48_000, "soundfile"),
        ("mp4_aac", 44_100, "pyav"),
        ("mp3", 44_100, "soundfile"),
    ],
)
def test_compressed_formats_are_decoded(kind: str, sample_rate: int, decoder: str) -> None:
    """Formats des navigateurs : WebM/Opus (Chrome, Firefox), Ogg/Opus, MP4/AAC (Safari), MP3."""
    decoded = decode_audio(encoded(kind, tone(440.0, seconds=1.0, sample_rate=sample_rate), sample_rate))
    assert decoded.decoder == decoder
    assert decoded.sample_rate == sample_rate
    assert decoded.channels == 1
    # Les codecs ajoutent un délai ou un remplissage de quelques dizaines de millisecondes.
    assert decoded.duration_s == pytest.approx(1.0, abs=0.1)
    assert dominant_frequency(decoded.samples[:, 0], sample_rate) == pytest.approx(440.0, abs=5.0)


def test_webm_is_described_for_the_user() -> None:
    decoded = decode_audio(encoded("webm_opus", tone(seconds=0.5, sample_rate=48_000), 48_000))
    assert decoded.description == "WebM/Matroska (opus)"


@pytest.mark.parametrize("kind", ["wav", "webm_opus"])
def test_stereo_channels_are_kept_until_preprocessing(kind: str) -> None:
    left, right = tone(440.0, sample_rate=48_000), tone(880.0, sample_rate=48_000)
    stereo = np.stack([left, right], axis=1)
    data = soundfile_bytes(stereo, 48_000) if kind == "wav" else encoded(kind, stereo, 48_000)
    decoded = decode_audio(data)
    assert decoded.channels == 2
    assert dominant_frequency(decoded.samples[:, 0], 48_000) == pytest.approx(440.0, abs=5.0)
    assert dominant_frequency(decoded.samples[:, 1], 48_000) == pytest.approx(880.0, abs=5.0)


def test_bytes_and_path_give_the_same_samples(tmp_path: Path) -> None:
    data = encoded("webm_opus", tone(sample_rate=48_000), 48_000)
    path = tmp_path / "recording.webm"
    path.write_bytes(data)
    assert np.array_equal(decode_audio(data).samples, decode_audio(path).samples)


def test_empty_content_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EmptyAudioError, match="vide"):
        decode_audio(b"")
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    with pytest.raises(EmptyAudioError, match="vide"):
        decode_audio(path)


def test_audio_without_any_sample_is_empty() -> None:
    with pytest.raises(EmptyAudioError, match="aucun échantillon"):
        decode_audio(soundfile_bytes(np.zeros(0, dtype=np.float32), 16_000))


@pytest.mark.parametrize(
    "data",
    [
        np.random.default_rng(0).bytes(4_000),
        b"Ceci n'est pas un fichier audio.\n" * 50,
        b"RIFF\x00\x00\x00\x00WAVEfmt ",  # en-tête WAV tronqué
    ],
    ids=["aléatoire", "texte", "en-tête tronqué"],
)
def test_unreadable_content_is_invalid(data: bytes) -> None:
    with pytest.raises(InvalidAudioError, match="illisible"):
        decode_audio(data)


def test_truncated_webm_is_invalid() -> None:
    data = encoded("webm_opus", tone(sample_rate=48_000), 48_000)
    with pytest.raises((InvalidAudioError, EmptyAudioError)):
        decode_audio(data[:60])


def test_long_wav_is_refused_from_its_header() -> None:
    data = soundfile_bytes(np.zeros(40 * 16_000, dtype=np.float32), 16_000)
    with pytest.raises(AudioTooLongError, match="40,0 s") as error:
        decode_audio(data, max_duration_s=30.0)
    assert error.value.code == "AUDIO_TOO_LONG"


def test_long_webm_is_refused_from_its_declared_duration() -> None:
    data = encoded("webm_opus", np.zeros(40 * 48_000, dtype=np.float32), 48_000)
    with pytest.raises(AudioTooLongError):
        decode_audio(data, max_duration_s=30.0)


def test_decoding_stops_at_the_limit_without_declared_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Les WebM de MediaRecorder n'annoncent pas leur durée : le décodage s'arrête de lui-même."""
    monkeypatch.setattr(decoding, "_declared_duration", lambda container, stream: None)
    data = encoded("webm_opus", np.zeros(40 * 48_000, dtype=np.float32), 48_000)
    with pytest.raises(AudioTooLongError, match="plus de") as error:
        decode_audio(data, max_duration_s=5.0)
    # Arrêt juste après la limite (5 s + 1 s de marge), loin des 40 s du fichier.
    assert error.value.duration_s < 7.0


def test_short_audio_is_decoded_without_limit_check() -> None:
    decoded = decode_audio(soundfile_bytes(tone(seconds=0.05), 16_000), max_duration_s=30.0)
    assert decoded.n_frames == 800


def test_every_compressed_format_of_the_test_suite_is_covered() -> None:
    assert set(COMPRESSED_FORMATS) == {"webm_opus", "ogg_opus", "mp4_aac", "mp3"}
