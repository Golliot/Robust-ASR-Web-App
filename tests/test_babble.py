"""Tests du babble, sur un mini-arbre LibriSpeech factice généré dans un dossier temporaire."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.augment.babble import MAX_TALKERS, MIN_TALKERS, BabbleBank, fit_length, normalize_rms

SAMPLE_RATE = 16_000
SPEAKERS = {
    "train-clean-100": range(100, 104),
    "train-clean-360": range(200, 204),
    "train-other-500": range(300, 304),
    "dev-clean": range(400, 405),
    "dev-other": range(500, 505),
    "test-clean": range(600, 605),
    "test-other": range(700, 705),
}


def speaker_of(path: Path) -> str:
    return path.name.split("-")[0]


def subset_of(path: Path) -> str:
    return path.parent.parent.parent.name


def write_flac(path: Path, n_samples: int, value: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.full(n_samples, value, dtype="float32"), SAMPLE_RATE, format="FLAC")


@pytest.fixture
def librispeech_root(tmp_path: Path) -> Path:
    root = tmp_path / "LibriSpeech"
    for subset, speakers in SPEAKERS.items():
        for speaker in speakers:
            for utterance in range(2):
                name = f"{speaker}-1-{utterance:04d}.flac"
                write_flac(root / subset / str(speaker) / "1" / name, 2_000)
    return root


@pytest.mark.parametrize("split", ["test", "test-clean", "test-other", ""])
def test_test_split_is_refused(librispeech_root: Path, split: str) -> None:
    with pytest.raises(ValueError, match="Split de babble invalide"):
        BabbleBank(split, librispeech_root)


# Écrit en toutes lettres : comparer à SPLIT_SUBSETS suivrait une fuite introduite dans cette constante.
@pytest.mark.parametrize(
    ("split", "expected"),
    [
        ("train", {"train-clean-100", "train-clean-360", "train-other-500"}),
        ("val", {"dev-clean", "dev-other"}),
    ],
)
def test_utterances_only_come_from_the_split(librispeech_root: Path, split: str, expected: set[str]) -> None:
    bank = BabbleBank(split, librispeech_root)
    assert {subset_of(path) for paths in bank.utterances.values() for path in paths} == expected


@pytest.mark.parametrize("target", [401, "401"])
def test_talkers_are_distinct_and_never_the_target(librispeech_root: Path, target: int | str) -> None:
    bank = BabbleBank("val", librispeech_root)
    generator = torch.Generator().manual_seed(0)
    counts = set()
    for _ in range(300):
        speakers = [speaker_of(path) for path in bank.pick_utterances(target, generator)]
        assert "401" not in speakers
        assert len(set(speakers)) == len(speakers)
        counts.add(len(speakers))
    assert counts == set(range(MIN_TALKERS, MAX_TALKERS + 1))


def test_missing_subset_fails(librispeech_root: Path) -> None:
    (librispeech_root / "dev-other").rename(librispeech_root / "moved")
    with pytest.raises(FileNotFoundError, match="dev-other"):
        BabbleBank("val", librispeech_root)


def test_too_few_speakers_fails(librispeech_root: Path) -> None:
    for speaker in (402, 403, 404):
        (librispeech_root / "dev-clean" / str(speaker) / "1").rename(librispeech_root / f"moved-{speaker}")
    with pytest.raises(ValueError, match="locuteurs"):
        BabbleBank("val", librispeech_root)


# 500 échantillons : recadrage, 5 000 : boucle. Un signal varié rend chaque position distincte.
@pytest.mark.parametrize("n_samples", [500, 5_000])
def test_build_is_reproducible_with_a_fixed_generator(librispeech_root: Path, n_samples: int) -> None:
    for subset in ("dev-clean", "dev-other"):
        for path in (librispeech_root / subset).glob("*/*/*.flac"):
            audio = np.random.default_rng(int(speaker_of(path))).uniform(-0.5, 0.5, 2_000).astype("float32")
            sf.write(path, audio, SAMPLE_RATE, format="FLAC")

    bank = BabbleBank("val", librispeech_root)
    first = bank.build(n_samples, 401, torch.Generator().manual_seed(3))
    torch.manual_seed(0)  # un tirage qui passerait par le générateur global changerait ici
    second = bank.build(n_samples, 401, torch.Generator().manual_seed(3))
    other = bank.build(n_samples, 401, torch.Generator().manual_seed(4))
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


@pytest.mark.parametrize("n_samples", [500, 2_000, 9_001])
def test_build_returns_the_requested_length(librispeech_root: Path, n_samples: int) -> None:
    babble = BabbleBank("val", librispeech_root).build(n_samples, 401, torch.Generator().manual_seed(0))
    assert babble.shape == (n_samples,)
    assert babble.dtype == torch.float32


def test_each_utterance_is_rms_normalized_before_the_sum(librispeech_root: Path) -> None:
    """Des niveaux très différents donnent le même résultat : chaque voix pèse autant."""
    for path in (librispeech_root / "dev-clean").glob("*/*/*.flac"):
        write_flac(path, 2_000, value=0.001 if int(speaker_of(path)) % 2 else 0.5)
    for path in (librispeech_root / "dev-other").glob("*/*/*.flac"):
        write_flac(path, 2_000, value=0.05)

    bank = BabbleBank("val", librispeech_root)
    generator = torch.Generator().manual_seed(0)
    n_talkers = len(bank.pick_utterances(401, generator))

    generator = torch.Generator().manual_seed(0)
    babble = bank.build(1_000, 401, generator)
    assert torch.allclose(babble, torch.full_like(babble, float(n_talkers)), atol=1e-3)


def test_normalize_rms_handles_silence() -> None:
    assert normalize_rms(torch.tensor([0.5, -0.5])).pow(2).mean().sqrt() == pytest.approx(1.0)
    assert torch.equal(normalize_rms(torch.zeros(10)), torch.zeros(10))


def test_fit_length_crops_a_contiguous_segment() -> None:
    audio = torch.arange(100.0)
    cropped = fit_length(audio, 10, torch.Generator().manual_seed(0))
    assert cropped.size(0) == 10
    assert torch.equal(cropped.diff(), torch.ones(9))


def test_fit_length_loops_short_audio_from_any_phase() -> None:
    audio = torch.arange(4.0)
    generator = torch.Generator().manual_seed(0)
    starts = set()
    for _ in range(50):
        looped = fit_length(audio, 11, generator)
        assert looped.size(0) == 11
        assert torch.equal(looped, (looped[0] + torch.arange(11.0)) % 4)
        starts.add(int(looped[0]))
    assert starts == {0, 1, 2, 3}


@pytest.mark.parametrize("n_samples", [4, 8, 9])
def test_fit_length_handles_exact_multiples(n_samples: int) -> None:
    assert fit_length(torch.arange(4.0), n_samples, torch.Generator().manual_seed(1)).size(0) == n_samples
