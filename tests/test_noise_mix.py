"""Tests du tirage d'entraînement en deux étages, avec un faux lecteur et un faux babble."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.augment.noise_mix import NoiseCategory, NoiseSampler, read_segment

N_DRAWS = 20_000
CATEGORIES = (
    NoiseCategory("noise", 0.4, (0.0, 15.0)),
    NoiseCategory("music", 0.3, (5.0, 15.0)),
    NoiseCategory("babble", 0.3, (13.0, 20.0)),
)
ENTRIES = [
    {"path": "noise/short.wav", "category": "noise", "duration": 2.0},
    {"path": "noise/long.wav", "category": "noise", "duration": 180.0},
    {"path": "music/a.wav", "category": "music", "duration": 60.0},
    {"path": "music/b.wav", "category": "music", "duration": 60.0},
    {"path": "speech/talk.wav", "category": "speech", "duration": 3600.0},
]


@dataclass
class FakeBabble:
    speakers: list[str] = field(default_factory=lambda: ["1", "2"])
    calls: list[tuple[int, int | str]] = field(default_factory=list)

    def build(self, n_samples: int, speaker: int | str, generator: torch.Generator | None = None) -> torch.Tensor:
        self.calls.append((n_samples, speaker))
        return torch.full((n_samples,), -1.0)


@dataclass
class FakeReader:
    paths: list[str] = field(default_factory=list)

    def __call__(self, path: str, n_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
        self.paths.append(path)
        return torch.full((n_samples,), float(path.startswith("music")))


def make_sampler(exponent: float = 1.0) -> tuple[NoiseSampler, FakeReader, FakeBabble]:
    reader, babble = FakeReader(), FakeBabble()
    return NoiseSampler(CATEGORIES, ENTRIES, babble, reader, exponent), reader, babble


def draw_all(sampler: NoiseSampler, seed: int = 0) -> list[tuple[str, float]]:
    """Renvoie (catégorie, SNR) pour N_DRAWS tirages, d'après la valeur du faux bruit."""
    generator = torch.Generator().manual_seed(seed)
    names = {0.0: "noise", 1.0: "music", -1.0: "babble"}
    draws = []
    for _ in range(N_DRAWS):
        noise, snr_db = sampler.sample(4, "42", generator)
        draws.append((names[noise[0].item()], snr_db))
    return draws


def test_categories_follow_their_probabilities() -> None:
    counts = Counter(name for name, _ in draw_all(make_sampler()[0]))
    assert counts["noise"] / N_DRAWS == pytest.approx(0.4, abs=0.015)
    assert counts["music"] / N_DRAWS == pytest.approx(0.3, abs=0.015)
    assert counts["babble"] / N_DRAWS == pytest.approx(0.3, abs=0.015)


def test_each_category_uses_its_own_snr_range() -> None:
    draws = draw_all(make_sampler()[0])
    for category in CATEGORIES:
        low, high = category.snr_db
        values = [snr for name, snr in draws if name == category.name]
        assert low <= min(values) and max(values) <= high
        # Une plage entièrement couverte, et non une partie seulement.
        assert min(values) < low + 0.05 * (high - low)
        assert max(values) > high - 0.05 * (high - low)
        assert sum(values) / len(values) == pytest.approx((low + high) / 2, abs=0.2)


@pytest.mark.parametrize(("exponent", "expected_ratio"), [(1.0, 90.0), (0.5, 90.0**0.5), (0.0, 1.0)])
def test_files_are_weighted_by_duration(exponent: float, expected_ratio: float) -> None:
    sampler, reader, _ = make_sampler(exponent)
    draw_all(sampler)
    counts = Counter(reader.paths)
    assert counts["noise/long.wav"] / counts["noise/short.wav"] == pytest.approx(expected_ratio, rel=0.25)


def test_every_second_of_a_category_has_the_same_chance() -> None:
    sampler, reader, _ = make_sampler()
    draw_all(sampler)
    counts = Counter(reader.paths)
    assert counts["music/a.wav"] / counts["music/b.wav"] == pytest.approx(1.0, abs=0.1)


def test_uncategorized_files_are_never_drawn() -> None:
    sampler, reader, _ = make_sampler()
    draw_all(sampler)
    assert "speech/talk.wav" not in reader.paths


def test_babble_receives_the_target_speaker_and_the_length() -> None:
    sampler, _, babble = make_sampler()
    draw_all(sampler)
    assert babble.calls
    assert set(babble.calls) == {(4, "42")}


def test_sampling_is_reproducible_with_a_generator() -> None:
    first = draw_all(make_sampler()[0], seed=1)
    second = draw_all(make_sampler()[0], seed=1)
    other = draw_all(make_sampler()[0], seed=2)
    assert first == second
    assert first != other


@pytest.mark.parametrize(
    ("categories", "entries", "exponent", "message"),
    [
        ([NoiseCategory("noise", 0.5, (0.0, 1.0))], ENTRIES, 1.0, "sommer à 1"),
        ([NoiseCategory("noise", 1.0, (5.0, 1.0))], ENTRIES, 1.0, "plage de SNR"),
        ([NoiseCategory("noise", 1.0, (0.0, 1.0))], ENTRIES, -1.0, "exposant"),
        ([NoiseCategory("noise", 1.0, (0.0, 1.0))], [], 1.0, "Aucun fichier MUSAN"),
    ],
)
def test_invalid_configuration_is_refused(
    categories: list[NoiseCategory],
    entries: list[dict],
    exponent: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        NoiseSampler(categories, entries, FakeBabble(), FakeReader(), exponent)


def test_summary_describes_each_category() -> None:
    summary = make_sampler()[0].summary()
    assert "noise   p = 0.40  SNR ~ U(0, 15) dB  2 fichiers" in summary
    assert "babble  p = 0.30  SNR ~ U(13, 20) dB  2 locuteurs" in summary


@pytest.mark.parametrize(("start", "n_samples"), [(0, 4), (3, 5), (6, 4)])
def test_read_segment_reads_from_start(tmp_path: Path, start: int, n_samples: int) -> None:
    path = tmp_path / "ramp.wav"
    sf.write(path, np.arange(10, dtype="float32") / 100, 16_000, subtype="FLOAT")
    segment = read_segment(path, start, n_samples)
    assert segment.size(0) == n_samples
    assert torch.allclose(segment, (start + torch.arange(n_samples)).float() / 100)


def test_read_segment_loops_short_files(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    sf.write(path, np.arange(3, dtype="float32") / 100, 16_000, subtype="FLOAT")
    assert torch.allclose(read_segment(path, 0, 7), torch.tensor([0, 1, 2, 0, 1, 2, 0.0]) / 100)


def test_read_segment_refuses_a_segment_outside_the_file(tmp_path: Path) -> None:
    path = tmp_path / "ramp.wav"
    sf.write(path, np.zeros(10, dtype="float32"), 16_000)
    with pytest.raises(ValueError, match="hors du fichier"):
        read_segment(path, 8, 5)
