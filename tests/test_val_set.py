"""Tests du jeu de validation figé, sur de mini-arbres factices dans un dossier temporaire."""

import json
import logging
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import build_val_set
from build_val_set import VAL_SETS, main
from src.augment.babble import cut, normalize_rms
from src.augment.val_set import (
    CLEAN,
    NO_NOISE,
    RIR_ONLY,
    ValCategory,
    ValSetSpec,
    category_wers,
    condition_name,
    draw_validation_noise,
    grid_summary_wer,
    load_val_set,
    render_noise,
)

SAMPLE_RATE = 16_000
VERSION = "val_v2"
N_DRAWS = 20_000
CATEGORIES = VAL_SETS[VERSION].categories


def noise_like(seed: int, n_samples: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.5, 0.5, n_samples).astype("float32")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for subset, speakers in (("dev-clean", range(100, 120)), ("dev-other", range(200, 220))):
        for speaker in speakers:
            for utterance in range(3):
                path = root / "LibriSpeech" / subset / str(speaker) / "1" / f"{speaker}-1-{utterance:04d}.flac"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, noise_like(speaker + utterance, 1_600 + 400 * utterance), SAMPLE_RATE, format="FLAC")
    # Un sous-ensemble de test présent sur le disque ne doit jamais être utilisé.
    path = root / "LibriSpeech" / "test-clean" / "900" / "1" / "900-1-0000.flac"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, noise_like(900, 1_600), SAMPLE_RATE, format="FLAC")

    musan = root / "musan"
    lines = []
    groups = (("noise", "free-sound", 12, 800), ("music", "fma", 12, 8_000), ("speech", "us-gov", 12, 8_000))
    for category, source, count, frames in groups:
        for index in range(count):
            relative = f"{category}/{source}/{index:03d}.wav"
            (musan / relative).parent.mkdir(parents=True, exist_ok=True)
            n_samples = frames * (index + 1)
            sf.write(musan / relative, noise_like(index, n_samples), SAMPLE_RATE)
            lines.append({"path": relative, "category": category, "source": source, "duration": n_samples / 16_000, "sr": 16_000})
    manifests = root / "manifests" / "musan"
    manifests.mkdir(parents=True)
    (manifests / "musan_val.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    (manifests / "musan_config.json").write_text(json.dumps({"musan_root": str(musan)}))

    for room in range(40):
        directory = root / "simulated_rirs_16k" / "smallroom" / f"Room{room:03d}"
        directory.mkdir(parents=True)
        for index in range(3):
            rir = np.zeros(400, dtype="float32")
            rir[index] = 1.0
            sf.write(directory / f"Room{room:03d}-{index}.wav", rir, SAMPLE_RATE)
    return root


def build(data_dir: Path, out_dir: Path | None = None) -> int:
    arguments = ["--data-dir", str(data_dir)]
    return main(arguments + (["--out-dir", str(out_dir)] if out_dir else []))


def built(data_dir: Path):
    assert build(data_dir) == 0
    return load_val_set(VERSION, data_dir)


def items_of(val_set) -> list[dict]:
    return [{"id": utterance_id} for utterance_id in val_set.entries]


# Génération unique


def test_two_builds_are_byte_identical(data_dir: Path, tmp_path: Path) -> None:
    assert build(data_dir, tmp_path / "a") == 0
    assert build(data_dir, tmp_path / "b") == 0
    for name in (f"{VERSION}.jsonl", f"{VERSION}_config.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_two_processes_with_different_hash_seeds_give_the_same_set(data_dir: Path, tmp_path: Path) -> None:
    outputs = []
    for hash_seed in ("1", "2"):
        out_dir = tmp_path / f"out_{hash_seed}"
        subprocess.run(
            [sys.executable, build_val_set.__file__, "--data-dir", str(data_dir), "--out-dir", str(out_dir)],
            check=True,
            capture_output=True,
            cwd=Path(build_val_set.__file__).parent,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        outputs.append((out_dir / f"{VERSION}.jsonl").read_bytes())
    assert outputs[0] == outputs[1]


def test_existing_version_is_never_overwritten(data_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert build(data_dir) == 0
    manifest = data_dir / "manifests" / "val" / f"{VERSION}.jsonl"
    before = manifest.read_bytes(), manifest.stat().st_mtime_ns
    assert build(data_dir) == 1
    assert "ne se régénère pas" in caplog.text
    assert (manifest.read_bytes(), manifest.stat().st_mtime_ns) == before


def test_modified_manifest_is_refused(data_dir: Path) -> None:
    assert build(data_dir) == 0
    manifest = data_dir / "manifests" / "val" / f"{VERSION}.jsonl"
    manifest.write_text(manifest.read_text().replace('"snr_db": 5', '"snr_db": 6', 1))
    with pytest.raises(ValueError, match="a changé depuis sa génération"):
        load_val_set(VERSION, data_dir)


def test_changed_musan_split_is_refused(data_dir: Path) -> None:
    """Un fichier retiré de la validation MUSAN peut être parti dans le train."""
    assert build(data_dir) == 0
    musan_val = data_dir / "manifests" / "musan" / "musan_val.jsonl"
    musan_val.write_text("".join(musan_val.read_text().splitlines(keepends=True)[1:]))
    with pytest.raises(ValueError, match="n'est plus le partage MUSAN"):
        load_val_set(VERSION, data_dir)


def test_missing_musan_split_is_refused_when_reading(data_dir: Path) -> None:
    assert build(data_dir) == 0
    (data_dir / "manifests" / "musan" / "musan_val.jsonl").unlink()
    with pytest.raises(ValueError, match="n'est plus le partage MUSAN"):
        load_val_set(VERSION, data_dir)


def test_identical_musan_split_is_accepted(data_dir: Path) -> None:
    """Seul le contenu compte : réécrire le même partage ne déclenche pas de refus."""
    assert build(data_dir) == 0
    musan_val = data_dir / "manifests" / "musan" / "musan_val.jsonl"
    musan_val.write_bytes(musan_val.read_bytes())
    assert load_val_set(VERSION, data_dir).spec == VAL_SETS[VERSION]


def test_missing_version_points_to_the_script(data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_val_set.py"):
        load_val_set(VERSION, data_dir)


def test_missing_musan_split_fails_clearly(data_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    (data_dir / "manifests" / "musan" / "musan_val.jsonl").unlink()
    assert build(data_dir) == 1
    assert "split_musan.py" in caplog.text


# Configuration figée


def test_val_v2_reference_values() -> None:
    """Écrit en toutes lettres : ces valeurs ne doivent plus jamais changer."""
    spec = VAL_SETS["val_v2"]
    assert (spec.p_noise, spec.p_rir, spec.duration_exponent, spec.seed) == (0.6, 0.3, 0.5, 1234)
    assert spec.subsets == ("dev-clean", "dev-other")
    assert [(c.name, c.probability, c.snrs_db) for c in spec.categories] == [
        ("noise", 0.4, (0, 5, 10)),
        ("music", 0.3, (5, 10, 15)),
        ("babble", 0.3, (13, 17, 20)),
    ]


def test_config_is_written_with_the_set_and_read_back(data_dir: Path) -> None:
    val_set = built(data_dir)
    assert val_set.spec == VAL_SETS[VERSION]
    assert val_set.sha256 == json.loads((data_dir / "manifests" / "val" / f"{VERSION}_config.json").read_text())["sha256"]


def test_reading_does_not_use_the_spec_of_the_script(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Modifier le script après coup ne change pas un jeu déjà généré."""
    assert build(data_dir) == 0
    before = load_val_set(VERSION, data_dir)
    monkeypatch.setitem(build_val_set.VAL_SETS, VERSION, replace(VAL_SETS[VERSION], p_rir=0.9, duration_exponent=0.0))
    after = load_val_set(VERSION, data_dir)
    assert after.spec == before.spec and after.spec.p_rir == 0.3
    assert after.entries == before.entries


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"subsets": ("test-clean",)}, "dev"),
        ({"subsets": ("dev-clean", "test-other")}, "dev"),
        ({"categories": (ValCategory("noise", 0.5, (0,), (0.0, 15.0)),)}, "sommer à 1"),
        ({"categories": (ValCategory("babble", 1.0, (0, 13), (13.0, 20.0)),)}, "hors de la plage"),
        ({"categories": (ValCategory("babble", 1.0, (13, 25), (13.0, 20.0)),)}, "hors de la plage"),
        ({"categories": (ValCategory("babble", 1.0, (), (13.0, 20.0)),)}, "hors de la plage"),
    ],
)
def test_invalid_spec_is_refused(change: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(VAL_SETS[VERSION], **change)


# Vue mélangée


def test_mixed_view_gives_one_condition_per_utterance(data_dir: Path) -> None:
    val_set = built(data_dir)
    items = items_of(val_set)
    assert len(val_set.mixed(items)) == len(items) == 120


def test_mixed_view_has_no_clean_condition(data_dir: Path) -> None:
    val_set = built(data_dir)
    names = {degradation.name for degradation in val_set.mixed(items_of(val_set))}
    assert NO_NOISE in names and CLEAN not in names and RIR_ONLY not in names


def test_mixed_view_keeps_reverberation_in_every_kind_of_condition(data_dir: Path) -> None:
    val_set = built(data_dir)
    mixed = val_set.mixed(items_of(val_set))
    assert any(d.rir for d in mixed if d.name == NO_NOISE)
    assert any(d.rir for d in mixed if d.name != NO_NOISE)
    assert any(d.rir is None for d in mixed)


def test_mixed_view_follows_the_stored_draws(data_dir: Path) -> None:
    val_set = built(data_dir)
    for degradation, entry in zip(val_set.mixed(items_of(val_set)), val_set.entries.values()):
        assert bool(degradation.rir) == entry["reverberant"]
        assert degradation.category == entry["category"]
        assert degradation.snr_db == entry["snr_db"]


def test_mixed_view_respects_item_order(data_dir: Path) -> None:
    val_set = built(data_dir)
    items = items_of(val_set)
    assert val_set.mixed(items[::-1]) == val_set.mixed(items)[::-1]


def test_unknown_utterance_is_refused(data_dir: Path) -> None:
    with pytest.raises(KeyError, match="absent"):
        built(data_dir).mixed([{"id": "1-1-0000"}])


def test_validation_draws_follow_training_proportions() -> None:
    generator = torch.Generator().manual_seed(0)
    draws = [draw_validation_noise(CATEGORIES, 0.6, generator) for _ in range(N_DRAWS)]
    counts = Counter(None if draw is None else draw[0].name for draw in draws)
    assert counts[None] / N_DRAWS == pytest.approx(0.4, abs=0.015)
    assert counts["noise"] / N_DRAWS == pytest.approx(0.24, abs=0.015)
    assert counts["music"] / N_DRAWS == pytest.approx(0.18, abs=0.015)
    assert counts["babble"] / N_DRAWS == pytest.approx(0.18, abs=0.015)


def test_validation_draws_use_each_grid_evenly() -> None:
    generator = torch.Generator().manual_seed(0)
    draws = [draw_validation_noise(CATEGORIES, 1.0, generator) for _ in range(N_DRAWS)]
    for category in CATEGORIES:
        counts = Counter(snr_db for drawn, snr_db in draws if drawn is category)
        assert set(counts) == set(category.snrs_db)
        share = category.probability * N_DRAWS / 3
        assert all(count == pytest.approx(share, rel=0.1) for count in counts.values())


# Vue grille


def test_grid_clean_is_really_clean(data_dir: Path) -> None:
    val_set = built(data_dir)
    for degradation in val_set.grid(items_of(val_set), CLEAN):
        assert degradation.rir is None and degradation.category is None and not degradation.sources


def test_grid_rir_condition_has_reverberation_only(data_dir: Path) -> None:
    val_set = built(data_dir)
    for degradation in val_set.grid(items_of(val_set), RIR_ONLY):
        assert degradation.rir is not None and degradation.category is None


def test_grid_noise_conditions_have_no_reverberation(data_dir: Path) -> None:
    val_set = built(data_dir)
    items = items_of(val_set)
    for name in val_set.grid_names:
        if name not in (CLEAN, RIR_ONLY):
            assert all(degradation.rir is None for degradation in val_set.grid(items, name))


def test_grid_lists_every_condition(data_dir: Path) -> None:
    assert built(data_dir).grid_names == [
        "clean", "rir_sim",
        "noise_0db", "noise_5db", "noise_10db",
        "music_5db", "music_10db", "music_15db",
        "babble_13db", "babble_17db", "babble_20db",
    ]  # fmt: skip


def test_same_noise_at_every_snr_of_a_category(data_dir: Path) -> None:
    val_set = built(data_dir)
    items = items_of(val_set)
    low, high = val_set.grid(items, "babble_13db"), val_set.grid(items, "babble_20db")
    assert [d.sources for d in low] == [d.sources for d in high]
    assert [d.snr_db for d in low] == [13] * len(items)


def test_mixed_and_grid_share_the_same_frozen_noise(data_dir: Path) -> None:
    val_set = built(data_dir)
    items = items_of(val_set)
    for item, degradation in zip(items, val_set.mixed(items)):
        if degradation.category:
            assert degradation.sources == val_set.grid([item], degradation.name)[0].sources


def test_unknown_condition_is_refused(data_dir: Path) -> None:
    with pytest.raises(ValueError, match="inconnue"):
        built(data_dir).grid([], "babble_0db")


# Fuites


def test_babble_never_contains_the_target_speaker(data_dir: Path) -> None:
    val_set = built(data_dir)
    for utterance_id, entry in val_set.entries.items():
        speakers = [Path(source["path"]).name.split("-")[0] for source in entry["sources"]["babble"]]
        assert 3 <= len(speakers) <= 7
        assert len(set(speakers)) == len(speakers)
        assert utterance_id.split("-")[0] not in speakers


def test_babble_comes_from_dev_only(data_dir: Path) -> None:
    val_set = built(data_dir)
    subsets = {
        Path(source["path"]).parts[0]
        for entry in val_set.entries.values()
        for source in entry["sources"]["babble"]
    }
    assert subsets == {"dev-clean", "dev-other"}


def test_test_speech_is_never_a_target(data_dir: Path) -> None:
    assert not any(utterance_id.startswith("900-") for utterance_id in built(data_dir).entries)


def test_musan_sources_match_their_category_and_exclude_speech(data_dir: Path) -> None:
    val_set = built(data_dir)
    for entry in val_set.entries.values():
        assert set(entry["sources"]) == {"noise", "music", "babble"}
        assert entry["sources"]["noise"][0]["path"].startswith("noise/")
        assert entry["sources"]["music"][0]["path"].startswith("music/")


def test_rir_rooms_are_reserved_and_rirs_come_from_them(data_dir: Path) -> None:
    val_set = built(data_dir)
    assert len(val_set.rir_rooms) == 4  # 10 % de 40 salles
    assert {str(Path(entry["rir"]).parent) for entry in val_set.entries.values()} <= set(val_set.rir_rooms)


def test_each_draw_has_its_own_generator() -> None:
    """Sans cela, le fichier de bruit et celui de musique d'un énoncé seraient corrélés."""
    spec = VAL_SETS[VERSION]
    purposes = ["mix", "rir", *[f"noise/{category.name}" for category in spec.categories]]
    seeds = {build_val_set.make_generator(spec, purpose, "1-1-0000").initial_seed() for purpose in purposes}
    assert len(seeds) == len(purposes)
    assert build_val_set.make_generator(spec, "mix", "1-1-0001").initial_seed() not in seeds


def test_build_uses_one_generator_per_category(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    purposes = set()
    original = build_val_set.make_generator

    def recording(spec: ValSetSpec, purpose: str, utterance_id: str) -> torch.Generator:
        purposes.add(purpose)
        return original(spec, purpose, utterance_id)

    monkeypatch.setattr(build_val_set, "make_generator", recording)
    assert build(data_dir) == 0
    assert purposes == {"mix", "rir", "noise/noise", "noise/music", "noise/babble"}


def test_files_are_weighted_by_the_frozen_exponent() -> None:
    weights = build_val_set.duration_weights([4.0, 100.0], VAL_SETS[VERSION].duration_exponent)
    assert (weights[1] / weights[0]).item() == pytest.approx(5.0)


# Rendu


def test_render_is_exact_and_needs_no_randomness(data_dir: Path) -> None:
    val_set = built(data_dir)
    items = items_of(val_set)[:10]
    for name in ("noise_5db", "music_10db", "babble_17db"):
        for item, degradation in zip(items, val_set.grid(items, name)):
            n_samples = val_set.entries[item["id"]]["frames"]
            first = render_noise(degradation, n_samples)
            torch.manual_seed(123)
            assert first.size(0) == n_samples and first.isfinite().all()
            assert torch.equal(first, render_noise(degradation, n_samples))


def test_render_normalizes_each_talker_before_summing(data_dir: Path) -> None:
    val_set = built(data_dir)
    degradation = val_set.grid(items_of(val_set)[:1], "babble_13db")[0]
    expected = torch.zeros(1_000)
    for path, start in degradation.sources:
        audio, _ = sf.read(path, dtype="float32")
        expected += cut(normalize_rms(torch.from_numpy(audio)), start, 1_000)
    assert torch.equal(render_noise(degradation, 1_000), expected)
    assert expected.pow(2).mean().sqrt() > 1.0  # un énoncé brut a ici un RMS d'environ 0,3


def test_render_uses_the_stored_offset(data_dir: Path) -> None:
    val_set = built(data_dir)
    degradation = val_set.grid(items_of(val_set)[:1], "music_10db")[0]
    path, start = degradation.sources[0]
    audio, _ = sf.read(path, dtype="float32")
    expected = torch.from_numpy(audio[start : start + 500])
    assert torch.allclose(render_noise(degradation, 500), expected)


# Résumé de la grille


def grid_wers() -> dict[str, float]:
    return {
        "clean": 0.10, "rir_sim": 0.99,
        "noise_0db": 0.30, "noise_5db": 0.20, "noise_10db": 0.10,
        "music_5db": 0.40, "music_10db": 0.30, "music_15db": 0.20,
        "babble_13db": 0.60, "babble_17db": 0.50, "babble_20db": 0.40,
    }  # fmt: skip


def test_grid_summary_matches_the_written_formula() -> None:
    expected = 0.4 * 0.10 + 0.6 * (0.4 * 0.20 + 0.3 * 0.30 + 0.3 * 0.50)
    assert grid_summary_wer(grid_wers(), CATEGORIES, 0.6) == pytest.approx(expected)


def test_grid_summary_ignores_rir_sim() -> None:
    wers = grid_wers()
    wers["rir_sim"] = 0.0
    assert grid_summary_wer(wers, CATEGORIES, 0.6) == grid_summary_wer(grid_wers(), CATEGORIES, 0.6)


def test_category_wers_average_over_snrs_on_either_base() -> None:
    assert category_wers(grid_wers(), CATEGORIES) == pytest.approx(
        {"clean": 0.10, "noise": 0.20, "music": 0.30, "babble": 0.50}
    )
    wers = {(NO_NOISE if name == CLEAN else name): wer for name, wer in grid_wers().items()}
    assert category_wers(wers, CATEGORIES, base=NO_NOISE)[NO_NOISE] == 0.10


def test_incomplete_results_are_refused() -> None:
    wers = grid_wers()
    del wers["clean"]
    with pytest.raises(ValueError, match="clean"):
        category_wers(wers, CATEGORIES)
    wers = {name: wer for name, wer in grid_wers().items() if not name.startswith("music")}
    with pytest.raises(ValueError, match="music"):
        category_wers(wers, CATEGORIES)


def test_condition_names_are_stable() -> None:
    assert condition_name("noise", 5) == "noise_5db"
    assert condition_name("babble", 13.0) == "babble_13db"


def test_logging_reports_what_was_written(data_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="build_val_set"):
        assert build(data_dir) == 0
    assert "120 énoncés" in caplog.text and "4 salles" in caplog.text
