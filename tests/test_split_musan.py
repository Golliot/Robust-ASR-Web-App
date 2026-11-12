"""Tests du partage MUSAN, sur un mini-arbre factice généré dans un dossier temporaire."""

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import split_musan
from split_musan import AudioFile, ValidationError, check_disjoint, check_groups, inventory, main, split_files

SCRIPT = Path(split_musan.__file__)
GROUP_SIZES = {
    "music/fma": 20,
    "music/rfm": 10,
    "noise/free-sound": 25,
    "noise/sound-bible": 10,
    "speech/librivox": 15,
    "speech/us-gov": 10,
}


def write_wav(path: Path, sample_rate: int = 16_000, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = sample_rate // 100
    audio = np.zeros(samples if channels == 1 else (samples, channels), dtype="float32")
    sf.write(path, audio, sample_rate)


@pytest.fixture
def musan_root(tmp_path: Path) -> Path:
    root = tmp_path / "musan"
    for group, size in GROUP_SIZES.items():
        for index in range(size):
            write_wav(root / group / f"{index:03d}.wav")
    (root / "README").write_text("pas un fichier audio")
    return root


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def split_paths(out_dir: Path) -> tuple[list[str], list[str]]:
    return tuple([entry["path"] for entry in read_manifest(out_dir / f"musan_{name}.jsonl")] for name in ("train", "val"))


def run_split(root: Path, out_dir: Path, *options: str) -> int:
    return main(["--root", str(root), "--out-dir", str(out_dir), *options])


def snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def fake_file(path: str, category: str = "noise", source: str = "free-sound") -> AudioFile:
    return AudioFile(path=path, category=category, source=source, duration=1.0, sr=16_000, channels=1)


# Déterminisme


def test_two_executions_give_identical_outputs(musan_root: Path, tmp_path: Path) -> None:
    """Deux vrais processus, avec des hachages Python différents, produisent les mêmes fichiers."""
    outputs = []
    for hash_seed in ("1", "2"):
        out_dir = tmp_path / f"out_{hash_seed}"
        subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(musan_root), "--out-dir", str(out_dir)],
            check=True,
            capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        outputs.append(out_dir)

    for name in ("musan_train.jsonl", "musan_val.jsonl", "musan_split_summary.md"):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()


def test_split_does_not_depend_on_input_order() -> None:
    files = [fake_file(f"noise/free-sound/{index}.wav") for index in range(50)]
    shuffled = files[:]
    random.Random(0).shuffle(shuffled)
    assert split_files(files, 0.2, seed=42) == split_files(shuffled, 0.2, seed=42)


def test_seed_changes_the_split(musan_root: Path, tmp_path: Path) -> None:
    assert run_split(musan_root, tmp_path / "a", "--seed", "1") == 0
    assert run_split(musan_root, tmp_path / "b", "--seed", "2") == 0
    assert split_paths(tmp_path / "a") != split_paths(tmp_path / "b")


# Recouvrement


def test_splits_do_not_overlap_and_cover_kept_files(musan_root: Path, tmp_path: Path) -> None:
    assert run_split(musan_root, tmp_path / "out") == 0
    train, val = split_paths(tmp_path / "out")
    assert not set(train) & set(val)
    assert len(train) + len(val) == sum(GROUP_SIZES.values()) - GROUP_SIZES["speech/librivox"]


def test_check_disjoint_detects_overlap() -> None:
    shared = fake_file("noise/free-sound/shared.wav")
    with pytest.raises(ValidationError, match="shared.wav"):
        check_disjoint([shared], [shared])


# LibriVox


def test_librivox_excluded_by_default(musan_root: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="split_musan"):
        assert run_split(musan_root, tmp_path / "out") == 0

    train, val = split_paths(tmp_path / "out")
    assert not any("librivox" in path.lower() for path in train + val)
    assert "LibriVox exclu : 15 fichiers" in caplog.text
    summary = (tmp_path / "out" / "musan_split_summary.md").read_text()
    assert "Sources exclues : speech/librivox (15 fichiers" in summary


def test_librivox_exclusion_ignores_case(musan_root: Path, tmp_path: Path) -> None:
    (musan_root / "speech" / "librivox").rename(musan_root / "speech" / "LibriVox")
    assert run_split(musan_root, tmp_path / "out") == 0
    train, val = split_paths(tmp_path / "out")
    assert not any("librivox" in path.lower() for path in train + val)


def test_keep_librivox(musan_root: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="split_musan"):
        assert run_split(musan_root, tmp_path / "out", "--keep-librivox") == 0

    train, val = split_paths(tmp_path / "out")
    assert sum("speech/librivox" in path for path in train + val) == 15
    assert "--keep-librivox" in caplog.text
    assert "Sources exclues : aucune" in (tmp_path / "out" / "musan_split_summary.md").read_text()


def test_keep_librivox_leaves_other_sources_unchanged(musan_root: Path, tmp_path: Path) -> None:
    assert run_split(musan_root, tmp_path / "without") == 0
    assert run_split(musan_root, tmp_path / "with", "--keep-librivox") == 0
    without = split_paths(tmp_path / "without")
    with_librivox = split_paths(tmp_path / "with")
    for kept, full in zip(without, with_librivox):
        assert kept == [path for path in full if "librivox" not in path]


def test_missing_librivox_is_reported(musan_root: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (musan_root / "speech" / "librivox").rename(musan_root / "speech" / "other-readings")
    with caplog.at_level(logging.INFO, logger="split_musan"):
        assert run_split(musan_root, tmp_path / "out") == 0
    assert "Aucun dossier LibriVox" in caplog.text


# Stratification


@pytest.mark.parametrize("val_ratio", [0.2, 0.3])
def test_ratio_applied_in_each_group(musan_root: Path, tmp_path: Path, val_ratio: float) -> None:
    assert run_split(musan_root, tmp_path / "out", "--val-ratio", str(val_ratio)) == 0
    train, val = split_paths(tmp_path / "out")

    for group, size in GROUP_SIZES.items():
        if group == "speech/librivox":
            continue
        n_train = sum(path.startswith(f"{group}/") for path in train)
        n_val = sum(path.startswith(f"{group}/") for path in val)
        assert n_train + n_val == size
        assert n_val == round(size * val_ratio)


def test_manifest_entries_describe_their_group(musan_root: Path, tmp_path: Path) -> None:
    assert run_split(musan_root, tmp_path / "out") == 0
    for entry in read_manifest(tmp_path / "out" / "musan_train.jsonl"):
        assert set(entry) == {"path", "category", "source", "duration", "sr"}
        category, source, _ = entry["path"].split("/")
        assert (entry["category"], entry["source"]) == (category, source)
        assert entry["sr"] == 16_000
        assert entry["duration"] == pytest.approx(0.01)


def test_small_group_warns_but_stays_in_both_splits(
    musan_root: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    write_wav(musan_root / "noise" / "tiny" / "a.wav")
    write_wav(musan_root / "noise" / "tiny" / "b.wav")
    with caplog.at_level(logging.INFO, logger="split_musan"):
        assert run_split(musan_root, tmp_path / "out") == 0

    train, val = split_paths(tmp_path / "out")
    assert sum("noise/tiny/" in path for path in train) == 1
    assert sum("noise/tiny/" in path for path in val) == 1
    assert "trop petit" in caplog.text and "noise/tiny" in caplog.text


def test_single_file_group_warns_and_stays_in_train(
    musan_root: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    write_wav(musan_root / "noise" / "lonely" / "only.wav")
    with caplog.at_level(logging.INFO, logger="split_musan"):
        assert run_split(musan_root, tmp_path / "out") == 0

    train, val = split_paths(tmp_path / "out")
    assert "noise/lonely/only.wav" in train
    assert "noise/lonely/only.wav" not in val
    assert "noise/lonely" in caplog.text


def test_check_groups_detects_missing_group() -> None:
    train = [fake_file("noise/free-sound/a.wav"), fake_file("noise/free-sound/b.wav")]
    with pytest.raises(ValidationError, match="noise/free-sound"):
        check_groups(train, [])


# Format et portabilité


@pytest.mark.parametrize(
    ("sample_rate", "channels", "expected"),
    [(8_000, 1, "8000 Hz, 1 canal"), (16_000, 2, "16000 Hz, 2 canal")],
)
def test_wrong_format_fails_and_lists_the_file(
    musan_root: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    sample_rate: int,
    channels: int,
    expected: str,
) -> None:
    write_wav(musan_root / "noise" / "free-sound" / "bad.wav", sample_rate, channels)
    assert run_split(musan_root, tmp_path / "out") == 1
    assert "noise/free-sound/bad.wav" in caplog.text
    assert expected in caplog.text
    assert not (tmp_path / "out").exists()


def test_excluded_files_are_not_format_checked(musan_root: Path, tmp_path: Path) -> None:
    write_wav(musan_root / "speech" / "librivox" / "bad.wav", 8_000)
    assert run_split(musan_root, tmp_path / "out") == 0


def test_unreadable_file_fails_clearly(musan_root: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (musan_root / "noise" / "free-sound" / "broken.wav").write_bytes(b"ceci n'est pas un wav")
    assert run_split(musan_root, tmp_path / "out") == 1
    assert "noise/free-sound/broken.wav" in caplog.text


def test_misplaced_file_fails_clearly(musan_root: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    write_wav(musan_root / "unknown" / "source" / "a.wav")
    assert run_split(musan_root, tmp_path / "out") == 1
    assert "unknown/source/a.wav" in caplog.text


def test_missing_root_fails_clearly(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert run_split(tmp_path / "absent", tmp_path / "out") == 1
    assert "download_noise.py" in caplog.text


def test_paths_are_relative_and_root_is_in_config(musan_root: Path, tmp_path: Path) -> None:
    assert run_split(musan_root, tmp_path / "out", "--seed", "7", "--val-ratio", "0.3") == 0

    config = json.loads((tmp_path / "out" / "musan_config.json").read_text())
    assert config == {"musan_root": str(musan_root), "val_ratio": 0.3, "seed": 7, "keep_librivox": False}
    for path in sum(split_paths(tmp_path / "out"), []):
        assert not Path(path).is_absolute()
        assert (Path(config["musan_root"]) / path).is_file()


def test_summary_reports_each_group_seed_and_ratio(musan_root: Path, tmp_path: Path) -> None:
    assert run_split(musan_root, tmp_path / "out", "--seed", "7") == 0
    summary = (tmp_path / "out" / "musan_split_summary.md").read_text()
    assert "Graine : 7" in summary
    assert "0.8/0.2" in summary
    assert "| music | fma | 16 |" in summary
    assert "| speech | us-gov | 8 |" in summary
    assert "| **Total** |  | 60 |" in summary


@pytest.mark.parametrize("value", ["0", "1", "-0.1", "1.5"])
def test_invalid_ratio_is_refused(value: str) -> None:
    with pytest.raises(SystemExit):
        split_musan.parse_args(["--val-ratio", value])


# Fichiers originaux


def test_originals_are_untouched(musan_root: Path, tmp_path: Path) -> None:
    before = snapshot(musan_root)
    assert run_split(musan_root, tmp_path / "out") == 0
    assert snapshot(musan_root) == before


def test_out_dir_inside_musan_is_refused(musan_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    before = snapshot(musan_root)
    assert run_split(musan_root, musan_root / "manifests") == 1
    assert "ne doit pas être dans MUSAN" in caplog.text
    assert snapshot(musan_root) == before


def test_inventory_never_reads_audio(musan_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("l'audio ne doit pas être chargé")

    monkeypatch.setattr(sf, "read", forbidden)
    monkeypatch.setattr(sf.SoundFile, "read", forbidden)
    assert len(inventory(musan_root)) == sum(GROUP_SIZES.values())
