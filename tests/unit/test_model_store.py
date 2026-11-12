"""Tests des versions de modèle : config.yaml, intégrité des fichiers, cohérence de la CMVN."""

from dataclasses import fields
from pathlib import Path

import pytest
import torch
import yaml

from src.asr import model_store
from src.asr.config import ModelConfig
from src.asr.model_store import (
    ModelConfigError,
    ModelIntegrityError,
    list_versions,
    load_weights,
    parse_spec,
    positional_limit_s,
    read_spec,
)
from tests.support.models import TINY_MODEL, tiny_config, tiny_state_dict, write_model_dir

NOTEBOOK_CONFIG = {
    "sample_rate": 16_000, "n_mels": 80, "n_fft": 512, "win_length": 400, "hop_length": 160,
    "freq_masks": 2, "freq_mask_width": 27, "time_masks": 2, "time_mask_width": 40,
    "d_model": 256, "n_heads": 4, "d_ffn": 2048, "encoder_layers": 12, "decoder_layers": 6,
    "conv_channels": 32, "dropout": 0.1, "qk_norm": True, "ctc_weight": 0.3, "label_smoothing": 0.1,
    "beam_size": 10,
}  # fmt: skip


def test_model_config_defaults_are_the_notebook_values() -> None:
    """Écrites en toutes lettres : le test de non-régression les compare aussi au Config du notebook."""
    assert ModelConfig().to_dict() == NOTEBOOK_CONFIG


def test_model_config_requires_every_field() -> None:
    values = dict(NOTEBOOK_CONFIG)
    del values["n_heads"]
    with pytest.raises(ValueError, match="champs absents : n_heads"):
        ModelConfig.from_dict(values)
    with pytest.raises(ValueError, match="champs inconnus : n_layers"):
        ModelConfig.from_dict({**NOTEBOOK_CONFIG, "n_layers": 3})


@pytest.mark.parametrize(("name", "value"), [("d_model", 256.0), ("d_model", True), ("qk_norm", 1), ("dropout", "0.1")])
def test_model_config_checks_types(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"model.{name}"):
        ModelConfig.from_dict({**NOTEBOOK_CONFIG, name: value})


def test_model_config_accepts_integers_for_floats() -> None:
    config = ModelConfig.from_dict({**NOTEBOOK_CONFIG, "dropout": 0})
    assert config.dropout == 0.0 and isinstance(config.dropout, float)
    assert [field.name for field in fields(config)] == list(NOTEBOOK_CONFIG)


def test_positional_limit_is_about_240_seconds() -> None:
    assert 239 < positional_limit_s(ModelConfig()) < 240


def test_valid_config_is_parsed(tmp_path: Path) -> None:
    spec = parse_spec(tiny_config(), tmp_path / "v1")
    assert spec.version == "v1"
    assert spec.model == ModelConfig.from_dict(TINY_MODEL)
    assert spec.decoding.method == "joint_beam_search"
    assert spec.limits.max_duration_s == 30.0
    assert spec.checkpoint_path == tmp_path / "v1" / "best.pt"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"version": "v2"}, "différente du nom du dossier"),
        ({"format": 2}, "format 2 non pris en charge"),
        ({"extra": 1}, "sections inconnues : extra"),
        ({"tokenizer": {"chars": "ABC"}}, "tokenizer.chars"),
        ({"files": {"checkpoint": "../best.pt", "cmvn": "cmvn.pt"}}, "sans chemin"),
        ({"files": {"checkpoint": "best.pt"}}, "files.cmvn"),
        ({"decoding": {"method": "greedy", "beam_size": 1, "ctc_weight": 0.3}}, "decoding.method"),
        ({"decoding": {"method": "joint_beam_search", "beam_size": 0, "ctc_weight": 0.3}}, "beam_size"),
        ({"decoding": {"method": "joint_beam_search", "beam_size": 4, "ctc_weight": 0.0}}, "ctc_weight"),
        (
            {"decoding": {"method": "joint_beam_search", "beam_size": 4, "ctc_weight": 0.3, "use_corrector": True}},
            "corrector_transcripts",
        ),
        ({"limits": {"min_duration_s": 0.01, "max_duration_s": 30.0}}, "256 échantillons"),
        ({"limits": {"min_duration_s": 0.1, "max_duration_s": 300.0}}, "encodage positionnel"),
        ({"limits": {"min_duration_s": 0.1}}, "limits.max_duration_s"),
    ],
)
def test_invalid_config_is_refused(tmp_path: Path, change: dict, message: str) -> None:
    raw = tiny_config()
    raw.update(change)
    with pytest.raises(ModelConfigError, match=message):
        parse_spec(raw, tmp_path / "v1")


def test_missing_section_is_refused(tmp_path: Path) -> None:
    raw = tiny_config()
    del raw["limits"]
    with pytest.raises(ModelConfigError, match="sections absentes : limits"):
        parse_spec(raw, tmp_path / "v1")


def test_read_spec_reports_missing_or_broken_files(tmp_path: Path) -> None:
    with pytest.raises(ModelConfigError, match="introuvable"):
        read_spec(tmp_path)
    (tmp_path / "config.yaml").write_text("model: [", encoding="utf-8")
    with pytest.raises(ModelConfigError, match="YAML valide"):
        read_spec(tmp_path)


def test_weights_are_loaded_and_verified(tmp_path: Path) -> None:
    directory = write_model_dir(tmp_path)
    weights = load_weights(read_spec(directory))
    expected = tiny_state_dict()
    assert weights.state_dict.keys() == expected.keys()
    assert all(torch.equal(weights.state_dict[name], tensor) for name, tensor in expected.items())
    assert weights.metadata == {"epoch": 3}


def test_training_checkpoint_is_accepted_without_its_optimizer(tmp_path: Path) -> None:
    checkpoint = {"model": tiny_state_dict(), "optimizer": {"state": {}}, "history": [], "epoch": 59, "val_set": "val_v2"}
    weights = load_weights(read_spec(write_model_dir(tmp_path, checkpoint=checkpoint)))
    assert weights.metadata == {"epoch": 59, "val_set": "val_v2"}


def test_plain_state_dict_is_accepted(tmp_path: Path) -> None:
    weights = load_weights(read_spec(write_model_dir(tmp_path, checkpoint=tiny_state_dict())))
    assert "encoder.norm.weight" in weights.state_dict


def test_modified_checkpoint_is_refused(tmp_path: Path) -> None:
    directory = write_model_dir(tmp_path)
    torch.save({"model": tiny_state_dict(seed=1)}, directory / "best.pt")
    with pytest.raises(ModelIntegrityError, match="empreinte"):
        load_weights(read_spec(directory))


def test_foreign_cmvn_is_refused(tmp_path: Path) -> None:
    state_dict = tiny_state_dict()
    cmvn = {"mean": state_dict["logmel.mean"] + 0.5, "std": state_dict["logmel.std"]}
    directory = write_model_dir(tmp_path, cmvn=cmvn)
    with pytest.raises(ModelIntegrityError, match="diffère de la CMVN du checkpoint"):
        load_weights(read_spec(directory))


def test_cmvn_file_fills_a_checkpoint_without_cmvn(tmp_path: Path) -> None:
    state_dict = tiny_state_dict()
    mean, std = state_dict.pop("logmel.mean"), state_dict.pop("logmel.std")
    directory = write_model_dir(tmp_path, checkpoint={"model": state_dict}, cmvn={"mean": mean, "std": std})
    weights = load_weights(read_spec(directory))
    assert torch.equal(weights.state_dict["logmel.mean"], mean)


def test_unrecognized_checkpoint_is_refused(tmp_path: Path) -> None:
    directory = write_model_dir(tmp_path, checkpoint={"weights": [1, 2, 3]})
    with pytest.raises(ModelIntegrityError, match="state_dict"):
        load_weights(read_spec(directory))


def test_missing_files_are_reported(tmp_path: Path) -> None:
    directory = write_model_dir(tmp_path)
    (directory / "cmvn.pt").unlink()
    with pytest.raises(ModelIntegrityError, match="fichiers absents .* cmvn.pt"):
        load_weights(read_spec(directory))


def test_checkpoint_is_read_without_arbitrary_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model_store.torch, "load", spy)
    load_weights(read_spec(write_model_dir(tmp_path)))
    assert calls == [True, True]


def test_list_versions_reports_each_version(tmp_path: Path) -> None:
    write_model_dir(tmp_path, "v1")
    broken = write_model_dir(tmp_path, "v2")
    (broken / "best.pt").unlink()
    (tmp_path / "v3").mkdir()
    (tmp_path / "v3" / "config.yaml").write_text(yaml.safe_dump({**tiny_config("v3"), "format": 9}), encoding="utf-8")
    (tmp_path / "notes").mkdir()  # sans config.yaml : ignoré

    entries = {entry.version: entry for entry in list_versions(tmp_path)}
    assert set(entries) == {"v1", "v2", "v3"}
    assert entries["v1"].error is None and entries["v1"].spec is not None
    assert "best.pt" in entries["v2"].error
    assert entries["v3"].spec is None and "format 9" in entries["v3"].error
    assert list_versions(tmp_path / "absent") == []
