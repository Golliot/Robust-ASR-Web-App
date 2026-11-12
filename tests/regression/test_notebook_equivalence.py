"""Non-régression du moteur : mêmes fichiers audio, mêmes résultats que le code du notebook.

Sur un même périphérique, le moteur doit reproduire le notebook à l'identique : prétraitement, CMVN,
poids, features, log-probabilités CTC, tokens du beam search et texte. Entre CPU (fp32) et GPU
(autocast bf16), les calculs diffèrent légèrement. Les critères adaptés et leurs seuils, environ 5 à
10 fois les écarts mesurés sur les fixtures, sont regroupés dans test_cpu_and_gpu_agree.
"""

import json
from dataclasses import fields
from pathlib import Path

import jiwer
import pytest
import torch

from notebook_reference import build_model, load_namespace
from src.asr.config import ModelConfig
from src.asr.decoding import joint_beam_search
from src.asr.engine import ASREngine
from src.asr.model import autocast
from src.asr.model_store import load_weights, read_spec
from src.audio.decoding import decode_audio
from src.audio.preprocessing import prepare_waveform

GOLDEN = json.loads((Path(__file__).parent / "golden_v1.json").read_text(encoding="utf-8"))
LIMITS = {"min_duration_s": 0.1, "max_duration_s": 30.0}


def engine_waveform(path: Path) -> torch.Tensor:
    return prepare_waveform(decode_audio(path), **LIMITS)


@pytest.fixture(scope="module")
def notebook_cpu() -> dict:
    return load_namespace(torch.device("cpu"))


@pytest.fixture(scope="module")
def state_dict(model_dir: Path) -> dict:
    return load_weights(read_spec(model_dir)).state_dict


@pytest.fixture(scope="module")
def notebook_model_cpu(notebook_cpu: dict, state_dict: dict) -> torch.nn.Module:
    return build_model(notebook_cpu, state_dict)


@pytest.fixture(scope="module")
def engine_cpu(model_dir: Path) -> ASREngine:
    return ASREngine.load(model_dir, device="cpu")


@torch.no_grad()
def ctc_log_probs(model, wav: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Features et log-probabilités CTC d'un énoncé, calculées comme dans le notebook."""
    wav = wav.to(device)[None]
    lens = torch.tensor([wav.size(1)], device=device)
    feats, feat_lens = model.features(wav, lens)
    with autocast(device.type):
        memory, _ = model.encode(feats, feat_lens)
        log_probs = model.ctc_head(memory).float().log_softmax(-1)
    return feats[0].cpu(), log_probs[0].cpu()


# --- Configuration, poids, CMVN, prétraitement ------------------------------------------------------


def test_model_config_is_the_notebook_config(notebook_cpu: dict) -> None:
    config = ModelConfig()
    for field in fields(config):
        assert getattr(config, field.name) == getattr(notebook_cpu["cfg"], field.name), field.name


def test_exported_config_is_the_notebook_config(model_dir: Path, notebook_cpu: dict) -> None:
    spec = read_spec(model_dir)
    assert spec.model == ModelConfig()
    assert spec.chars == notebook_cpu["tok"].chars
    assert spec.decoding.beam_size == notebook_cpu["cfg"].beam_size
    assert spec.decoding.ctc_weight == notebook_cpu["cfg"].ctc_weight


def test_weights_and_cmvn_are_those_of_the_research_checkpoint(state_dict: dict, research_dir: Path) -> None:
    source = torch.load(research_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)["model"]
    assert state_dict.keys() == source.keys()
    for name, tensor in source.items():
        assert torch.equal(state_dict[name], tensor), name
    cmvn = torch.load(research_dir / "data" / "manifests" / "cmvn.pt", map_location="cpu", weights_only=True)
    assert torch.equal(state_dict["logmel.mean"], cmvn["mean"])
    assert torch.equal(state_dict["logmel.std"], cmvn["std"])


def test_preprocessing_matches_load_audio(notebook_cpu: dict, references: list[dict]) -> None:
    for item in references:
        assert torch.equal(engine_waveform(item["path"]), notebook_cpu["load_audio"](str(item["path"]))), item["id"]


def test_engine_model_has_the_notebook_parameters(engine_cpu: ASREngine, notebook_model_cpu) -> None:
    expected = notebook_model_cpu.state_dict()
    actual = engine_cpu.model.state_dict()
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name], tensor) for name, tensor in expected.items())


# --- CPU : identité stricte -------------------------------------------------------------------------


def test_features_and_ctc_log_probs_are_identical_on_cpu(engine_cpu, notebook_model_cpu, references) -> None:
    cpu = torch.device("cpu")
    for item in references:
        wav = engine_waveform(item["path"])
        feats, log_probs = ctc_log_probs(engine_cpu.model, wav, cpu)
        expected_feats, expected_log_probs = ctc_log_probs(notebook_model_cpu, wav, cpu)
        assert torch.equal(feats, expected_feats), item["id"]
        assert torch.equal(log_probs, expected_log_probs), item["id"]


def test_beam_search_tokens_are_identical_on_cpu(engine_cpu, notebook_cpu, notebook_model_cpu, references) -> None:
    for item in references:
        wav = engine_waveform(item["path"])[None]
        lens = torch.tensor([wav.size(1)])
        expected = notebook_cpu["joint_beam_search"](notebook_model_cpu, wav, lens)
        assert joint_beam_search(engine_cpu.model, wav, lens) == expected, item["id"]


def test_transcriptions_match_the_notebook_on_cpu(engine_cpu, notebook_cpu, notebook_model_cpu, references) -> None:
    for item in references:
        expected = notebook_cpu["transcribe"](str(item["path"]))
        assert engine_cpu.transcribe_file(item["path"]).text == expected, item["id"]
        assert expected == GOLDEN["utterances"][item["id"]]["cpu"], item["id"]


def test_golden_file_matches_the_exported_model(model_dir: Path) -> None:
    assert GOLDEN["checkpoint_sha256"] == read_spec(model_dir).sha256["checkpoint"]


# --- GPU : identité avec le notebook, tolérances avec le CPU ---------------------------------------


@pytest.mark.gpu
def test_transcriptions_match_the_notebook_on_gpu(model_dir: Path, state_dict: dict, references) -> None:
    cuda = torch.device("cuda")
    notebook = load_namespace(cuda)
    notebook_model = build_model(notebook, state_dict)
    engine = ASREngine.load(model_dir, device="cuda")
    for item in references:
        wav = engine_waveform(item["path"]).to(cuda)[None]
        lens = torch.tensor([wav.size(1)], device=cuda)
        assert joint_beam_search(engine.model, wav, lens) == notebook["joint_beam_search"](notebook_model, wav, lens)
        expected = notebook["transcribe"](str(item["path"]))
        assert engine.transcribe_file(item["path"]).text == expected, item["id"]
        assert expected == GOLDEN["utterances"][item["id"]]["cuda"], item["id"]


@pytest.mark.gpu
def test_cpu_and_gpu_agree(model_dir: Path, engine_cpu: ASREngine, references) -> None:
    """Critères adaptés au bf16, avec leurs seuils.

    Écarts mesurés sur les 5 fixtures :
    - features : au plus 1,2e-3 ;
    - divergence KL moyenne par trame : au plus 1,7e-4 (maximum 9,4e-3) ;
    - argmax identique sur 98,8 % des trames au moins ;
    - textes identiques.
    """
    engine_gpu = ASREngine.load(model_dir, device="cuda")
    cpu_texts, gpu_texts = [], []
    for item in references:
        wav = engine_waveform(item["path"])
        feats_cpu, log_probs_cpu = ctc_log_probs(engine_cpu.model, wav, torch.device("cpu"))
        feats_gpu, log_probs_gpu = ctc_log_probs(engine_gpu.model, wav, torch.device("cuda"))
        # Features en float32 des deux côtés (hors autocast) : seules les FFT diffèrent.
        assert (feats_cpu - feats_gpu).abs().max() < 1e-2, item["id"]
        kl = (log_probs_cpu.exp() * (log_probs_cpu - log_probs_gpu)).sum(-1)
        assert kl.mean() < 1e-3, item["id"]
        assert kl.max() < 0.1, item["id"]
        agreement = (log_probs_cpu.argmax(-1) == log_probs_gpu.argmax(-1)).float().mean()
        assert agreement >= 0.97, item["id"]
        cpu_texts.append(engine_cpu.transcribe_file(item["path"]).text)
        gpu_texts.append(engine_gpu.transcribe_file(item["path"]).text)
    assert jiwer.wer(cpu_texts, gpu_texts) <= 0.02
