"""Tests du moteur sur un mini-modèle : API Python, limites, verrou et repli CPU sans rechargement."""

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.asr import engine as engine_module
from src.asr import model_store
from src.asr.engine import ASREngine, Transcription
from src.audio.errors import AudioTooLongError, AudioTooShortError, InvalidAudioError
from src.runtime.device_manager import DeviceInfo
from tests.support.audio import encoded, soundfile_bytes, tone
from tests.support.models import write_model_dir


@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    return write_model_dir(tmp_path)


@pytest.fixture
def engine(tiny_dir: Path) -> ASREngine:
    return ASREngine.load(tiny_dir, device="cpu")


def test_signal_is_transcribed(engine: ASREngine) -> None:
    result = engine.transcribe(tone(seconds=1.0))
    assert isinstance(result, Transcription)
    assert isinstance(result.text, str)
    assert result.audio_duration == 1.0
    assert result.device == "cpu"
    assert result.model_version == "v1"
    assert result.sample_rate == 16_000
    assert result.inference_time > 0
    assert result.preprocessing_time >= 0
    assert result.processing_time == pytest.approx(result.inference_time + result.preprocessing_time)
    assert result.input_format is None


def test_numpy_and_torch_signals_give_the_same_text(engine: ASREngine) -> None:
    signal = tone(seconds=1.0)
    assert engine.transcribe(signal).text == engine.transcribe(torch.from_numpy(signal)).text


def test_signal_at_another_rate_is_resampled(engine: ASREngine) -> None:
    result = engine.transcribe(tone(seconds=1.0, sample_rate=48_000), sample_rate=48_000)
    assert result.audio_duration == pytest.approx(1.0)


def test_file_and_bytes_give_the_same_text(engine: ASREngine, tmp_path: Path) -> None:
    path = tmp_path / "speech.wav"
    sf.write(path, tone(seconds=1.0), 16_000)
    from_file, from_bytes = engine.transcribe_file(path), engine.transcribe_bytes(path.read_bytes())
    assert from_file.text == from_bytes.text
    assert from_file.input_format == "WAV (PCM_16)"


def test_browser_formats_are_accepted(engine: ASREngine) -> None:
    result = engine.transcribe_bytes(encoded("webm_opus", tone(seconds=1.0, sample_rate=48_000), 48_000))
    assert result.input_format == "WebM/Matroska (opus)"
    assert result.audio_duration == pytest.approx(1.0, abs=0.1)


def test_duration_limits(engine: ASREngine) -> None:
    with pytest.raises(AudioTooShortError):
        engine.transcribe(np.zeros(800, dtype=np.float32))
    with pytest.raises(AudioTooLongError):
        engine.transcribe(np.zeros(31 * 16_000, dtype=np.float32))
    with pytest.raises(AudioTooLongError):
        engine.transcribe_bytes(soundfile_bytes(np.zeros(45 * 16_000, dtype=np.float32), 16_000))


def test_application_can_only_lower_the_model_limit(tiny_dir: Path) -> None:
    lowered = ASREngine.load(tiny_dir, device="cpu", max_duration_s=5.0)
    assert lowered.max_duration_s == 5.0
    with pytest.raises(AudioTooLongError, match="limite est de 5 s"):
        lowered.transcribe(np.zeros(6 * 16_000, dtype=np.float32))
    assert ASREngine.load(tiny_dir, device="cpu", max_duration_s=100.0).max_duration_s == 30.0


@pytest.mark.parametrize(
    "signal",
    [np.zeros(16_000, dtype=np.int16), np.zeros((2, 2, 16_000), dtype=np.float32), np.full(16_000, np.nan, np.float32)],
    ids=["entiers", "3D", "NaN"],
)
def test_invalid_signals_are_refused(engine: ASREngine, signal: np.ndarray) -> None:
    with pytest.raises(InvalidAudioError):
        engine.transcribe(signal)


def test_inferences_never_overlap(engine: ASREngine, monkeypatch: pytest.MonkeyPatch) -> None:
    """Quatre threads appellent le moteur en même temps : une seule inférence tourne à la fois."""
    active, peak, lock = 0, 0, threading.Lock()
    original = engine_module.joint_beam_search

    def tracked(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        assert engine.busy
        time.sleep(0.05)
        try:
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(engine_module, "joint_beam_search", tracked)
    threads = [threading.Thread(target=engine.transcribe, args=(tone(seconds=0.5),)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak == 1
    assert not engine.busy


def test_gpu_failure_falls_back_without_reloading(tiny_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Échec sur GPU : le modèle passe sur CPU, et le checkpoint n'a été lu qu'une fois."""
    loaded = []
    original_load = model_store.torch.load
    monkeypatch.setattr(
        model_store.torch, "load", lambda path, **kwargs: loaded.append(Path(path).name) or original_load(path, **kwargs)
    )
    fake_gpu = DeviceInfo("cuda", "GPU factice", "auto", "bf16-autocast", torch.__version__, "13.0", None, (12, 0), 1, 1)
    monkeypatch.setattr(engine_module, "select_device", lambda requested: fake_gpu)
    original_warmup = ASREngine._warmup

    def warmup(self, model, device):
        if device.type == "cuda":
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        original_warmup(self, model, device)

    monkeypatch.setattr(ASREngine, "_warmup", warmup)
    engine = ASREngine.load(tiny_dir, device="auto")
    assert engine.device_info.type == "cpu"
    assert "no kernel image" in engine.device_info.fallback_reason
    assert loaded == ["best.pt", "cmvn.pt"]
    assert engine.transcribe(tone(seconds=0.5)).device == "cpu"


def test_greedy_decoding_does_not_run_the_beam_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = write_model_dir(
        tmp_path, decoding={"method": "ctc_greedy", "beam_size": 1, "ctc_weight": 0.3, "use_corrector": False}
    )
    monkeypatch.setattr(engine_module, "joint_beam_search", lambda *args, **kwargs: pytest.fail("beam search appelé"))
    assert isinstance(ASREngine.load(directory, device="cpu").transcribe(tone()).text, str)


def test_corrector_is_applied_when_enabled(tmp_path: Path) -> None:
    pytest.importorskip("rapidfuzz")
    directory = write_model_dir(
        tmp_path,
        files={"checkpoint": "best.pt", "cmvn": "cmvn.pt", "corrector_transcripts": "transcripts.txt"},
        decoding={"method": "joint_beam_search", "beam_size": 2, "ctc_weight": 0.3, "use_corrector": True},
    )
    (directory / "transcripts.txt").write_text("HELLO WORLD\nHELLO WORLD\n", encoding="utf-8")
    engine = ASREngine.load(directory, device="cpu")
    assert engine.corrector("HELO WORLD") == "HELLO WORLD"
    engine.corrector = lambda text: "CORRIGÉ"
    assert engine.transcribe(tone()).text == "CORRIGÉ"


def test_describe_exposes_the_effective_limits(tiny_dir: Path) -> None:
    description = ASREngine.load(tiny_dir, device="cpu", max_duration_s=12.0).describe()
    assert description["version"] == "v1"
    assert description["limits"] == {"min_duration_s": 0.1, "max_duration_s": 12.0}
    assert description["decoding"]["method"] == "joint_beam_search"
    assert description["checkpoint"] == {"epoch": 3}
