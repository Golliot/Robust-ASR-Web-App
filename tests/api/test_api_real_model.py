"""Tests d'intégration de l'API avec le modèle exporté (ignorés si models/v1 est absent)."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jiwer
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.core.settings import Settings
from backend.app.main import create_app
from tests.api.conftest import BASE_URL, wait_until_ready
from tests.support.audio import encoded

GOLDEN = json.loads((Path(__file__).parents[1] / "regression" / "golden_v1.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client(model_dir: Path):
    settings = Settings(models_dir=model_dir.parent, model_version=model_dir.name, device="auto", log_format="text")
    with TestClient(create_app(settings), base_url=BASE_URL) as client:
        assert wait_until_ready(client, timeout_s=120)["status"] == "ready"
        yield client


def transcribe(client: TestClient, data: bytes, name: str):
    return client.post("/api/transcribe", files={"file": (name, data, "application/octet-stream")})


def test_fixtures_are_transcribed_like_the_notebook(client: TestClient, references) -> None:
    device = client.get("/api/health").json()["device"]
    for item in references:
        response = transcribe(client, item["path"].read_bytes(), item["file"])
        assert response.status_code == 200, response.text
        assert response.json()["transcription"] == GOLDEN["utterances"][item["id"]][device], item["id"]


def test_browser_recording_is_transcribed(client: TestClient, references) -> None:
    """Enregistrement de type Chrome : WebM/Opus, 48 kHz, stéréo, issu d'un énoncé LibriSpeech."""
    import torch
    import torchaudio

    hypotheses, texts = [], []
    for item in references:
        audio, _ = sf.read(item["path"], dtype="float32")
        upsampled = torchaudio.functional.resample(torch.from_numpy(audio), 16_000, 48_000).numpy()
        response = transcribe(client, encoded("webm_opus", np.stack([upsampled] * 2, axis=1), 48_000), "chrome.webm")
        assert response.status_code == 200, response.text
        assert response.json()["input_format"] == "WebM/Matroska (opus)"
        hypotheses.append(response.json()["transcription"])
        texts.append(item["text"])
    # Opus à 48 kHz et un double rééchantillonnage dégradent à peine : WER mesuré de 6 % sur ces fixtures.
    assert jiwer.wer(texts, hypotheses) < 0.15


def test_parallel_requests_never_run_together(client: TestClient, references) -> None:
    data = references[0]["path"].read_bytes()
    with ThreadPoolExecutor(max_workers=3) as pool:
        responses = list(pool.map(lambda _: transcribe(client, data, "a.flac"), range(3)))
    statuses = sorted(response.status_code for response in responses)
    assert set(statuses) <= {200, 409} and 200 in statuses
    assert all(response.json()["error"]["code"] == "ENGINE_BUSY" for response in responses if response.status_code == 409)
