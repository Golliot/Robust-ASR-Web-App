"""Tests de l'API REST sur un mini-modèle : réponses, erreurs HTTP, sécurité, concurrence, délais."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.core.settings import Settings, SettingsError
from tests.support.audio import encoded, soundfile_bytes, tone

WAV = soundfile_bytes(tone(seconds=1.0), 16_000)


def post(client: TestClient, data: bytes, name: str = "audio.wav", **kwargs):
    return client.post("/api/transcribe", files={"file": (name, data, "application/octet-stream")}, **kwargs)


def assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    assert body["error"]["request_id"] == response.headers["x-request-id"]
    assert "Traceback" not in response.text
    return body["error"]


# --- Transcription ----------------------------------------------------------------------------------


def test_valid_upload_is_transcribed(client: TestClient) -> None:
    response = post(client, WAV)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {
        "transcription", "processing_time", "inference_time", "preprocessing_time", "audio_duration",
        "device", "model_version", "sample_rate", "input_format", "request_id",
    }  # fmt: skip
    assert isinstance(body["transcription"], str)
    assert body["audio_duration"] == 1.0
    assert body["device"] == "cpu"
    assert body["model_version"] == "v1"
    assert body["sample_rate"] == 16_000
    assert body["input_format"] == "WAV (PCM_16)"
    assert body["processing_time"] >= body["inference_time"] > 0
    assert body["request_id"] == response.headers["x-request-id"]


@pytest.mark.parametrize("kind", ["webm_opus", "ogg_opus", "mp4_aac", "mp3"])
def test_browser_formats_are_accepted(client: TestClient, kind: str) -> None:
    data = encoded(kind, np.stack([tone(sample_rate=48_000)] * 2, axis=1), 48_000)
    response = post(client, data, name=f"enregistrement.{kind.split('_')[0]}")
    assert response.status_code == 200, response.text
    assert response.json()["audio_duration"] == pytest.approx(1.0, abs=0.1)


def test_successive_requests_are_independent(client: TestClient) -> None:
    first = [post(client, WAV).json()["transcription"] for _ in range(3)]
    assert len(set(first)) == 1
    assert post(client, b"abc").status_code == 422
    assert post(client, WAV).json()["transcription"] == first[0]


def test_incoming_request_id_is_kept(client: TestClient) -> None:
    response = post(client, WAV, headers={"X-Request-ID": "essai-42"})
    assert response.headers["x-request-id"] == "essai-42"
    assert response.json()["request_id"] == "essai-42"


# --- Erreurs ----------------------------------------------------------------------------------------


def test_empty_file(client: TestClient) -> None:
    assert_error(post(client, b""), 400, "EMPTY_FILE")


def test_missing_file_field(client: TestClient) -> None:
    assert_error(client.post("/api/transcribe", data={"autre": "x"}), 400, "INVALID_REQUEST")


@pytest.mark.parametrize(
    "data",
    [b"pas de l'audio" * 100, np.random.default_rng(1).bytes(5_000), WAV[:30]],
    ids=["texte", "aléatoire", "tronqué"],
)
def test_corrupted_file(client: TestClient, data: bytes) -> None:
    error = assert_error(post(client, data), 422, "INVALID_AUDIO")
    assert "illisible" in error["message"]


def test_audio_without_samples(client: TestClient) -> None:
    assert_error(post(client, soundfile_bytes(np.zeros(0, dtype=np.float32), 16_000)), 422, "EMPTY_AUDIO")


def test_too_short_audio(client: TestClient) -> None:
    assert_error(post(client, soundfile_bytes(tone(seconds=0.05), 16_000)), 422, "AUDIO_TOO_SHORT")


def test_too_long_audio(client: TestClient) -> None:
    long_wav = soundfile_bytes(np.zeros(35 * 16_000, dtype=np.int16), 16_000, subtype="PCM_16")
    error = assert_error(post(client, long_wav), 422, "AUDIO_TOO_LONG")
    assert "30 s" in error["message"]


def test_application_limit_on_duration(make_client) -> None:
    client = make_client(max_duration_s=2.0)
    assert_error(post(client, soundfile_bytes(tone(seconds=3.0), 16_000)), 422, "AUDIO_TOO_LONG")
    assert client.get("/api/system").json()["limits"]["max_duration_s"] == 2.0


def test_too_large_file_with_declared_size(make_client) -> None:
    client = make_client(max_upload_mb=0.05)
    error = assert_error(post(client, soundfile_bytes(np.zeros(160_000, dtype=np.float32), 16_000)), 413, "FILE_TOO_LARGE")
    assert "0,05 Mo" in error["message"]


def test_too_large_file_sent_in_chunks(make_client) -> None:
    """Sans Content-Length, le corps est compté au fil de la lecture."""
    client = make_client(max_upload_mb=0.05)
    boundary = "limite"
    head = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'.encode()

    def chunks():
        yield head
        for _ in range(40):
            yield b"\x00" * 8_192
        yield f"\r\n--{boundary}--\r\n".encode()

    response = client.post(
        "/api/transcribe", content=chunks(), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    assert_error(response, 413, "FILE_TOO_LARGE")


def test_unknown_route_and_method(client: TestClient) -> None:
    assert_error(client.get("/api/inconnu"), 404, "NOT_FOUND")
    assert_error(client.get("/api/transcribe"), 405, "METHOD_NOT_ALLOWED")


def test_unexpected_error_hides_its_details(make_client, gated) -> None:
    class Broken:
        def __init__(self, engine):
            self.engine = engine

        def __getattr__(self, name):
            return getattr(self.engine, name)

        def transcribe_bytes(self, data):
            raise RuntimeError("détail interne secret /srv/models")

    client = make_client(engine_factory=lambda settings: Broken(gated(settings)))
    response = post(client, WAV)
    error = assert_error(response, 500, "INTERNAL_ERROR")
    assert "secret" not in response.text and "RuntimeError" not in response.text
    assert "journaux" in error["message"]


# --- Sécurité ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [{"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"}, {"Origin": "https://exemple.org"}, {"Origin": "null"}],
)
def test_requests_from_another_site_are_refused(client: TestClient, headers: dict) -> None:
    assert_error(post(client, WAV, headers=headers), 403, "FORBIDDEN_ORIGIN")


def test_requests_from_the_interface_are_accepted(client: TestClient) -> None:
    headers = {"Sec-Fetch-Site": "same-origin", "Origin": "http://localhost"}
    assert post(client, WAV, headers=headers).status_code == 200


def test_unknown_host_is_refused(client: TestClient) -> None:
    """Protection contre le « DNS rebinding » : un autre nom de domaine pointant sur 127.0.0.1."""
    assert_error(client.get("/api/health", headers={"Host": "attaquant.example"}), 400, "INVALID_HOST")
    assert client.get("/api/health", headers={"Host": "127.0.0.1:8080"}).status_code == 200


# --- Concurrence et délais --------------------------------------------------------------------------


def test_concurrent_transcription_is_refused(make_client, gated) -> None:
    client = make_client(engine_factory=gated)
    engine = gated.holder["engine"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(post, client, WAV)
        assert engine.entered.wait(timeout=10)
        second = post(client, WAV)
        assert_error(second, 409, "ENGINE_BUSY")
        assert second.headers["retry-after"] == "1"
        engine.gate.set()
        assert first.result(timeout=30).status_code == 200
    assert post(client, WAV).status_code == 200


def test_timeout_keeps_the_engine_reserved_until_the_end(make_client, gated) -> None:
    client = make_client(engine_factory=gated, inference_timeout_s=0.2)
    engine = gated.holder["engine"]
    assert_error(post(client, WAV), 504, "INFERENCE_TIMEOUT")
    # Le calcul continue : aucune autre inférence ne démarre tant qu'il n'est pas terminé.
    assert_error(post(client, WAV), 409, "ENGINE_BUSY")
    engine.gate.set()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = post(client, WAV)
        if response.status_code != 409:
            break
        time.sleep(0.02)
    assert response.status_code == 200


# --- Santé, système, modèles ------------------------------------------------------------------------


def test_health_while_loading_then_ready(make_client, models_dir) -> None:
    gate = threading.Event()

    def slow(settings):
        gate.wait(timeout=10)
        from src.asr.engine import ASREngine

        return ASREngine.load(settings.model_dir, device="cpu")

    client = make_client(engine_factory=slow, wait=False)
    response = client.get("/api/health")
    assert response.status_code == 503 and response.json()["status"] == "loading"
    assert_error(post(client, WAV), 503, "MODEL_NOT_READY")
    gate.set()
    from tests.api.conftest import wait_until_ready

    assert wait_until_ready(client)["status"] == "ready"
    assert client.get("/api/health").json() == {"status": "ready", "model_version": "v1", "device": "cpu", "detail": None}


def test_model_error_is_reported(make_client, models_dir) -> None:
    (models_dir / "v1" / "best.pt").unlink()
    client = make_client()
    response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["status"] == "error"
    assert "best.pt" in response.json()["detail"]
    error = assert_error(post(client, WAV), 503, "MODEL_NOT_READY")
    assert "best.pt" in error["message"]


def test_system_information(client: TestClient) -> None:
    body = client.get("/api/system").json()
    assert body["status"] == "ready"
    assert body["device"]["type"] == "cpu"
    assert body["device"]["requested"] == "cpu"
    assert body["model"]["version"] == "v1"
    assert body["limits"] == {"max_upload_mb": 25.0, "min_duration_s": 0.1, "max_duration_s": 30.0, "inference_timeout_s": 120.0}
    assert ".webm" in body["accepted_extensions"]
    assert body["runtime"]["torch"]


def test_models_listing(client: TestClient, models_dir) -> None:
    from tests.support.models import write_model_dir

    write_model_dir(models_dir, "v2", seed=1)
    (models_dir / "v3").mkdir()
    (models_dir / "v3" / "config.yaml").write_text("format: 9", encoding="utf-8")
    body = client.get("/api/models").json()
    statuses = {entry["version"]: entry["status"] for entry in body["models"]}
    assert body["loaded"] == "v1"
    assert statuses == {"v1": "loaded", "v2": "available", "v3": "invalid"}


def test_docs_can_be_disabled(make_client) -> None:
    assert make_client().get("/api/openapi.json").status_code == 200
    assert make_client(enable_docs=False).get("/api/openapi.json").status_code == 404


# --- Réglages ---------------------------------------------------------------------------------------


def test_settings_from_environment() -> None:
    settings = Settings.from_env(
        {
            "ASR_MODELS_DIR": "/models",
            "ASR_DEVICE": "CPU",
            "ASR_MAX_UPLOAD_MB": "10",
            "ASR_MAX_DURATION_S": "20",
            "ASR_ALLOWED_HOSTS": "localhost, machine.local",
            "ASR_ENABLE_DOCS": "false",
        }
    )
    assert settings.model_dir.as_posix() == "/models/v1"
    assert settings.device == "cpu"
    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.max_duration_s == 20.0
    assert settings.allowed_hosts == ("localhost", "machine.local")
    assert settings.enable_docs is False
    assert Settings.from_env({}) == Settings()


@pytest.mark.parametrize(
    "environ",
    [{"ASR_DEVICE": "tpu"}, {"ASR_MAX_UPLOAD_MB": "-1"}, {"ASR_PORT": "abc"}, {"ASR_MODEL_VERSION": "../v1"}],
)
def test_invalid_settings_are_refused(environ: dict) -> None:
    with pytest.raises(SettingsError):
        Settings.from_env(environ)
