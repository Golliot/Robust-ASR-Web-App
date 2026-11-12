"""Fixtures des tests d'API : l'application réelle, avec un mini-modèle ou le modèle exporté.

Le moteur n'est pas simulé : la chaîne complète (décodage, conversion, modèle, beam search) tourne
sur un mini-modèle aléatoire. Seuls certains tests enveloppent le moteur, pour bloquer une
inférence ou provoquer une erreur inattendue.
"""

import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.settings import Settings
from backend.app.main import create_app
from src.asr.engine import ASREngine
from tests.support.models import write_model_dir

BASE_URL = "http://localhost"  # hôte autorisé par défaut, comme l'interface servie par Nginx


def wait_until_ready(client: TestClient, timeout_s: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = client.get("/api/health")
        if response.json()["status"] != "loading":
            return response.json()
        time.sleep(0.02)
    raise TimeoutError("le modèle ne s'est pas chargé à temps")


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    write_model_dir(tmp_path / "models", "v1")
    return tmp_path / "models"


@pytest.fixture
def make_client(models_dir: Path) -> Iterator[Callable[..., TestClient]]:
    """Fabrique de clients : réglages et moteur modifiables, chargement attendu."""
    clients = []

    def make(engine_factory=None, wait: bool = True, **overrides) -> TestClient:
        settings = Settings(**{"models_dir": models_dir, "device": "cpu", "log_format": "text", **overrides})
        client = TestClient(create_app(settings, engine_factory), base_url=BASE_URL)
        client.__enter__()
        clients.append(client)
        if wait:
            wait_until_ready(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


class GatedEngine:
    """Enveloppe un vrai moteur : chaque transcription attend que le test ouvre la barrière."""

    def __init__(self, engine: ASREngine) -> None:
        self.engine = engine
        self.entered = threading.Event()
        self.gate = threading.Event()

    def __getattr__(self, name: str):
        return getattr(self.engine, name)

    def transcribe_bytes(self, data: bytes):
        self.entered.set()
        if not self.gate.wait(timeout=30):
            raise RuntimeError("barrière jamais ouverte")
        return self.engine.transcribe_bytes(data)


@pytest.fixture
def gated(models_dir: Path) -> Callable[[Settings], GatedEngine]:
    holder = {}

    def factory(settings: Settings) -> GatedEngine:
        holder["engine"] = GatedEngine(ASREngine.load(settings.model_dir, device="cpu"))
        return holder["engine"]

    factory.holder = holder
    return factory
