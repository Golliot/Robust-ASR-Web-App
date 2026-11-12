"""Configuration commune des tests : marqueurs, fixtures audio et modèle exporté.

Les tests de recherche (test_*.py à la racine de tests/) n'utilisent rien d'ici.
"""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "librispeech"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: exige un GPU CUDA utilisable (ignoré sinon)")
    config.addinivalue_line("markers", "model: exige le modèle exporté dans models/v1 (ignoré sinon)")


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("gpu"):
        from src.runtime.device_manager import probe_cuda

        probe = probe_cuda()
        if not probe.usable:
            pytest.skip(f"GPU inutilisable : {probe.reason}")


@pytest.fixture(scope="session")
def references() -> list[dict]:
    """Énoncés LibriSpeech de tests/fixtures, avec leur transcription de référence."""
    items = json.loads((FIXTURES / "references.json").read_text(encoding="utf-8"))
    for item in items:
        item["path"] = FIXTURES / item["file"]
    return items


@pytest.fixture(scope="session")
def model_dir() -> Path:
    """Le modèle exporté (scripts/export_model.py) ; les tests qui l'exigent sont ignorés sans lui."""
    directory = Path(os.environ.get("ASR_TEST_MODEL_DIR", ROOT / "models" / "v1"))
    if not (directory / "config.yaml").is_file() or not (directory / "best.pt").is_file():
        pytest.skip(f"modèle absent de {directory} : lancez scripts/export_model.py")
    return directory


@pytest.fixture(scope="session")
def research_dir() -> Path:
    """Le dépôt de recherche d'origine (checkpoint complet, CMVN), s'il est présent à côté."""
    directory = Path(os.environ.get("ROBUST_ASR_RESEARCH_DIR", ROOT.parent / "Robust ASR"))
    if not (directory / "checkpoints" / "best.pt").is_file():
        pytest.skip(f"dépôt de recherche absent de {directory}")
    return directory
