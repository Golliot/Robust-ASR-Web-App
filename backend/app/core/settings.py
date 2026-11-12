"""Réglages du backend, lus dans les variables d'environnement ASR_* (voir .env.example).

Toutes ont une valeur par défaut adaptée au développement local ; l'image Docker fixe celles qui
diffèrent (ASR_MODELS_DIR=/models, ASR_HOST=0.0.0.0). Une valeur invalide arrête le démarrage avec
un message explicite plutôt que de laisser l'application tourner avec un réglage inattendu.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEVICES = ("auto", "cuda", "cpu")
LOG_FORMATS = ("json", "text")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class SettingsError(ValueError):
    """Variable d'environnement invalide."""


@dataclass(frozen=True)
class Settings:
    models_dir: Path = Path("models")  # dossier des versions : models/<version>/config.yaml
    model_version: str = "v1"
    device: str = "auto"  # auto : GPU si utilisable, sinon CPU
    max_upload_mb: float = 25.0
    max_duration_s: float | None = None  # None : limite de la version de modèle (30 s)
    inference_timeout_s: float = 120.0
    num_threads: int | None = None  # threads CPU de PyTorch ; None : choix de PyTorch
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")
    log_level: str = "INFO"
    log_format: str = "json"
    enable_docs: bool = True  # /api/docs et /api/openapi.json
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def model_dir(self) -> Path:
        return self.models_dir / self.model_version

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        defaults = cls()

        def text(name: str, default: str) -> str:
            value = env.get(name, "").strip()
            return value or default

        def number(name: str, default, kind=float, minimum: float = 0.0, optional: bool = False):
            raw = env.get(name, "").strip()
            if not raw:
                return default
            if optional and raw.lower() in {"none", "auto"}:
                return None
            try:
                value = kind(raw)
            except ValueError:
                raise SettingsError(f"{name} doit être un nombre (reçu : {raw!r})") from None
            if value <= minimum:
                raise SettingsError(f"{name} doit être strictement supérieur à {minimum:g} (reçu : {raw})")
            return value

        def choice(name: str, default: str, choices: tuple[str, ...], transform=str.lower) -> str:
            value = transform(text(name, default))
            if value not in choices:
                raise SettingsError(f"{name} doit valoir {', '.join(choices)} (reçu : {value!r})")
            return value

        def flag(name: str, default: bool) -> bool:
            raw = env.get(name, "").strip().lower()
            if not raw:
                return default
            if raw in {"1", "true", "yes", "on", "oui"}:
                return True
            if raw in {"0", "false", "no", "off", "non"}:
                return False
            raise SettingsError(f"{name} doit valoir true ou false (reçu : {raw!r})")

        hosts = env.get("ASR_ALLOWED_HOSTS", "").replace(",", " ").split()
        version = text("ASR_MODEL_VERSION", defaults.model_version)
        if Path(version).name != version or version.startswith("."):
            raise SettingsError(f"ASR_MODEL_VERSION doit être un nom de dossier (reçu : {version!r})")

        return cls(
            models_dir=Path(text("ASR_MODELS_DIR", str(defaults.models_dir))),
            model_version=version,
            device=choice("ASR_DEVICE", defaults.device, DEVICES),
            max_upload_mb=number("ASR_MAX_UPLOAD_MB", defaults.max_upload_mb),
            max_duration_s=number("ASR_MAX_DURATION_S", defaults.max_duration_s, optional=True),
            inference_timeout_s=number("ASR_INFERENCE_TIMEOUT_S", defaults.inference_timeout_s),
            num_threads=number("ASR_NUM_THREADS", defaults.num_threads, kind=int, optional=True),
            allowed_hosts=tuple(hosts) or defaults.allowed_hosts,
            log_level=choice("ASR_LOG_LEVEL", defaults.log_level, LOG_LEVELS, transform=str.upper),
            log_format=choice("ASR_LOG_FORMAT", defaults.log_format, LOG_FORMATS),
            enable_docs=flag("ASR_ENABLE_DOCS", defaults.enable_docs),
            host=text("ASR_HOST", defaults.host),
            port=number("ASR_PORT", defaults.port, kind=int),
        )
