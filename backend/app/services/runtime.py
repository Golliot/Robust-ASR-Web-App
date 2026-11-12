"""Propriétaire unique du moteur ASR dans le processus du backend.

- Le modèle est chargé une seule fois, en arrière-plan, au démarrage. Pendant ce temps, le serveur
  répond déjà : /api/health indique « loading », puis « ready » ou « error » avec la raison.
- Le décodage et l'inférence s'exécutent dans un exécuteur à un seul thread : la boucle ASGI n'est
  jamais bloquée, et les tâches ne peuvent pas s'y chevaucher.
- Une garde refuse immédiatement (409) une transcription pendant qu'une autre est en cours, au lieu de
  laisser les requêtes s'empiler. Le moteur a en plus son propre verrou.
- Au-delà du délai, l'API répond 504. Un calcul PyTorch ne peut pas être interrompu : la garde reste
  donc prise jusqu'à sa fin réelle, et deux inférences ne tournent jamais en même temps.
"""

import asyncio
import enum
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from backend.app.core.errors import ApiError
from backend.app.core.settings import Settings
from src.asr.engine import ASREngine, Transcription
from src.asr.model_store import ModelConfigError, ModelIntegrityError
from src.audio.errors import format_limit
from src.runtime.device_manager import DeviceUnavailableError

logger = logging.getLogger(__name__)

EngineFactory = Callable[[Settings], Any]


class EngineState(enum.StrEnum):
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


def load_engine(settings: Settings) -> ASREngine:
    return ASREngine.load(
        settings.model_dir,
        device=settings.device,
        max_duration_s=settings.max_duration_s,
        num_threads=settings.num_threads,
    )


def loading_error_message(error: Exception) -> str:
    """Raison affichée à l'utilisateur. Les erreurs prévues décrivent déjà le problème et sa solution."""
    if isinstance(error, (ModelConfigError, ModelIntegrityError, DeviceUnavailableError)):
        return str(error)
    if isinstance(error, FileNotFoundError):
        return f"Fichier introuvable : {error.filename}"
    return f"Échec du chargement du modèle ({type(error).__name__}). Le détail figure dans les journaux du backend."


class EngineRuntime:
    def __init__(self, settings: Settings, engine_factory: EngineFactory | None = None) -> None:
        self.settings = settings
        self._factory = engine_factory or load_engine
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self.state = EngineState.LOADING
        self.error: str | None = None
        self.engine = None
        self.started_at = time.time()
        self.loaded_at: float | None = None
        self._busy = False

    def start(self) -> asyncio.Future:
        """Lance le chargement du modèle dans l'exécuteur ; la boucle reste disponible."""
        return asyncio.get_running_loop().run_in_executor(self._executor, self._load)

    def _load(self) -> None:
        logger.info("Chargement du modèle %s", self.settings.model_version)
        try:
            engine = self._factory(self.settings)
        except Exception as error:  # noqa: BLE001 - l'erreur est exposée par /api/health
            logger.exception("Échec du chargement du modèle")
            self.error = loading_error_message(error)
            self.state = EngineState.ERROR
            return
        self.engine = engine
        self.loaded_at = time.time()
        self.state = EngineState.READY

    @property
    def ready(self) -> bool:
        return self.state is EngineState.READY

    @property
    def busy(self) -> bool:
        return self._busy

    def require_engine(self):
        if self.state is EngineState.LOADING:
            raise ApiError(503, "MODEL_NOT_READY", "Le modèle est en cours de chargement. Réessayez dans un instant.",
                           {"Retry-After": "2"})  # fmt: skip
        if self.state is EngineState.ERROR:
            raise ApiError(503, "MODEL_NOT_READY", f"Le modèle n'a pas pu être chargé : {self.error}")
        return self.engine

    async def transcribe(self, data: bytes) -> Transcription:
        engine = self.require_engine()
        if self._busy:
            raise ApiError(409, "ENGINE_BUSY", "Une transcription est déjà en cours. Réessayez dans quelques secondes.",
                           {"Retry-After": "1"})  # fmt: skip
        self._busy = True
        future = asyncio.get_running_loop().run_in_executor(self._executor, engine.transcribe_bytes, data)
        future.add_done_callback(self._release)
        timeout = self.settings.inference_timeout_s
        try:
            # shield : à l'expiration du délai, le calcul continue ; seule l'attente s'arrête.
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except TimeoutError:
            logger.warning("Transcription au-delà du délai de %s s", timeout, extra={"timeout_s": timeout})
            raise ApiError(
                504,
                "INFERENCE_TIMEOUT",
                f"La transcription a dépassé le délai de {format_limit(timeout)} s. "
                "Essayez un enregistrement plus court, ou une machine avec GPU.",
            ) from None

    def _release(self, future: asyncio.Future) -> None:
        self._busy = False
        if not future.cancelled():
            future.exception()  # une exception non attendue (après un 504) ne doit pas être signalée en double

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
