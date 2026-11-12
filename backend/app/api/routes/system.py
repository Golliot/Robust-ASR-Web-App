"""GET /api/system : périphérique de calcul, versions, limites et formats acceptés."""

import os
import platform
import time

import av
import torch
import torchaudio
from fastapi import APIRouter, Depends

from backend.app import __version__
from backend.app.api.dependencies import get_runtime
from backend.app.schemas.api import SystemResponse
from backend.app.services.runtime import EngineRuntime

router = APIRouter(tags=["système"])

SUPPORTED_FORMATS = ["WAV", "FLAC", "OGG (Vorbis, Opus)", "MP3", "WebM (Opus)", "MP4 / M4A (AAC)"]
ACCEPTED_EXTENSIONS = [".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".webm", ".m4a", ".mp4", ".aac"]


@router.get("/system", response_model=SystemResponse, summary="Informations système")
async def system(runtime: EngineRuntime = Depends(get_runtime)) -> SystemResponse:
    """Périphérique retenu (et raison d'un repli sur CPU), modèle chargé, versions et limites."""
    engine = runtime.engine if runtime.ready else None
    settings = runtime.settings
    return SystemResponse(
        status=runtime.state.value,
        detail=runtime.error,
        application_version=__version__,
        device=engine.device_info.as_dict() if engine else None,
        model=engine.describe() if engine else None,
        runtime={
            "python": platform.python_version(),
            "platform": platform.platform(terse=True),
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "pyav": av.__version__,
            "cpu_count": os.cpu_count(),
        },
        limits={
            "max_upload_mb": settings.max_upload_mb,
            "min_duration_s": engine.min_duration_s if engine else None,
            "max_duration_s": engine.max_duration_s if engine else None,
            "inference_timeout_s": settings.inference_timeout_s,
        },
        supported_formats=SUPPORTED_FORMATS,
        accepted_extensions=ACCEPTED_EXTENSIONS,
        uptime_s=round(time.time() - runtime.started_at, 1),
    )
