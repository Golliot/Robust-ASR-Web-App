"""POST /api/transcribe : transcription d'un fichier audio envoyé en multipart/form-data."""

import logging

from fastapi import APIRouter, Depends, File, UploadFile

from backend.app.api.dependencies import get_runtime
from backend.app.core.errors import ApiError
from backend.app.core.logging import request_id_var
from backend.app.schemas.api import ErrorResponse, TranscriptionResponse
from backend.app.services.runtime import EngineRuntime
from src.audio.errors import format_limit

logger = logging.getLogger(__name__)
router = APIRouter(tags=["transcription"])

ERRORS = {
    400: {"model": ErrorResponse, "description": "Requête invalide ou fichier vide"},
    403: {"model": ErrorResponse, "description": "Requête venant d'un autre site"},
    409: {"model": ErrorResponse, "description": "Une transcription est déjà en cours"},
    413: {"model": ErrorResponse, "description": "Fichier trop volumineux"},
    422: {"model": ErrorResponse, "description": "Audio illisible, vide, trop court ou trop long"},
    503: {"model": ErrorResponse, "description": "Modèle pas encore prêt"},
    504: {"model": ErrorResponse, "description": "Délai de transcription dépassé"},
}


@router.post("/transcribe", response_model=TranscriptionResponse, responses=ERRORS, summary="Transcrire un fichier audio")
async def transcribe(
    file: UploadFile = File(description="Fichier audio : WAV, FLAC, OGG, MP3, WebM, MP4/M4A…"),
    runtime: EngineRuntime = Depends(get_runtime),
) -> TranscriptionResponse:
    """Décode le fichier, le convertit en 16 kHz mono et le transcrit. L'audio n'est jamais conservé."""
    try:
        data = await file.read()
    finally:
        await file.close()
    if not data:
        raise ApiError(400, "EMPTY_FILE", "Le fichier envoyé est vide.")
    limit_mb = runtime.settings.max_upload_mb
    if len(data) > runtime.settings.max_upload_bytes:
        raise ApiError(413, "FILE_TOO_LARGE", f"Fichier trop volumineux : la limite est de {format_limit(limit_mb)} Mo.")

    result = await runtime.transcribe(data)
    del data  # rien n'est conservé au-delà de la requête
    logger.info(
        "Transcription terminée",
        extra={
            "size_bytes": file.size,
            "input_format": result.input_format,
            "audio_duration": round(result.audio_duration, 3),
            "inference_time": round(result.inference_time, 3),
            "preprocessing_time": round(result.preprocessing_time, 3),
            "device": result.device,
            "model_version": result.model_version,
        },
    )
    return TranscriptionResponse(
        transcription=result.text,
        processing_time=round(result.processing_time, 3),
        inference_time=round(result.inference_time, 3),
        preprocessing_time=round(result.preprocessing_time, 3),
        audio_duration=round(result.audio_duration, 3),
        device=result.device,
        model_version=result.model_version,
        sample_rate=result.sample_rate,
        input_format=result.input_format,
        request_id=request_id_var.get(),
    )
