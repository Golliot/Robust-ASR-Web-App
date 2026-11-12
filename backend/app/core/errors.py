"""Erreurs de l'API : un seul format de réponse, jamais de trace Python.

    {"error": {"code": "AUDIO_TOO_LONG", "message": "L'audio dure 42,0 s : …", "request_id": "…"}}

`code` est stable : le frontend s'appuie dessus. `message` est destiné à l'utilisateur. La trace
d'une erreur inattendue n'apparaît que dans les logs, sous le même identifiant de requête.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.core.logging import request_id_var
from src.audio.errors import AudioError

logger = logging.getLogger(__name__)

INTERNAL_ERROR_MESSAGE = "Erreur interne du serveur. Le détail figure dans les journaux du backend."
HTTP_ERRORS = {
    404: ("NOT_FOUND", "Ressource introuvable."),
    405: ("METHOD_NOT_ALLOWED", "Méthode HTTP non autorisée sur cette ressource."),
}


class ApiError(Exception):
    """Erreur prévue, renvoyée telle quelle au client."""

    def __init__(self, status_code: int, code: str, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers


def error_response(status_code: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}
    return JSONResponse(body, status_code=status_code, headers=headers)


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError) -> JSONResponse:
        return error_response(error.status_code, error.code, error.message, error.headers)

    @app.exception_handler(AudioError)
    async def audio_error(request: Request, error: AudioError) -> JSONResponse:
        logger.info("Audio refusé : %s", error.code, extra={"error_code": error.code})
        return error_response(422, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        return error_response(
            400,
            "INVALID_REQUEST",
            "Requête invalide : envoyez le fichier audio dans le champ « file » d'un formulaire multipart/form-data.",
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        code, message = HTTP_ERRORS.get(error.status_code, (f"HTTP_{error.status_code}", "Requête refusée."))
        return error_response(error.status_code, code, message, getattr(error, "headers", None))
