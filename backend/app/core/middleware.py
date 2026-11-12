"""Middlewares ASGI, du plus extérieur au plus intérieur.

RequestContextMiddleware   identifiant de requête (en-tête X-Request-ID), journal des accès, et
                           dernier rempart : toute exception inattendue devient une erreur 500 JSON,
                           sans trace Python.
SecurityMiddleware         en-tête Host limité aux hôtes autorisés (contre le « DNS rebinding »), et
                           requêtes d'écriture refusées si elles viennent d'un autre site : une page
                           ouverte dans le navigateur ne peut pas utiliser l'API. curl reste utilisable.
BodySizeLimitMiddleware    corps de requête borné, que la taille soit annoncée ou non (envoi par blocs).
"""

import logging
import re
import time
import uuid
from urllib.parse import urlsplit

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.app.core.errors import INTERNAL_ERROR_MESSAGE, error_response
from backend.app.core.logging import request_id_var
from src.audio.errors import format_limit

logger = logging.getLogger("backend.access")

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


def _hostname(host: str) -> str:
    """« localhost:8080 » -> « localhost », « [::1]:8080 » -> « ::1 »."""
    host = host.strip().lower()
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = _header(scope, b"x-request-id")
        request_id = incoming if incoming and _REQUEST_ID.match(incoming) else uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 500
        response_started = False

        async def send_with_id(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                status, response_started = message["status"], True
                message.setdefault("headers", [])
                message["headers"] = [*message["headers"], (b"x-request-id", request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            logger.exception("Erreur inattendue")
            if not response_started:
                await error_response(500, "INTERNAL_ERROR", INTERNAL_ERROR_MESSAGE)(scope, receive, send_with_id)
        finally:
            path = scope.get("path", "")
            if path != "/api/health" or status != 200:  # les sondes de santé ne remplissent pas les logs
                logger.info(
                    "%s %s %s",
                    scope.get("method"),
                    path,
                    status,
                    extra={
                        "method": scope.get("method"),
                        "path": path,
                        "status": status,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    },
                )
            request_id_var.reset(token)


class SecurityMiddleware:
    def __init__(self, app: ASGIApp, allowed_hosts: tuple[str, ...]) -> None:
        self.app = app
        self.allowed_hosts = {_hostname(host) for host in allowed_hosts}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        host = _header(scope, b"host") or ""
        if "*" not in self.allowed_hosts and _hostname(host) not in self.allowed_hosts:
            response = error_response(400, "INVALID_HOST", "Hôte non autorisé : utilisez l'adresse locale de l'application.")
            await response(scope, receive, send)
            return

        if scope.get("method") in _UNSAFE_METHODS and self._cross_site(scope, host):
            response = error_response(
                403, "FORBIDDEN_ORIGIN", "Requête refusée : elle ne provient pas de l'interface de l'application."
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

    @staticmethod
    def _cross_site(scope: Scope, host: str) -> bool:
        # Fetch Metadata : envoyé par les navigateurs récents, absent pour curl ou un script.
        fetch_site = _header(scope, b"sec-fetch-site")
        if fetch_site is not None and fetch_site.lower() not in {"same-origin", "none"}:
            return True
        origin = _header(scope, b"origin")
        if origin is not None and origin != "null":
            return urlsplit(origin).netloc.lower() != host.lower()
        return origin == "null"


class BodyTooLargeError(Exception):
    pass


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int, max_upload_mb: float) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.message = f"Fichier trop volumineux : la limite est de {format_limit(max_upload_mb)} Mo."

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        too_large = error_response(413, "FILE_TOO_LARGE", self.message)
        declared = _header(scope, b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await too_large(scope, receive, send)
            return

        received, exceeded, response_started = 0, False, False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    raise BodyTooLargeError
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if exceeded:
                return  # la réponse de l'application (erreur de lecture du corps) est remplacée par 413
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except BodyTooLargeError:
            pass
        if exceeded and not response_started:
            await too_large(scope, receive, send)
