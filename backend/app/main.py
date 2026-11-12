"""Construction de l'application FastAPI.

Le moteur est créé une seule fois, dans le cycle de vie de l'application, et chargé en arrière-plan.
Les routes ne contiennent aucune logique ASR : elles délèguent à EngineRuntime, qui délègue à
src.asr.engine.ASREngine.
"""

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from backend.app import __version__
from backend.app.api.routes import health, models, system, transcribe
from backend.app.core.errors import install_exception_handlers
from backend.app.core.middleware import BodySizeLimitMiddleware, RequestContextMiddleware, SecurityMiddleware
from backend.app.core.settings import Settings
from backend.app.services.runtime import EngineFactory, EngineRuntime

# Enveloppe multipart (en-têtes, délimiteurs, nom du fichier) en plus du fichier lui-même.
MULTIPART_OVERHEAD_BYTES = 64 * 1024


def create_app(settings: Settings | None = None, engine_factory: EngineFactory | None = None) -> FastAPI:
    """Application prête à servir. Sans argument, les réglages viennent des variables ASR_*."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = EngineRuntime(settings, engine_factory)
        app.state.runtime = runtime
        runtime.start()
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(
        title="Robust ASR",
        summary="Transcription locale de la parole anglaise, robuste au bruit et à la réverbération.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if settings.enable_docs else None,
    )
    app.state.settings = settings
    install_exception_handlers(app)

    api = APIRouter(prefix="/api")
    for module in (transcribe, system, health, models):
        api.include_router(module.router)
    app.include_router(api)

    # Le dernier ajouté est le plus extérieur : contexte de requête, puis sécurité, puis taille.
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
        max_upload_mb=settings.max_upload_mb,
    )
    app.add_middleware(SecurityMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(RequestContextMiddleware)
    return app
