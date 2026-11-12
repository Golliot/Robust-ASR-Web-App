"""GET /api/models : versions de modèle présentes dans le dossier des modèles, et version chargée."""

from fastapi import APIRouter, Depends

from backend.app.api.dependencies import get_runtime
from backend.app.schemas.api import ModelEntry, ModelsResponse
from backend.app.services.runtime import EngineRuntime
from src.asr.model_store import list_versions

router = APIRouter(tags=["modèles"])


@router.get("/models", response_model=ModelsResponse, summary="Versions de modèle")
async def models(runtime: EngineRuntime = Depends(get_runtime)) -> ModelsResponse:
    """Chaque version trouvée, avec son état : chargée, disponible, ou invalide (et pourquoi)."""
    loaded = runtime.engine.spec.version if runtime.ready else None
    entries = []
    for entry in list_versions(runtime.settings.models_dir):
        if entry.spec is None:
            entries.append(ModelEntry(version=entry.version, status="invalid", error=entry.error))
            continue
        summary = runtime.engine.describe() if entry.version == loaded else entry.spec.summary()
        status = "loaded" if entry.version == loaded else "invalid" if entry.error else "available"
        entries.append(
            ModelEntry(
                version=entry.version,
                status=status,
                error=entry.error,
                description=summary["description"],
                language=summary["language"],
                decoding=summary["decoding"],
                limits=summary["limits"],
                provenance=summary["provenance"],
            )
        )
    return ModelsResponse(loaded=loaded, models=entries)
