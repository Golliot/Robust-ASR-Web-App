"""GET /api/health : état du backend et disponibilité du modèle (sonde Docker et frontend)."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from backend.app.api.dependencies import get_runtime
from backend.app.schemas.api import HealthResponse
from backend.app.services.runtime import EngineRuntime

router = APIRouter(tags=["système"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="État du backend",
    responses={503: {"model": HealthResponse, "description": "Modèle en chargement ou en erreur"}},
)
async def health(runtime: EngineRuntime = Depends(get_runtime)) -> JSONResponse:
    """200 quand le modèle est prêt ; 503 pendant son chargement ou s'il a échoué, avec la raison."""
    engine = runtime.engine if runtime.ready else None
    body = HealthResponse(
        status=runtime.state.value,
        model_version=runtime.settings.model_version,
        device=engine.device_info.type if engine else None,
        detail=runtime.error,
    )
    return JSONResponse(body.model_dump(), status_code=200 if runtime.ready else 503)
