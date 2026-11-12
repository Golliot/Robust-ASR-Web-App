"""Dépendances FastAPI : accès aux objets créés une fois au démarrage (réglages, moteur)."""

from fastapi import Request

from backend.app.core.settings import Settings
from backend.app.services.runtime import EngineRuntime


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_runtime(request: Request) -> EngineRuntime:
    return request.app.state.runtime
