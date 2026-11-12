"""Modèles Pydantic des réponses de l'API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str = Field(description="Code stable de l'erreur, par exemple AUDIO_TOO_LONG.")
    message: str = Field(description="Message destiné à l'utilisateur.")
    request_id: str | None = Field(description="Identifiant de la requête, repris dans les logs.")


class ErrorResponse(BaseModel):
    error: ErrorBody


class TranscriptionResponse(BaseModel):
    transcription: str = Field(description="Texte reconnu : anglais, en majuscules, sans ponctuation.")
    processing_time: float = Field(description="Durée du traitement côté serveur (décodage et inférence), en s.")
    inference_time: float = Field(description="Durée de l'inférence seule, en s.")
    preprocessing_time: float = Field(description="Durée du décodage du fichier et de sa conversion, en s.")
    audio_duration: float = Field(description="Durée de l'audio, en s.")
    device: Literal["cuda", "cpu"] = Field(description="Périphérique de calcul utilisé.")
    model_version: str
    sample_rate: int = Field(description="Fréquence d'échantillonnage du modèle, en Hz.")
    input_format: str | None = Field(description="Format détecté du fichier reçu.")
    request_id: str | None


class HealthResponse(BaseModel):
    status: Literal["loading", "ready", "error"]
    model_version: str | None = None
    device: Literal["cuda", "cpu"] | None = None
    detail: str | None = Field(default=None, description="Raison de l'échec du chargement, le cas échéant.")


class DeviceSchema(BaseModel):
    type: Literal["cuda", "cpu"]
    name: str
    requested: str
    precision: str
    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    capability: str | None
    total_memory_mb: int | None
    num_threads: int
    fallback_reason: str | None = Field(description="Pourquoi CUDA n'a pas été retenu, le cas échéant.")


class ModelSummary(BaseModel):
    version: str
    description: str
    language: str
    sample_rate: int
    decoding: dict[str, Any]
    limits: dict[str, float]
    provenance: dict[str, Any]


class LimitsSchema(BaseModel):
    max_upload_mb: float
    min_duration_s: float | None
    max_duration_s: float | None
    inference_timeout_s: float


class RuntimeSchema(BaseModel):
    python: str
    platform: str
    torch: str
    torchaudio: str
    pyav: str
    cpu_count: int | None


class SystemResponse(BaseModel):
    status: Literal["loading", "ready", "error"]
    detail: str | None
    application_version: str
    device: DeviceSchema | None
    model: ModelSummary | None
    runtime: RuntimeSchema
    limits: LimitsSchema
    supported_formats: list[str]
    accepted_extensions: list[str]
    uptime_s: float


class ModelEntry(BaseModel):
    version: str
    status: Literal["loaded", "available", "invalid"]
    error: str | None = None
    description: str | None = None
    language: str | None = None
    decoding: dict[str, Any] | None = None
    limits: dict[str, float] | None = None
    provenance: dict[str, Any] | None = None


class ModelsResponse(BaseModel):
    loaded: str | None = Field(description="Version chargée, ou None si aucune ne l'est encore.")
    models: list[ModelEntry]
