// Types des réponses de l'API (miroir des schémas Pydantic de backend/app/schemas/api.py).

export type EngineStatus = "loading" | "ready" | "error";
export type DeviceType = "cuda" | "cpu";

export interface ApiErrorBody {
  code: string;
  message: string;
  request_id: string | null;
}

export interface TranscriptionResponse {
  transcription: string;
  processing_time: number;
  inference_time: number;
  preprocessing_time: number;
  audio_duration: number;
  device: DeviceType;
  model_version: string;
  sample_rate: number;
  input_format: string | null;
  request_id: string | null;
}

export interface HealthResponse {
  status: EngineStatus;
  model_version: string | null;
  device: DeviceType | null;
  detail: string | null;
}

export interface DeviceInfo {
  type: DeviceType;
  name: string;
  requested: string;
  precision: string;
  torch_version: string;
  cuda_version: string | null;
  cudnn_version: number | null;
  capability: string | null;
  total_memory_mb: number | null;
  num_threads: number;
  fallback_reason: string | null;
}

export interface ModelSummary {
  version: string;
  description: string;
  language: string;
  sample_rate: number;
  decoding: Record<string, unknown>;
  limits: Record<string, number>;
  provenance: Record<string, unknown>;
}

export interface SystemResponse {
  status: EngineStatus;
  detail: string | null;
  application_version: string;
  device: DeviceInfo | null;
  model: ModelSummary | null;
  runtime: {
    python: string;
    platform: string;
    torch: string;
    torchaudio: string;
    pyav: string;
    cpu_count: number | null;
  };
  limits: {
    max_upload_mb: number;
    min_duration_s: number | null;
    max_duration_s: number | null;
    inference_timeout_s: number;
  };
  supported_formats: string[];
  accepted_extensions: string[];
  uptime_s: number;
}

export interface ModelEntry {
  version: string;
  status: "loaded" | "available" | "invalid";
  error: string | null;
  description: string | null;
  language: string | null;
  decoding: Record<string, unknown> | null;
  limits: Record<string, number> | null;
  provenance: Record<string, unknown> | null;
}

export interface ModelsResponse {
  loaded: string | null;
  models: ModelEntry[];
}
