// Client de l'API REST. Toutes les requêtes passent par /api, sur la même origine que l'interface.

import type {
  ApiErrorBody,
  HealthResponse,
  ModelsResponse,
  SystemResponse,
  TranscriptionResponse,
} from "../types/api";

/** Erreur affichable : `code` reprend celui du backend, ou NETWORK_ERROR / CLIENT_TIMEOUT. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | null;

  constructor(message: string, status: number, code: string, requestId: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

const GENERIC_MESSAGES: Record<number, string> = {
  413: "Fichier trop volumineux.",
  502: "Le service de transcription est injoignable.",
  503: "Le service de transcription n'est pas prêt.",
  504: "Le service de transcription n'a pas répondu à temps.",
};

function isErrorEnvelope(body: unknown): body is { error: ApiErrorBody } {
  return typeof body === "object" && body !== null && "error" in body;
}

async function request<T>(path: string, init: RequestInit, timeoutMs: number, accept: number[] = []): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`/api${path}`, {
      ...init,
      signal: controller.signal,
      headers: { Accept: "application/json", ...init.headers },
    });
    const body: unknown = await response.json().catch(() => null);
    if (response.ok || accept.includes(response.status)) {
      return body as T;
    }
    if (isErrorEnvelope(body)) {
      throw new ApiError(body.error.message, response.status, body.error.code, body.error.request_id);
    }
    const message = GENERIC_MESSAGES[response.status] ?? `Erreur inattendue du serveur (HTTP ${response.status}).`;
    throw new ApiError(message, response.status, `HTTP_${response.status}`);
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("Le serveur n'a pas répondu à temps.", 0, "CLIENT_TIMEOUT");
    }
    throw new ApiError("Impossible de joindre le serveur. Vérifiez que l'application est démarrée.", 0, "NETWORK_ERROR");
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  /** État du modèle : renvoyé aussi pendant le chargement ou après un échec (HTTP 503). */
  health: () => request<HealthResponse>("/health", {}, 5_000, [503]),

  system: () => request<SystemResponse>("/system", {}, 10_000),

  models: () => request<ModelsResponse>("/models", {}, 10_000),

  /** Envoie l'audio tel quel : le décodage et la conversion se font côté serveur. */
  transcribe(audio: Blob, filename: string, timeoutMs: number): Promise<TranscriptionResponse> {
    const form = new FormData();
    form.append("file", audio, filename);
    return request<TranscriptionResponse>("/transcribe", { method: "POST", body: form }, timeoutMs);
  },
};
