// État du backend partagé par toutes les vues : santé du modèle (sondée régulièrement) et
// informations système (périphérique, limites, formats), chargées dès que le modèle est prêt.

import { computed, ref } from "vue";
import { api, ApiError } from "../services/api";
import type { HealthResponse, SystemResponse } from "../types/api";

const DEFAULT_MAX_DURATION_S = 30;
const DEFAULT_MAX_UPLOAD_MB = 25;
const DEFAULT_TIMEOUT_S = 120;

const health = ref<HealthResponse | null>(null);
const system = ref<SystemResponse | null>(null);
const networkError = ref<string | null>(null);
let timer: ReturnType<typeof setTimeout> | undefined;
let started = false;

async function poll(): Promise<void> {
  try {
    health.value = await api.health();
    networkError.value = null;
    if (health.value.status === "ready" && system.value?.status !== "ready") {
      system.value = await api.system();
    }
  } catch (error) {
    networkError.value = error instanceof ApiError ? error.message : "Serveur injoignable.";
  }
  clearTimeout(timer);
  // Sondage rapide tant que le modèle charge ou que le serveur est injoignable.
  timer = setTimeout(poll, health.value?.status === "ready" && !networkError.value ? 30_000 : 2_000);
}

export function useSystemStatus() {
  if (!started) {
    started = true;
    void poll();
  }
  const ready = computed(() => health.value?.status === "ready" && !networkError.value);
  const limits = computed(() => ({
    maxDurationS: system.value?.limits.max_duration_s ?? DEFAULT_MAX_DURATION_S,
    maxUploadMb: system.value?.limits.max_upload_mb ?? DEFAULT_MAX_UPLOAD_MB,
    timeoutS: system.value?.limits.inference_timeout_s ?? DEFAULT_TIMEOUT_S,
  }));

  async function refresh(): Promise<void> {
    system.value = await api.system();
    await poll();
  }

  return { health, system, networkError, ready, limits, refresh };
}
