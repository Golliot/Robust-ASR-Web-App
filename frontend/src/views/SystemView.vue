<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useSystemStatus } from "../composables/useSystemStatus";
import { api, ApiError } from "../services/api";
import { formatDuration, formatLimit } from "../services/format";
import type { ModelsResponse } from "../types/api";

const { health, system, networkError, refresh } = useSystemStatus();
const models = ref<ModelsResponse | null>(null);
const error = ref<string | null>(null);
const refreshing = ref(false);

async function load(): Promise<void> {
  refreshing.value = true;
  error.value = null;
  try {
    await refresh();
    models.value = await api.models();
  } catch (cause) {
    error.value = cause instanceof ApiError ? cause.message : "Informations indisponibles.";
  } finally {
    refreshing.value = false;
  }
}
onMounted(load);

const device = computed(() => system.value?.device ?? null);
const provenance = computed(() => system.value?.model?.provenance ?? {});
const decoding = computed(() => system.value?.model?.decoding ?? {});

function percent(value: unknown): string {
  return typeof value === "number" ? `${formatLimit(value * 100)} %` : "—";
}
const statusLabel: Record<string, string> = { loading: "chargement en cours", ready: "prêt", error: "en erreur" };
const modelStatus: Record<string, string> = { loaded: "chargé", available: "disponible", invalid: "invalide" };
</script>

<template>
  <div class="page">
    <header class="page-header">
      <div>
        <h1>Système</h1>
        <p class="muted">Matériel utilisé, modèle chargé et limites de l'application.</p>
      </div>
      <button class="button small" type="button" :disabled="refreshing" @click="load">Actualiser</button>
    </header>

    <div v-if="networkError || error" class="alert error">{{ networkError ?? error }}</div>

    <section class="card">
      <h2 class="card-title">Périphérique de calcul</h2>
      <p v-if="!device" class="muted">
        {{ health?.status === "error" ? `Modèle non chargé : ${health.detail}` : "Disponible une fois le modèle chargé." }}
      </p>
      <template v-else>
        <div class="device">
          <span class="device-type" :class="device.type">{{ device.type === "cuda" ? "GPU" : "CPU" }}</span>
          <strong>{{ device.name }}</strong>
        </div>
        <p v-if="device.fallback_reason" class="alert warning">
          CUDA n'est pas utilisé : {{ device.fallback_reason }}
        </p>
        <dl class="definition-list">
          <dt>Demande</dt>
          <dd>{{ device.requested }}</dd>
          <dt>Précision</dt>
          <dd>{{ device.precision === "bf16-autocast" ? "bfloat16 (autocast), comme à l'entraînement" : "float32" }}</dd>
          <template v-if="device.type === 'cuda'">
            <dt>Capacité CUDA</dt>
            <dd>{{ device.capability }}</dd>
            <dt>Mémoire GPU</dt>
            <dd>{{ device.total_memory_mb ? `${formatLimit(device.total_memory_mb / 1024)} Go` : "—" }}</dd>
          </template>
          <dt>Threads CPU</dt>
          <dd>{{ device.num_threads }}</dd>
          <dt>PyTorch</dt>
          <dd>{{ device.torch_version }}{{ device.cuda_version ? ` (CUDA ${device.cuda_version})` : "" }}</dd>
        </dl>
      </template>
    </section>

    <section class="card">
      <h2 class="card-title">Modèle</h2>
      <p v-if="!system?.model" class="muted">
        État : {{ statusLabel[health?.status ?? "loading"] }}{{ health?.detail ? ` — ${health.detail}` : "" }}
      </p>
      <dl v-else class="definition-list">
        <dt>Version</dt>
        <dd>{{ system.model.version }}</dd>
        <dt>Description</dt>
        <dd>{{ system.model.description }}</dd>
        <dt>Langue</dt>
        <dd>{{ system.model.language === "en" ? "anglais" : system.model.language }}</dd>
        <dt>Décodage</dt>
        <dd>
          {{ decoding.method === "joint_beam_search" ? "beam search joint CTC / attention" : "CTC glouton" }}
          <span v-if="decoding.method === 'joint_beam_search'" class="muted">
            (largeur {{ decoding.beam_size }}, poids CTC {{ decoding.ctc_weight }})</span
          >
        </dd>
        <template v-if="provenance.epoch !== undefined">
          <dt>Entraînement</dt>
          <dd>
            époque {{ Number(provenance.epoch) + 1 }}, WER de validation {{ percent(provenance.best_wer) }}
            <span class="muted">({{ provenance.val_set }})</span>
          </dd>
        </template>
        <template v-if="provenance.source_sha256">
          <dt>Checkpoint source</dt>
          <dd class="mono">{{ String(provenance.source_sha256).slice(0, 16) }}…</dd>
        </template>
      </dl>
      <div v-if="models?.models.length" class="versions">
        <h3 class="small muted">Versions présentes</h3>
        <ul>
          <li v-for="entry in models.models" :key="entry.version">
            <strong>{{ entry.version }}</strong> — {{ modelStatus[entry.status] }}
            <span v-if="entry.error" class="muted small">({{ entry.error }})</span>
          </li>
        </ul>
      </div>
    </section>

    <section v-if="system" class="card">
      <h2 class="card-title">Limites et formats</h2>
      <dl class="definition-list">
        <dt>Durée de l'audio</dt>
        <dd>
          de {{ formatLimit(system.limits.min_duration_s ?? 0.1) }} à
          {{ formatLimit(system.limits.max_duration_s ?? 30) }} s
        </dd>
        <dt>Taille des fichiers</dt>
        <dd>{{ formatLimit(system.limits.max_upload_mb) }} Mo au maximum</dd>
        <dt>Délai de transcription</dt>
        <dd>{{ formatLimit(system.limits.inference_timeout_s) }} s</dd>
        <dt>Formats</dt>
        <dd>{{ system.supported_formats.join(", ") }}</dd>
      </dl>
    </section>

    <section v-if="system" class="card">
      <h2 class="card-title">Logiciels</h2>
      <dl class="definition-list">
        <dt>Application</dt>
        <dd>{{ system.application_version }}</dd>
        <dt>Python</dt>
        <dd>{{ system.runtime.python }}</dd>
        <dt>PyTorch / torchaudio</dt>
        <dd>{{ system.runtime.torch }} / {{ system.runtime.torchaudio }}</dd>
        <dt>PyAV (FFmpeg)</dt>
        <dd>{{ system.runtime.pyav }}</dd>
        <dt>Plateforme</dt>
        <dd>{{ system.runtime.platform }}</dd>
        <dt>Démarré depuis</dt>
        <dd>{{ formatDuration(system.uptime_s) }}</dd>
      </dl>
    </section>
  </div>
</template>

<style scoped>
.page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 18px;
}
.page-header h1 {
  font-size: 1.6rem;
}
.page-header p {
  margin: 6px 0 0;
}
.device {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
  flex-wrap: wrap;
}
.device-type {
  font-weight: 700;
  font-size: 0.8rem;
  padding: 3px 9px;
  border-radius: 6px;
  background: var(--surface-muted);
}
.device-type.cuda {
  background: var(--success-soft);
  color: var(--success);
}
.alert.warning {
  margin: 0 0 14px;
}
.versions {
  margin-top: 16px;
}
.versions ul {
  margin: 6px 0 0;
  padding-left: 18px;
}
</style>
