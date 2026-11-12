<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from "vue";
import AudioPreview from "../components/AudioPreview.vue";
import AudioRecorder from "../components/AudioRecorder.vue";
import ErrorAlert from "../components/ErrorAlert.vue";
import FileDropZone from "../components/FileDropZone.vue";
import ProcessingIndicator from "../components/ProcessingIndicator.vue";
import TranscriptionResult from "../components/TranscriptionResult.vue";
import { useSystemStatus } from "../composables/useSystemStatus";
import { extensionFor } from "../composables/useRecorder";
import { api, ApiError } from "../services/api";
import { formatDuration, formatLimit } from "../services/format";
import { useHistory } from "../services/history";
import type { TranscriptionResponse } from "../types/api";
import type { SelectedAudio } from "../types/audio";

const { health, system, networkError, ready, limits } = useSystemStatus();
const history = useHistory();

const mode = ref<"record" | "import">("record");
const audio = ref<SelectedAudio | null>(null);
const processing = ref(false);
const result = ref<TranscriptionResponse | null>(null);
const error = ref<ApiError | null>(null);

// Contrôles côté navigateur : un retour immédiat. Le serveur reste seul juge.
const validationError = computed(() => {
  if (!audio.value) return null;
  if (audio.value.blob.size > limits.value.maxUploadMb * 1024 * 1024) {
    return `Fichier trop volumineux : la limite est de ${formatLimit(limits.value.maxUploadMb)} Mo.`;
  }
  const duration = audio.value.durationS;
  if (duration !== null && duration > limits.value.maxDurationS + 0.5) {
    return `Cet audio dure ${formatDuration(duration)} : la limite est de ${formatLimit(limits.value.maxDurationS)} s.`;
  }
  return null;
});
const canTranscribe = computed(() => !!audio.value && ready.value && !processing.value && !validationError.value);

function probeDuration(url: string, target: SelectedAudio): void {
  const element = new Audio();
  element.preload = "metadata";
  element.onloadedmetadata = () => {
    // Les WebM de MediaRecorder n'annoncent pas toujours leur durée (Infinity) : elle reste inconnue.
    if (audio.value === target && Number.isFinite(element.duration)) target.durationS = element.duration;
  };
  element.src = url;
}

function select(blob: Blob, filename: string, source: SelectedAudio["source"], durationS: number | null = null): void {
  discard();
  const selected: SelectedAudio = { blob, url: URL.createObjectURL(blob), filename, source, durationS };
  audio.value = selected;
  if (durationS === null) probeDuration(selected.url, selected);
}

function onRecorded(blob: Blob, mimeType: string): void {
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  select(blob, `enregistrement-${stamp}.${extensionFor(mimeType)}`, "micro");
}

function discard(): void {
  if (audio.value) URL.revokeObjectURL(audio.value.url);
  audio.value = null;
  result.value = null;
  error.value = null;
}

async function transcribe(): Promise<void> {
  if (!audio.value || !canTranscribe.value) return;
  const current = audio.value;
  processing.value = true;
  result.value = null;
  error.value = null;
  try {
    // Délai du navigateur : celui du serveur, plus une marge pour l'envoi.
    const response = await api.transcribe(current.blob, current.filename, (limits.value.timeoutS + 30) * 1000);
    result.value = response;
    current.durationS = response.audio_duration;
    history.add(response, current.source, current.filename);
  } catch (cause) {
    error.value = cause instanceof ApiError ? cause : new ApiError("Erreur inattendue.", 0, "UNKNOWN");
  } finally {
    processing.value = false;
  }
}

onBeforeUnmount(() => {
  if (audio.value) URL.revokeObjectURL(audio.value.url);
});
</script>

<template>
  <div class="page">
    <header class="page-header">
      <h1>Transcription</h1>
      <p class="muted">
        Enregistrez votre voix ou importez un fichier : il est transcrit localement par le modèle Robust ASR, puis
        oublié.
      </p>
    </header>

    <div v-if="networkError" class="alert error">{{ networkError }}</div>
    <div v-else-if="health?.status === 'loading'" class="alert info">Chargement du modèle en cours…</div>
    <div v-else-if="health?.status === 'error'" class="alert error">
      Le modèle n'a pas pu être chargé : {{ health.detail }}
    </div>

    <section class="card">
      <div class="tabs" role="tablist" aria-label="Source de l'audio">
        <button
          type="button"
          role="tab"
          :aria-selected="mode === 'record'"
          :class="{ active: mode === 'record' }"
          @click="mode = 'record'"
        >
          Enregistrer
        </button>
        <button
          type="button"
          role="tab"
          :aria-selected="mode === 'import'"
          :class="{ active: mode === 'import' }"
          @click="mode = 'import'"
        >
          Importer un fichier
        </button>
      </div>

      <AudioRecorder
        v-if="mode === 'record'"
        :max-duration-s="limits.maxDurationS"
        :disabled="processing"
        @started="discard"
        @recorded="onRecorded"
      />
      <FileDropZone
        v-else
        :accept="system?.accepted_extensions ?? []"
        :max-upload-mb="limits.maxUploadMb"
        :max-duration-s="limits.maxDurationS"
        :disabled="processing"
        @selected="(file) => select(file, file.name, 'fichier')"
      />
    </section>

    <section v-if="audio" class="card">
      <h2 class="card-title">Audio à transcrire</h2>
      <AudioPreview :audio="audio" @discard="discard" />
      <p v-if="validationError" class="alert warning">{{ validationError }}</p>
      <div class="transcribe-row">
        <button class="button primary" type="button" :disabled="!canTranscribe" @click="transcribe">
          Transcrire
        </button>
        <ProcessingIndicator v-if="processing" :device="health?.device ?? null" />
      </div>
    </section>

    <ErrorAlert v-if="error" class="spaced" :message="error.message" :code="error.code" :request-id="error.requestId" />

    <section v-if="result" class="card">
      <TranscriptionResult :result="result" :device-name="system?.device?.name" />
    </section>

    <p class="muted small note">
      Le modèle reconnaît l'anglais lu (LibriSpeech) et écrit en majuscules, sans ponctuation. Au-delà de
      {{ formatLimit(limits.maxDurationS) }} s, sa transcription se dégrade fortement : l'audio est donc refusé.
    </p>
  </div>
</template>

<style scoped>
.page > * + * {
  margin-top: 18px;
}
.page-header h1 {
  font-size: 1.6rem;
}
.page-header p {
  margin: 6px 0 0;
}
.tabs {
  display: inline-flex;
  padding: 4px;
  gap: 4px;
  border-radius: 11px;
  background: var(--surface-muted);
  margin-bottom: 8px;
}
.tabs button {
  border: none;
  background: transparent;
  color: var(--text-muted);
  font: inherit;
  font-weight: 600;
  padding: 7px 14px;
  border-radius: 8px;
  cursor: pointer;
}
.tabs button.active {
  background: var(--surface);
  color: var(--text);
  box-shadow: var(--shadow);
}
.transcribe-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 16px;
  margin-top: 16px;
}
.alert.warning {
  margin: 12px 0 0;
}
.note {
  margin-top: 22px;
}
</style>
