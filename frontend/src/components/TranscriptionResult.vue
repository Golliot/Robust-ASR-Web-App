<script setup lang="ts">
import { ref } from "vue";
import type { TranscriptionResponse } from "../types/api";
import { deviceLabel, formatDuration, formatSeconds, realTimeFactor } from "../services/format";

const props = defineProps<{ result: TranscriptionResponse; deviceName?: string | null }>();

const copied = ref(false);
async function copy(): Promise<void> {
  await navigator.clipboard.writeText(props.result.transcription);
  copied.value = true;
  setTimeout(() => (copied.value = false), 1500);
}
</script>

<template>
  <section class="result" aria-live="polite">
    <div class="card-title">
      <h2 class="card-title-text">Transcription</h2>
      <button class="button small" type="button" :disabled="!result.transcription" @click="copy">
        {{ copied ? "Copié" : "Copier" }}
      </button>
    </div>
    <p v-if="result.transcription" class="text" data-testid="transcription">{{ result.transcription }}</p>
    <p v-else class="text empty">Aucune parole reconnue dans cet enregistrement.</p>

    <dl class="grid metrics">
      <div class="metric">
        <dt>Temps de traitement</dt>
        <dd>{{ formatSeconds(result.processing_time) }}</dd>
      </div>
      <div class="metric">
        <dt>Durée de l'audio</dt>
        <dd>{{ formatDuration(result.audio_duration) }}</dd>
      </div>
      <div class="metric">
        <dt>Facteur temps réel</dt>
        <dd>{{ realTimeFactor(result.processing_time, result.audio_duration) }}</dd>
      </div>
      <div class="metric">
        <dt>Périphérique</dt>
        <dd :title="deviceName ?? undefined">{{ deviceLabel(result.device) }}</dd>
      </div>
      <div class="metric">
        <dt>Modèle</dt>
        <dd>{{ result.model_version }}</dd>
      </div>
      <div class="metric">
        <dt>Format reçu</dt>
        <dd>{{ result.input_format ?? "—" }}</dd>
      </div>
    </dl>
  </section>
</template>

<style scoped>
.card-title-text {
  font-size: inherit;
  font-weight: inherit;
}
.text {
  font-size: 1.2rem;
  line-height: 1.6;
  margin: 0 0 18px;
  letter-spacing: 0.01em;
  overflow-wrap: anywhere;
}
.empty {
  color: var(--text-muted);
  font-size: 1rem;
}
</style>
