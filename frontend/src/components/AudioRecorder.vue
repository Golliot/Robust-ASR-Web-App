<script setup lang="ts">
import { computed, toRef } from "vue";
import { useRecorder } from "../composables/useRecorder";
import { formatClock, formatLimit } from "../services/format";

const props = defineProps<{ maxDurationS: number; disabled?: boolean }>();
const emit = defineEmits<{ started: []; recorded: [audio: Blob, mimeType: string] }>();

const recorder = useRecorder(toRef(props, "maxDurationS"), (audio, mimeType) => emit("recorded", audio, mimeType));
const { state, elapsed, level, error, stop } = recorder;

async function start(): Promise<void> {
  await recorder.start();
  if (state.value === "recording") emit("started");
}

const remaining = computed(() => Math.max(0, props.maxDurationS - elapsed.value));
const progress = computed(() => Math.min(100, (elapsed.value / props.maxDurationS) * 100));
</script>

<template>
  <div class="recorder">
    <template v-if="state !== 'recording'">
      <button
        class="record-button"
        type="button"
        :disabled="disabled || state === 'requesting'"
        @click="start"
        aria-label="Démarrer l'enregistrement"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-2.08A7 7 0 0 0 19 12h-2Z" />
        </svg>
      </button>
      <p class="hint">
        <strong>{{ state === "requesting" ? "Autorisation du micro…" : "Cliquez pour enregistrer" }}</strong>
        <span class="muted small">Parlez en anglais, {{ formatLimit(maxDurationS) }} s au maximum.</span>
      </p>
    </template>

    <template v-else>
      <div class="live" aria-live="polite">
        <span class="rec-dot" aria-hidden="true"></span>
        <span class="clock mono">{{ formatClock(elapsed) }}</span>
        <span class="muted small">/ {{ formatClock(maxDurationS) }}</span>
      </div>
      <div class="bar" role="progressbar" :aria-valuenow="Math.round(progress)" aria-valuemin="0" aria-valuemax="100">
        <div class="bar-fill" :style="{ width: `${progress}%` }"></div>
      </div>
      <div class="meter" aria-hidden="true">
        <div class="meter-fill" :style="{ transform: `scaleX(${level})` }"></div>
      </div>
      <p v-if="remaining <= 5" class="muted small">Arrêt automatique dans {{ Math.ceil(remaining) }} s</p>
      <button class="button primary" type="button" @click="stop">
        <span class="stop-square" aria-hidden="true"></span> Arrêter
      </button>
    </template>

    <p v-if="error" class="alert error" role="alert">{{ error }}</p>
  </div>
</template>

<style scoped>
.recorder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 14px;
  padding: 18px 0 6px;
  text-align: center;
}
.record-button {
  width: 84px;
  height: 84px;
  border-radius: 50%;
  border: none;
  background: var(--record);
  color: #fff;
  cursor: pointer;
  display: grid;
  place-items: center;
  box-shadow: 0 0 0 8px color-mix(in srgb, var(--record) 15%, transparent);
  transition: transform 0.1s;
}
.record-button:hover:not(:disabled) {
  transform: scale(1.04);
}
.record-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.record-button svg {
  width: 36px;
  height: 36px;
  fill: currentColor;
}
.hint {
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin: 0;
}
.live {
  display: flex;
  align-items: baseline;
  gap: 8px;
}
.clock {
  font-size: 2rem;
  font-weight: 700;
}
.rec-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--record);
  align-self: center;
  animation: blink 1s ease-in-out infinite;
}
.bar,
.meter {
  width: min(420px, 100%);
  height: 6px;
  border-radius: 999px;
  background: var(--surface-muted);
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  background: var(--accent);
  transition: width 0.1s linear;
}
.meter-fill {
  height: 100%;
  background: var(--success);
  transform-origin: left;
  transition: transform 0.08s linear;
}
.stop-square {
  width: 12px;
  height: 12px;
  border-radius: 2px;
  background: currentColor;
}
.alert {
  margin: 0;
}
@keyframes blink {
  50% {
    opacity: 0.25;
  }
}
</style>
