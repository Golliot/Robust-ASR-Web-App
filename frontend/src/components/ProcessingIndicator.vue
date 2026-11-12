<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";

defineProps<{ device: string | null }>();

const elapsed = ref(0);
let timer: ReturnType<typeof setInterval> | undefined;
onMounted(() => {
  const started = performance.now();
  timer = setInterval(() => (elapsed.value = (performance.now() - started) / 1000), 200);
});
onBeforeUnmount(() => clearInterval(timer));
</script>

<template>
  <div class="processing" role="status" aria-live="polite">
    <span class="spinner" aria-hidden="true"></span>
    <div>
      <strong>Transcription en cours… {{ Math.floor(elapsed) }} s</strong>
      <p v-if="device === 'cpu'" class="muted small">
        Calcul sur CPU : comptez jusqu'à environ une seconde par seconde d'audio.
      </p>
    </div>
  </div>
</template>

<style scoped>
.processing {
  display: flex;
  align-items: center;
  gap: 14px;
}
.processing p {
  margin: 2px 0 0;
}
.spinner {
  flex: none;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: 3px solid var(--accent-soft);
  border-top-color: var(--accent);
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
