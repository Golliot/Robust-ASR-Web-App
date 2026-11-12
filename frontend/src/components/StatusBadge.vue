<script setup lang="ts">
import { computed } from "vue";
import type { HealthResponse } from "../types/api";
import { deviceLabel } from "../services/format";

const props = defineProps<{ health: HealthResponse | null; networkError: string | null }>();

const view = computed(() => {
  if (props.networkError) return { tone: "error", label: "Serveur injoignable" };
  if (!props.health) return { tone: "pending", label: "Connexion…" };
  if (props.health.status === "loading") return { tone: "pending", label: "Chargement du modèle…" };
  if (props.health.status === "error") return { tone: "error", label: "Modèle indisponible" };
  return { tone: "ready", label: `Prêt · ${deviceLabel(props.health.device)}` };
});
</script>

<template>
  <span class="status" :class="view.tone" role="status" aria-live="polite">
    <span class="dot" aria-hidden="true"></span>{{ view.label }}
  </span>
</template>

<style scoped>
.status {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 5px 12px;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
  white-space: nowrap;
  background: var(--surface-muted);
  color: var(--text-muted);
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
}
.ready {
  background: var(--success-soft);
  color: var(--success);
}
.error {
  background: var(--danger-soft);
  color: var(--danger);
}
.pending .dot {
  animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
  50% {
    opacity: 0.3;
  }
}
</style>
