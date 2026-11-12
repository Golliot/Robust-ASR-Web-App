<script setup lang="ts">
import { formatBytes, formatDuration } from "../services/format";
import type { SelectedAudio } from "../types/audio";

defineProps<{ audio: SelectedAudio }>();
defineEmits<{ discard: [] }>();
</script>

<template>
  <div class="preview">
    <audio class="player" :src="audio.url" controls preload="metadata"></audio>
    <div class="meta">
      <span class="name" :title="audio.filename">{{ audio.source === "micro" ? "Enregistrement" : audio.filename }}</span>
      <span class="muted small">
        {{ formatDuration(audio.durationS) }} · {{ formatBytes(audio.blob.size) }} · {{ audio.blob.type || "type inconnu" }}
      </span>
    </div>
    <div class="actions">
      <a class="button small" :href="audio.url" :download="audio.filename">Télécharger l'audio</a>
      <button class="button small danger" type="button" @click="$emit('discard')">Supprimer</button>
    </div>
  </div>
</template>

<style scoped>
.preview {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.player {
  width: 100%;
}
.meta {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 4px 12px;
}
.name {
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 100%;
}
.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
</style>
