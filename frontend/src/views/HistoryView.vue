<script setup lang="ts">
import { ref } from "vue";
import { deviceLabel, formatDate, formatDuration, formatSeconds } from "../services/format";
import { useHistory } from "../services/history";

const { entries, remove, clear } = useHistory();
const copiedId = ref<string | null>(null);

async function copy(id: string, text: string): Promise<void> {
  await navigator.clipboard.writeText(text);
  copiedId.value = id;
  setTimeout(() => (copiedId.value = null), 1500);
}

function clearAll(): void {
  if (window.confirm("Effacer tout l'historique de ce navigateur ?")) clear();
}
</script>

<template>
  <div class="page">
    <header class="page-header">
      <div>
        <h1>Historique</h1>
        <p class="muted">Les transcriptions de ce navigateur. L'audio n'est jamais conservé.</p>
      </div>
      <button v-if="entries.length" class="button small danger" type="button" @click="clearAll">Tout effacer</button>
    </header>

    <div v-if="!entries.length" class="card empty">
      <p class="muted">Aucune transcription pour l'instant.</p>
      <RouterLink class="button primary" to="/">Transcrire un audio</RouterLink>
    </div>

    <ol v-else class="list">
      <li v-for="entry in entries" :key="entry.id" class="card entry">
        <div class="entry-head">
          <span class="muted small">
            {{ formatDate(entry.createdAt) }} · {{ entry.source === "micro" ? "micro" : entry.filename }}
          </span>
          <div class="entry-actions">
            <button class="button small" type="button" :disabled="!entry.text" @click="copy(entry.id, entry.text)">
              {{ copiedId === entry.id ? "Copié" : "Copier" }}
            </button>
            <button class="button small danger" type="button" :aria-label="`Supprimer`" @click="remove(entry.id)">
              Supprimer
            </button>
          </div>
        </div>
        <p class="text">{{ entry.text || "Aucune parole reconnue." }}</p>
        <p class="muted small">
          {{ formatDuration(entry.audioDuration) }} d'audio · {{ formatSeconds(entry.processingTime) }} de traitement ·
          {{ deviceLabel(entry.device) }} · modèle {{ entry.modelVersion }}
        </p>
      </li>
    </ol>
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
.empty {
  text-align: center;
}
.list {
  list-style: none;
  margin: 0;
  padding: 0;
}
.entry-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.entry-actions {
  display: flex;
  gap: 8px;
}
.text {
  font-size: 1.05rem;
  margin: 10px 0 6px;
  overflow-wrap: anywhere;
}
.entry p:last-child {
  margin: 0;
}
</style>
