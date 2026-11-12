// Historique local des transcriptions : texte et mesures seulement, jamais l'audio.
// Stocké dans le navigateur (localStorage) ; rien n'est conservé côté serveur.

import { ref } from "vue";
import type { TranscriptionResponse } from "../types/api";

export const HISTORY_KEY = "robust-asr.history.v1";
export const HISTORY_LIMIT = 50;

export type AudioSource = "micro" | "fichier";

export interface HistoryEntry {
  id: string;
  createdAt: number;
  source: AudioSource;
  filename: string;
  text: string;
  audioDuration: number;
  processingTime: number;
  device: string;
  modelVersion: string;
}

function read(storage: Storage): HistoryEntry[] {
  try {
    const parsed: unknown = JSON.parse(storage.getItem(HISTORY_KEY) ?? "[]");
    return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : [];
  } catch {
    return []; // contenu illisible : on repart d'un historique vide
  }
}

function write(storage: Storage, entries: HistoryEntry[]): void {
  try {
    storage.setItem(HISTORY_KEY, JSON.stringify(entries));
  } catch {
    // stockage plein ou désactivé : l'historique reste en mémoire pour la session
  }
}

export function createHistory(storage: Storage = window.localStorage) {
  const entries = ref<HistoryEntry[]>(read(storage));

  function add(result: TranscriptionResponse, source: AudioSource, filename: string): HistoryEntry {
    const entry: HistoryEntry = {
      id: result.request_id ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      createdAt: Date.now(),
      source,
      filename,
      text: result.transcription,
      audioDuration: result.audio_duration,
      processingTime: result.processing_time,
      device: result.device,
      modelVersion: result.model_version,
    };
    entries.value = [entry, ...entries.value].slice(0, HISTORY_LIMIT);
    write(storage, entries.value);
    return entry;
  }

  function remove(id: string): void {
    entries.value = entries.value.filter((entry) => entry.id !== id);
    write(storage, entries.value);
  }

  function clear(): void {
    entries.value = [];
    write(storage, entries.value);
  }

  return { entries, add, remove, clear };
}

let shared: ReturnType<typeof createHistory> | null = null;

/** Historique partagé par toutes les vues. */
export function useHistory() {
  shared ??= createHistory();
  return shared;
}
