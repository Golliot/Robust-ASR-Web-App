import { beforeEach, describe, expect, it } from "vitest";
import { createHistory, HISTORY_KEY, HISTORY_LIMIT } from "../services/history";
import type { TranscriptionResponse } from "../types/api";

function result(text: string, id: string | null = null): TranscriptionResponse {
  return {
    transcription: text,
    processing_time: 0.5,
    inference_time: 0.4,
    preprocessing_time: 0.1,
    audio_duration: 3,
    device: "cpu",
    model_version: "v1",
    sample_rate: 16000,
    input_format: "WAV (PCM_16)",
    request_id: id,
  };
}

describe("historique local", () => {
  beforeEach(() => localStorage.clear());

  it("ajoute en tête et persiste dans le navigateur", () => {
    const history = createHistory(localStorage);
    history.add(result("FIRST", "a"), "micro", "a.webm");
    history.add(result("SECOND", "b"), "fichier", "b.wav");
    expect(history.entries.value.map((entry) => entry.text)).toEqual(["SECOND", "FIRST"]);
    expect(createHistory(localStorage).entries.value).toHaveLength(2);
  });

  it("ne conserve que le texte et les mesures", () => {
    const entry = createHistory(localStorage).add(result("HELLO", "x"), "micro", "a.webm");
    expect(Object.keys(entry).sort()).toEqual(
      ["audioDuration", "createdAt", "device", "filename", "id", "modelVersion", "processingTime", "source", "text"].sort(),
    );
  });

  it("borne le nombre d'entrées", () => {
    const history = createHistory(localStorage);
    for (let index = 0; index < HISTORY_LIMIT + 5; index++) history.add(result(`T${index}`, `${index}`), "micro", "a");
    expect(history.entries.value).toHaveLength(HISTORY_LIMIT);
    expect(history.entries.value[0].text).toBe(`T${HISTORY_LIMIT + 4}`);
  });

  it("supprime une entrée ou tout l'historique", () => {
    const history = createHistory(localStorage);
    history.add(result("A", "a"), "micro", "a");
    history.add(result("B", "b"), "micro", "b");
    history.remove("a");
    expect(history.entries.value.map((entry) => entry.id)).toEqual(["b"]);
    history.clear();
    expect(createHistory(localStorage).entries.value).toEqual([]);
  });

  it("ignore un contenu illisible", () => {
    localStorage.setItem(HISTORY_KEY, "{pas du json");
    expect(createHistory(localStorage).entries.value).toEqual([]);
  });
});
