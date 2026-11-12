import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ErrorAlert from "../components/ErrorAlert.vue";
import TranscriptionResult from "../components/TranscriptionResult.vue";
import type { TranscriptionResponse } from "../types/api";

const result: TranscriptionResponse = {
  transcription: "HE HOPED THERE WOULD BE STEW FOR DINNER",
  processing_time: 1.42,
  inference_time: 1.4,
  preprocessing_time: 0.02,
  audio_duration: 4.85,
  device: "cuda",
  model_version: "v1",
  sample_rate: 16000,
  input_format: "WebM/Matroska (opus)",
  request_id: "abc",
};

describe("TranscriptionResult", () => {
  it("affiche le texte et les mesures demandées", () => {
    const wrapper = mount(TranscriptionResult, { props: { result, deviceName: "NVIDIA GeForce RTX 5060 Ti" } });
    expect(wrapper.get("[data-testid=transcription]").text()).toBe(result.transcription);
    const text = wrapper.text();
    expect(text).toContain("1,42 s"); // temps de traitement
    expect(text).toContain("4,9 s"); // durée de l'audio
    expect(text).toContain("GPU"); // périphérique
    expect(text).toContain("v1"); // version du modèle
    expect(text).toContain("WebM/Matroska (opus)");
  });

  it("signale une transcription vide", () => {
    const wrapper = mount(TranscriptionResult, { props: { result: { ...result, transcription: "" } } });
    expect(wrapper.text()).toContain("Aucune parole reconnue");
  });
});

describe("ErrorAlert", () => {
  it("affiche le message, le code et l'identifiant de requête", () => {
    const wrapper = mount(ErrorAlert, { props: { message: "Fichier illisible.", code: "INVALID_AUDIO", requestId: "r1" } });
    expect(wrapper.attributes("role")).toBe("alert");
    expect(wrapper.text()).toContain("Fichier illisible.");
    expect(wrapper.text()).toContain("INVALID_AUDIO");
    expect(wrapper.text()).toContain("r1");
  });
});
