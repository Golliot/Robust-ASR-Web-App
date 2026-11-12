// Enregistrement au micro avec MediaRecorder. Le navigateur choisit son format (WebM/Opus sous
// Chrome et Firefox, MP4/AAC sous Safari) : la conversion en 16 kHz mono se fait côté serveur.

import { onBeforeUnmount, ref, type Ref } from "vue";

/** Formats préférés, du plus courant au plus rare. */
export const PREFERRED_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/ogg;codecs=opus",
  "audio/mp4",
  "audio/webm",
  "audio/ogg",
];

/** Marge avant la limite du serveur : l'encodeur peut allonger légèrement l'enregistrement. */
export const STOP_MARGIN_S = 0.5;

export type RecorderState = "idle" | "requesting" | "recording";

export function pickMimeType(isSupported: (type: string) => boolean): string | undefined {
  return PREFERRED_MIME_TYPES.find((type) => isSupported(type));
}

export function extensionFor(mimeType: string): string {
  if (mimeType.includes("ogg")) return "ogg";
  if (mimeType.includes("mp4") || mimeType.includes("aac")) return "m4a";
  if (mimeType.includes("wav")) return "wav";
  return "webm";
}

export function microphoneErrorMessage(error: unknown): string {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Accès au micro refusé. Autorisez-le dans les réglages du navigateur pour ce site, puis réessayez.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") return "Aucun micro détecté sur cette machine.";
  if (name === "NotReadableError") return "Le micro est déjà utilisé par une autre application.";
  return "Impossible de démarrer l'enregistrement.";
}

export function useRecorder(maxDurationS: Ref<number>, onRecorded: (audio: Blob, mimeType: string) => void) {
  const state = ref<RecorderState>("idle");
  const elapsed = ref(0);
  const level = ref(0);
  const error = ref<string | null>(null);

  let recorder: MediaRecorder | null = null;
  let stream: MediaStream | null = null;
  let audioContext: AudioContext | null = null;
  let frame = 0;
  let startedAt = 0;

  const supported = typeof window !== "undefined" && "MediaRecorder" in window && !!navigator.mediaDevices?.getUserMedia;

  function release(): void {
    cancelAnimationFrame(frame);
    stream?.getTracks().forEach((track) => track.stop());
    void audioContext?.close().catch(() => undefined);
    stream = null;
    audioContext = null;
    level.value = 0;
  }

  function monitor(analyser: AnalyserNode): void {
    const samples = new Float32Array(analyser.fftSize);
    const tick = () => {
      elapsed.value = (performance.now() - startedAt) / 1000;
      analyser.getFloatTimeDomainData(samples);
      let sum = 0;
      for (const value of samples) sum += value * value;
      // Niveau RMS sur une échelle de -60 à 0 dBFS, pour la jauge.
      const db = 10 * Math.log10(sum / samples.length + 1e-12);
      level.value = Math.min(1, Math.max(0, (db + 60) / 60));
      if (elapsed.value >= maxDurationS.value - STOP_MARGIN_S) {
        stop();
        return;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
  }

  async function start(): Promise<void> {
    error.value = null;
    if (!window.isSecureContext || !supported) {
      error.value = window.isSecureContext
        ? "Ce navigateur ne permet pas l'enregistrement audio. Importez plutôt un fichier."
        : "L'enregistrement exige un contexte sécurisé : ouvrez l'application à l'adresse http://localhost.";
      return;
    }
    state.value = "requesting";
    try {
      // Pas de réduction de bruit ni d'annulation d'écho : le modèle a été entraîné sur de l'audio brut,
      // bruité et réverbéré. Le contrôle de gain garde un niveau d'entrée raisonnable.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true },
      });
    } catch (cause) {
      state.value = "idle";
      error.value = microphoneErrorMessage(cause);
      return;
    }

    const mimeType = pickMimeType((type) => MediaRecorder.isTypeSupported(type));
    recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks: Blob[] = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    recorder.onstop = () => {
      const type = recorder?.mimeType || mimeType || "audio/webm";
      release();
      state.value = "idle";
      if (chunks.length) onRecorded(new Blob(chunks, { type }), type);
    };

    audioContext = new AudioContext();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    audioContext.createMediaStreamSource(stream).connect(analyser);

    recorder.start(250);
    startedAt = performance.now();
    elapsed.value = 0;
    state.value = "recording";
    monitor(analyser);
  }

  function stop(): void {
    cancelAnimationFrame(frame);
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }

  onBeforeUnmount(() => {
    if (recorder && recorder.state !== "inactive") {
      recorder.onstop = null;
      recorder.stop();
    }
    release();
  });

  return { state, elapsed, level, error, supported, start, stop };
}
