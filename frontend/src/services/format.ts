// Mise en forme des nombres, durées et tailles pour l'affichage (conventions françaises).

const decimals = (digits: number) =>
  new Intl.NumberFormat("fr-FR", { minimumFractionDigits: digits, maximumFractionDigits: digits });

/** 4.85 -> « 4,9 s » ; 75 -> « 1 min 15 s ». */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${decimals(1).format(seconds)} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds - minutes * 60);
  return `${minutes} min ${String(rest).padStart(2, "0")} s`;
}

/** Temps de calcul : 0.4321 -> « 0,43 s ». */
export function formatSeconds(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  return `${decimals(seconds < 10 ? 2 : 1).format(seconds)} s`;
}

/** Chronomètre : 12.3 -> « 00:12 ». */
export function formatClock(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`;
}

/** 1536 -> « 1,5 Ko » ; 2_500_000 -> « 2,4 Mo ». */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} o`;
  if (bytes < 1024 * 1024) return `${decimals(1).format(bytes / 1024)} Ko`;
  return `${decimals(1).format(bytes / 1024 / 1024)} Mo`;
}

/** Facteur temps réel : temps de calcul / durée de l'audio (< 1 : plus rapide que le temps réel). */
export function realTimeFactor(processing: number, audio: number): string {
  if (!(audio > 0)) return "—";
  return decimals(2).format(processing / audio);
}

export function formatLimit(value: number): string {
  return new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 2 }).format(value);
}

export function deviceLabel(type: string | null | undefined): string {
  if (type === "cuda") return "GPU";
  if (type === "cpu") return "CPU";
  return "—";
}

export function formatDate(timestamp: number): string {
  return new Intl.DateTimeFormat("fr-FR", { dateStyle: "medium", timeStyle: "short" }).format(timestamp);
}
