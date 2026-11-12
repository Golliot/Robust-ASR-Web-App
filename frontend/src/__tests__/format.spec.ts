import { describe, expect, it } from "vitest";
import { deviceLabel, formatBytes, formatClock, formatDuration, formatSeconds, realTimeFactor } from "../services/format";

const nbsp = (text: string) => text.replace(/ | /g, " ");

describe("mise en forme", () => {
  it("affiche les durées en secondes puis en minutes", () => {
    expect(formatDuration(4.85)).toBe("4,9 s");
    expect(formatDuration(75)).toBe("1 min 15 s");
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(Infinity)).toBe("—");
  });

  it("affiche les temps de calcul et le chronomètre", () => {
    expect(formatSeconds(0.4321)).toBe("0,43 s");
    expect(formatClock(12.9)).toBe("00:12");
    expect(formatClock(75)).toBe("01:15");
  });

  it("affiche les tailles", () => {
    expect(formatBytes(512)).toBe("512 o");
    expect(formatBytes(1536)).toBe("1,5 Ko");
    expect(nbsp(formatBytes(2.5 * 1024 * 1024))).toBe("2,5 Mo");
  });

  it("calcule le facteur temps réel", () => {
    expect(realTimeFactor(1.42, 4.85)).toBe("0,29");
    expect(realTimeFactor(1, 0)).toBe("—");
  });

  it("nomme le périphérique", () => {
    expect(deviceLabel("cuda")).toBe("GPU");
    expect(deviceLabel("cpu")).toBe("CPU");
    expect(deviceLabel(null)).toBe("—");
  });
});
