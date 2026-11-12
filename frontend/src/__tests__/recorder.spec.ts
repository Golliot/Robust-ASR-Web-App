import { describe, expect, it } from "vitest";
import { extensionFor, microphoneErrorMessage, pickMimeType, PREFERRED_MIME_TYPES } from "../composables/useRecorder";

describe("enregistrement", () => {
  it("préfère WebM/Opus, puis Ogg, puis MP4 (Safari)", () => {
    expect(pickMimeType(() => true)).toBe("audio/webm;codecs=opus");
    expect(pickMimeType((type) => type.startsWith("audio/ogg"))).toBe("audio/ogg;codecs=opus");
    expect(pickMimeType((type) => type === "audio/mp4")).toBe("audio/mp4");
    expect(pickMimeType(() => false)).toBeUndefined();
    expect(PREFERRED_MIME_TYPES[0]).toBe("audio/webm;codecs=opus");
  });

  it("nomme le fichier selon le format du navigateur", () => {
    expect(extensionFor("audio/webm;codecs=opus")).toBe("webm");
    expect(extensionFor("audio/ogg;codecs=opus")).toBe("ogg");
    expect(extensionFor("audio/mp4")).toBe("m4a");
  });

  it("explique les refus d'accès au micro", () => {
    expect(microphoneErrorMessage(new DOMException("x", "NotAllowedError"))).toContain("refusé");
    expect(microphoneErrorMessage(new DOMException("x", "NotFoundError"))).toContain("Aucun micro");
    expect(microphoneErrorMessage(new Error("autre"))).toContain("Impossible");
  });
});
