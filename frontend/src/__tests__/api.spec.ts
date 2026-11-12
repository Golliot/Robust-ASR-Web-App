import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../services/api";

function respond(status: number, body: unknown) {
  return vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status }));
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("client de l'API", () => {
  it("envoie l'audio en multipart dans le champ file", async () => {
    const fetchMock = respond(200, { transcription: "HELLO", device: "cuda" });
    vi.stubGlobal("fetch", fetchMock);
    // happy-dom ne conserve pas le nom de fichier dans FormData : on vérifie l'appel lui-même.
    const append = vi.spyOn(FormData.prototype, "append");
    const audio = new Blob(["abc"], { type: "audio/webm" });
    const result = await api.transcribe(audio, "voix.webm", 1000);
    expect(result.transcription).toBe("HELLO");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/transcribe");
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    expect(append).toHaveBeenCalledWith("file", audio, "voix.webm");
    append.mockRestore();
  });

  it("transmet le code et le message d'erreur du backend", async () => {
    const error = { code: "AUDIO_TOO_LONG", message: "L'audio dure 42,0 s.", request_id: "abc123" };
    vi.stubGlobal("fetch", respond(422, { error }));
    const failure = await api.transcribe(new Blob(["x"]), "a.wav", 1000).catch((e) => e);
    expect(failure).toBeInstanceOf(ApiError);
    expect(failure).toMatchObject({ status: 422, code: "AUDIO_TOO_LONG", message: error.message, requestId: "abc123" });
  });

  it("explique une erreur sans corps JSON (Nginx, proxy)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>", { status: 502 })));
    await expect(api.system()).rejects.toMatchObject({ code: "HTTP_502", message: expect.stringContaining("injoignable") });
  });

  it("signale un serveur injoignable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(api.models()).rejects.toMatchObject({ code: "NETWORK_ERROR" });
  });

  it("abandonne une requête trop longue", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) =>
            init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError"))),
          ),
      ),
    );
    const pending = api.transcribe(new Blob(["x"]), "a.wav", 500).catch((e) => e);
    await vi.advanceTimersByTimeAsync(600);
    expect(await pending).toMatchObject({ code: "CLIENT_TIMEOUT" });
  });

  it("lit l'état du modèle même pendant son chargement (503)", async () => {
    vi.stubGlobal("fetch", respond(503, { status: "loading", model_version: "v1", device: null, detail: null }));
    expect(await api.health()).toMatchObject({ status: "loading" });
  });
});
