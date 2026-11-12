/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// En développement, /api est relayé vers le backend lancé localement (python -m backend.app).
// En production, Nginx sert dist/ et relaie /api : le navigateur ne voit qu'une seule origine.
const backend = process.env.ASR_BACKEND_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [vue()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": { target: backend } },
  },
  build: {
    target: "es2022",
    sourcemap: false,
  },
  test: {
    environment: "happy-dom",
    include: ["src/**/*.spec.ts"],
  },
});
