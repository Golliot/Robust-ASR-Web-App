<script setup lang="ts">
import { RouterLink, RouterView } from "vue-router";
import StatusBadge from "./components/StatusBadge.vue";
import { useSystemStatus } from "./composables/useSystemStatus";

const { health, networkError } = useSystemStatus();
</script>

<template>
  <div class="app">
    <header class="app-header">
      <div class="container header-inner">
        <RouterLink to="/" class="brand" aria-label="Robust ASR, accueil">
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <rect width="32" height="32" rx="8" />
            <g>
              <rect x="6" y="13" width="3" height="6" rx="1.5" />
              <rect x="11" y="9" width="3" height="14" rx="1.5" />
              <rect x="16" y="6" width="3" height="20" rx="1.5" />
              <rect x="21" y="11" width="3" height="10" rx="1.5" />
            </g>
          </svg>
          Robust ASR
        </RouterLink>
        <nav class="nav" aria-label="Navigation principale">
          <RouterLink to="/">Transcription</RouterLink>
          <RouterLink to="/historique">Historique</RouterLink>
          <RouterLink to="/systeme">Système</RouterLink>
        </nav>
        <StatusBadge class="status" :health="health" :network-error="networkError" />
      </div>
    </header>

    <main class="container main">
      <RouterView />
    </main>

    <footer class="container footer muted small">
      Traitement 100 % local : l'audio ne quitte pas cette machine et n'est pas conservé après la transcription.
    </footer>
  </div>
</template>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
.app-header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 10;
}
.header-inner {
  display: flex;
  align-items: center;
  gap: 20px;
  min-height: 62px;
  flex-wrap: wrap;
}
.brand {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  font-weight: 700;
  font-size: 1.05rem;
  color: var(--text);
  text-decoration: none;
}
.brand svg {
  width: 28px;
  height: 28px;
}
.brand rect:first-child {
  fill: var(--accent);
}
.brand g {
  fill: var(--on-accent);
}
.nav {
  display: flex;
  gap: 4px;
  flex: 1;
}
.nav a {
  padding: 7px 12px;
  border-radius: 8px;
  color: var(--text-muted);
  text-decoration: none;
  font-weight: 600;
}
.nav a:hover {
  background: var(--surface-muted);
  color: var(--text);
}
.nav a.router-link-exact-active {
  background: var(--accent-soft);
  color: var(--accent);
}
.main {
  flex: 1;
  padding-top: 28px;
  padding-bottom: 28px;
}
.footer {
  padding-bottom: 24px;
  text-align: center;
}
@media (max-width: 640px) {
  .header-inner {
    gap: 8px 12px;
    padding-top: 10px;
    padding-bottom: 10px;
  }
  .status {
    margin-left: auto;
  }
  .nav {
    order: 3;
    flex-basis: 100%;
    overflow-x: auto;
  }
}
</style>
