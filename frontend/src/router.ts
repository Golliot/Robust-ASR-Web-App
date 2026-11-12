import { createRouter, createWebHistory } from "vue-router";
import TranscriptionView from "./views/TranscriptionView.vue";

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "transcription", component: TranscriptionView, meta: { title: "Transcription" } },
    {
      path: "/historique",
      name: "historique",
      component: () => import("./views/HistoryView.vue"),
      meta: { title: "Historique" },
    },
    { path: "/systeme", name: "systeme", component: () => import("./views/SystemView.vue"), meta: { title: "Système" } },
    { path: "/:chemin(.*)*", redirect: "/" },
  ],
});

router.afterEach((to) => {
  document.title = to.meta.title ? `${String(to.meta.title)} · Robust ASR` : "Robust ASR";
});
