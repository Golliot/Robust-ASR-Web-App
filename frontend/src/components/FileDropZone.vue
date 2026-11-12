<script setup lang="ts">
import { computed, ref } from "vue";
import { formatLimit } from "../services/format";

const props = defineProps<{ accept: string[]; maxUploadMb: number; maxDurationS: number; disabled?: boolean }>();
const emit = defineEmits<{ selected: [file: File] }>();

const dragging = ref(false);
const input = ref<HTMLInputElement | null>(null);
const acceptAttribute = computed(() => ["audio/*", ...props.accept].join(","));

function choose(files: FileList | null | undefined): void {
  const file = files?.[0];
  if (file) emit("selected", file);
  if (input.value) input.value.value = "";
}

function onDrop(event: DragEvent): void {
  dragging.value = false;
  if (!props.disabled) choose(event.dataTransfer?.files);
}
</script>

<template>
  <label
    class="dropzone"
    :class="{ dragging, disabled }"
    @dragover.prevent="dragging = !disabled"
    @dragleave.prevent="dragging = false"
    @drop.prevent="onDrop"
  >
    <input
      ref="input"
      class="visually-hidden"
      type="file"
      :accept="acceptAttribute"
      :disabled="disabled"
      @change="choose(($event.target as HTMLInputElement).files)"
    />
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3 7 8h3v6h4V8h3l-5-5Zm-7 14v3h14v-3h2v5H3v-5h2Z" />
    </svg>
    <strong>Glissez un fichier audio ici, ou cliquez pour parcourir</strong>
    <span class="muted small">
      WAV, FLAC, OGG, MP3, WebM, M4A… · {{ formatLimit(maxUploadMb) }} Mo et {{ formatLimit(maxDurationS) }} s au maximum
    </span>
  </label>
</template>

<style scoped>
.dropzone {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 30px 16px;
  border: 2px dashed var(--border);
  border-radius: var(--radius);
  text-align: center;
  cursor: pointer;
  transition:
    border-color 0.15s,
    background 0.15s;
}
.dropzone:hover,
.dropzone.dragging {
  border-color: var(--accent);
  background: var(--accent-soft);
}
.dropzone:focus-within {
  outline: 3px solid color-mix(in srgb, var(--accent) 45%, transparent);
}
.dropzone.disabled {
  opacity: 0.55;
  cursor: not-allowed;
}
svg {
  width: 30px;
  height: 30px;
  fill: var(--accent);
}
</style>
