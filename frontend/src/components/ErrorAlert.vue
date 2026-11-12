<script setup lang="ts">
defineProps<{ message: string; code?: string | null; requestId?: string | null }>();
</script>

<template>
  <div class="alert error" role="alert">
    <strong>{{ message }}</strong>
    <div v-if="code || requestId" class="details mono">
      <span v-if="code">{{ code }}</span>
      <span v-if="requestId"> · requête {{ requestId }}</span>
    </div>
  </div>
</template>

<style scoped>
.details {
  margin-top: 4px;
  opacity: 0.8;
}
</style>
