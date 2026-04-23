<template>
  <span class="version-badge">
    <span class="pulse-dot"></span>
    v{{ version }}
  </span>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  version: { type: String, default: '0.1.0' }
})
</script>

<style scoped>
.version-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.375rem 0.875rem;
  background: var(--vp-c-brand-soft);
  border: 1px solid rgba(63, 131, 248, 0.2);
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--vp-c-brand);
  margin-bottom: 1rem;
}

.pulse-dot {
  width: 6px;
  height: 6px;
  background: #22c55e;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
</style>
