<template>
  <div class="code-block" :class="{ 'with-filename': filename }">
    <div v-if="filename" class="code-header">
      <span class="filename">{{ filename }}</span>
      <button class="copy-btn" @click="copyCode">{{ copied ? '✓' : 'Copy' }}</button>
    </div>
    <div class="code-content">
      <slot />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  filename: { type: String, default: '' }
})

const copied = ref(false)

const copyCode = () => {
  copied.value = true
  setTimeout(() => copied.value = false, 2000)
}
</script>

<style scoped>
.code-block {
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--vp-c-border);
  margin: 1.5rem 0;
}

.code-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: var(--vp-c-bg-soft);
  border-bottom: 1px solid var(--vp-c-border);
}

.filename {
  font-family: var(--vp-font-mono);
  font-size: 13px;
  color: var(--vp-c-text-2);
}

.copy-btn {
  padding: 4px 12px;
  font-size: 12px;
  font-weight: 500;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-border);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.copy-btn:hover {
  border-color: var(--vp-c-brand);
  color: var(--vp-c-brand);
}

.code-content {
  background: var(--vp-c-bg-alt);
}

.code-content :deep(pre) {
  margin: 0;
  background: transparent !important;
}
</style>
