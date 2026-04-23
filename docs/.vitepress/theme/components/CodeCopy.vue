<template>
  <div class="code-copy-wrapper">
    <slot></slot>
    <button class="copy-btn" @click="copyCode" :class="{ copied }">
      <svg v-if="!copied" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
      <svg v-else xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="20 6 9 17 4 12"></polyline>
      </svg>
      <span>{{ copied ? 'Copied!' : 'Copy' }}</span>
    </button>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const copied = ref(false)

const copyCode = async () => {
  const codeBlock = document.querySelector('.code-copy-wrapper pre code')
  if (!codeBlock) return

  try {
    await navigator.clipboard.writeText(codeBlock.textContent)
    copied.value = true
    setTimeout(() => {
      copied.value = false
    }, 2000)
  } catch (err) {
    console.error('Failed to copy:', err)
  }
}
</script>

<style scoped>
.code-copy-wrapper {
  position: relative;
}

.copy-btn {
  position: absolute;
  top: 0.75rem;
  right: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.375rem 0.75rem;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-border);
  border-radius: 6px;
  color: var(--vp-c-text-3);
  font-size: 0.75rem;
  font-weight: 500;
  cursor: pointer;
  opacity: 0;
  transition: all 0.2s;
  z-index: 10;
}

.code-copy-wrapper:hover .copy-btn {
  opacity: 1;
}

.copy-btn:hover {
  background: var(--vp-c-brand-soft);
  border-color: var(--vp-c-brand);
  color: var(--vp-c-brand);
}

.copy-btn.copied {
  background: rgba(34, 197, 94, 0.1);
  border-color: rgba(34, 197, 94, 0.5);
  color: #22c55e;
  opacity: 1;
}
</style>
