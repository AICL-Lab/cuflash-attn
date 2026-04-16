<template>
  <div class="language-switcher" ref="switcherRef">
    <button class="lang-button" @click="isOpen = !isOpen">
      <span class="lang-label">{{ currentLang.label }}</span>
      <svg class="chevron" :class="{ open: isOpen }" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="6 9 12 15 18 9"/>
      </svg>
    </button>
    <Transition name="dropdown">
      <div v-show="isOpen" class="lang-dropdown">
        <a v-for="lang in availableLangs" :key="lang.code" :href="getLangLink(lang)" class="lang-option">
          <span>{{ lang.label }}</span>
        </a>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useData } from 'vitepress'

const { lang } = useData()
const isOpen = ref(false)

const languages = [
  { code: 'root', label: 'English', link: '/en/' },
  { code: 'zh', label: '简体中文', link: '/zh/' }
]

const currentLang = computed(() => languages.find(l => l.code === lang.value) || languages[0])
const availableLangs = computed(() => languages.filter(l => l.code !== currentLang.value.code))

const getLangLink = (lang) => {
  if (typeof window === 'undefined') return lang.link
  const path = window.location.pathname
  return path.replace(/^\/[^/]+\//, lang.link)
}
</script>

<style scoped>
.language-switcher {
  position: relative;
  margin-left: 1rem;
}

.lang-button {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 12px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  color: var(--vp-c-text);
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.lang-button:hover {
  border-color: var(--vp-c-brand);
}

.lang-label {
  font-size: 13px;
}

.chevron {
  width: 14px;
  height: 14px;
  transition: transform 0.2s;
}

.chevron.open {
  transform: rotate(180deg);
}

.lang-dropdown {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
  min-width: 160px;
  z-index: 100;
}

.lang-option {
  display: block;
  padding: 10px 16px;
  color: var(--vp-c-text);
  text-decoration: none;
  font-size: 14px;
  transition: all 0.2s;
}

.lang-option:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-brand);
}

.dropdown-enter-active, .dropdown-leave-active {
  transition: all 0.2s;
}

.dropdown-enter-from, .dropdown-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}
</style>
