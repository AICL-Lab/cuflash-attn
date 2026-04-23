<template>
  <div v-if="!isRootPage" class="language-switcher" ref="switcherRef">
    <button class="lang-button" @click="isOpen = !isOpen">
      <span class="lang-label">{{ currentLang.label }}</span>
      <svg class="chevron" :class="{ open: isOpen }" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="6 9 12 15 18 9"/>
      </svg>
    </button>
    <Transition name="dropdown">
      <div v-show="isOpen" class="lang-dropdown">
        <a v-for="lang in availableLangs" :key="lang.code" :href="getLangLink(lang)" class="lang-option" @click="isOpen = false">
          <span>{{ lang.label }}</span>
        </a>
        <div class="lang-divider"></div>
        <a href="/cuflash-attn/" class="lang-option lang-home">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
          <span>Language / 语言</span>
        </a>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useData } from 'vitepress'

const { lang } = useData()
const isOpen = ref(false)
const isRootPage = ref(false)

onMounted(() => {
  // 检测是否在根页面（语言选择页）
  const path = window.location.pathname
  isRootPage.value = path === '/' || path === '/cuflash-attn/' || path === '/cuflash-attn' || path === ''
})

const languages = [
  { code: 'en', label: 'English', link: '/en/' },
  { code: 'zh', label: '简体中文', link: '/zh/' }
]

const currentLang = computed(() => languages.find(l => l.code === lang.value) || languages[0])
const availableLangs = computed(() => languages.filter(l => l.code !== currentLang.value.code))

const getLangLink = (targetLang) => {
  if (typeof window === 'undefined') return targetLang.link
  const path = window.location.pathname
  // Remove base path if present
  const basePath = '/cuflash-attn'
  let cleanPath = path.startsWith(basePath) ? path.slice(basePath.length) : path
  // Replace current locale with target locale
  const currentCode = currentLang.value.code
  if (cleanPath.startsWith(`/${currentCode}/`)) {
    cleanPath = cleanPath.replace(`/${currentCode}/`, targetLang.link)
  } else {
    // If not in a locale path, just go to the target lang root
    return targetLang.link
  }
  return cleanPath
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

.lang-divider {
  height: 1px;
  background: var(--vp-c-border);
  margin: 4px 0;
}

.lang-home {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--vp-c-text-3);
}

.dropdown-enter-active, .dropdown-leave-active {
  transition: all 0.2s;
}

.dropdown-enter-from, .dropdown-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}
</style>
