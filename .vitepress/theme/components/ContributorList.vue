<template>
  <div class="contributor-list">
    <h3 v-if="title">{{ title }}</h3>
    <div class="contributors">
      <a v-for="contributor in contributors" :key="contributor.login" :href="contributor.html_url" target="_blank" class="contributor">
        <img :src="contributor.avatar_url" :alt="contributor.login" />
        <span>{{ contributor.login }}</span>
      </a>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const props = defineProps({
  title: { type: String, default: 'Contributors' },
  repo: { type: String, default: 'LessUp/cuflash-attn' }
})

const contributors = ref([])

onMounted(async () => {
  try {
    const response = await fetch(`https://api.github.com/repos/${props.repo}/contributors?per_page=10`)
    if (response.ok) {
      contributors.value = await response.json()
    }
  } catch (error) {
    console.error('Failed to fetch contributors:', error)
  }
})
</script>

<style scoped>
.contributor-list {
  margin: 2rem 0;
  padding: 1.5rem;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
}

.contributor-list h3 {
  margin-bottom: 1rem;
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.contributors {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
}

.contributor {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  text-decoration: none;
  transition: all 0.2s;
}

.contributor:hover {
  border-color: var(--vp-c-brand);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(63, 131, 248, 0.2);
}

.contributor img {
  width: 32px;
  height: 32px;
  border-radius: 50%;
}

.contributor span {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--vp-c-text-1);
}
</style>
