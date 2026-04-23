<template>
  <div class="benchmark-chart">
    <h3 v-if="title">{{ title }}</h3>
    <div class="chart-container">
      <div v-for="(item, index) in data" :key="index" class="chart-bar-group">
        <div class="bar" :style="{ height: getBarHeight(item.value), background: item.color || defaultColor }">
          <span class="bar-value">{{ item.value }}</span>
        </div>
        <span class="bar-label">{{ item.label }}</span>
      </div>
    </div>
    <p v-if="description" class="chart-description">{{ description }}</p>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: { type: String, default: '' },
  description: { type: String, default: '' },
  data: { type: Array, required: true },
  maxValue: { type: Number, default: 100 },
  defaultColor: { type: String, default: 'linear-gradient(135deg, #3f83f8, #60a5fa)' }
})

const getBarHeight = (value) => {
  const percentage = (value / props.maxValue) * 100
  return `${Math.max(percentage, 5)}%`
}
</script>

<style scoped>
.benchmark-chart {
  margin: 2rem 0;
  padding: 1.5rem;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
}

.chart-container {
  display: flex;
  align-items: flex-end;
  gap: 1.5rem;
  height: 300px;
  padding: 1rem 0;
  overflow-x: auto;
}

.chart-bar-group {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
  min-width: 80px;
}

.bar {
  width: 100%;
  max-width: 60px;
  border-radius: 6px 6px 0 0;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding-top: 0.5rem;
  transition: all 0.3s;
  position: relative;
}

.bar:hover {
  transform: scaleY(1.05);
}

.bar-value {
  font-size: 0.75rem;
  font-weight: 600;
  color: white;
}

.bar-label {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  text-align: center;
}

.chart-description {
  margin-top: 1rem;
  font-size: 0.875rem;
  color: var(--vp-c-text-3);
  font-style: italic;
}
</style>
