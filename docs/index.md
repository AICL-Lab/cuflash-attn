---
layout: page
title: Select Language
description: CuFlash-Attn - High-performance CUDA C++ FlashAttention implementation. Choose your preferred language.
---

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  // 如果用户之前访问过特定语言，自动跳转
  const lang = localStorage.getItem('preferred-lang')
  if (lang === 'en') {
    window.location.href = '/cuflash-attn/en/'
  } else if (lang === 'zh') {
    window.location.href = '/cuflash-attn/zh/'
  }
})

function setLanguage(lang) {
  localStorage.setItem('preferred-lang', lang)
}
</script>

<div class="landing-container">
  <!-- Hero Section -->
  <div class="landing-hero">
    <div class="hero-glow"></div>
    <div class="hero-content">
      <div class="hero-icon">
        <svg viewBox="0 0 128 128" width="120" height="120" fill="none" xmlns="http://www.w3.org/2000/svg">
          <!-- GPU/CUDA inspired logo -->
          <rect x="16" y="24" width="96" height="80" rx="12" fill="url(#gpuGradient)" stroke="url(#gpuStroke)" stroke-width="2"/>
          <rect x="28" y="36" width="20" height="16" rx="4" fill="rgba(255,255,255,0.2)"/>
          <rect x="54" y="36" width="46" height="16" rx="4" fill="rgba(255,255,255,0.2)"/>
          <rect x="28" y="60" width="72" height="32" rx="4" fill="rgba(255,255,255,0.15)"/>
          <circle cx="44" cy="76" r="8" fill="url(#flashGradient)"/>
          <circle cx="68" cy="76" r="8" fill="url(#flashGradient)"/>
          <circle cx="92" cy="76" r="8" fill="url(#flashGradient)"/>
          <defs>
            <linearGradient id="gpuGradient" x1="16" y1="24" x2="112" y2="104" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stop-color="#3f83f8"/>
              <stop offset="50%" stop-color="#60a5fa"/>
              <stop offset="100%" stop-color="#a78bfa"/>
            </linearGradient>
            <linearGradient id="gpuStroke" x1="16" y1="24" x2="112" y2="104" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stop-color="#60a5fa"/>
              <stop offset="100%" stop-color="#3f83f8"/>
            </linearGradient>
            <linearGradient id="flashGradient" x1="36" y1="68" x2="52" y2="84" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stop-color="#fbbf24"/>
              <stop offset="100%" stop-color="#f59e0b"/>
            </linearGradient>
          </defs>
        </svg>
      </div>
      <h1 class="hero-title">CuFlash-Attn</h1>
      <p class="hero-tagline">High-Performance CUDA FlashAttention</p>
      <p class="hero-description">
        A from-scratch implementation with O(N) memory, FP32/FP16 support, and full training capabilities
      </p>
    </div>
  </div>

  <!-- Language Selection -->
  <div class="lang-selection">
    <p class="lang-hint">Please select your language / 请选择语言</p>
    <div class="lang-buttons">
      <a href="/cuflash-attn/en/" class="lang-btn" @click="setLanguage('en')">
        <span class="lang-flag">🇺🇸</span>
        <span class="lang-name">English</span>
        <span class="lang-desc">Documentation in English</span>
      </a>
      <a href="/cuflash-attn/zh/" class="lang-btn" @click="setLanguage('zh')">
        <span class="lang-flag">🇨🇳</span>
        <span class="lang-name">简体中文</span>
        <span class="lang-desc">中文文档</span>
      </a>
    </div>
  </div>

  <!-- Features Preview -->
  <div class="features-preview">
    <div class="feature-item">
      <div class="feature-icon">⚡</div>
      <h3>O(N) Memory</h3>
      <p>Linear memory complexity using FlashAttention algorithm — handle 16K+ sequences</p>
    </div>
    <div class="feature-item">
      <div class="feature-icon">🔢</div>
      <h3>Dual Precision</h3>
      <p>Full FP32 & FP16 support for both forward and backward passes</p>
    </div>
    <div class="feature-item">
      <div class="feature-icon">🔁</div>
      <h3>Training Ready</h3>
      <p>Complete forward/backward with gradient computation</p>
    </div>
    <div class="feature-item">
      <div class="feature-icon">🚀</div>
      <h3>Multi-Architecture</h3>
      <p>Optimized for NVIDIA GPUs from V100 (sm_70) to H100 (sm_90)</p>
    </div>
  </div>

  <!-- Links -->
  <div class="landing-links">
    <a href="https://github.com/LessUp/cuflash-attn" class="landing-link" target="_blank">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
      GitHub Repository
    </a>
    <a href="https://github.com/LessUp/cuflash-attn/releases" class="landing-link" target="_blank">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 15 4-8H8l4 8Z"></path><path d="M8 15a4 4 0 1 0 8 0"></path></svg>
      Releases
    </a>
  </div>
</div>

<style scoped>
.landing-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 4rem 2rem;
  background: linear-gradient(180deg, var(--vp-c-bg) 0%, var(--vp-c-bg-alt) 100%);
}

/* Hero Section */
.landing-hero {
  position: relative;
  text-align: center;
  margin-bottom: 3rem;
}

.hero-glow {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 600px;
  height: 600px;
  background: radial-gradient(circle, rgba(63, 131, 248, 0.15) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
}

.hero-content {
  position: relative;
  z-index: 1;
}

.hero-icon {
  margin-bottom: 1.5rem;
  display: flex;
  justify-content: center;
}

.hero-title {
  font-size: 4rem;
  font-weight: 800;
  background: linear-gradient(135deg, #3f83f8 0%, #60a5fa 50%, #a78bfa 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 0.5rem 0;
  letter-spacing: -0.02em;
}

.hero-tagline {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--vp-c-text-2);
  margin: 0 0 1rem 0;
}

.hero-description {
  font-size: 1.125rem;
  color: var(--vp-c-text-3);
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.6;
}

/* Language Selection */
.lang-selection {
  text-align: center;
  margin-bottom: 3rem;
}

.lang-hint {
  font-size: 0.875rem;
  color: var(--vp-c-text-3);
  margin-bottom: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.lang-buttons {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  justify-content: center;
}

.lang-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem 3rem;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-border);
  border-radius: 16px;
  text-decoration: none;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  min-width: 200px;
}

.lang-btn:hover {
  transform: translateY(-4px);
  border-color: var(--vp-c-brand);
  box-shadow: 0 20px 40px -12px rgba(63, 131, 248, 0.25);
}

.lang-flag {
  font-size: 2rem;
  margin-bottom: 0.75rem;
}

.lang-name {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  margin-bottom: 0.25rem;
}

.lang-desc {
  font-size: 0.875rem;
  color: var(--vp-c-text-3);
}

/* Features Preview */
.features-preview {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1.5rem;
  max-width: 1000px;
  margin-bottom: 3rem;
}

.feature-item {
  text-align: center;
  padding: 1.5rem;
}

.feature-icon {
  font-size: 2rem;
  margin-bottom: 0.75rem;
}

.feature-item h3 {
  font-size: 1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin: 0 0 0.5rem 0;
}

.feature-item p {
  font-size: 0.875rem;
  color: var(--vp-c-text-3);
  margin: 0;
  line-height: 1.5;
}

/* Links */
.landing-links {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
  justify-content: center;
}

.landing-link {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  color: var(--vp-c-text-2);
  text-decoration: none;
  font-size: 0.875rem;
  font-weight: 500;
  transition: color 0.2s;
}

.landing-link:hover {
  color: var(--vp-c-brand);
}

.landing-link svg {
  opacity: 0.7;
}

/* Responsive */
@media (max-width: 768px) {
  .landing-container {
    padding: 2rem 1rem;
  }

  .hero-title {
    font-size: 2.5rem;
  }

  .hero-tagline {
    font-size: 1.125rem;
  }

  .hero-description {
    font-size: 1rem;
  }

  .lang-buttons {
    flex-direction: column;
    align-items: center;
  }

  .lang-btn {
    width: 100%;
    max-width: 300px;
  }

  .features-preview {
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
  }

  .feature-item {
    padding: 1rem;
  }
}

@media (max-width: 480px) {
  .features-preview {
    grid-template-columns: 1fr;
  }
}

/* Dark mode adjustments */
.dark .hero-glow {
  background: radial-gradient(circle, rgba(96, 165, 250, 0.2) 0%, transparent 70%);
}

.dark .lang-btn:hover {
  box-shadow: 0 20px 40px -12px rgba(96, 165, 250, 0.3);
}
</style>
