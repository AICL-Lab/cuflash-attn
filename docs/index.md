---
layout: page
title: CuFlash-Attn
description: From-scratch CUDA FlashAttention reference implementation
---

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  // Auto-detect browser language and redirect
  const lang = navigator.language || navigator.userLanguage
  const preferred = localStorage.getItem('preferred-lang')
  
  // Prefer saved preference, then browser language
  if (preferred === 'en') {
    window.location.href = '/cuflash-attn/en/'
  } else if (preferred === 'zh') {
    window.location.href = '/cuflash-attn/zh/'
  } else if (lang && lang.toLowerCase().startsWith('zh')) {
    window.location.href = '/cuflash-attn/zh/'
  } else {
    window.location.href = '/cuflash-attn/en/'
  }
})

function setLanguage(lang) {
  localStorage.setItem('preferred-lang', lang)
}
</script>

<div class="hero">
  <div class="hero-logo">
    <svg viewBox="0 0 120 120" width="96" height="96">
      <defs>
        <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:#76B900"/>
          <stop offset="100%" style="stop-color:#4A7600"/>
        </linearGradient>
      </defs>
      <polygon points="60,8 108,36 108,84 60,112 12,84 12,36" fill="url(#logoGrad)" opacity="0.9"/>
      <polygon points="60,24 92,44 92,76 60,96 28,76 28,44" fill="#1a1a1a" opacity="0.3"/>
      <text x="60" y="68" text-anchor="middle" fill="#fff" font-family="Inter, sans-serif" font-weight="700" font-size="24">FA</text>
    </svg>
  </div>
  <h1 class="hero-title">CuFlash-Attn</h1>
  <p class="hero-tagline">From-scratch CUDA FlashAttention Reference Implementation</p>
  <p class="hero-version">v0.3.0 Stable Baseline</p>
</div>

<div class="lang-selector">
  <p class="lang-prompt">Select your preferred language</p>
  <div class="lang-grid">
    <a href="/cuflash-attn/en/" class="lang-card" @click="setLanguage('en')">
      <div class="lang-icon">
        <svg viewBox="0 0 24 24" width="32" height="32" fill="currentColor">
          <path d="M12.87 15.07l2.54-2.79 3.74-7.19.92 1.01-3.74 7.18-2.54 2.79zm4.13-3.68l2.54 2.79 3.74 7.19-.92-1.01-3.74-7.18-2.54-2.79zM9.13 15.07l-2.54-2.79-3.74-7.19-.92 1.01 3.74 7.18 2.54 2.79zm-4.13-3.68l-2.54 2.79-3.74 7.19.92-1.01 3.74-7.18 2.54-2.79z"/>
          <circle cx="12" cy="12" r="3" fill="currentColor"/>
        </svg>
      </div>
      <div class="lang-content">
        <strong>English</strong>
        <span>Documentation for international developers</span>
      </div>
      <div class="lang-arrow">
        <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
          <path d="M7 10l5 5 5-5H7z"/>
        </svg>
      </div>
    </a>
    <a href="/cuflash-attn/zh/" class="lang-card" @click="setLanguage('zh')">
      <div class="lang-icon">
        <svg viewBox="0 0 24 24" width="32" height="32" fill="currentColor">
          <path d="M12.87 15.07l2.54-2.79 3.74-7.19.92 1.01-3.74 7.18-2.54 2.79zm4.13-3.68l2.54 2.79 3.74 7.19-.92-1.01-3.74-7.18-2.54-2.79zM9.13 15.07l-2.54-2.79-3.74-7.19-.92 1.01 3.74 7.18 2.54 2.79zm-4.13-3.68l-2.54 2.79-3.74 7.19.92-1.01 3.74-7.18 2.54-2.79z"/>
          <circle cx="12" cy="12" r="3" fill="currentColor"/>
        </svg>
      </div>
      <div class="lang-content">
        <strong>简体中文</strong>
        <span>面向中文读者的完整文档</span>
      </div>
      <div class="lang-arrow">
        <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
          <path d="M7 10l5 5 5-5H7z"/>
        </svg>
      </div>
    </a>
  </div>
</div>

<div class="features">
  <div class="feature">
    <div class="feature-icon">⚡</div>
    <div class="feature-text">
      <strong>O(N) Memory</strong>
      <span>Tiled algorithm with logarithmic softmax</span>
    </div>
  </div>
  <div class="feature">
    <div class="feature-icon">🎯</div>
    <div class="feature-text">
      <strong>Spec-Driven</strong>
      <span>All design traced to OpenSpec</span>
    </div>
  </div>
  <div class="feature">
    <div class="feature-icon">🔧</div>
    <div class="feature-text">
      <strong>FP32/FP16</strong>
      <span>Forward and backward kernels</span>
    </div>
  </div>
</div>

<div class="links-section">
  <h3>Quick Links</h3>
  <div class="quick-links">
    <a href="https://github.com/AICL-Lab/cuflash-attn" class="quick-link">
      <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
        <path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.87 1.52 2.34 1.07 2.91.83.09-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.69 0 0 .84-.27 2.75 1.02A9.32 9.32 0 0112 6.8c.85 0 1.7.11 2.5.33 1.91-1.29 2.75-1.02 2.75-1.02.55 1.4.2 2.44.1 2.69.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85v2.73c0 .27.16.58.66.5A10.008 10.008 0 0022 12c0-5.523-4.477-10-10-10z"/>
      </svg>
      <span>GitHub</span>
    </a>
    <a href="https://github.com/AICL-Lab/cuflash-attn/releases" class="quick-link">
      <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
        <path d="M7 18h2v-2H7v2zm4 0h2v-2h-2v2zm4 0h2v-2h-2v2zM7 14h2v-2H7v2zm4 0h2v-2h-2v2zm4 0h2v-2h-2v2zM7 10h2V8H7v2zm4 0h2V8h-2v2zm4 0h2V8h-2v2zM5 22h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2zM5 4h14v16H5V4z"/>
      </svg>
      <span>Releases</span>
    </a>
    <a href="https://github.com/AICL-Lab/cuflash-attn/tree/master/openspec/specs" class="quick-link">
      <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
        <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm4 18H6V4h7v5h5v11z"/>
      </svg>
      <span>OpenSpec</span>
    </a>
  </div>
</div>

<style scoped>
.hero {
  text-align: center;
  padding: 2.5rem 0 1.5rem;
}

.hero-logo {
  margin-bottom: 1rem;
}

.hero-title {
  font-size: 2.5rem;
  font-weight: 800;
  margin: 0;
  background: linear-gradient(135deg, var(--vp-c-brand-1) 0%, var(--vp-c-brand-2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-tagline {
  font-size: 1.1rem;
  color: var(--vp-c-text-2);
  margin: 0.5rem 0 0;
  font-weight: 500;
}

.hero-version {
  display: inline-block;
  margin-top: 0.75rem;
  padding: 0.25rem 0.75rem;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: 600;
}

.lang-selector {
  margin: 2rem 0;
}

.lang-prompt {
  text-align: center;
  color: var(--vp-c-text-2);
  font-size: 0.95rem;
  margin-bottom: 1rem;
}

.lang-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  max-width: 600px;
  margin: 0 auto;
}

@media (max-width: 640px) {
  .lang-grid {
    grid-template-columns: 1fr;
  }
}

.lang-card {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1.25rem 1.5rem;
  border: 2px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg);
  color: inherit;
  text-decoration: none;
  transition: all 0.2s ease;
}

.lang-card:hover {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-bg-soft);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}

.lang-icon {
  flex-shrink: 0;
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--vp-c-brand-soft);
  border-radius: 10px;
  color: var(--vp-c-brand-1);
}

.lang-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.lang-content strong {
  font-size: 1.1rem;
  font-weight: 600;
}

.lang-content span {
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
}

.lang-arrow {
  flex-shrink: 0;
  color: var(--vp-c-text-3);
  transition: transform 0.2s ease;
}

.lang-card:hover .lang-arrow {
  transform: translateX(4px);
  color: var(--vp-c-brand-1);
}

.features {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin: 2.5rem 0;
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
}

@media (max-width: 768px) {
  .features {
    grid-template-columns: 1fr;
  }
}

.feature {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.feature-icon {
  font-size: 1.5rem;
}

.feature-text {
  display: flex;
  flex-direction: column;
}

.feature-text strong {
  font-size: 0.95rem;
  font-weight: 600;
}

.feature-text span {
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
}

.links-section {
  margin-top: 2rem;
  text-align: center;
}

.links-section h3 {
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
  font-weight: 500;
  margin-bottom: 1rem;
}

.quick-links {
  display: flex;
  justify-content: center;
  gap: 1.5rem;
  flex-wrap: wrap;
}

.quick-link {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--vp-c-text-2);
  text-decoration: none;
  font-size: 0.9rem;
  transition: color 0.2s ease;
}

.quick-link:hover {
  color: var(--vp-c-brand-1);
}

.quick-link svg {
  opacity: 0.8;
}
</style>
