---
layout: page
title: Choose Language
description: CuFlash-Attn documentation entry point.
---

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  const preferred = localStorage.getItem('preferred-lang')
  if (preferred === 'en') {
    window.location.href = '/cuflash-attn/en/'
  } else if (preferred === 'zh') {
    window.location.href = '/cuflash-attn/zh/'
  }
})

function setLanguage(lang) {
  localStorage.setItem('preferred-lang', lang)
}
</script>

# CuFlash-Attn

**OpenSpec-driven CUDA FlashAttention reference implementation** with FP32/FP16 support, forward and backward kernels, and a stable `v0.3.0` baseline aimed at long-term handoff and archival quality.

<div class="lang-grid">
  <a href="/cuflash-attn/en/" class="lang-card" @click="setLanguage('en')">
    <strong>English</strong>
    <span>Guide, build notes, API reference, troubleshooting, and project status.</span>
  </a>
  <a href="/cuflash-attn/zh/" class="lang-card" @click="setLanguage('zh')">
    <strong>简体中文</strong>
    <span>面向中文读者的上手指南、构建说明、接口文档、故障排除与项目状态。</span>
  </a>
</div>

## What this site is for

- explain the supported API and build surface
- point to the canonical OpenSpec design and verification sources
- provide a clean entry point for integration, review, and handoff work

## Canonical project links

| Resource | Link |
| --- | --- |
| Repository | [LessUp/cuflash-attn](https://github.com/LessUp/cuflash-attn) |
| Releases | [GitHub Releases](https://github.com/LessUp/cuflash-attn/releases) |
| OpenSpec specs | [`openspec/specs/`](https://github.com/LessUp/cuflash-attn/tree/master/openspec/specs) |

<style scoped>
.lang-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}

.lang-card {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding: 1.5rem;
  border: 1px solid var(--vp-c-border);
  border-radius: 16px;
  background: var(--vp-c-bg-soft);
  color: inherit;
  text-decoration: none;
  transition: border-color 0.2s ease, transform 0.2s ease, background 0.2s ease;
}

.lang-card:hover {
  border-color: var(--vp-c-brand);
  background: var(--vp-c-bg-elv);
  transform: translateY(-2px);
}

.lang-card strong {
  font-size: 1.125rem;
}

.lang-card span {
  color: var(--vp-c-text-2);
  line-height: 1.6;
}
</style>
