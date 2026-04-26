import { defineConfig } from 'vitepress'

const sharedHead = [
  ['meta', { name: 'theme-color', content: '#3f83f8' }],
  ['meta', { property: 'og:type', content: 'website' }],
  ['meta', { property: 'og:site_name', content: 'CuFlash-Attn' }],
  ['link', { rel: 'icon', href: '/favicon.svg', type: 'image/svg+xml' }],
  ['link', { rel: 'alternate icon', href: '/favicon.ico', type: 'image/png', sizes: '16x16' }],
  ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
  ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
  ['link', {
    href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;600;700&display=swap',
    rel: 'stylesheet'
  }]
]

const enNav = [
  { text: 'Guide', link: '/en/guide/quick-start', activeMatch: '/en/guide/' },
  { text: 'Build', link: '/en/building', activeMatch: '/en/building' },
  { text: 'API', link: '/en/api-reference', activeMatch: '/en/api' },
  { text: 'Troubleshooting', link: '/en/troubleshooting', activeMatch: '/en/troubleshooting' },
  {
    text: 'Project',
    items: [
      { text: 'Project Status', link: '/en/project-status' },
      { text: 'Releases', link: 'https://github.com/LessUp/cuflash-attn/releases' },
      { text: 'Specs', link: 'https://github.com/LessUp/cuflash-attn/tree/master/openspec/specs' }
    ]
  }
]

const zhNav = [
  { text: '指南', link: '/zh/guide/quick-start', activeMatch: '/zh/guide/' },
  { text: '构建', link: '/zh/building', activeMatch: '/zh/building' },
  { text: 'API 参考', link: '/zh/api-reference', activeMatch: '/zh/api' },
  { text: '故障排除', link: '/zh/troubleshooting', activeMatch: '/zh/troubleshooting' },
  {
    text: '项目',
    items: [
      { text: '项目状态', link: '/zh/project-status' },
      { text: '发布版本', link: 'https://github.com/LessUp/cuflash-attn/releases' },
      { text: '规范文档', link: 'https://github.com/LessUp/cuflash-attn/tree/master/openspec/specs' }
    ]
  }
]

const enSidebar = {
  '/en/': [
    {
      text: 'Getting Started',
      collapsed: false,
      items: [
        { text: 'Overview', link: '/en/' },
        { text: 'Quick Start', link: '/en/guide/quick-start' },
        { text: 'Building from Source', link: '/en/building' }
      ]
    },
    {
      text: 'Reference',
      collapsed: false,
      items: [
        { text: 'API Reference', link: '/en/api-reference' },
        { text: 'Algorithm Deep Dive', link: '/en/algorithm' },
        { text: 'Troubleshooting', link: '/en/troubleshooting' }
      ]
    },
    {
      text: 'Project',
      collapsed: false,
      items: [
        { text: 'Project Status', link: '/en/project-status' }
      ]
    }
  ]
}

const zhSidebar = {
  '/zh/': [
    {
      text: '开始',
      collapsed: false,
      items: [
        { text: '概览', link: '/zh/' },
        { text: '快速开始', link: '/zh/guide/quick-start' },
        { text: '从源码构建', link: '/zh/building' }
      ]
    },
    {
      text: '参考',
      collapsed: false,
      items: [
        { text: 'API 参考', link: '/zh/api-reference' },
        { text: '算法详解', link: '/zh/algorithm' },
        { text: '故障排除', link: '/zh/troubleshooting' }
      ]
    },
    {
      text: '项目',
      collapsed: false,
      items: [
        { text: '项目状态', link: '/zh/project-status' }
      ]
    }
  ]
}

export default defineConfig({
  title: 'CuFlash-Attn',
  titleTemplate: ':title | CuFlash-Attn',
  description: 'OpenSpec-driven CUDA C++ FlashAttention reference implementation',
  lang: 'en-US',
  base: '/cuflash-attn/',
  head: sharedHead,

  locales: {
    en: {
      label: 'English',
      lang: 'en',
      link: '/en/',
      themeConfig: {
        nav: enNav,
        sidebar: enSidebar,
        outline: { label: 'On this page' },
        docFooter: { prev: 'Previous', next: 'Next' },
        editLink: {
          pattern: 'https://github.com/LessUp/cuflash-attn/edit/master/docs/:path',
          text: 'Edit this page on GitHub'
        },
        lastUpdated: { text: 'Last updated' }
      }
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      themeConfig: {
        nav: zhNav,
        sidebar: zhSidebar,
        outline: { label: '本页目录', level: 'deep' },
        docFooter: { prev: '上一页', next: '下一页' },
        editLink: {
          pattern: 'https://github.com/LessUp/cuflash-attn/edit/master/docs/:path',
          text: '在 GitHub 上编辑此页面'
        },
        lastUpdated: { text: '最后更新' },
        returnToTopLabel: '返回顶部',
        sidebarMenuLabel: '菜单',
        darkModeSwitchLabel: '外观'
      }
    }
  },

  themeConfig: {
    logo: {
      light: '/logo-light.svg',
      dark: '/logo-dark.svg',
      alt: 'CuFlash-Attn'
    },
    siteTitle: 'CuFlash-Attn',
    socialLinks: [
      { icon: 'github', link: 'https://github.com/LessUp/cuflash-attn' }
    ],
    footer: {
      message: 'Stable v0.3.0 baseline • OpenSpec-driven CUDA FlashAttention reference.',
      copyright: 'Copyright © 2026 LessUp.'
    },
    editLink: {
      pattern: 'https://github.com/LessUp/cuflash-attn/edit/master/docs/:path',
      text: 'Edit this page on GitHub'
    },
    lastUpdated: {
      text: 'Last updated',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'medium'
      }
    },
    search: {
      provider: 'local'
    }
  },

  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    },
    lineNumbers: true
  },

  vite: {
    resolve: {
      alias: {
        '@': '/.vitepress'
      }
    }
  },

  srcDir: '.',
  srcExclude: ['**/(README|CHANGELOG|LICENSE|package)*'],
  lastUpdated: true,
  cleanUrls: true
})
