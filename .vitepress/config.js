import { defineConfig } from 'vitepress'
import { withPwa } from '@vite-pwa/vitepress'

// Google Analytics (set environment variable VITE_GA_ID or leave empty to disable)
const GA_ID = process.env.VITE_GA_ID || ''

// Algolia Search configuration (set environment variables or leave empty to use local search fallback)
const ALGOLIA_APP_ID = process.env.VITE_ALGOLIA_APP_ID || ''
const ALGOLIA_API_KEY = process.env.VITE_ALGOLIA_API_KEY || ''

// 共享的 SEO 元数据
const sharedHead = [
  ['meta', { name: 'theme-color', content: '#3f83f8' }],
  ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
  ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'black' }],
  ['meta', { name: 'msapplication-TileColor', content: '#3f83f8' }],
  ['link', { rel: 'icon', href: '/favicon.svg', type: 'image/svg+xml' }],
  ['link', { rel: 'alternate icon', href: '/favicon.ico', type: 'image/png', sizes: '16x16' }],
  ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
  ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
  ['link', { href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;600;700&display=swap', rel: 'stylesheet' }],
  ...(GA_ID ? [
    ['script', { async: '', src: `https://www.googletagmanager.com/gtag/js?id=${GA_ID}` }],
    ['script', {}, `
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', '${GA_ID}');
    `]
  ] : []),
]

// 英文导航
const enNav = [
  { text: 'Guide', link: '/en/guide/quick-start', activeMatch: '/en/guide/' },
  { text: 'API Reference', link: '/en/api-reference', activeMatch: '/en/api' },
  { text: 'Algorithm', link: '/en/algorithm', activeMatch: '/en/algorithm' },
  { text: 'Specs', link: '/specs/', activeMatch: '/specs/' },
  {
    text: 'v0.2.0',
    items: [
      { text: 'Changelog', link: 'https://github.com/LessUp/cuflash-attn/blob/master/CHANGELOG.md' },
      { text: 'Releases', link: 'https://github.com/LessUp/cuflash-attn/releases' }
    ]
  }
]

// 中文导航
const zhNav = [
  { text: '指南', link: '/zh/guide/quick-start', activeMatch: '/zh/guide/' },
  { text: 'API 参考', link: '/zh/api-reference', activeMatch: '/zh/api' },
  { text: '算法详解', link: '/zh/algorithm', activeMatch: '/zh/algorithm' },
  {
    text: 'v0.2.0',
    items: [
      { text: '更新日志', link: 'https://github.com/LessUp/cuflash-attn/blob/master/CHANGELOG.md' },
      { text: '发布版本', link: 'https://github.com/LessUp/cuflash-attn/releases' }
    ]
  }
]

// 英文侧边栏
const enSidebar = {
  '/en/': [
    {
      text: 'Getting Started',
      collapsed: false,
      items: [
        { text: 'Introduction', link: '/en/' },
        { text: 'Quick Start', link: '/en/guide/quick-start' },
        { text: 'Building from Source', link: '/en/building' },
      ]
    },
    {
      text: 'API Reference',
      collapsed: false,
      items: [
        { text: 'API Overview', link: '/en/api-reference' },
      ]
    },
    {
      text: 'Core Concepts',
      collapsed: false,
      items: [
        { text: 'FlashAttention Algorithm', link: '/en/algorithm' },
      ]
    },
    {
      text: 'Help',
      collapsed: false,
      items: [
        { text: 'Troubleshooting', link: '/en/troubleshooting' },
      ]
    }
  ],
  '/specs/': [
    {
      text: 'Specifications',
      collapsed: false,
      items: [
        { text: 'Overview', link: '/specs/' },
        { text: 'Product Requirements', link: '/specs/product/001-flash-attention-core' },
        { text: 'Architecture RFC', link: '/specs/rfc/001-core-architecture' },
        { text: 'API Specification', link: '/specs/api/001-public-api' },
        { text: 'Testing Specification', link: '/specs/testing/001-test-specification' },
      ]
    }
  ]
}

// 中文侧边栏
const zhSidebar = {
  '/zh/': [
    {
      text: '开始',
      collapsed: false,
      items: [
        { text: '简介', link: '/zh/' },
        { text: '快速开始', link: '/zh/guide/quick-start' },
        { text: '从源码构建', link: '/zh/building' },
      ]
    },
    {
      text: 'API 参考',
      collapsed: false,
      items: [
        { text: 'API 概述', link: '/zh/api-reference' },
      ]
    },
    {
      text: '核心概念',
      collapsed: false,
      items: [
        { text: 'FlashAttention 算法', link: '/zh/algorithm' },
      ]
    },
    {
      text: '帮助',
      collapsed: false,
      items: [
        { text: '故障排除', link: '/zh/troubleshooting' },
      ]
    }
  ]
}

// Search configuration - uses Algolia if configured, otherwise falls back to local search
const searchConfig = ALGOLIA_APP_ID && ALGOLIA_API_KEY ? {
  provider: 'algolia',
  options: {
    appId: ALGOLIA_APP_ID,
    apiKey: ALGOLIA_API_KEY,
    indexName: 'cuflash-attn',
    locales: {
      zh: {
        placeholder: '搜索文档',
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            searchBox: {
              resetButtonTitle: '清除查询条件',
              resetButtonAriaLabel: '清除查询条件',
              cancelButtonText: '取消',
              cancelButtonAriaLabel: '取消'
            },
            startScreen: {
              recentSearchesTitle: '搜索历史',
              noRecentSearchesText: '没有搜索历史',
              saveRecentSearchButtonTitle: '保存至搜索历史',
              removeRecentSearchButtonTitle: '从搜索历史中移除',
              favoriteSearchesTitle: '收藏',
              removeFavoriteSearchButtonTitle: '从收藏中移除'
            },
            errorScreen: {
              titleText: '无法获取结果',
              helpText: '你可能需要检查你的网络连接'
            },
            footer: {
              selectText: '选择',
              navigateText: '切换',
              closeText: '关闭',
              searchByText: '搜索提供者'
            },
            noResultsScreen: {
              noResultsText: '无法找到相关结果',
              suggestedQueryText: '你可以尝试查询',
              reportMissingResultsText: '你认为该查询应该有结果？',
              reportMissingResultsLinkText: '点击反馈'
            }
          }
        }
      }
    }
  }
} : {
  // Local search fallback - VitePress built-in
  provider: 'local'
}

export default withPwa(
  defineConfig({
    // 站点元数据
    title: 'CuFlash-Attn',
    titleTemplate: ':title - CuFlash-Attn',
    description: 'High-performance CUDA C++ FlashAttention implementation from scratch',
    lang: 'en-US',

    // 基础路径（GitHub Pages 子路径）
    base: '/cuflash-attn/',

    // 头信息
    head: sharedHead,

    // 国际化配置
    locales: {
      root: {
        label: 'English',
        lang: 'en',
        themeConfig: {
          nav: enNav,
          sidebar: enSidebar,
          outline: { label: 'On this page' },
          docFooter: { prev: 'Previous', next: 'Next' },
          editLink: {
            pattern: 'https://github.com/LessUp/cuflash-attn/edit/master/docs/:path',
            text: 'Edit this page on GitHub'
          },
          lastUpdated: { text: 'Last updated' },
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
          darkModeSwitchLabel: '外观',
        }
      }
    },

    // 主题配置
    themeConfig: {
      // Logo
      logo: {
        light: '/logo-light.svg',
        dark: '/logo-dark.svg',
        alt: 'CuFlash-Attn'
      },

      // 站点标题
      siteTitle: 'CuFlash-Attn',

      // 导航栏
      nav: enNav,

      // 侧边栏
      sidebar: enSidebar,

      // 右侧目录
      outline: {
        level: 'deep',
        label: 'On this page'
      },

      // 社交链接
      socialLinks: [
        { icon: 'github', link: 'https://github.com/LessUp/cuflash-attn' },
        {
          icon: {
            svg: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10s10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93c0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41c0 2.08-.8 3.97-2.1 5.39z"/></svg>'
          },
          link: 'https://lessup.github.io/cuflash-attn/'
        }
      ],

      // 页脚
      footer: {
        message: 'Released under the MIT License.',
        copyright: 'Copyright © 2026 LessUp. Built with VitePress.'
      },

      // 编辑链接
      editLink: {
        pattern: 'https://github.com/LessUp/cuflash-attn/edit/master/docs/:path',
        text: 'Edit this page on GitHub'
      },

      // 最近更新
      lastUpdated: {
        text: 'Last updated',
        formatOptions: {
          dateStyle: 'full',
          timeStyle: 'medium'
        }
      },

      // 搜索配置
      search: searchConfig,
    },

    // Markdown 配置
    markdown: {
      theme: {
        light: 'github-light',
        dark: 'github-dark'
      },
      lineNumbers: true,
      config: (md) => {
        // 可以在这里添加自定义 markdown-it 插件
      }
    },

    // Vite 配置
    vite: {
      // 别名
      resolve: {
        alias: {
          '@': '/.vitepress'
        }
      },
      // CSS
      css: {
        preprocessorOptions: {
          scss: {
            additionalData: `
              @use "./.vitepress/theme/custom.scss" as *;
            `
          }
        }
      },
      // 优化
      build: {
        chunkSizeWarningLimit: 1000,
        rollupOptions: {
          output: {
            manualChunks: {
              'vendor': ['vue'],
            }
          }
        }
      }
    },

    // 源目录
    srcDir: '.',

    // 源文件排除
    srcExclude: ['**/(README|CHANGELOG|LICENSE|package)*'],

    // Sitemap
    sitemap: {
      hostname: 'https://lessup.github.io/cuflash-attn'
    },

    // 最后更新时间
    lastUpdated: true,

    // 清理 URL（去掉 .html）
    cleanUrls: true,

    // 源目录
    srcDir: '.',

    // 源文件排除
    srcExclude: ['**/(README|CHANGELOG|LICENSE|package)*'],
  }),

  // PWA 配置
  {
    // PWA 选项
    pwa: {
      registerType: 'autoUpdate',
      manifest: {
        name: 'CuFlash-Attn Documentation',
        short_name: 'CuFlash-Attn',
        description: 'High-performance CUDA C++ FlashAttention implementation',
        theme_color: '#3f83f8',
        background_color: '#ffffff',
        icons: [
          {
            src: '/pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: '/pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable'
          }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,svg,png,ico,woff2}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/fonts\.googleapis\.com\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'google-fonts-cache',
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 * 365 // 365 days
              },
              cacheableResponse: {
                statuses: [0, 200]
              }
            }
          },
          {
            urlPattern: /^https:\/\/fonts\.gstatic\.com\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'gstatic-fonts-cache',
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 * 365 // 365 days
              },
              cacheableResponse: {
                statuses: [0, 200]
              }
            }
          }
        ]
      },
      devOptions: {
        enabled: false
      }
    }
  }
)
