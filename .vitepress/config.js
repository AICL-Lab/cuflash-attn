import { defineConfig } from 'vitepress'
import { withPwa } from '@vite-pwa/vitepress'

// 获取当前语言的基础路径
const getBase = (lang) => lang === 'en' ? '/' : `/${lang}/`

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
  ['script', { async: '', src: 'https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX' }],
  ['script', {}, `
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-XXXXXXXXXX');
  `],
]

// 英文导航
const enNav = [
  { text: 'Guide', link: '/en/guide/', activeMatch: '/en/guide/' },
  { text: 'API Reference', link: '/en/api/', activeMatch: '/en/api/' },
  { text: 'Algorithm', link: '/en/algorithm', activeMatch: '/en/algorithm' },
  { 
    text: 'v0.2.0',
    items: [
      { text: 'Changelog', link: '/en/changelog' },
      { text: 'Releases', link: 'https://github.com/LessUp/cuflash-attn/releases' }
    ]
  }
]

// 中文导航
const zhNav = [
  { text: '指南', link: '/zh/guide/', activeMatch: '/zh/guide/' },
  { text: 'API 参考', link: '/zh/api/', activeMatch: '/zh/api/' },
  { text: '算法详解', link: '/zh/algorithm', activeMatch: '/zh/algorithm' },
  { 
    text: 'v0.2.0',
    items: [
      { text: '更新日志', link: '/zh/changelog' },
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
        { text: 'Installation', link: '/en/guide/installation' },
      ]
    },
    {
      text: 'Core Concepts',
      collapsed: false,
      items: [
        { text: 'FlashAttention Algorithm', link: '/en/algorithm' },
        { text: 'Memory Efficiency', link: '/en/guide/memory-efficiency' },
        { text: 'Performance Tuning', link: '/en/guide/performance' },
      ]
    },
    {
      text: 'API Reference',
      collapsed: false,
      items: [
        { text: 'Overview', link: '/en/api/' },
        { text: 'Forward Pass', link: '/en/api/forward' },
        { text: 'Backward Pass', link: '/en/api/backward' },
        { text: 'Error Handling', link: '/en/api/errors' },
        { text: 'C ABI', link: '/en/api/c-abi' },
      ]
    },
    {
      text: 'Development',
      collapsed: false,
      items: [
        { text: 'Building from Source', link: '/en/guide/building' },
        { text: 'Testing', link: '/en/guide/testing' },
        { text: 'Contributing', link: '/en/guide/contributing' },
      ]
    },
    {
      text: 'Help',
      collapsed: false,
      items: [
        { text: 'Troubleshooting', link: '/en/troubleshooting' },
        { text: 'FAQ', link: '/en/guide/faq' },
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
        { text: '安装指南', link: '/zh/guide/installation' },
      ]
    },
    {
      text: '核心概念',
      collapsed: false,
      items: [
        { text: 'FlashAttention 算法', link: '/zh/algorithm' },
        { text: '内存优化', link: '/zh/guide/memory-efficiency' },
        { text: '性能调优', link: '/zh/guide/performance' },
      ]
    },
    {
      text: 'API 参考',
      collapsed: false,
      items: [
        { text: '概述', link: '/zh/api/' },
        { text: '前向传播', link: '/zh/api/forward' },
        { text: '反向传播', link: '/zh/api/backward' },
        { text: '错误处理', link: '/zh/api/errors' },
        { text: 'C ABI', link: '/zh/api/c-abi' },
      ]
    },
    {
      text: '开发',
      collapsed: false,
      items: [
        { text: '从源码构建', link: '/zh/guide/building' },
        { text: '测试', link: '/zh/guide/testing' },
        { text: '贡献指南', link: '/zh/guide/contributing' },
      ]
    },
    {
      text: '帮助',
      collapsed: false,
      items: [
        { text: '故障排除', link: '/zh/troubleshooting' },
        { text: '常见问题', link: '/zh/guide/faq' },
      ]
    }
  ]
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
      search: {
        provider: 'algolia',
        options: {
          appId: 'YOUR_APP_ID',
          apiKey: 'YOUR_SEARCH_API_KEY',
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
      },
      
      // 广告（可选）
      // carbonAds: {
      //   code: 'your-carbon-code',
      //   placement: 'your-carbon-placement'
      // }
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
    
    // 路径重写（保持与旧 URL 兼容）
    rewrites: {
      'docs/en/:page': 'en/:page',
      'docs/zh/:page': 'zh/:page',
      'docs/:page': 'en/:page',
    },
    
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
