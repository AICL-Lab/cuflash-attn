import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import './custom.css'
import './animation.css'

// 自定义组件
import LanguageSwitcher from './components/LanguageSwitcher.vue'
import FeatureCard from './components/FeatureCard.vue'
import BenchmarkChart from './components/BenchmarkChart.vue'
import CodeCopy from './components/CodeCopy.vue'
import VersionBadge from './components/VersionBadge.vue'
import ContributorList from './components/ContributorList.vue'

export default {
  extends: DefaultTheme,
  
  Layout() {
    return h(DefaultTheme.Layout, null, {
      // 导航栏右侧扩展
      'nav-bar-content-after': () => h(LanguageSwitcher),
      
      // 文档页面前扩展
      'doc-before': () => h(VersionBadge),
      
      // 页脚前扩展
      'layout-bottom': () => h(ContributorList),
      
      // 404 页面
      'not-found': () => h('div', { class: 'not-found-custom' }, [
        h('h1', '404'),
        h('p', 'Page not found'),
        h('a', { href: '/cuflash-attn/' }, 'Go Home')
      ])
    })
  },
  
  enhanceApp({ app, router, siteData }) {
    // 注册全局组件
    app.component('FeatureCard', FeatureCard)
    app.component('BenchmarkChart', BenchmarkChart)
    app.component('CodeCopy', CodeCopy)
    app.component('VersionBadge', VersionBadge)
    
    // 添加自定义指令
    app.directive('animate', {
      mounted(el, binding) {
        el.classList.add('animate-on-scroll')
        el.dataset.animate = binding.value || 'fade-up'
      }
    })
    
    // 路由增强 - 仅在客户端执行
    if (typeof document !== 'undefined') {
      router.onBeforeRouteChange = (to) => {
        // 页面切换动画
        document.documentElement.classList.add('page-transitioning')
      }
      
      router.onAfterRouteChanged = (to) => {
        document.documentElement.classList.remove('page-transitioning')
        
        // 代码复制按钮
        setTimeout(() => {
          document.querySelectorAll('div[class*="language-"]').forEach(block => {
            if (!block.querySelector('.copy-code-button')) {
              const button = document.createElement('button')
              button.className = 'copy-code-button'
              button.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`
              button.addEventListener('click', () => {
                const code = block.querySelector('code')?.textContent || ''
                navigator.clipboard.writeText(code)
                button.classList.add('copied')
                setTimeout(() => button.classList.remove('copied'), 2000)
              })
              block.appendChild(button)
            }
          })
        }, 100)
      }
    }
  }
}
