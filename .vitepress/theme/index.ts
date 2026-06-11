import DefaultTheme from 'vitepress/theme'
import './custom.css'
import { onMounted } from 'vue'

let translateInitialized = false

export default {
  extends: DefaultTheme,
  setup() {
    onMounted(() => {
      if (translateInitialized) return
      translateInitialized = true

      const translateDiv = document.createElement('div')
      translateDiv.id = 'google_translate_element'

      const insertTranslate = () => {
        const navBarContent = document.querySelector('.VPNavBarExtra') ||
                              document.querySelector('.VPNavBarHamburger')?.parentElement
        if (navBarContent && !document.getElementById('google_translate_element')) {
          const wrapper = document.createElement('div')
          wrapper.className = 'translate-wrapper'
          wrapper.appendChild(translateDiv)
          navBarContent.parentElement?.insertBefore(wrapper, navBarContent)

          if (!document.getElementById('google-translate-script')) {
            const script = document.createElement('script')
            script.id = 'google-translate-script'
            script.src = '//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit'
            document.body.appendChild(script)

            ;(window as any).googleTranslateElementInit = function() {
              new (window as any).google.translate.TranslateElement({
                pageLanguage: 'en',
                includedLanguages: 'ja,zh-CN,zh-TW,ko,es,fr,de,pt,ru,ar,hi',
                layout: (window as any).google.translate.TranslateElement.InlineLayout.SIMPLE,
                autoDisplay: false
              }, 'google_translate_element')
            }
          }
        }
      }

      insertTranslate()
      setTimeout(insertTranslate, 1000)
    })
  }
}
