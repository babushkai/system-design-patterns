import { withMermaid } from 'vitepress-plugin-mermaid'
import bookManifest from '../book.manifest.json'

/**
 * Part and chapter order lives in `book.manifest.json`, the same file that drives the
 * PDF/EPUB build (`scripts/book_structure.py --chapters`) and the README table of
 * contents. Add a chapter there and every surface below follows it.
 */
type Locale = 'en' | 'ja'
type LocalizedTitle = Record<Locale, string>
type BookChapter = { route: string; title: LocalizedTitle }
type BookPart = {
  number: number
  dir: string
  title: LocalizedTitle
  chapters: BookChapter[]
}

const parts = (bookManifest as unknown as { parts: BookPart[] }).parts

const routeLink = (route: string, locale: Locale) => `/${locale === 'ja' ? 'ja/' : ''}${route}`

const sidebarFor = (locale: Locale) =>
  parts.map((part, index) => ({
    text: `${part.number}. ${part.title[locale]}`,
    // The first part stays open so the book opens on its foundations.
    collapsed: index > 0,
    items: part.chapters.map(chapter => ({
      text: chapter.title[locale],
      link: routeLink(chapter.route, locale)
    }))
  }))

const sidebarEN = sidebarFor('en')
const sidebarJA = sidebarFor('ja')
const firstChapterLink = (locale: Locale) => routeLink(parts[0].chapters[0].route, locale)

export default withMermaid({
  title: 'System Design Patterns',
  description: 'An architecture fieldbook for reliable distributed, data, ML, and AI systems',

  // Base URL for the custom domain
  base: '/',

  // README is the repository landing page, not a localized documentation route.
  srcExclude: ['README.md'],

  // Generated-link integrity is checked after every production build.
  ignoreDeadLinks: true,

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', sizes: 'any', href: '/favicon-book.svg' }],
    ['meta', { name: 'theme-color', content: '#f3efe6', media: '(prefers-color-scheme: light)' }],
    ['meta', { name: 'theme-color', content: '#151411', media: '(prefers-color-scheme: dark)' }],
    ['meta', { name: 'color-scheme', content: 'light dark' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'System Design Patterns' }],
    ['meta', { property: 'og:description', content: 'An architecture fieldbook for reliable distributed, data, ML, and AI systems.' }],
    ['meta', { name: 'twitter:card', content: 'summary' }],
  ],

  locales: {
    root: {
      label: 'English',
      lang: 'en',
      themeConfig: {
        nav: [
          { text: 'Read', link: firstChapterLink('en') },
          { text: 'PDF / EPUB', link: 'https://github.com/babushkai/system-design-patterns/releases/latest' },
        ],
        sidebar: { '/': sidebarEN },
        editLink: {
          pattern: 'https://github.com/babushkai/system-design-patterns/edit/main/:path',
          text: 'Edit this page on GitHub'
        },
        docFooter: {
          prev: 'Previous',
          next: 'Next'
        },
        returnToTopLabel: 'Back to top',
        footer: {
          message: 'A practical reference for distributed system design. Released under the MIT License.',
          copyright: 'Copyright 2024-present Babushkai'
        },
      }
    },
    ja: {
      label: '日本語',
      lang: 'ja',
      themeConfig: {
        nav: [
          { text: '読む', link: firstChapterLink('ja') },
          { text: 'PDF / EPUB', link: 'https://github.com/babushkai/system-design-patterns/releases/latest' },
        ],
        sidebar: { '/ja/': sidebarJA },
        editLink: {
          pattern: 'https://github.com/babushkai/system-design-patterns/edit/main/:path',
          text: 'GitHubでこのページを編集'
        },
        docFooter: { prev: '前へ', next: '次へ' },
        returnToTopLabel: 'トップに戻る',
        outlineTitle: 'このページの内容',
        footer: {
          message: 'MITライセンスの下で公開。Babushkaiコミュニティが構築。',
          copyright: 'Copyright 2024-present Babushkai'
        },
      }
    }
  },

  themeConfig: {
    siteTitle: 'System Design Patterns',
    logo: {
      src: '/favicon-book.svg',
      alt: ''
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/babushkai/system-design-patterns' }
    ],

    search: {
      provider: 'local'
    },

    outline: {
      level: 2,
      label: 'On this page'
    }
  },

  markdown: {
    math: true,
    lineNumbers: true,
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },

  lastUpdated: true,
  cleanUrls: true,
mermaid: {
  theme: 'base',
  themeVariables: {
    primaryColor: '#8A4B37',
    primaryTextColor: '#FFFDF8',
    primaryBorderColor: '#653427',
    lineColor: '#47675E',
    secondaryColor: '#E9E0D1',
    tertiaryColor: '#F2E6CF',
    fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
  },
},
mermaidPlugin: {
  class: 'mermaid',
},
})
