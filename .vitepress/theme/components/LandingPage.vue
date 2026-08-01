<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import { useRouter, withBase } from 'vitepress'

type Locale = 'en' | 'ja'

const props = withDefaults(defineProps<{ locale?: Locale }>(), {
  locale: 'en',
})

const copy = {
  en: {
    kicker: 'Architecture Fieldbook',
    titleLead: 'System Design',
    titleAccent: 'Patterns',
    subject: 'Distributed systems · Data · ML · AI',
    edition: 'Fieldbook edition',
    openHint: 'Open the book',
    openLabel: 'Open System Design Patterns and begin with ACID Transactions',
    opening: 'Opening the book',
    chapter: 'Chapter 01',
    chapterTitle: 'ACID Transactions',
    destination: '/01-foundations/01-acid-transactions',
  },
  ja: {
    kicker: 'Architecture Fieldbook',
    titleLead: 'システム設計',
    titleAccent: 'パターン',
    subject: '分散システム · データ · ML · AI',
    edition: 'フィールドブック版',
    openHint: '本を開く',
    openLabel: 'システム設計パターンを開き、ACIDトランザクションから読み始める',
    opening: '本を開いています',
    chapter: '第1章',
    chapterTitle: 'ACIDトランザクション',
    destination: '/ja/01-foundations/01-acid-transactions',
  },
} satisfies Record<Locale, {
  kicker: string
  titleLead: string
  titleAccent: string
  subject: string
  edition: string
  openHint: string
  openLabel: string
  opening: string
  chapter: string
  chapterTitle: string
  destination: string
}>

const page = computed(() => copy[props.locale])
const destination = computed(() => withBase(page.value.destination))
const isOpening = ref(false)
const router = useRouter()

let navigationTimer: number | undefined

function beginOpening() {
  if (isOpening.value) return

  isOpening.value = true
  navigationTimer = window.setTimeout(() => {
    void router.go(destination.value)
  }, 720)
}

function openBook(event: MouseEvent) {
  if (
    event.button !== 0 ||
    event.metaKey ||
    event.ctrlKey ||
    event.shiftKey ||
    event.altKey
  ) {
    return
  }

  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    return
  }

  event.preventDefault()
  beginOpening()
}

onBeforeUnmount(() => {
  if (navigationTimer !== undefined) {
    window.clearTimeout(navigationTimer)
  }
})
</script>

<template>
  <main class="sdp-home book-landing" :lang="locale">
    <a
      class="book-entry"
      :class="{ 'is-opening': isOpening }"
      :href="destination"
      target="_self"
      :aria-label="page.openLabel"
      :aria-busy="isOpening"
      @click="openBook"
    >
      <span class="book-status" aria-live="polite">
        {{ isOpening ? page.opening : '' }}
      </span>

      <span class="book-volume">
        <span class="book-back-cover" aria-hidden="true" />

        <span class="book-pages" aria-hidden="true">
          <span class="book-page-edge book-page-edge-one" />
          <span class="book-page-edge book-page-edge-two" />
          <span class="book-first-page">
            <span class="inside-chapter">{{ page.chapter }}</span>
            <strong>{{ page.chapterTitle }}</strong>
            <span class="inside-rule" />
          </span>
        </span>

        <span class="book-front-cover">
          <span class="cover-face">
            <span class="cover-kicker">{{ page.kicker }}</span>
            <h1 class="cover-title">
              <span>{{ page.titleLead }}</span>
              <strong>{{ page.titleAccent }}</strong>
            </h1>
            <span class="cover-plate">
              <img
                :src="withBase('/art/systems-constellation-960.webp')"
                :srcset="`${withBase('/art/systems-constellation-640.webp')} 640w, ${withBase('/art/systems-constellation-960.webp')} 960w`"
                sizes="(max-width: 680px) 180px, 205px"
                alt=""
                width="960"
                height="960"
                decoding="async"
                fetchpriority="high"
              >
            </span>
            <span class="cover-subject">{{ page.subject }}</span>
            <span class="cover-footer">
              <span>{{ page.edition }}</span>
              <span class="cover-open-hint">{{ page.openHint }} <b>→</b></span>
            </span>
          </span>

          <span class="cover-inside" aria-hidden="true">
            <span class="inside-mark" />
          </span>
        </span>
      </span>
    </a>
  </main>
</template>

<style>
.VPDoc:has(.book-landing),
.VPDoc:has(.book-landing) .main,
.VPDoc:has(.book-landing) .container,
.VPDoc:has(.book-landing) .content,
.VPDoc:has(.book-landing) .content-container {
  width: 100% !important;
  min-width: 0 !important;
  max-width: none !important;
  box-sizing: border-box;
  padding: 0 !important;
}

.VPDoc:has(.book-landing) {
  overflow: clip;
}

.Layout:has(.book-landing) .VPFooter {
  display: none;
}
</style>

<style scoped>
.book-landing {
  --cover: #3f2923;
  --cover-deep: #281914;
  --cover-ink: #f0e5ce;
  --paper: #eee5d2;
  --paper-deep: #d6c8ad;
  min-height: calc(100svh - var(--vp-nav-height, 64px) - 1px);
  color: var(--vp-c-text-1);
  background:
    radial-gradient(circle at 50% 43%, color-mix(in srgb, var(--sdp-accent) 5%, transparent), transparent 34%),
    var(--vp-c-bg);
}

.book-entry {
  display: grid;
  min-height: inherit;
  place-items: center;
  box-sizing: border-box;
  padding: clamp(24px, 5vw, 64px);
  color: inherit;
  text-decoration: none;
  cursor: pointer;
  perspective: 1800px;
  -webkit-tap-highlight-color: transparent;
}

.book-entry:focus-visible {
  outline: none;
}

.book-entry:focus-visible .book-volume {
  outline: 3px solid var(--sdp-accent);
  outline-offset: 8px;
}

.book-status {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  border: 0;
}

.book-volume {
  position: relative;
  display: block;
  width: min(390px, calc(100vw - 72px), 56svh);
  aspect-ratio: 0.72;
  border-radius: 3px 10px 10px 3px;
  transform: rotateX(1.5deg) rotateY(-5deg);
  transform-style: preserve-3d;
  transition: transform 680ms cubic-bezier(0.2, 0.72, 0.18, 1);
}

.book-back-cover,
.book-pages,
.book-front-cover,
.cover-face,
.cover-inside {
  position: absolute;
  inset: 0;
  display: block;
  border-radius: inherit;
}

.book-back-cover {
  background: var(--cover-deep);
  box-shadow:
    4px 20px 38px rgb(28 20 15 / 22%),
    18px 22px 42px rgb(28 20 15 / 18%);
  transform: translateZ(-10px);
}

.book-pages {
  inset: 5px 5px 5px 9px;
  overflow: hidden;
  background: var(--paper);
  border: 1px solid rgb(84 64 40 / 16%);
  border-radius: 2px 8px 8px 2px;
  box-shadow:
    inset -8px 0 12px rgb(77 57 35 / 8%),
    3px 3px 0 var(--paper-deep),
    6px 6px 0 color-mix(in srgb, var(--paper-deep) 75%, #7d6a4b);
  transform: translateZ(-2px);
}

.book-page-edge {
  position: absolute;
  right: 0;
  left: 0;
  height: 1px;
  background: rgb(95 74 49 / 16%);
}

.book-page-edge-one {
  top: 34%;
}

.book-page-edge-two {
  top: 71%;
}

.book-first-page {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 12%;
  color: #3f352a;
  font-family: var(--sdp-font-reading);
  text-align: center;
  background:
    linear-gradient(90deg, rgb(94 70 42 / 8%), transparent 11%),
    var(--paper);
}

.inside-chapter {
  margin-bottom: 14px;
  color: #77634b;
  font-family: var(--sdp-font-ui);
  font-size: 0.68rem;
  font-weight: 650;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.book-first-page strong {
  max-width: 12ch;
  font-family: var(--sdp-font-display);
  font-size: clamp(1.6rem, 4vw, 2.25rem);
  font-weight: 560;
  line-height: 1.08;
}

.inside-rule {
  width: 32px;
  height: 1px;
  margin-top: 22px;
  background: #9d6e4a;
}

.book-front-cover {
  z-index: 3;
  transform: translateZ(2px);
  transform-origin: left center;
  transform-style: preserve-3d;
  transition: transform 680ms cubic-bezier(0.2, 0.72, 0.18, 1);
}

.cover-face,
.cover-inside {
  overflow: hidden;
  backface-visibility: hidden;
  -webkit-backface-visibility: hidden;
}

.cover-face {
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr) auto auto;
  gap: clamp(12px, 2vw, 18px);
  box-sizing: border-box;
  padding: clamp(28px, 5vw, 44px) clamp(26px, 4.5vw, 40px) clamp(24px, 4vw, 34px);
  color: var(--cover-ink);
  background:
    linear-gradient(90deg, rgb(255 255 255 / 4%), transparent 8%, transparent 93%, rgb(0 0 0 / 12%)),
    var(--cover);
  border: 1px solid rgb(239 224 193 / 16%);
  box-shadow:
    inset 10px 0 18px rgb(0 0 0 / 16%),
    inset 1px 0 rgb(255 255 255 / 8%),
    2px 16px 30px rgb(28 20 15 / 24%);
}

.cover-face::before {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 16px;
  width: 1px;
  background: rgb(237 217 180 / 19%);
  box-shadow: 4px 0 12px rgb(0 0 0 / 12%);
  content: '';
}

.cover-kicker,
.cover-subject,
.cover-footer {
  position: relative;
  z-index: 1;
  font-family: var(--sdp-font-ui);
  font-weight: 650;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}

.cover-kicker {
  font-size: clamp(0.58rem, 1.4vw, 0.68rem);
}

.cover-title {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  font-family: var(--sdp-font-display);
  font-size: clamp(2.15rem, 7vw, 3.35rem);
  font-weight: 520;
  line-height: 0.93;
  letter-spacing: -0.045em;
  margin: 0 !important;
}

.cover-title strong {
  color: #dca472;
  font: inherit;
  font-style: italic;
}

.book-landing:lang(ja) .cover-title {
  font-size: clamp(2rem, 6.5vw, 3rem);
  line-height: 1.08;
  letter-spacing: -0.04em;
}

.book-landing:lang(ja) .cover-title strong {
  font-style: normal;
}

.cover-plate {
  position: relative;
  align-self: center;
  justify-self: center;
  width: min(100%, 205px);
  aspect-ratio: 1;
  overflow: hidden;
  border: 1px solid rgb(239 224 193 / 22%);
  background: #d8cbae;
}

.cover-plate img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
  filter: sepia(0.12) saturate(0.72) contrast(1.04);
}

.cover-subject {
  font-size: clamp(0.52rem, 1.2vw, 0.62rem);
  line-height: 1.5;
}

.cover-footer {
  display: flex;
  gap: 12px;
  align-items: center;
  justify-content: space-between;
  font-size: clamp(0.5rem, 1.1vw, 0.58rem);
}

.cover-open-hint {
  color: #e4ad7b;
}

.cover-open-hint b {
  display: inline-block;
  font-size: 0.85rem;
  transition: transform 180ms ease;
}

.book-entry:hover .cover-open-hint b {
  transform: translateX(3px);
}

.cover-inside {
  background:
    linear-gradient(90deg, rgb(70 50 31 / 15%), transparent 14%),
    var(--paper);
  border: 1px solid rgb(84 64 40 / 18%);
  transform: rotateY(180deg);
}

.inside-mark {
  position: absolute;
  inset: 10%;
  border: 1px solid rgb(117 88 54 / 14%);
}

.book-entry.is-opening {
  cursor: wait;
}

.book-entry.is-opening .book-volume {
  transform: translateX(11%) rotateX(0.5deg) rotateY(0deg) scale(0.96);
}

.book-entry.is-opening .book-front-cover {
  transform: translateZ(2px) rotateY(-82deg);
}

.book-entry.is-opening .book-first-page {
  animation: page-arrive 750ms 180ms ease both;
}

@keyframes page-arrive {
  from {
    opacity: 0.66;
    transform: translateX(-5px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

:global(.dark) .book-landing {
  --cover: #34231e;
  --cover-deep: #1d1411;
  background:
    radial-gradient(circle at 50% 43%, rgb(198 132 79 / 6%), transparent 34%),
    var(--vp-c-bg);
}

@media (max-width: 680px) {
  .book-entry {
    padding: 24px 20px 34px;
  }

  .book-volume {
    width: min(316px, calc(100vw - 48px));
  }

  .book-entry.is-opening .book-volume {
    transform: translateX(8%) rotateX(0deg) rotateY(0deg) scale(0.94);
  }

  .book-entry.is-opening .book-front-cover {
    transform: translateZ(2px) rotateY(-78deg);
  }

  .cover-face {
    gap: 13px;
    padding: 27px 26px 23px 29px;
  }

  .cover-plate {
    width: min(100%, 180px);
  }
}

@media (max-height: 720px) and (min-width: 681px) {
  .book-volume {
    width: min(330px, calc(100vw - 72px));
  }

  .cover-face {
    gap: 12px;
    padding: 28px 30px 24px 34px;
  }

  .cover-plate {
    width: min(100%, 184px);
  }
}

@media (max-height: 540px) {
  .book-entry {
    padding-block: 12px;
  }

  .book-volume {
    width: min(390px, calc(100vw - 72px), 50svh);
  }
}

@media (prefers-reduced-motion: reduce) {
  .book-volume,
  .book-front-cover,
  .cover-open-hint b {
    transition-duration: 120ms;
  }

  .book-entry.is-opening .book-volume {
    transform: none;
  }

  .book-entry.is-opening .book-front-cover {
    transform: none;
  }

  .book-entry.is-opening .book-first-page {
    animation: none;
  }
}

@media (forced-colors: active) {
  .book-entry:focus-visible .book-volume {
    outline-color: Highlight;
  }

  .cover-face,
  .book-pages,
  .book-back-cover {
    border: 2px solid CanvasText;
  }

  .cover-plate img {
    opacity: 0.3;
  }
}
</style>
