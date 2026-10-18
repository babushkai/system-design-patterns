<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'
import bookManifest from '../../../book.manifest.json'

type Locale = 'en' | 'ja'

const props = withDefaults(defineProps<{ locale?: Locale }>(), {
  locale: 'en',
})

type LocalizedTitle = Record<Locale, string>
type BookChapter = { route: string; title: LocalizedTitle }
type BookPart = {
  number: number
  dir: string
  title: LocalizedTitle
  chapters: BookChapter[]
}

const parts = (bookManifest as unknown as { parts: BookPart[] }).parts

/**
 * The volume opens on whichever chapter the manifest lists first, so this object
 * can never disagree with the sidebar or the print order.
 */
const opening = computed(() => ({ part: parts[0], chapter: parts[0].chapters[0] }))
const destination = computed(() => withBase(`/${opening.value.chapter.route}`))

const copy = {
  en: {
    kicker: 'Architecture Fieldbook',
    titleLead: 'System Design',
    titleAccent: 'Patterns',
    subject: 'Distributed systems · Data · ML · AI',
    edition: 'Fieldbook edition',
    hint: 'Open the book',
    insideLabel: (part: string) => `${part} · In this edition`,
    ariaLabel: (title: string) => `Open the fieldbook and begin with ${title}`,
  },
  ja: {
    kicker: 'Architecture Fieldbook',
    titleLead: 'システム設計',
    titleAccent: 'パターン',
    subject: '分散システム · データ · ML · AI',
    edition: 'フィールドブック版',
    hint: '本を開く',
    insideLabel: (part: string) => `${part} · 本編より`,
    ariaLabel: (title: string) => `フィールドブックを開き、${title}から読む`,
  },
} satisfies Record<Locale, {
  kicker: string
  titleLead: string
  titleAccent: string
  subject: string
  edition: string
  hint: string
  insideLabel: (part: string) => string
  ariaLabel: (title: string) => string
}>

const page = computed(() => copy[props.locale])
const openingTitle = computed(() => opening.value.chapter.title[props.locale])
const partLabel = computed(() =>
  props.locale === 'ja'
    ? `第${opening.value.part.number}部`
    : `Part ${opening.value.part.number}`
)
</script>

<template>
  <a
    class="book-object"
    :href="destination"
    :lang="locale"
    :aria-label="page.ariaLabel(openingTitle)"
  >
    <span class="book-volume">
      <span class="book-back-cover" aria-hidden="true" />

      <span class="book-pages" aria-hidden="true">
        <span class="book-page-edge book-page-edge-one" />
        <span class="book-page-edge book-page-edge-two" />
        <span class="book-first-page">
          <span class="inside-chapter">{{ page.insideLabel(partLabel) }}</span>
          <strong>{{ openingTitle }}</strong>
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
              sizes="(max-width: 1023px) 300px, 340px"
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
            <span class="cover-open-hint">{{ page.hint }} <b>→</b></span>
          </span>
        </span>

        <span class="cover-inside" aria-hidden="true">
          <span class="inside-mark" />
        </span>
      </span>
    </span>
  </a>
</template>

<style scoped>
.book-object {
  --cover: #3f2923;
  --cover-deep: #281914;
  --cover-ink: #f0e5ce;
  --paper: #eee5d2;
  --paper-deep: #d6c8ad;
  display: block;
  width: 100%;
  color: inherit;
  text-decoration: none;
  perspective: 1800px;
  -webkit-tap-highlight-color: transparent;
}

:global(.dark) .book-object {
  --cover: #34231e;
  --cover-deep: #1d1411;
}

.book-object:focus-visible {
  outline: none;
}

.book-object:focus-visible .book-volume {
  outline: 3px solid var(--sdp-accent);
  outline-offset: 8px;
}

.book-volume {
  position: relative;
  display: block;
  width: 100%;
  max-width: 340px;
  aspect-ratio: 0.72;
  border-radius: 3px 10px 10px 3px;
  transform: rotateX(1.5deg) rotateY(-5deg);
  transform-style: preserve-3d;
  transition: transform 460ms cubic-bezier(0.2, 0.72, 0.18, 1);
}

/* The cover cracks open on hover, and the first page underneath names the chapter
   the link actually opens. */
.book-object:hover .book-volume,
.book-object:focus-visible .book-volume {
  transform: rotateX(1deg) rotateY(-11deg) translate3d(0, -5px, 0);
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
  font-size: 0.6rem;
  font-weight: 650;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.book-first-page strong {
  max-width: 12ch;
  font-family: var(--sdp-font-display);
  font-size: clamp(1.4rem, 3.4vw, 2rem);
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
  transition: transform 460ms cubic-bezier(0.2, 0.72, 0.18, 1);
}

.book-object:hover .book-front-cover,
.book-object:focus-visible .book-front-cover {
  transform: translateZ(2px) rotateY(-14deg);
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
  padding: clamp(24px, 4vw, 36px) clamp(22px, 3.6vw, 32px) clamp(20px, 3.4vw, 28px);
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
  font-size: clamp(0.54rem, 1.2vw, 0.64rem);
}

.cover-title {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  margin: 0 !important;
  font-family: var(--sdp-font-display);
  font-size: clamp(1.85rem, 4.6vw, 2.7rem);
  font-weight: 520;
  line-height: 0.93;
  letter-spacing: -0.045em;
}

.cover-title strong {
  color: #dca472;
  font: inherit;
  font-style: italic;
}

.book-object:lang(ja) .cover-title {
  font-size: clamp(1.7rem, 4.2vw, 2.4rem);
  line-height: 1.08;
  letter-spacing: -0.04em;
}

.book-object:lang(ja) .cover-title strong {
  font-style: normal;
}

.cover-plate {
  position: relative;
  align-self: center;
  justify-self: center;
  width: min(100%, 240px);
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
  font-size: clamp(0.5rem, 1.1vw, 0.6rem);
  line-height: 1.5;
}

.cover-footer {
  display: flex;
  gap: 12px;
  align-items: center;
  justify-content: space-between;
  font-size: clamp(0.48rem, 1vw, 0.56rem);
}

.cover-open-hint {
  color: #e4ad7b;
}

.cover-open-hint b {
  display: inline-block;
  font-size: 0.85rem;
  transition: transform 180ms ease;
}

.book-object:hover .cover-open-hint b {
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

@media (prefers-reduced-motion: reduce) {
  .book-volume,
  .book-front-cover,
  .cover-open-hint b {
    transition-duration: 1ms;
  }

  .book-object:hover .book-volume,
  .book-object:focus-visible .book-volume {
    transform: rotateX(1.5deg) rotateY(-5deg);
  }

  .book-object:hover .book-front-cover,
  .book-object:focus-visible .book-front-cover {
    transform: translateZ(2px);
  }
}

@media (forced-colors: active) {
  .book-object:focus-visible .book-volume {
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
