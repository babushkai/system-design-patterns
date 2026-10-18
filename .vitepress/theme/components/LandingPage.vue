<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'
import BookVolume from './BookVolume.vue'
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
const totalChapters = parts.reduce((sum, part) => sum + part.chapters.length, 0)

/** One entry per question a reader arrives with. Routes are checked against the manifest. */
const doors = [
  { id: 'correctness', route: '01-foundations/01-acid-transactions' },
  { id: 'load', route: '04-caching/01-cache-strategies' },
  { id: 'async', route: '18-workflow-job-systems/01-workflow-system-fundamentals' },
  { id: 'retrieval', route: '14-search-systems/01-inverted-indexes' },
  { id: 'intelligence', route: '16-ml-systems/01-ml-system-fundamentals' },
  { id: 'signoff', route: '20-production-readiness/01-design-documents-and-decision-records' },
].map((door, index) => ({ ...door, order: index + 1 }))

const chaptersByRoute = new Map<string, { part: BookPart; chapter: BookChapter }>()
for (const part of parts) {
  for (const chapter of part.chapters) {
    chaptersByRoute.set(chapter.route, { part, chapter })
  }
}

const copy = {
  en: {
    scale: `${parts.length} parts · ${totalChapters} chapters · English + 日本語`,
    pitch:
      'Mechanism-level reasoning for distributed, data, ML, and AI systems. Every chapter is written as a production contract: the invariant, the arithmetic behind it, and the evidence that defends it.',
    doorsHeading: 'Start with your question',
    partsHeading: 'The whole edition',
    footnote:
      'Parts are grouped by role rather than difficulty. Every chapter opens with a TL;DR and closes with three to eight curated neighbours, so any page can be the first one you open.',
    search: `Search ${totalChapters} chapters`,
    source: 'PDF / EPUB',
    otherLocale: '日本語で読む',
    partLabel: (part: BookPart) => `Part ${part.number}`,
    chapterCount: (count: number) => `${count} ${count === 1 ? 'chapter' : 'chapters'}`,
    doors: {
      correctness: {
        question: 'Is this data still correct under concurrent writes?',
        answer:
          'Transactions, isolation levels, replication and conflict resolution, traced through the interleavings that break each guarantee.',
      },
      load: {
        question: 'What gives way first when the load grows?',
        answer:
          'Cache economics, invalidation races, partitioning and edge delivery, with the capacity arithmetic behind every choice.',
      },
      async: {
        question: 'How does work survive past the request?',
        answer:
          'Queues, worker pools, durable execution, commit protocols for retries and idempotency, leases and recovery.',
      },
      retrieval: {
        question: 'How do you find the right thing quickly?',
        answer:
          'Inverted indexes, lexical query execution, vector retrieval, ranking, and typeahead under real traffic.',
      },
      intelligence: {
        question: 'How do models and agents reach production?',
        answer:
          'Feature stores, GPU inference, serving, offline and online evaluation, retrieval, and platforms for coding agents.',
      },
      signoff: {
        question: 'Is this design ready to be signed off?',
        answer:
          'Design documents and ADRs, review governance, supply-chain risk, verification strategy, and lawful retention.',
      },
    },
  },
  ja: {
    scale: `全${parts.length}部・合計${totalChapters}章・日英対応`,
    pitch:
      '分散システム・データ・ML・AIの仕組みを、契約として読む一冊。不変条件、その背後の計算、主張を支える根拠まで、一章ごとに揃えてあります。',
    doorsHeading: '問いから入る',
    partsHeading: '全編の構成',
    footnote:
      '部は難易度ではなく役割で分けてあります。各章の冒頭にTL;DR、末尾に関連する3〜8章を理由付きで置いてあるので、どのページから開いても構いません。',
    search: `全${totalChapters}章を検索`,
    source: 'PDF / EPUB',
    otherLocale: 'Read in English',
    partLabel: (part: BookPart) => `第${part.number}部`,
    chapterCount: (count: number) => `${count}章`,
    doors: {
      correctness: {
        question: '同時書き込みの下で、データはまだ正しいか？',
        answer: 'トランザクション、分離レベル、レプリケーション、競合解決。保証を壊す干渉まで追跡する。',
      },
      load: {
        question: '負荷が増えたとき、最初にどこが壊れるか？',
        answer: 'キャッシュの経済性、無効化の競合、パーティショニング、エッジ配信と、その容量計算。',
      },
      async: {
        question: 'リクエストを超えた仕事をどう進めるか？',
        answer: 'キュー、ワーカープール、永続実行、リトライと冪等性、リースと復旧。',
      },
      retrieval: {
        question: '正しいものをどう素早く見つけるか？',
        answer: '転置インデックス、語彙クエリ実行、ベクトル検索、ランキング、入力補完。',
      },
      intelligence: {
        question: 'モデルとエージェントをどう本番に出すか？',
        answer: 'フィーチャーストア、GPU推論、サービング、評価、検索、そしてコーディングエージェントの基盤。',
      },
      signoff: {
        question: 'その設計は承認できる状態か？',
        answer: '設計文書とADR、レビューとガバナンス、サプライチェーン、検証戦略、データの保持。',
      },
    },
  },
} satisfies Record<Locale, {
  scale: string
  pitch: string
  doorsHeading: string
  partsHeading: string
  footnote: string
  search: string
  source: string
  otherLocale: string
  partLabel: (part: BookPart) => string
  chapterCount: (count: number) => string
  doors: Record<string, { question: string; answer: string }>
}>

const page = computed(() => copy[props.locale])

/**
 * Doors resolve through the manifest. A route that stops existing falls back to the
 * first chapter of Part 1 rather than shipping a dead link, and says so in development.
 */
const resolvedDoors = computed(() =>
  doors.map((door) => {
    const found = chaptersByRoute.get(door.route) ?? { part: parts[0], chapter: parts[0].chapters[0] }
    if (import.meta.env.DEV && !chaptersByRoute.has(door.route)) {
      console.warn(`[landing] door "${door.id}" points at a route that is no longer in the manifest: ${door.route}`)
    }
    return {
      id: door.id,
      order: door.order,
      href: withBase(`/${found.chapter.route}`),
      meta: `${page.value.partLabel(found.part)} · ${found.chapter.title[props.locale]}`,
      ...page.value.doors[door.id],
    }
  })
)

const index = computed(() =>
  parts.map((part) => ({
    number: String(part.number).padStart(2, '0'),
    href: withBase(`/${part.chapters[0].route}`),
    title: part.title[props.locale],
    count: page.value.chapterCount(part.chapters.length),
  }))
)

const localeSwitch = computed(() => withBase(props.locale === 'ja' ? '/' : '/ja/'))

const searchKeys = ref('⌘K')
onMounted(() => {
  if (!/Mac|iPhone|iPod|iPad/i.test(navigator.platform)) {
    searchKeys.value = 'Ctrl K'
  }
})

function focusSearch() {
  document.querySelector<HTMLButtonElement>('.VPNavBarSearch button')?.click()
}
</script>

<template>
  <main class="sdp-home sdp-entrance" :lang="locale">
    <div class="entrance-shell">
      <header class="entrance-rail">
        <BookVolume :locale="locale" />

        <p class="entrance-scale">{{ page.scale }}</p>
        <p class="entrance-pitch">{{ page.pitch }}</p>

        <ul class="entrance-utility">
          <li>
            <button type="button" @click="focusSearch">
              {{ page.search }}
              <kbd>{{ searchKeys }}</kbd>
            </button>
          </li>
          <li>
            <a
              href="https://github.com/babushkai/system-design-patterns/releases/latest"
              target="_blank"
              rel="noopener"
              >{{ page.source }}</a
            >
          </li>
          <li>
            <a :href="localeSwitch">{{ page.otherLocale }}</a>
          </li>
        </ul>
      </header>

      <div class="entrance-index">
        <section aria-labelledby="entrance-doors">
          <h2 id="entrance-doors" class="entrance-heading">{{ page.doorsHeading }}</h2>
          <div class="door-grid">
            <a
              v-for="door in resolvedDoors"
              :key="door.id"
              class="door"
              :href="door.href"
            >
              <span class="door-question">{{ door.question }}</span>
              <span class="door-answer">{{ door.answer }}</span>
              <span class="door-meta">{{ door.meta }} <b>→</b></span>
            </a>
          </div>
        </section>

        <section aria-labelledby="entrance-parts">
          <h2 id="entrance-parts" class="entrance-heading">{{ page.partsHeading }}</h2>
          <ol class="parts-index">
            <li v-for="partEntry in index" :key="partEntry.number">
              <a class="part-row" :href="partEntry.href">
                <span class="part-number">{{ partEntry.number }}</span>
                <span class="part-title">{{ partEntry.title }}</span>
                <span class="part-count">{{ partEntry.count }}</span>
              </a>
            </li>
          </ol>
          <p class="entrance-footnote">{{ page.footnote }}</p>
        </section>
      </div>
    </div>
  </main>
</template>

<style scoped>
.sdp-entrance {
  min-height: calc(100svh - var(--vp-nav-height, 64px) - 1px);
  color: var(--vp-c-text-1);
  background:
    radial-gradient(circle at 18% 8%, color-mix(in srgb, var(--sdp-accent) 7%, transparent), transparent 42%),
    var(--vp-c-bg);
}

.entrance-shell {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: clamp(44px, 6vw, 84px);
  box-sizing: border-box;
  max-width: 1240px;
  margin: 0 auto;
  padding: clamp(32px, 5vw, 72px) clamp(20px, 5vw, 64px) clamp(56px, 7vw, 104px);
}

@media (min-width: 1024px) {
  .entrance-shell {
    grid-template-columns: minmax(288px, 356px) minmax(0, 1fr);
    align-items: start;
  }

  .entrance-rail {
    position: sticky;
    top: calc(var(--vp-nav-height) + 32px);
  }
}

.entrance-scale {
  margin: clamp(24px, 3vw, 32px) 0 0;
  color: var(--sdp-accent);
  font-family: var(--sdp-font-ui);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}

.entrance-pitch {
  max-width: 36ch;
  margin: 14px 0 0;
  font-family: var(--sdp-font-reading);
  font-size: 1.02rem;
  line-height: 1.66;
}

.sdp-entrance:lang(ja) .entrance-pitch {
  max-width: none;
  font-family: var(--sdp-font-reading-ja);
  font-size: 0.95rem;
  line-height: 1.9;
}

.entrance-utility {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: clamp(24px, 3vw, 32px) 0 0;
  padding: 0;
  list-style: none;
}

.entrance-utility a,
.entrance-utility button {
  display: inline-flex;
  gap: 9px;
  align-items: center;
  padding: 9px 13px;
  color: var(--sdp-ink);
  font-family: var(--sdp-font-ui);
  font-size: 0.7rem;
  font-weight: 650;
  letter-spacing: 0.05em;
  text-decoration: none;
  background: color-mix(in srgb, var(--sdp-paper-raised) 70%, transparent);
  border: 1px solid var(--sdp-rule);
  border-radius: 3px;
  cursor: pointer;
  transition: border-color 160ms ease, color 160ms ease;
}

.entrance-utility a:hover,
.entrance-utility button:hover {
  color: var(--sdp-accent);
  border-color: color-mix(in srgb, var(--sdp-accent) 62%, var(--sdp-rule));
}

.entrance-utility kbd {
  padding: 2px 5px;
  color: var(--sdp-ink-muted);
  font-family: var(--sdp-font-mono);
  font-size: 0.6rem;
  background: var(--sdp-paper-sunken);
  border: 1px solid var(--sdp-rule);
  border-radius: 2px;
}

.entrance-heading {
  display: flex;
  gap: 16px;
  align-items: center;
  margin: 0 0 clamp(20px, 2.4vw, 28px);
  color: var(--sdp-ink-muted);
  font-family: var(--sdp-font-ui);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.17em;
  text-transform: uppercase;
}

.entrance-heading::after {
  flex: 1;
  height: 1px;
  background: var(--sdp-rule);
  content: '';
}

.entrance-index section + section {
  margin-top: clamp(48px, 6vw, 72px);
}

.door-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(252px, 1fr));
  gap: clamp(20px, 2.4vw, 30px) clamp(26px, 3vw, 44px);
}

.door {
  display: flex;
  flex-direction: column;
  gap: 9px;
  padding-top: 17px;
  color: inherit;
  text-decoration: none;
  border-top: 1px solid var(--sdp-rule);
}

.door-question {
  color: var(--vp-c-text-1);
  font-family: var(--sdp-font-reading);
  font-size: 1.07rem;
  line-height: 1.3;
  letter-spacing: -0.012em;
  text-decoration: none;
  transition: color 160ms ease;
}

.sdp-entrance:lang(ja) .door-question {
  font-family: var(--sdp-font-reading-ja);
  font-size: 1rem;
  line-height: 1.6;
  letter-spacing: 0;
}

.door-answer {
  color: var(--sdp-ink-muted);
  font-family: var(--sdp-font-ui);
  font-size: 0.83rem;
  line-height: 1.62;
}

.sdp-entrance:lang(ja) .door-answer {
  font-size: 0.8rem;
  line-height: 1.95;
}

.door-meta {
  margin-top: auto;
  padding-top: 6px;
  color: color-mix(in srgb, var(--sdp-accent) 78%, var(--sdp-ink-muted));
  font-family: var(--sdp-font-ui);
  font-size: 0.63rem;
  font-weight: 650;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.door-meta b {
  display: inline-block;
  font-weight: 400;
  transition: transform 160ms ease;
}

.door:hover .door-question {
  color: var(--sdp-accent);
}

.door:hover .door-meta b {
  transform: translateX(3px);
}

.parts-index {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(276px, 1fr));
  gap: 0 clamp(30px, 4vw, 58px);
  margin: 0;
  padding: 0;
  list-style: none;
}

.part-row {
  display: flex;
  gap: 12px;
  align-items: baseline;
  padding: 10px 0;
  color: inherit;
  text-decoration: none;
  border-bottom: 1px solid color-mix(in srgb, var(--sdp-rule) 64%, transparent);
}

.part-number {
  color: var(--sdp-accent);
  font-family: var(--sdp-font-mono);
  font-size: 0.68rem;
  font-variant-numeric: tabular-nums;
}

.part-title {
  flex: 1;
  font-size: 0.92rem;
  line-height: 1.4;
  transition: color 160ms ease;
}

.sdp-entrance:lang(ja) .part-title {
  font-family: var(--sdp-font-reading-ja);
  font-size: 0.87rem;
}

.part-count {
  color: var(--sdp-ink-muted);
  font-family: var(--sdp-font-ui);
  font-size: 0.66rem;
  letter-spacing: 0.06em;
  white-space: nowrap;
}

.part-row:hover .part-title {
  color: var(--sdp-accent);
}

.entrance-footnote {
  max-width: 74ch;
  margin: clamp(22px, 3vw, 30px) 0 0;
  color: var(--sdp-ink-muted);
  font-size: 0.8rem;
  line-height: 1.68;
}

.sdp-entrance:lang(ja) .entrance-footnote {
  font-size: 0.78rem;
  line-height: 2;
}

@media (max-width: 640px) {
  .door-grid,
  .parts-index {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (forced-colors: active) {
  .door,
  .part-row,
  .entrance-utility a,
  .entrance-utility button,
  .entrance-utility kbd {
    border-color: CanvasText;
    background: Canvas;
  }
}
</style>
