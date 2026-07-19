<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'

type Locale = 'en' | 'ja'
type LinkCard = {
  eyebrow: string
  title: string
  body: string
  href: string
  external?: boolean
}
type Domain = {
  number: string
  title: string
  body: string
  href: string
}

const props = withDefaults(defineProps<{ locale?: Locale }>(), {
  locale: 'en',
})

const domains: Record<Locale, Domain[]> = {
  en: [
    { number: '01', title: 'Foundations', body: 'Guarantees and failure', href: '/01-foundations/01-acid-transactions' },
    { number: '02', title: 'Distributed Databases', body: 'Replication and consensus', href: '/02-distributed-databases/01-single-leader-replication' },
    { number: '03', title: 'Storage Engines', body: 'Indexes and persistence', href: '/03-storage-engines/01-b-trees' },
    { number: '04', title: 'Caching', body: 'Latency and invalidation', href: '/04-caching/01-cache-strategies' },
    { number: '05', title: 'Messaging', body: 'Delivery and ordering', href: '/05-messaging/01-message-queues' },
    { number: '06', title: 'Scaling', body: 'Load and protection', href: '/06-scaling/01-load-balancing' },
    { number: '07', title: 'Real-Time', body: 'Streams and presence', href: '/07-real-time/01-polling' },
    { number: '08', title: 'Case Studies', body: 'Production evidence', href: '/08-case-studies/01-twitter' },
    { number: '09', title: 'Whitepapers', body: 'Foundational designs', href: '/09-whitepapers/01-mapreduce' },
    { number: '10', title: 'Security', body: 'Trust and identity', href: '/10-security/01-authentication-fundamentals' },
    { number: '11', title: 'Observability & Operations', body: 'Telemetry, reliability, cost, and response', href: '/11-observability/01-distributed-tracing' },
    { number: '12', title: 'Service Connectivity', body: 'Discovery, gateways, mesh, and APIs', href: '/12-service-mesh/01-service-discovery' },
    { number: '13', title: 'Data Pipelines', body: 'Batch and streams', href: '/13-data-pipelines/01-batch-processing' },
    { number: '14', title: 'Search Systems', body: 'Retrieve and rank', href: '/14-search-systems/01-inverted-indexes' },
    { number: '15', title: 'Deployment', body: 'Release and recover', href: '/15-deployment/01-deployment-strategies' },
    { number: '16', title: 'ML Systems', body: 'Features, serving, drift', href: '/16-ml-systems/01-ml-system-fundamentals' },
    { number: '17', title: 'LLM Systems', body: 'Agents and context', href: '/17-llm-systems/01-agent-fundamentals' },
    { number: '18', title: 'Workflow & Jobs', body: 'Workers and recovery', href: '/18-workflow-job-systems/01-workflow-system-fundamentals' },
    { number: '19', title: 'Coding Agent Platforms', body: 'Controlled software change', href: '/19-compound-engineering/01-compound-engineering-fundamentals' },
  ],
  ja: [
    { number: '01', title: '基礎', body: '保証と障害', href: '/ja/01-foundations/01-acid-transactions' },
    { number: '02', title: '分散データベース', body: '複製と合意', href: '/ja/02-distributed-databases/01-single-leader-replication' },
    { number: '03', title: 'ストレージエンジン', body: '索引と永続化', href: '/ja/03-storage-engines/01-b-trees' },
    { number: '04', title: 'キャッシュ', body: 'レイテンシと無効化', href: '/ja/04-caching/01-cache-strategies' },
    { number: '05', title: 'メッセージング', body: '配信と順序', href: '/ja/05-messaging/01-message-queues' },
    { number: '06', title: 'スケーリング', body: '負荷と保護', href: '/ja/06-scaling/01-load-balancing' },
    { number: '07', title: 'リアルタイム', body: 'ストリームとプレゼンス', href: '/ja/07-real-time/01-polling' },
    { number: '08', title: 'ケーススタディ', body: '本番の証拠', href: '/ja/08-case-studies/01-twitter' },
    { number: '09', title: 'ホワイトペーパー', body: '基礎設計', href: '/ja/09-whitepapers/01-mapreduce' },
    { number: '10', title: 'セキュリティ', body: '信頼と認証', href: '/ja/10-security/01-authentication-fundamentals' },
    { number: '11', title: 'オブザーバビリティ', body: '追跡と対応', href: '/ja/11-observability/01-distributed-tracing' },
    { number: '12', title: 'サービスメッシュ', body: '実行時トラフィック', href: '/ja/12-service-mesh/01-service-discovery' },
    { number: '13', title: 'データパイプライン', body: 'バッチとストリーム', href: '/ja/13-data-pipelines/01-batch-processing' },
    { number: '14', title: '検索システム', body: '検索と順位付け', href: '/ja/14-search-systems/01-inverted-indexes' },
    { number: '15', title: 'デプロイメント', body: 'リリースと復旧', href: '/ja/15-deployment/01-deployment-strategies' },
    { number: '16', title: 'MLシステム', body: '特徴量・サービング・ドリフト', href: '/ja/16-ml-systems/01-ml-system-fundamentals' },
    { number: '17', title: 'LLMシステム', body: 'エージェントと文脈', href: '/ja/17-llm-systems/01-agent-fundamentals' },
    { number: '18', title: 'ワークフローとジョブ', body: '実行と復旧', href: '/ja/18-workflow-job-systems/01-workflow-system-fundamentals' },
    { number: '19', title: 'コンパウンドエンジニアリング', body: 'AIネイティブワークフロー', href: '/ja/19-compound-engineering/01-compound-engineering-fundamentals' },
  ],
}

const content = {
  en: {
    titleHtml: 'Design systems that keep their <em>promises.</em>',
    eyebrow: ['Architecture Fieldbook', 'Distributed · Data · ML · AI Systems'],
    deck: 'A deeply technical field guide for engineers who need more than interview diagrams: invariants, trade-offs, failure modes, capacity math, rollout mechanics, and production case studies.',
    edition: 'Fieldbook edition · Evidence-first technical reference',
    primaryCta: { label: 'Start with foundations', href: '/01-foundations/01-acid-transactions' },
    secondaryCta: { label: 'Explore ML systems', href: '/16-ml-systems/01-ml-system-fundamentals' },
    downloadCta: { label: 'Download PDF / EPUB', href: 'https://github.com/babushkai/system-design-patterns/releases/latest', external: true },
    statsLabel: 'Reference statistics',
    stats: [
      ['160', 'Articles'],
      ['19', 'Design domains'],
      ['13', 'Case studies'],
      ['15', 'Paper notes'],
    ],
    mapHeading: ['Complete map', 'Nineteen domains, one connected system.'],
    routesHeading: ['Reading routes', 'Choose the pressure you are designing against.'],
    routes: [
      { eyebrow: 'Correctness path', title: 'Transactions, isolation, CAP, consistency, time, partitions.', body: 'Use when the system must preserve invariants under concurrency and failure.', href: '/01-foundations/01-acid-transactions' },
      { eyebrow: 'State path', title: 'Storage engines, replication, indexes, sharding, encoding.', body: 'Use when data layout and write/read paths dominate the architecture.', href: '/03-storage-engines/01-b-trees' },
      { eyebrow: 'Flow path', title: 'Queues, ordering, delivery guarantees, outbox, backpressure.', body: 'Use when throughput, asynchronous boundaries, and retries decide reliability.', href: '/05-messaging/01-message-queues' },
      { eyebrow: 'Intelligence path', title: 'ML platforms, features, serving, monitoring, experiments, governance.', body: 'Use when statistical systems meet production control planes.', href: '/16-ml-systems/01-ml-system-fundamentals' },
    ] satisfies LinkCard[],
    evidenceEyebrow: 'Evidence shelf',
    evidenceHeading: 'Use papers and production systems as calibration data.',
    evidenceBody: 'The goal is not to memorize diagrams. The goal is to recognize load-bearing decisions and defend them with mechanisms, numbers, and failure analysis.',
    evidenceLinks: [
      { eyebrow: 'Paper trail', title: 'Raft, Dynamo, Spanner, Kafka, FoundationDB', body: 'Read the original mechanisms behind modern infrastructure.', href: '/09-whitepapers/07-raft' },
      { eyebrow: 'Production systems', title: 'Netflix, Slack, Discord, Stripe, Cloudflare', body: 'Compare abstractions against systems that served real traffic.', href: '/08-case-studies/04-netflix' },
      { eyebrow: 'Modern systems', title: 'RAG, agents, orchestration, ML control planes', body: 'Extend classic systems thinking into AI-native architectures.', href: '/17-llm-systems/04-rag-patterns' },
    ] satisfies LinkCard[],
  },
  ja: {
    titleHtml: '約束を守るシステムを<em>設計する。</em>',
    eyebrow: ['Architecture Fieldbook', '分散 · データ · ML · AI システム'],
    deck: '面接用の図ではなく、設計レビューと本番判断のための技術フィールドブック。整合性、障害、容量、ロールアウト、監視、ML/AIシステムまで、実装上のトレードオフを深掘りする。',
    edition: 'フィールドブック版 · 根拠重視の技術リファレンス',
    primaryCta: { label: '基礎から始める', href: '/ja/01-foundations/01-acid-transactions' },
    secondaryCta: { label: 'MLシステムを見る', href: '/ja/16-ml-systems/01-ml-system-fundamentals' },
    downloadCta: { label: 'PDF / EPUB', href: 'https://github.com/babushkai/system-design-patterns/releases/latest', external: true },
    statsLabel: 'リファレンス統計',
    stats: [
      ['160', '記事'],
      ['19', '設計領域'],
      ['13', '本番事例'],
      ['15', '論文ノート'],
    ],
    mapHeading: ['Complete map', '19領域を、ひとつの接続されたシステムとして読む。'],
    routesHeading: ['Reading routes', '設計で向き合う圧力から読む。'],
    routes: [
      { eyebrow: 'Correctness path', title: 'トランザクション、分離、CAP、整合性、時間、分断。', body: '並行性と障害の中で不変条件を守る設計に。', href: '/ja/01-foundations/01-acid-transactions' },
      { eyebrow: 'State path', title: 'ストレージエンジン、複製、索引、シャーディング、エンコーディング。', body: 'データ配置と読み書き経路が支配的な設計に。', href: '/ja/03-storage-engines/01-b-trees' },
      { eyebrow: 'Flow path', title: 'キュー、順序、配信保証、Saga、バックプレッシャー。', body: 'スループット、非同期境界、再試行が信頼性を決める設計に。', href: '/ja/05-messaging/01-message-queues' },
      { eyebrow: 'Intelligence path', title: 'ML基盤、特徴量、サービング、監視、実験、ガバナンス。', body: '統計的システムと本番制御プレーンが交わる設計に。', href: '/ja/16-ml-systems/01-ml-system-fundamentals' },
    ] satisfies LinkCard[],
    evidenceEyebrow: 'Evidence shelf',
    evidenceHeading: '論文と本番システムを、設計判断のキャリブレーションに使う。',
    evidenceBody: '目的は図を暗記することではない。負荷を支える判断を見抜き、メカニズム、数値、障害分析で説明できるようにすること。',
    evidenceLinks: [
      { eyebrow: 'Paper trail', title: 'Raft、Dynamo、Spanner、Kafka、FoundationDB', body: '現代インフラの背後にある原論文のメカニズムを読む。', href: '/ja/09-whitepapers/07-raft' },
      { eyebrow: 'Production systems', title: 'Netflix、Slack、Discord、Stripe、Cloudflare', body: '抽象化を実トラフィックに耐えたシステムと比較する。', href: '/ja/08-case-studies/04-netflix' },
      { eyebrow: 'Modern systems', title: 'RAG、エージェント、オーケストレーション、ML制御プレーン', body: '古典的なシステム思考をAIネイティブアーキテクチャへ拡張する。', href: '/ja/17-llm-systems/04-rag-patterns' },
    ] satisfies LinkCard[],
  },
}

const page = computed(() => content[props.locale])
const domainList = computed(() => domains[props.locale])

function hrefFor(card: { href: string; external?: boolean }) {
  return card.external ? card.href : withBase(card.href)
}

function linkRel(card: { external?: boolean }) {
  return card.external ? 'noreferrer' : undefined
}

function linkTarget(card: { external?: boolean }) {
  return card.external ? '_blank' : undefined
}
</script>

<template>
  <main class="sdp-home" :lang="locale">
    <section class="sdp-hero" aria-labelledby="sdp-title">
      <div class="sdp-shell">
        <p class="sdp-eyebrow">
          <span>{{ page.eyebrow[0] }}</span>
          <span>{{ page.eyebrow[1] }}</span>
        </p>
        <h1 id="sdp-title" v-html="page.titleHtml" />
        <p class="sdp-deck">{{ page.deck }}</p>
        <nav class="sdp-actions" aria-label="Primary paths">
          <a class="sdp-cta" :href="hrefFor(page.primaryCta)">{{ page.primaryCta.label }} <b aria-hidden="true">→</b></a>
          <a class="sdp-link" :href="hrefFor(page.secondaryCta)">{{ page.secondaryCta.label }}</a>
          <a
            class="sdp-link"
            :href="hrefFor(page.downloadCta)"
            :target="linkTarget(page.downloadCta)"
            :rel="linkRel(page.downloadCta)"
          >{{ page.downloadCta.label }}</a>
        </nav>
        <div class="sdp-stats" :aria-label="page.statsLabel">
          <p v-for="stat in page.stats" :key="stat[1]" class="sdp-stat">
            <span class="home-stat-number">{{ stat[0] }}</span>
            <span class="home-stat-label">{{ stat[1] }}</span>
          </p>
        </div>
        <p class="sdp-edition">{{ page.edition }}</p>
      </div>
    </section>

    <section class="sdp-section" aria-labelledby="sdp-map-title">
      <div class="sdp-shell">
        <header class="sdp-section-head">
          <p>{{ page.mapHeading[0] }}</p>
          <h2 id="sdp-map-title">{{ page.mapHeading[1] }}</h2>
        </header>
        <nav class="sdp-domain-map" aria-label="Architecture sections">
          <a v-for="domain in domainList" :key="domain.number" class="sdp-domain" :href="withBase(domain.href)">
            <span>{{ domain.number }}</span>
            <strong>{{ domain.title }}</strong>
            <small>{{ domain.body }}</small>
          </a>
        </nav>
      </div>
    </section>

    <section class="sdp-section" aria-labelledby="sdp-routes-title">
      <div class="sdp-shell">
        <header class="sdp-section-head">
          <p>{{ page.routesHeading[0] }}</p>
          <h2 id="sdp-routes-title">{{ page.routesHeading[1] }}</h2>
        </header>
        <nav class="sdp-routes" aria-label="Curated reading routes">
          <a v-for="route in page.routes" :key="route.eyebrow" class="sdp-route" :href="hrefFor(route)">
            <span class="sdp-route-eyebrow">{{ route.eyebrow }}</span>
            <span class="sdp-route-copy">
              <strong>{{ route.title }}</strong>
              <small>{{ route.body }}</small>
            </span>
            <b aria-hidden="true">→</b>
          </a>
        </nav>
      </div>
    </section>

    <section class="sdp-section" aria-labelledby="sdp-evidence-title">
      <div class="sdp-shell sdp-evidence">
        <header class="sdp-evidence-copy">
          <p>{{ page.evidenceEyebrow }}</p>
          <h2 id="sdp-evidence-title">{{ page.evidenceHeading }}</h2>
          <p class="sdp-evidence-body">{{ page.evidenceBody }}</p>
        </header>
        <nav class="sdp-evidence-list" aria-label="Evidence links">
          <a v-for="link in page.evidenceLinks" :key="link.eyebrow" class="sdp-evidence-item" :href="hrefFor(link)">
            <span class="sdp-evidence-eyebrow">{{ link.eyebrow }}</span>
            <span class="sdp-evidence-copy-inner">
              <strong>{{ link.title }}</strong>
              <small>{{ link.body }}</small>
            </span>
            <b aria-hidden="true">→</b>
          </a>
        </nav>
      </div>
    </section>
  </main>
</template>

<style>
.VPDoc:has(.sdp-home) .content,
.VPDoc:has(.sdp-home) .content-container,
.VPDoc:has(.sdp-home) .container,
.VPDoc:has(.sdp-home) .main,
.VPDoc:has(.sdp-home) {
  width: 100% !important;
  min-width: 0 !important;
  max-width: none !important;
  box-sizing: border-box;
  padding: 0 !important;
  overflow-x: clip;
}
.VPDoc:has(.sdp-home) .vp-doc h1,
.VPDoc:has(.sdp-home) .vp-doc h2,
.VPDoc:has(.sdp-home) .vp-doc h3,
.VPDoc:has(.sdp-home) .vp-doc p {
  margin: 0;
}
</style>

<style scoped>
.sdp-home {
  --sdp-line: var(--vp-c-divider);
  --sdp-accent: var(--vp-c-brand-1);
  --sdp-serif: var(--sdp-font-display, 'Newsreader', 'Iowan Old Style', Georgia, serif);
  width: 100%;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  font-feature-settings: 'ss01' on, 'cv01' on;
}
.sdp-home a {
  color: inherit;
  text-decoration: none;
}
.sdp-home a:focus-visible {
  border-radius: 2px;
  outline: 2px solid var(--sdp-accent);
  outline-offset: 3px;
}
.sdp-shell {
  width: min(1120px, calc(100% - 48px));
  margin: 0 auto;
}

/* Hero */
.sdp-hero {
  padding: clamp(80px, 10vw, 136px) 0 clamp(56px, 7vw, 84px);
}
.sdp-eyebrow {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 12px;
  align-items: center;
  margin: 0 0 30px !important;
  color: var(--vp-c-text-3);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}
.sdp-eyebrow span {
  display: inline-flex;
  align-items: center;
}
.sdp-eyebrow span + span::before {
  width: 3px;
  height: 3px;
  margin-right: 12px;
  border-radius: 999px;
  background: var(--vp-c-text-3);
  content: '';
}
.sdp-hero h1 {
  max-width: 24ch;
  font-size: clamp(2.6rem, 6vw, 4.75rem);
  font-weight: 600;
  line-height: 1.04;
  text-wrap: balance;
}
.sdp-home:lang(en) .sdp-hero h1 {
  letter-spacing: -0.02em;
}
.sdp-hero h1 :deep(em) {
  font-family: var(--sdp-serif);
  font-style: italic;
  font-weight: 500;
  letter-spacing: 0;
}
.sdp-home:lang(ja) .sdp-hero h1 {
  font-size: clamp(2.2rem, 5vw, 3.9rem);
  line-height: 1.18;
}
.sdp-home:lang(ja) .sdp-hero h1 :deep(em) {
  font-family: serif;
  font-style: normal;
  font-weight: 600;
}
.sdp-deck {
  max-width: 62ch;
  margin: 26px 0 0 !important;
  color: var(--vp-c-text-2);
  font-size: clamp(1rem, 1.4vw, 1.125rem);
  line-height: 1.7;
  text-wrap: pretty;
}
.sdp-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 14px 28px;
  align-items: center;
  margin-top: 40px;
}
.sdp-home .sdp-cta {
  display: inline-flex;
  min-height: 46px;
  align-items: center;
  gap: 10px;
  border-radius: 3px;
  padding: 0 22px;
  color: var(--vp-c-bg);
  background: var(--vp-c-text-1);
  font-size: 0.92rem;
  font-weight: 600;
  transition: background-color 0.15s ease;
}
.sdp-home .sdp-cta:hover {
  color: var(--vp-c-bg);
  background: var(--vp-c-text-2);
}
.sdp-cta b {
  font-weight: 400;
}
.sdp-home .sdp-link {
  color: var(--vp-c-text-1);
  font-size: 0.92rem;
  font-weight: 500;
  text-decoration: underline;
  text-decoration-color: var(--sdp-line);
  text-decoration-thickness: 1px;
  text-underline-offset: 5px;
  transition: color 0.15s ease, text-decoration-color 0.15s ease;
}
.sdp-link:hover {
  color: var(--sdp-accent);
  text-decoration-color: currentColor;
}

/* Stats strip */
.sdp-stats {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 24px;
  margin-top: clamp(56px, 8vw, 96px);
  border-top: 1px solid var(--sdp-line);
  padding-top: 28px;
}
.sdp-stat {
  display: grid;
  gap: 7px;
  margin: 0 !important;
}
.home-stat-number {
  font-size: clamp(1.7rem, 2.6vw, 2.3rem);
  font-weight: 600;
  line-height: 1;
  letter-spacing: -0.01em;
}
.home-stat-label {
  color: var(--vp-c-text-3);
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.sdp-edition {
  margin: 40px 0 0 !important;
  color: var(--vp-c-text-3);
  font-size: 0.66rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

/* Sections */
.sdp-section {
  border-top: 1px solid var(--sdp-line);
  padding: clamp(64px, 8vw, 104px) 0;
}
.sdp-section-head {
  display: grid;
  grid-template-columns: minmax(160px, 0.28fr) minmax(0, 0.72fr);
  gap: 16px clamp(20px, 4vw, 64px);
  align-items: baseline;
  margin-bottom: clamp(36px, 5vw, 60px);
}
.sdp-section-head p,
.sdp-evidence-copy > p:first-child {
  margin: 0 !important;
  color: var(--vp-c-text-3);
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}
.sdp-section-head h2,
.sdp-evidence-copy h2 {
  max-width: 32ch;
  font-size: clamp(1.6rem, 3vw, 2.4rem);
  font-weight: 600;
  line-height: 1.18;
  text-wrap: balance;
}
.sdp-home:lang(en) .sdp-section-head h2,
.sdp-home:lang(en) .sdp-evidence-copy h2 {
  letter-spacing: -0.015em;
}

/* Domain map */
.sdp-domain-map {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0 40px;
}
.sdp-domain {
  display: grid;
  grid-template-columns: 2.6em minmax(0, 1fr);
  align-items: baseline;
  border-top: 1px solid var(--sdp-line);
  padding: 18px 0 24px;
}
.sdp-domain span {
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono, ui-monospace, monospace);
  font-size: 0.72rem;
}
.sdp-domain strong {
  font-size: 0.98rem;
  font-weight: 600;
  line-height: 1.3;
  transition: color 0.15s ease;
}
.sdp-domain small {
  grid-column: 2;
  margin-top: 5px;
  color: var(--vp-c-text-2);
  font-size: 0.84rem;
  line-height: 1.5;
}
.sdp-domain:hover strong {
  color: var(--sdp-accent);
}

/* Reading routes */
.sdp-routes {
  display: grid;
  border-bottom: 1px solid var(--sdp-line);
}
.sdp-route {
  display: grid;
  grid-template-columns: minmax(150px, 0.28fr) minmax(0, 1fr) auto;
  gap: 10px clamp(20px, 4vw, 64px);
  align-items: baseline;
  border-top: 1px solid var(--sdp-line);
  padding: 28px 0;
}
.sdp-route-eyebrow,
.sdp-evidence-eyebrow {
  color: var(--vp-c-text-3);
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.sdp-route strong,
.sdp-evidence-item strong {
  display: block;
  font-size: 1.05rem;
  font-weight: 600;
  line-height: 1.4;
  transition: color 0.15s ease;
}
.sdp-route small,
.sdp-evidence-item small {
  display: block;
  margin-top: 6px;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
  line-height: 1.55;
}
.sdp-route b,
.sdp-evidence-item b {
  color: var(--vp-c-text-3);
  font-size: 1.1rem;
  font-weight: 400;
  transition: color 0.15s ease, transform 0.15s ease;
}
.sdp-route:hover strong,
.sdp-evidence-item:hover strong {
  color: var(--sdp-accent);
}
.sdp-route:hover b,
.sdp-evidence-item:hover b {
  color: var(--sdp-accent);
  transform: translateX(4px);
}

/* Evidence shelf */
.sdp-evidence {
  display: grid;
  grid-template-columns: minmax(0, 0.38fr) minmax(0, 0.62fr);
  gap: clamp(36px, 5vw, 88px);
  align-items: start;
}
.sdp-evidence-copy h2 {
  margin-top: 16px !important;
}
.sdp-evidence-body {
  margin: 18px 0 0 !important;
  color: var(--vp-c-text-2);
  font-size: 0.95rem;
  line-height: 1.65;
}
.sdp-evidence-list {
  display: grid;
  border-bottom: 1px solid var(--sdp-line);
}
.sdp-evidence-item {
  display: grid;
  grid-template-columns: minmax(140px, 0.3fr) minmax(0, 1fr) auto;
  gap: 10px clamp(16px, 3vw, 40px);
  align-items: baseline;
  border-top: 1px solid var(--sdp-line);
  padding: 24px 0;
}

/* Responsive */
@media (max-width: 980px) {
  .sdp-section-head,
  .sdp-evidence {
    grid-template-columns: 1fr;
  }
  .sdp-domain-map {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .sdp-route,
  .sdp-evidence-item {
    grid-template-columns: minmax(0, 1fr) auto;
  }
  .sdp-route-eyebrow,
  .sdp-evidence-eyebrow {
    grid-column: 1 / -1;
  }
}
@media (max-width: 680px) {
  .sdp-shell {
    width: calc(100% - 40px);
  }
  .sdp-hero {
    padding: 56px 0 48px;
  }
  .sdp-hero h1 {
    font-size: clamp(2.2rem, 11vw, 2.9rem);
  }
  .sdp-actions {
    align-items: stretch;
    flex-direction: column;
  }
  .sdp-cta {
    justify-content: center;
  }
  .sdp-stats {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 24px 16px;
  }
  .sdp-domain-map {
    grid-template-columns: 1fr;
  }
  .sdp-domain {
    padding: 15px 0 18px;
  }
  .sdp-section {
    padding: 52px 0;
  }
}
@media (prefers-reduced-motion: reduce) {
  .sdp-home .sdp-cta,
  .sdp-home .sdp-link,
  .sdp-domain strong,
  .sdp-route strong,
  .sdp-route b,
  .sdp-evidence-item strong,
  .sdp-evidence-item b {
    transition: none;
  }
  .sdp-route:hover b,
  .sdp-evidence-item:hover b {
    transform: none;
  }
}
</style>
