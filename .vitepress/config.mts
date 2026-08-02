import { withMermaid } from 'vitepress-plugin-mermaid'

// Shared sidebar configuration
const sidebarEN = [
  {
    text: '1. Foundations',
    collapsed: false,
    items: [
      { text: 'ACID Transactions', link: '/01-foundations/01-acid-transactions' },
      { text: 'Isolation Levels', link: '/01-foundations/02-isolation-levels' },
      { text: 'CAP Theorem', link: '/01-foundations/03-cap-theorem' },
      { text: 'Consistency Models', link: '/01-foundations/04-consistency-models' },
      { text: 'Distributed Time', link: '/01-foundations/05-distributed-time' },
      { text: 'Failure Semantics and Recovery', link: '/01-foundations/06-failure-modes' },
      { text: 'Idempotency', link: '/01-foundations/08-idempotency' },
      { text: 'Distributed Locks', link: '/01-foundations/09-distributed-locks' },
      { text: 'Capacity Planning & Estimation', link: '/01-foundations/10-capacity-planning' },
    ]
  },
  {
    text: '2. Distributed Databases',
    collapsed: true,
    items: [
      { text: 'Single-Leader Replication', link: '/02-distributed-databases/01-single-leader-replication' },
      { text: 'Multi-Leader Replication', link: '/02-distributed-databases/02-multi-leader-replication' },
      { text: 'Leaderless Replication', link: '/02-distributed-databases/03-leaderless-replication' },
      { text: 'Conflict Resolution', link: '/02-distributed-databases/04-conflict-resolution' },
      { text: 'Partitioning Strategies', link: '/02-distributed-databases/05-partitioning-strategies' },
      { text: 'Secondary Indexes', link: '/02-distributed-databases/06-secondary-indexes' },
      { text: 'Distributed Transactions', link: '/02-distributed-databases/07-distributed-transactions' },
      { text: 'Consensus Algorithms', link: '/02-distributed-databases/08-consensus-algorithms' },
      { text: 'Leader Election', link: '/02-distributed-databases/09-leader-election' },
      { text: 'Workload-Driven Data Modeling', link: '/02-distributed-databases/10-data-modeling' },
    ]
  },
  {
    text: '3. Storage Engines',
    collapsed: true,
    items: [
      { text: 'B-Trees', link: '/03-storage-engines/01-b-trees' },
      { text: 'LSM Trees', link: '/03-storage-engines/02-lsm-trees' },
      { text: 'SSTables & Compaction', link: '/03-storage-engines/03-sstables-compaction' },
      { text: 'Write-Ahead Logging', link: '/03-storage-engines/04-write-ahead-logging' },
      { text: 'Bloom Filters', link: '/03-storage-engines/05-bloom-filters' },
      { text: 'Column Storage', link: '/03-storage-engines/06-column-storage' },
      { text: 'Data Encoding', link: '/03-storage-engines/07-data-encoding' },
      { text: 'Object Storage & Commit Protocols', link: '/03-storage-engines/08-object-storage' },
    ]
  },
  {
    text: '4. Caching',
    collapsed: true,
    items: [
      { text: 'Semantics & Economics', link: '/04-caching/01-cache-strategies' },
      { text: 'Invalidation & Coherence', link: '/04-caching/02-cache-invalidation' },
      { text: 'Distributed Cache Internals', link: '/04-caching/03-distributed-caching' },
      { text: 'Stampede, Cold Start & Warming', link: '/04-caching/04-cache-stampede' },
    ]
  },
  {
    text: '5. Messaging',
    collapsed: true,
    items: [
      { text: 'Message Queue Architecture', link: '/05-messaging/01-message-queues' },
      { text: 'Publish-Subscribe Architecture', link: '/05-messaging/02-pub-sub' },
      { text: 'Message Ordering', link: '/05-messaging/03-message-ordering' },
      { text: 'Delivery & Effect Boundaries', link: '/05-messaging/04-delivery-guarantees' },
      { text: 'Event Sourcing & Domain Logs', link: '/05-messaging/05-event-sourcing' },
      { text: 'CQRS & Projection Architecture', link: '/05-messaging/06-cqrs' },
      { text: 'Outbox, Inbox & CDC', link: '/05-messaging/07-outbox-pattern' },
      { text: 'Poison Messages & Redrive', link: '/05-messaging/08-dead-letter-queues' },
    ]
  },
  {
    text: '6. Scaling',
    collapsed: true,
    items: [
      { text: 'Load Balancing', link: '/06-scaling/01-load-balancing' },
      { text: 'Horizontal vs Vertical', link: '/06-scaling/02-horizontal-vertical' },
      { text: 'Database Sharding', link: '/06-scaling/03-database-sharding' },
      { text: 'CDN Architecture', link: '/06-scaling/04-cdn-architecture' },
      { text: 'Rate Limiting', link: '/06-scaling/05-rate-limiting' },
      { text: 'Circuit Breakers', link: '/06-scaling/06-circuit-breakers' },
      { text: 'Backpressure', link: '/06-scaling/07-backpressure' },
      { text: 'Auto-Scaling', link: '/06-scaling/08-auto-scaling' },
      { text: 'Multi-Region Architecture', link: '/06-scaling/09-multi-region-architecture' },
      { text: 'Retries & Hedging', link: '/06-scaling/10-retries-timeouts-hedging' },
      { text: 'Cell-Based Architecture', link: '/06-scaling/11-cell-based-architecture' },
      { text: 'Multi-Tenant Isolation & Lifecycle', link: '/06-scaling/12-multi-tenancy' },
      { text: 'DNS & Connections', link: '/06-scaling/13-dns-and-connection-management' },
      { text: 'Network Transport Internals', link: '/06-scaling/14-network-transport-internals' },
    ]
  },
  {
    text: '7. Real-Time',
    collapsed: true,
    items: [
      { text: 'Client Delivery Transports', link: '/07-real-time/01-polling' },
      { text: 'WebRTC', link: '/07-real-time/05-webrtc' },
      { text: 'Presence', link: '/07-real-time/06-presence' },
      { text: 'Collaborative Sync & CRDTs', link: '/07-real-time/07-crdts-collaborative-editing' },
    ]
  },
  {
    text: '8. Case Studies',
    collapsed: true,
    items: [
      { text: 'Twitter', link: '/08-case-studies/01-twitter' },
      { text: 'Instagram', link: '/08-case-studies/02-instagram' },
      { text: 'Uber', link: '/08-case-studies/03-uber' },
      { text: 'Netflix', link: '/08-case-studies/04-netflix' },
      { text: 'Slack', link: '/08-case-studies/05-slack' },
      { text: 'Stripe', link: '/08-case-studies/06-stripe' },
      { text: 'Dropbox', link: '/08-case-studies/07-dropbox' },
      { text: 'Discord', link: '/08-case-studies/08-discord' },
      { text: 'Google Maps', link: '/08-case-studies/09-google-maps' },
      { text: 'WhatsApp', link: '/08-case-studies/10-whatsapp' },
      { text: 'Figma', link: '/08-case-studies/11-figma' },
      { text: 'Cloudflare', link: '/08-case-studies/12-cloudflare' },
      { text: 'LLM Inference Platforms', link: '/08-case-studies/13-llm-inference-platforms' },
    ]
  },
  {
    text: '9. Whitepapers',
    collapsed: true,
    items: [
      { text: 'MapReduce', link: '/09-whitepapers/01-mapreduce' },
      { text: 'Dynamo', link: '/09-whitepapers/02-dynamo' },
      { text: 'Bigtable', link: '/09-whitepapers/03-bigtable' },
      { text: 'Spanner', link: '/09-whitepapers/04-spanner' },
      { text: 'TAO', link: '/09-whitepapers/05-tao' },
      { text: 'Kafka', link: '/09-whitepapers/06-kafka' },
      { text: 'Raft', link: '/09-whitepapers/07-raft' },
      { text: 'Chubby', link: '/09-whitepapers/08-chubby' },
      { text: 'Aurora', link: '/09-whitepapers/09-aurora' },
      { text: 'CockroachDB', link: '/09-whitepapers/10-cockroachdb' },
      { text: 'Zanzibar', link: '/09-whitepapers/11-zanzibar' },
      { text: 'Monarch', link: '/09-whitepapers/12-monarch' },
      { text: 'FoundationDB', link: '/09-whitepapers/13-foundationdb' },
      { text: 'DynamoDB (2022)', link: '/09-whitepapers/14-dynamodb-2022' },
      { text: 'The Transformer', link: '/09-whitepapers/15-attention-transformers' },
    ]
  },
  {
    text: '10. Security',
    collapsed: true,
    items: [
      { text: 'Authentication Systems', link: '/10-security/01-authentication-fundamentals' },
      { text: 'OAuth 2.0 and OpenID Connect', link: '/10-security/02-oauth2-openid-connect' },
      { text: 'JOSE and JWT Verification', link: '/10-security/03-jwt-tokens' },
      { text: 'API Threat Boundaries', link: '/10-security/04-api-security' },
      { text: 'Zero-Trust Workload Architecture', link: '/10-security/05-zero-trust-architecture' },
      { text: 'Cryptographic Key Architecture', link: '/10-security/06-encryption' },
      { text: 'Authorization at Scale', link: '/10-security/07-authorization-patterns' },
    ]
  },
  {
    text: '11. Observability and Operations',
    collapsed: true,
    items: [
      { text: 'Tracing & Telemetry Pipelines', link: '/11-observability/01-distributed-tracing' },
      { text: 'Metrics Systems & Monitoring', link: '/11-observability/02-metrics-monitoring' },
      { text: 'Production Logging', link: '/11-observability/03-logging' },
      { text: 'Alert Evaluation & Notification', link: '/11-observability/04-alerting' },
      { text: 'SLOs & Error-Budget Control', link: '/11-observability/05-slos-error-budgets' },
      { text: 'FinOps & Cost Engineering', link: '/11-observability/06-finops-cost-engineering' },
      { text: 'Incident Command & Learning', link: '/11-observability/07-incident-management' },
    ]
  },
  {
    text: '12. Service Connectivity and APIs',
    collapsed: true,
    items: [
      { text: 'Discovery & Control-Plane State', link: '/12-service-mesh/01-service-discovery' },
      { text: 'Edge Gateway & API Mediation', link: '/12-service-mesh/02-api-gateway' },
      { text: 'Mesh Data & Control Planes', link: '/12-service-mesh/03-sidecar-pattern' },
      { text: 'API Design & Evolution', link: '/12-service-mesh/04-api-design-patterns' },
    ]
  },
  {
    text: '13. Data Pipelines',
    collapsed: true,
    items: [
      { text: 'Batch Execution', link: '/13-data-pipelines/01-batch-processing' },
      { text: 'Stream Execution', link: '/13-data-pipelines/02-stream-processing' },
      { text: 'CDC: Snapshot, Tail & Repair', link: '/13-data-pipelines/04-change-data-capture' },
      { text: 'Lakehouse Table Formats', link: '/13-data-pipelines/05-lakehouse-table-formats' },
    ]
  },
  {
    text: '14. Search Systems',
    collapsed: true,
    items: [
      { text: 'Index Architecture & Internals', link: '/14-search-systems/01-inverted-indexes' },
      { text: 'Lexical Query Execution', link: '/14-search-systems/02-full-text-search' },
      { text: 'Vector Retrieval Systems', link: '/14-search-systems/03-vector-search' },
      { text: 'Ranking & Evaluation', link: '/14-search-systems/04-ranking-algorithms' },
      { text: 'Typeahead & Autocomplete', link: '/14-search-systems/06-typeahead-autocomplete' },
    ]
  },
  {
    text: '15. Deployment',
    collapsed: true,
    items: [
      { text: 'Progressive Delivery', link: '/15-deployment/01-deployment-strategies' },
      { text: 'Feature-Flag Control Planes', link: '/15-deployment/02-feature-flags' },
      { text: 'Database Migrations', link: '/15-deployment/03-database-migrations' },
      { text: 'Delivery Control Planes & GitOps', link: '/15-deployment/04-cicd-gitops' },
      { text: 'Disaster Recovery & Reconstruction', link: '/15-deployment/05-disaster-recovery' },
      { text: 'Service & Platform Migration', link: '/15-deployment/06-migration-strategies' },
    ]
  },
  {
    text: '16. ML Systems',
    collapsed: true,
    items: [
      { text: 'ML System Fundamentals', link: '/16-ml-systems/01-ml-system-fundamentals' },
      { text: 'Feature Stores', link: '/16-ml-systems/02-feature-stores' },
      { text: 'Model Serving', link: '/16-ml-systems/03-model-serving' },
      { text: 'Model Monitoring', link: '/16-ml-systems/04-model-monitoring' },
      { text: 'Training Pipelines', link: '/16-ml-systems/05-training-pipelines' },
      { text: 'Model Deployment & Rollouts', link: '/16-ml-systems/06-model-deployment-rollouts' },
      { text: 'Recommendation Systems', link: '/16-ml-systems/07-recommendation-systems' },
      { text: 'Online Experiments', link: '/16-ml-systems/08-online-experiments' },
      { text: 'ML Risk & Governance', link: '/16-ml-systems/09-ml-risk-governance' },
      { text: 'Label & Ground-Truth Systems', link: '/16-ml-systems/10-label-ground-truth-systems' },
      { text: 'Dataset Management & Versioning', link: '/16-ml-systems/11-dataset-management-versioning' },
      { text: 'Offline Evaluation & Metrics', link: '/16-ml-systems/12-offline-evaluation-metrics' },
      { text: 'Model Registry & Metadata', link: '/16-ml-systems/13-model-registry-metadata' },
      { text: 'ML Capacity & Cost Planning', link: '/16-ml-systems/14-ml-capacity-cost-planning' },
      { text: 'Distributed Training Internals', link: '/16-ml-systems/15-distributed-training-internals' },
    ]
  },
  {
    text: '17. LLM Systems',
    collapsed: true,
    items: [
      { text: 'Agent Fundamentals', link: '/17-llm-systems/01-agent-fundamentals' },
      { text: 'Orchestration Patterns', link: '/17-llm-systems/02-orchestration-patterns' },
      { text: 'Multi-Agent Systems', link: '/17-llm-systems/03-multi-agent-systems' },
      { text: 'RAG Patterns', link: '/17-llm-systems/04-rag-patterns' },
      { text: 'LLM Infrastructure', link: '/17-llm-systems/05-llm-infrastructure' },
      { text: 'Prompt Engineering', link: '/17-llm-systems/06-prompt-engineering' },
      { text: 'Fine-Tuning Patterns', link: '/17-llm-systems/07-fine-tuning-patterns' },
      { text: 'Context Management', link: '/17-llm-systems/08-context-management' },
      { text: 'Harness Engineering', link: '/17-llm-systems/09-harness-engineering' },
      { text: 'LLM Evaluation', link: '/17-llm-systems/10-llm-evaluation' },
      { text: 'GPU Inference Internals', link: '/17-llm-systems/11-gpu-inference-internals' },
      { text: 'Agent Inference', link: '/17-llm-systems/12-agent-inference' },
    ]
  },
  {
    text: '18. Workflow & Job Systems',
    collapsed: true,
    items: [
      { text: 'Workflow Fundamentals', link: '/18-workflow-job-systems/01-workflow-system-fundamentals' },
      { text: 'Background Jobs & Workers', link: '/18-workflow-job-systems/02-background-jobs-worker-pools' },
      { text: 'Distributed Scheduling & Timers', link: '/18-workflow-job-systems/03-distributed-cron-scheduling' },
      { text: 'Durable Execution', link: '/18-workflow-job-systems/04-durable-execution-workflow-engines' },
      { text: 'DAG Orchestration', link: '/18-workflow-job-systems/05-dag-orchestration' },
      { text: 'Effect Commit Protocols', link: '/18-workflow-job-systems/06-retry-idempotency-compensation' },
      { text: 'Priority, Fairness & Backpressure', link: '/18-workflow-job-systems/07-priority-fairness-backpressure' },
      { text: 'Leases, Heartbeats & Recovery', link: '/18-workflow-job-systems/08-leases-heartbeats-recovery' },
      { text: 'Observability & Replay', link: '/18-workflow-job-systems/09-workflow-observability-replay' },
    ]
  },
  {
    text: '19. Engineering Systems for Coding Agents',
    collapsed: true,
    items: [
      { text: 'Platform Fundamentals', link: '/19-compound-engineering/01-compound-engineering-fundamentals' },
      { text: 'Tool & Runtime Contracts', link: '/19-compound-engineering/02-coding-agent-tool-design' },
      { text: 'Context & Policy Plane', link: '/19-compound-engineering/03-agent-context-engineering' },
      { text: 'Repository Architecture', link: '/19-compound-engineering/04-ai-native-software-architecture' },
      { text: 'Verification & Governance', link: '/19-compound-engineering/05-quality-engineering-with-ai-agents' },
      { text: 'Parallel Development', link: '/19-compound-engineering/06-compound-development-workflows' },
    ]
  },
]

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
          { text: 'Read', link: '/01-foundations/01-acid-transactions' },
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
          { text: '読む', link: '/ja/01-foundations/01-acid-transactions' },
          { text: 'PDF / EPUB', link: 'https://github.com/babushkai/system-design-patterns/releases/latest' },
        ],
        sidebar: {
          '/ja/': [
            {
              text: '1. 基礎',
              collapsed: false,
              items: [
                { text: 'ACIDトランザクション', link: '/ja/01-foundations/01-acid-transactions' },
                { text: '分離レベル', link: '/ja/01-foundations/02-isolation-levels' },
                { text: 'CAP定理', link: '/ja/01-foundations/03-cap-theorem' },
                { text: '整合性モデル', link: '/ja/01-foundations/04-consistency-models' },
                { text: '分散時間', link: '/ja/01-foundations/05-distributed-time' },
                { text: '障害セマンティクス・検知・復旧', link: '/ja/01-foundations/06-failure-modes' },
                { text: '冪等性', link: '/ja/01-foundations/08-idempotency' },
                { text: '分散ロック', link: '/ja/01-foundations/09-distributed-locks' },
                { text: 'キャパシティプランニング', link: '/ja/01-foundations/10-capacity-planning' },
              ]
            },
            {
              text: '2. 分散データベース',
              collapsed: true,
              items: [
                { text: 'シングルリーダーレプリケーション', link: '/ja/02-distributed-databases/01-single-leader-replication' },
                { text: 'マルチリーダーレプリケーション', link: '/ja/02-distributed-databases/02-multi-leader-replication' },
                { text: 'リーダーレスレプリケーション', link: '/ja/02-distributed-databases/03-leaderless-replication' },
                { text: 'コンフリクト解決', link: '/ja/02-distributed-databases/04-conflict-resolution' },
                { text: 'パーティショニング戦略', link: '/ja/02-distributed-databases/05-partitioning-strategies' },
                { text: 'セカンダリインデックス', link: '/ja/02-distributed-databases/06-secondary-indexes' },
                { text: '分散トランザクション', link: '/ja/02-distributed-databases/07-distributed-transactions' },
                { text: 'コンセンサスアルゴリズム', link: '/ja/02-distributed-databases/08-consensus-algorithms' },
                { text: 'リーダー選出', link: '/ja/02-distributed-databases/09-leader-election' },
                { text: 'データモデリング', link: '/ja/02-distributed-databases/10-data-modeling' },
              ]
            },
            {
              text: '3. ストレージエンジン',
              collapsed: true,
              items: [
                { text: 'B木', link: '/ja/03-storage-engines/01-b-trees' },
                { text: 'LSM木', link: '/ja/03-storage-engines/02-lsm-trees' },
                { text: 'SSTableとコンパクション', link: '/ja/03-storage-engines/03-sstables-compaction' },
                { text: '先行書き込みログ', link: '/ja/03-storage-engines/04-write-ahead-logging' },
                { text: 'ブルームフィルタ', link: '/ja/03-storage-engines/05-bloom-filters' },
                { text: 'カラムストレージ', link: '/ja/03-storage-engines/06-column-storage' },
                { text: 'データエンコーディング', link: '/ja/03-storage-engines/07-data-encoding' },
                { text: 'オブジェクトストレージ', link: '/ja/03-storage-engines/08-object-storage' },
              ]
            },
            {
              text: '4. キャッシング',
              collapsed: true,
              items: [
                { text: 'キャッシュのセマンティクスと経済性', link: '/ja/04-caching/01-cache-strategies' },
                { text: '無効化とコヒーレンス', link: '/ja/04-caching/02-cache-invalidation' },
                { text: '分散キャッシュの内部構造', link: '/ja/04-caching/03-distributed-caching' },
                { text: 'スタンピード、コールドスタート、ウォーミング', link: '/ja/04-caching/04-cache-stampede' },
              ]
            },
            {
              text: '5. メッセージング',
              collapsed: true,
              items: [
                { text: 'メッセージキュー', link: '/ja/05-messaging/01-message-queues' },
                { text: 'Pub/Sub', link: '/ja/05-messaging/02-pub-sub' },
                { text: 'メッセージ順序', link: '/ja/05-messaging/03-message-ordering' },
                { text: '配信保証', link: '/ja/05-messaging/04-delivery-guarantees' },
                { text: 'イベントソーシング', link: '/ja/05-messaging/05-event-sourcing' },
                { text: 'CQRS', link: '/ja/05-messaging/06-cqrs' },
                { text: 'Outboxパターン', link: '/ja/05-messaging/07-outbox-pattern' },
                { text: 'デッドレターキュー', link: '/ja/05-messaging/08-dead-letter-queues' },
              ]
            },
            {
              text: '6. スケーリング',
              collapsed: true,
              items: [
                { text: 'ロードバランシング', link: '/ja/06-scaling/01-load-balancing' },
                { text: '水平vs垂直スケーリング', link: '/ja/06-scaling/02-horizontal-vertical' },
                { text: 'データベースシャーディング', link: '/ja/06-scaling/03-database-sharding' },
                { text: 'CDNアーキテクチャ', link: '/ja/06-scaling/04-cdn-architecture' },
                { text: 'レート制限', link: '/ja/06-scaling/05-rate-limiting' },
                { text: 'サーキットブレーカー', link: '/ja/06-scaling/06-circuit-breakers' },
                { text: 'バックプレッシャー', link: '/ja/06-scaling/07-backpressure' },
                { text: 'オートスケーリング', link: '/ja/06-scaling/08-auto-scaling' },
                { text: 'マルチリージョン', link: '/ja/06-scaling/09-multi-region-architecture' },
                { text: 'リトライとヘッジング', link: '/ja/06-scaling/10-retries-timeouts-hedging' },
                { text: 'セルベースアーキテクチャ', link: '/ja/06-scaling/11-cell-based-architecture' },
                { text: 'マルチテナンシー', link: '/ja/06-scaling/12-multi-tenancy' },
                { text: 'DNSとコネクション管理', link: '/ja/06-scaling/13-dns-and-connection-management' },
                { text: 'ネットワークトランスポートの内部構造', link: '/ja/06-scaling/14-network-transport-internals' },
              ]
            },
            {
              text: '7. リアルタイム',
              collapsed: true,
              items: [
                { text: 'クライアント配信トランスポート', link: '/ja/07-real-time/01-polling' },
                { text: 'WebRTC', link: '/ja/07-real-time/05-webrtc' },
                { text: 'プレゼンス', link: '/ja/07-real-time/06-presence' },
                { text: 'CRDTと共同編集', link: '/ja/07-real-time/07-crdts-collaborative-editing' },
              ]
            },
            {
              text: '8. ケーススタディ',
              collapsed: true,
              items: [
                { text: 'Twitter', link: '/ja/08-case-studies/01-twitter' },
                { text: 'Instagram', link: '/ja/08-case-studies/02-instagram' },
                { text: 'Uber', link: '/ja/08-case-studies/03-uber' },
                { text: 'Netflix', link: '/ja/08-case-studies/04-netflix' },
                { text: 'Slack', link: '/ja/08-case-studies/05-slack' },
                { text: 'Stripe', link: '/ja/08-case-studies/06-stripe' },
                { text: 'Dropbox', link: '/ja/08-case-studies/07-dropbox' },
                { text: 'Discord', link: '/ja/08-case-studies/08-discord' },
                { text: 'Google Maps', link: '/ja/08-case-studies/09-google-maps' },
                { text: 'WhatsApp', link: '/ja/08-case-studies/10-whatsapp' },
                { text: 'Figma', link: '/ja/08-case-studies/11-figma' },
                { text: 'Cloudflare', link: '/ja/08-case-studies/12-cloudflare' },
                { text: 'LLM推論基盤', link: '/ja/08-case-studies/13-llm-inference-platforms' },
              ]
            },
            {
              text: '9. ホワイトペーパー',
              collapsed: true,
              items: [
                { text: 'MapReduce', link: '/ja/09-whitepapers/01-mapreduce' },
                { text: 'Dynamo', link: '/ja/09-whitepapers/02-dynamo' },
                { text: 'Bigtable', link: '/ja/09-whitepapers/03-bigtable' },
                { text: 'Spanner', link: '/ja/09-whitepapers/04-spanner' },
                { text: 'TAO', link: '/ja/09-whitepapers/05-tao' },
                { text: 'Kafka', link: '/ja/09-whitepapers/06-kafka' },
                { text: 'Raft', link: '/ja/09-whitepapers/07-raft' },
                { text: 'Chubby', link: '/ja/09-whitepapers/08-chubby' },
                { text: 'Aurora', link: '/ja/09-whitepapers/09-aurora' },
                { text: 'CockroachDB', link: '/ja/09-whitepapers/10-cockroachdb' },
                { text: 'Zanzibar', link: '/ja/09-whitepapers/11-zanzibar' },
                { text: 'Monarch', link: '/ja/09-whitepapers/12-monarch' },
                { text: 'FoundationDB', link: '/ja/09-whitepapers/13-foundationdb' },
                { text: 'DynamoDB (2022)', link: '/ja/09-whitepapers/14-dynamodb-2022' },
                { text: 'Transformer', link: '/ja/09-whitepapers/15-attention-transformers' },
              ]
            },
            {
              text: '10. セキュリティ',
              collapsed: true,
              items: [
                { text: '認証の基礎', link: '/ja/10-security/01-authentication-fundamentals' },
                { text: 'OAuth2とOpenID Connect', link: '/ja/10-security/02-oauth2-openid-connect' },
                { text: 'JWTトークン', link: '/ja/10-security/03-jwt-tokens' },
                { text: 'APIセキュリティ', link: '/ja/10-security/04-api-security' },
                { text: 'ゼロトラストアーキテクチャ', link: '/ja/10-security/05-zero-trust-architecture' },
                { text: '暗号化', link: '/ja/10-security/06-encryption' },
                { text: '認可パターン', link: '/ja/10-security/07-authorization-patterns' },
              ]
            },
            {
              text: '11. オブザーバビリティ',
              collapsed: true,
              items: [
                { text: '分散トレーシング', link: '/ja/11-observability/01-distributed-tracing' },
                { text: 'メトリクスとモニタリング', link: '/ja/11-observability/02-metrics-monitoring' },
                { text: 'ロギング', link: '/ja/11-observability/03-logging' },
                { text: 'アラート', link: '/ja/11-observability/04-alerting' },
                { text: 'SLOとエラーバジェット', link: '/ja/11-observability/05-slos-error-budgets' },
                { text: 'FinOpsとコスト工学', link: '/ja/11-observability/06-finops-cost-engineering' },
                { text: 'インシデント管理', link: '/ja/11-observability/07-incident-management' },
              ]
            },
            {
              text: '12. サービスメッシュ',
              collapsed: true,
              items: [
                { text: 'サービスディスカバリ', link: '/ja/12-service-mesh/01-service-discovery' },
                { text: 'APIゲートウェイ', link: '/ja/12-service-mesh/02-api-gateway' },
                { text: 'サイドカーパターン', link: '/ja/12-service-mesh/03-sidecar-pattern' },
                { text: 'API設計パターン', link: '/ja/12-service-mesh/04-api-design-patterns' },
              ]
            },
            {
              text: '13. データパイプライン',
              collapsed: true,
              items: [
                { text: 'バッチ処理', link: '/ja/13-data-pipelines/01-batch-processing' },
                { text: 'ストリーム処理', link: '/ja/13-data-pipelines/02-stream-processing' },
                { text: 'チェンジデータキャプチャ', link: '/ja/13-data-pipelines/04-change-data-capture' },
                { text: 'レイクハウス', link: '/ja/13-data-pipelines/05-lakehouse-table-formats' },
              ]
            },
            {
              text: '14. 検索システム',
              collapsed: true,
              items: [
                { text: '検索インデックスのアーキテクチャ', link: '/ja/14-search-systems/01-inverted-indexes' },
                { text: '字句クエリ実行', link: '/ja/14-search-systems/02-full-text-search' },
                { text: 'ベクトル検索システム', link: '/ja/14-search-systems/03-vector-search' },
                { text: 'ランキングと評価', link: '/ja/14-search-systems/04-ranking-algorithms' },
                { text: 'タイプアヘッド', link: '/ja/14-search-systems/06-typeahead-autocomplete' },
              ]
            },
            {
              text: '15. デプロイメント',
              collapsed: true,
              items: [
                { text: 'デプロイメント戦略', link: '/ja/15-deployment/01-deployment-strategies' },
                { text: 'フィーチャーフラグ', link: '/ja/15-deployment/02-feature-flags' },
                { text: 'DBマイグレーション', link: '/ja/15-deployment/03-database-migrations' },
                { text: 'CI/CDとGitOps', link: '/ja/15-deployment/04-cicd-gitops' },
                { text: 'ディザスタリカバリ', link: '/ja/15-deployment/05-disaster-recovery' },
                { text: 'マイグレーション戦略', link: '/ja/15-deployment/06-migration-strategies' },
              ]
            },
            {
              text: '16. MLシステム',
              collapsed: true,
              items: [
                { text: 'MLシステム基礎', link: '/ja/16-ml-systems/01-ml-system-fundamentals' },
                { text: 'フィーチャーストア', link: '/ja/16-ml-systems/02-feature-stores' },
                { text: 'モデルサービング', link: '/ja/16-ml-systems/03-model-serving' },
                { text: 'モデルモニタリング', link: '/ja/16-ml-systems/04-model-monitoring' },
                { text: 'トレーニングパイプライン', link: '/ja/16-ml-systems/05-training-pipelines' },
                { text: 'モデルデプロイとロールアウト', link: '/ja/16-ml-systems/06-model-deployment-rollouts' },
                { text: '推薦システム', link: '/ja/16-ml-systems/07-recommendation-systems' },
                { text: 'オンライン実験', link: '/ja/16-ml-systems/08-online-experiments' },
                { text: 'MLリスクとガバナンス', link: '/ja/16-ml-systems/09-ml-risk-governance' },
                { text: 'ラベルとグラウンドトゥルース', link: '/ja/16-ml-systems/10-label-ground-truth-systems' },
                { text: 'データセット管理とバージョニング', link: '/ja/16-ml-systems/11-dataset-management-versioning' },
                { text: 'オフライン評価とメトリクス設計', link: '/ja/16-ml-systems/12-offline-evaluation-metrics' },
                { text: 'モデルレジストリとMLメタデータ', link: '/ja/16-ml-systems/13-model-registry-metadata' },
                { text: 'MLキャパシティとコストプランニング', link: '/ja/16-ml-systems/14-ml-capacity-cost-planning' },
                { text: '分散学習の内部構造', link: '/ja/16-ml-systems/15-distributed-training-internals' },
              ]
            },
            {
              text: '17. LLMシステム',
              collapsed: true,
              items: [
                { text: 'エージェント基礎', link: '/ja/17-llm-systems/01-agent-fundamentals' },
                { text: 'オーケストレーション', link: '/ja/17-llm-systems/02-orchestration-patterns' },
                { text: 'マルチエージェント', link: '/ja/17-llm-systems/03-multi-agent-systems' },
                { text: 'RAGパターン', link: '/ja/17-llm-systems/04-rag-patterns' },
                { text: 'LLMインフラ', link: '/ja/17-llm-systems/05-llm-infrastructure' },
                { text: 'プロンプトエンジニアリング', link: '/ja/17-llm-systems/06-prompt-engineering' },
                { text: 'ファインチューニング', link: '/ja/17-llm-systems/07-fine-tuning-patterns' },
                { text: 'コンテキスト管理', link: '/ja/17-llm-systems/08-context-management' },
                { text: 'ハーネスエンジニアリング', link: '/ja/17-llm-systems/09-harness-engineering' },
                { text: 'LLM評価', link: '/ja/17-llm-systems/10-llm-evaluation' },
                { text: 'GPU推論の内部構造', link: '/ja/17-llm-systems/11-gpu-inference-internals' },
                { text: 'エージェント推論', link: '/ja/17-llm-systems/12-agent-inference' },
              ]
            },
            {
              text: '18. ワークフローとジョブシステム',
              collapsed: true,
              items: [
                { text: 'ワークフローシステム基礎', link: '/ja/18-workflow-job-systems/01-workflow-system-fundamentals' },
                { text: 'バックグラウンドジョブとワーカー', link: '/ja/18-workflow-job-systems/02-background-jobs-worker-pools' },
                { text: '分散cronとスケジューリング', link: '/ja/18-workflow-job-systems/03-distributed-cron-scheduling' },
                { text: 'Durable Execution', link: '/ja/18-workflow-job-systems/04-durable-execution-workflow-engines' },
                { text: 'DAGオーケストレーション', link: '/ja/18-workflow-job-systems/05-dag-orchestration' },
                { text: 'リトライ・冪等性・補償', link: '/ja/18-workflow-job-systems/06-retry-idempotency-compensation' },
                { text: '優先度・公平性・Backpressure', link: '/ja/18-workflow-job-systems/07-priority-fairness-backpressure' },
                { text: 'リース・Heartbeat・復旧', link: '/ja/18-workflow-job-systems/08-leases-heartbeats-recovery' },
                { text: '観測性とリプレイ', link: '/ja/18-workflow-job-systems/09-workflow-observability-replay' },
              ]
            },
            {
              text: '19. コンパウンドエンジニアリング',
              collapsed: true,
              items: [
                { text: '基礎', link: '/ja/19-compound-engineering/01-compound-engineering-fundamentals' },
                { text: 'コーディングエージェントツール設計', link: '/ja/19-compound-engineering/02-coding-agent-tool-design' },
                { text: 'エージェントコンテキストエンジニアリング', link: '/ja/19-compound-engineering/03-agent-context-engineering' },
                { text: 'AIネイティブアーキテクチャ', link: '/ja/19-compound-engineering/04-ai-native-software-architecture' },
                { text: 'AIエージェント品質エンジニアリング', link: '/ja/19-compound-engineering/05-quality-engineering-with-ai-agents' },
                { text: 'コンパウンド開発ワークフロー', link: '/ja/19-compound-engineering/06-compound-development-workflows' },
              ]
            }
          ]
        },
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
