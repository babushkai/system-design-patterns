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
      { text: 'Failure Modes', link: '/01-foundations/06-failure-modes' },
      { text: 'Network Partitions', link: '/01-foundations/07-network-partitions' },
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
      { text: 'Data Modeling', link: '/02-distributed-databases/10-data-modeling' },
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
      { text: 'Object Storage', link: '/03-storage-engines/08-object-storage' },
    ]
  },
  {
    text: '4. Caching',
    collapsed: true,
    items: [
      { text: 'Cache Strategies', link: '/04-caching/01-cache-strategies' },
      { text: 'Cache Invalidation', link: '/04-caching/02-cache-invalidation' },
      { text: 'Distributed Caching', link: '/04-caching/03-distributed-caching' },
      { text: 'Cache Stampede', link: '/04-caching/04-cache-stampede' },
      { text: 'Multi-Tier Caching', link: '/04-caching/05-multi-tier-caching' },
      { text: 'Cache Warming', link: '/04-caching/06-cache-warming' },
    ]
  },
  {
    text: '5. Messaging',
    collapsed: true,
    items: [
      { text: 'Message Queues', link: '/05-messaging/01-message-queues' },
      { text: 'Pub/Sub Systems', link: '/05-messaging/02-pub-sub' },
      { text: 'Message Ordering', link: '/05-messaging/03-message-ordering' },
      { text: 'Delivery Guarantees', link: '/05-messaging/04-delivery-guarantees' },
      { text: 'Event Sourcing', link: '/05-messaging/05-event-sourcing' },
      { text: 'CQRS', link: '/05-messaging/06-cqrs' },
      { text: 'Outbox Pattern', link: '/05-messaging/07-outbox-pattern' },
      { text: 'Dead Letter Queues', link: '/05-messaging/08-dead-letter-queues' },
      { text: 'Saga Pattern', link: '/05-messaging/09-saga-pattern' },
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
      { text: 'Multi-Tenancy', link: '/06-scaling/12-multi-tenancy' },
      { text: 'DNS & Connections', link: '/06-scaling/13-dns-and-connection-management' },
      { text: 'Network Transport Internals', link: '/06-scaling/14-network-transport-internals' },
    ]
  },
  {
    text: '7. Real-Time',
    collapsed: true,
    items: [
      { text: 'Polling', link: '/07-real-time/01-polling' },
      { text: 'Long Polling', link: '/07-real-time/02-long-polling' },
      { text: 'Server-Sent Events', link: '/07-real-time/03-server-sent-events' },
      { text: 'WebSockets', link: '/07-real-time/04-websockets' },
      { text: 'WebRTC', link: '/07-real-time/05-webrtc' },
      { text: 'Presence', link: '/07-real-time/06-presence' },
      { text: 'CRDTs & Collaboration', link: '/07-real-time/07-crdts-collaborative-editing' },
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
      { text: 'Authentication Fundamentals', link: '/10-security/01-authentication-fundamentals' },
      { text: 'OAuth2 & OpenID Connect', link: '/10-security/02-oauth2-openid-connect' },
      { text: 'JWT Tokens', link: '/10-security/03-jwt-tokens' },
      { text: 'API Security', link: '/10-security/04-api-security' },
      { text: 'Zero Trust Architecture', link: '/10-security/05-zero-trust-architecture' },
      { text: 'Encryption', link: '/10-security/06-encryption' },
      { text: 'Authorization Patterns', link: '/10-security/07-authorization-patterns' },
    ]
  },
  {
    text: '11. Observability',
    collapsed: true,
    items: [
      { text: 'Distributed Tracing', link: '/11-observability/01-distributed-tracing' },
      { text: 'Metrics & Monitoring', link: '/11-observability/02-metrics-monitoring' },
      { text: 'Logging', link: '/11-observability/03-logging' },
      { text: 'Alerting', link: '/11-observability/04-alerting' },
      { text: 'SLOs & Error Budgets', link: '/11-observability/05-slos-error-budgets' },
      { text: 'FinOps & Cost', link: '/11-observability/06-finops-cost-engineering' },
      { text: 'Incident Management', link: '/11-observability/07-incident-management' },
    ]
  },
  {
    text: '12. Service Mesh',
    collapsed: true,
    items: [
      { text: 'Service Discovery', link: '/12-service-mesh/01-service-discovery' },
      { text: 'API Gateway', link: '/12-service-mesh/02-api-gateway' },
      { text: 'Sidecar Pattern', link: '/12-service-mesh/03-sidecar-pattern' },
      { text: 'API Design Patterns', link: '/12-service-mesh/04-api-design-patterns' },
    ]
  },
  {
    text: '13. Data Pipelines',
    collapsed: true,
    items: [
      { text: 'Batch Processing', link: '/13-data-pipelines/01-batch-processing' },
      { text: 'Stream Processing', link: '/13-data-pipelines/02-stream-processing' },
      { text: 'Lambda & Kappa Architecture', link: '/13-data-pipelines/03-lambda-kappa-architecture' },
      { text: 'Change Data Capture', link: '/13-data-pipelines/04-change-data-capture' },
      { text: 'Lakehouse & Table Formats', link: '/13-data-pipelines/05-lakehouse-table-formats' },
    ]
  },
  {
    text: '14. Search Systems',
    collapsed: true,
    items: [
      { text: 'Inverted Indexes', link: '/14-search-systems/01-inverted-indexes' },
      { text: 'Full-Text Search', link: '/14-search-systems/02-full-text-search' },
      { text: 'Vector Search', link: '/14-search-systems/03-vector-search' },
      { text: 'Ranking Algorithms', link: '/14-search-systems/04-ranking-algorithms' },
      { text: 'Search Relevance Tuning', link: '/14-search-systems/05-search-relevance-tuning' },
      { text: 'Typeahead & Autocomplete', link: '/14-search-systems/06-typeahead-autocomplete' },
    ]
  },
  {
    text: '15. Deployment',
    collapsed: true,
    items: [
      { text: 'Deployment Strategies', link: '/15-deployment/01-deployment-strategies' },
      { text: 'Feature Flags', link: '/15-deployment/02-feature-flags' },
      { text: 'Database Migrations', link: '/15-deployment/03-database-migrations' },
      { text: 'CI/CD & GitOps', link: '/15-deployment/04-cicd-gitops' },
      { text: 'Disaster Recovery', link: '/15-deployment/05-disaster-recovery' },
      { text: 'Migration Strategies', link: '/15-deployment/06-migration-strategies' },
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
      { text: 'Distributed Cron & Scheduling', link: '/18-workflow-job-systems/03-distributed-cron-scheduling' },
      { text: 'Durable Execution', link: '/18-workflow-job-systems/04-durable-execution-workflow-engines' },
      { text: 'DAG Orchestration', link: '/18-workflow-job-systems/05-dag-orchestration' },
      { text: 'Retry, Idempotency & Compensation', link: '/18-workflow-job-systems/06-retry-idempotency-compensation' },
      { text: 'Priority, Fairness & Backpressure', link: '/18-workflow-job-systems/07-priority-fairness-backpressure' },
      { text: 'Leases, Heartbeats & Recovery', link: '/18-workflow-job-systems/08-leases-heartbeats-recovery' },
      { text: 'Observability & Replay', link: '/18-workflow-job-systems/09-workflow-observability-replay' },
    ]
  },
  {
    text: '19. Compound Engineering',
    collapsed: true,
    items: [
      { text: 'Fundamentals', link: '/19-compound-engineering/01-compound-engineering-fundamentals' },
      { text: 'Coding Agent Tool Design', link: '/19-compound-engineering/02-coding-agent-tool-design' },
      { text: 'Agent Context Engineering', link: '/19-compound-engineering/03-agent-context-engineering' },
      { text: 'AI-Native Architecture', link: '/19-compound-engineering/04-ai-native-software-architecture' },
      { text: 'Quality Engineering', link: '/19-compound-engineering/05-quality-engineering-with-ai-agents' },
      { text: 'Compound Workflows', link: '/19-compound-engineering/06-compound-development-workflows' },
    ]
  },
]

export default withMermaid({
  title: 'System Design Patterns',
  description: 'An architecture fieldbook for reliable distributed, data, ML, and AI systems',

  // Base URL for the custom domain
  base: '/',

  // Ignore dead links in README.md (original repo content)
  ignoreDeadLinks: true,

  head: [
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+JP:wght@400;500;700&family=Newsreader:ital,opsz,wght@1,6..72,400..600&display=swap' }],
    // favicon is served from the custom domain root
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/logo.svg' }],
    ['meta', { name: 'theme-color', content: '#0f172a' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'System Design Patterns' }],
    ['meta', { property: 'og:description', content: 'An architecture fieldbook for reliable distributed, data, ML, and AI systems.' }],
    ['meta', { property: 'og:image', content: 'https://design.babushkai.com/logo.svg' }],
    ['meta', { name: 'twitter:card', content: 'summary' }],
  ],

  locales: {
    root: {
      label: 'English',
      lang: 'en',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/' },
          { text: 'Guide', link: '/01-foundations/01-acid-transactions' },
          {
            text: 'Sections',
            items: [
              { text: 'Foundations', link: '/01-foundations/01-acid-transactions' },
              { text: 'Distributed Databases', link: '/02-distributed-databases/01-single-leader-replication' },
              { text: 'Storage Engines', link: '/03-storage-engines/01-b-trees' },
              { text: 'Caching', link: '/04-caching/01-cache-strategies' },
              { text: 'Messaging', link: '/05-messaging/01-message-queues' },
              { text: 'Scaling', link: '/06-scaling/01-load-balancing' },
              { text: 'Real-Time', link: '/07-real-time/01-polling' },
              { text: 'Case Studies', link: '/08-case-studies/01-twitter' },
              { text: 'Whitepapers', link: '/09-whitepapers/01-mapreduce' },
              { text: 'Security', link: '/10-security/01-authentication-fundamentals' },
              { text: 'Observability', link: '/11-observability/01-distributed-tracing' },
              { text: 'Service Mesh', link: '/12-service-mesh/01-service-discovery' },
              { text: 'Data Pipelines', link: '/13-data-pipelines/01-batch-processing' },
              { text: 'Search Systems', link: '/14-search-systems/01-inverted-indexes' },
              { text: 'Deployment', link: '/15-deployment/01-deployment-strategies' },
              { text: 'ML Systems', link: '/16-ml-systems/01-ml-system-fundamentals' },
              { text: 'LLM Systems', link: '/17-llm-systems/01-agent-fundamentals' },
              { text: 'Workflow & Job Systems', link: '/18-workflow-job-systems/01-workflow-system-fundamentals' },
              { text: 'Compound Engineering', link: '/19-compound-engineering/01-compound-engineering-fundamentals' },
            ]
          },
          { text: 'GitHub', link: 'https://github.com/babushkai/system-design-patterns' }
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
          { text: 'ホーム', link: '/ja/' },
          { text: 'ガイド', link: '/ja/01-foundations/01-acid-transactions' },
          { text: 'GitHub', link: 'https://github.com/babushkai/system-design-patterns' }
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
                { text: '障害モード', link: '/ja/01-foundations/06-failure-modes' },
                { text: 'ネットワーク分断', link: '/ja/01-foundations/07-network-partitions' },
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
                { text: 'キャッシュ戦略', link: '/ja/04-caching/01-cache-strategies' },
                { text: 'キャッシュ無効化', link: '/ja/04-caching/02-cache-invalidation' },
                { text: '分散キャッシュ', link: '/ja/04-caching/03-distributed-caching' },
                { text: 'キャッシュスタンピード', link: '/ja/04-caching/04-cache-stampede' },
                { text: 'マルチティアキャッシュ', link: '/ja/04-caching/05-multi-tier-caching' },
                { text: 'キャッシュウォーミング', link: '/ja/04-caching/06-cache-warming' },
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
                { text: 'Sagaパターン', link: '/ja/05-messaging/09-saga-pattern' },
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
                { text: 'ポーリング', link: '/ja/07-real-time/01-polling' },
                { text: 'ロングポーリング', link: '/ja/07-real-time/02-long-polling' },
                { text: 'Server-Sent Events', link: '/ja/07-real-time/03-server-sent-events' },
                { text: 'WebSocket', link: '/ja/07-real-time/04-websockets' },
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
                { text: 'Lambda/Kappaアーキテクチャ', link: '/ja/13-data-pipelines/03-lambda-kappa-architecture' },
                { text: 'チェンジデータキャプチャ', link: '/ja/13-data-pipelines/04-change-data-capture' },
                { text: 'レイクハウス', link: '/ja/13-data-pipelines/05-lakehouse-table-formats' },
              ]
            },
            {
              text: '14. 検索システム',
              collapsed: true,
              items: [
                { text: '転置インデックス', link: '/ja/14-search-systems/01-inverted-indexes' },
                { text: '全文検索', link: '/ja/14-search-systems/02-full-text-search' },
                { text: 'ベクトル検索', link: '/ja/14-search-systems/03-vector-search' },
                { text: 'ランキングアルゴリズム', link: '/ja/14-search-systems/04-ranking-algorithms' },
                { text: '検索関連性チューニング', link: '/ja/14-search-systems/05-search-relevance-tuning' },
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
    logo: '/logo.svg',
    siteTitle: 'System Design Patterns',

    socialLinks: [
      { icon: 'github', link: 'https://github.com/babushkai/system-design-patterns' }
    ],

    search: {
      provider: 'local'
    },

    outline: {
      level: [2, 3],
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
    primaryColor: '#2563EB',
    primaryTextColor: '#ffffff',
    primaryBorderColor: '#1D4ED8',
    lineColor: '#0F766E',
    secondaryColor: '#DBEAFE',
    tertiaryColor: '#FEF3C7',
    fontFamily: 'Inter, sans-serif',
  },
},
mermaidPlugin: {
  class: 'mermaid',
},
})
