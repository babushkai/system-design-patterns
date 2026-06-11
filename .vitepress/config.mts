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
    ]
  },
  {
    text: '12. Service Mesh',
    collapsed: true,
    items: [
      { text: 'Service Discovery', link: '/12-service-mesh/01-service-discovery' },
      { text: 'API Gateway', link: '/12-service-mesh/02-api-gateway' },
      { text: 'Sidecar Pattern', link: '/12-service-mesh/03-sidecar-pattern' },
    ]
  },
  {
    text: '13. Data Pipelines',
    collapsed: true,
    items: [
      { text: 'Batch Processing', link: '/13-data-pipelines/01-batch-processing' },
      { text: 'Stream Processing', link: '/13-data-pipelines/02-stream-processing' },
      { text: 'Lambda & Kappa Architecture', link: '/13-data-pipelines/03-lambda-kappa-architecture' },
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
    ]
  },
  {
    text: '16. LLM Systems',
    collapsed: true,
    items: [
      { text: 'Agent Fundamentals', link: '/16-llm-systems/01-agent-fundamentals' },
      { text: 'Orchestration Patterns', link: '/16-llm-systems/02-orchestration-patterns' },
      { text: 'Multi-Agent Systems', link: '/16-llm-systems/03-multi-agent-systems' },
      { text: 'RAG Patterns', link: '/16-llm-systems/04-rag-patterns' },
      { text: 'LLM Infrastructure', link: '/16-llm-systems/05-llm-infrastructure' },
      { text: 'Prompt Engineering', link: '/16-llm-systems/06-prompt-engineering' },
      { text: 'Fine-Tuning Patterns', link: '/16-llm-systems/07-fine-tuning-patterns' },
      { text: 'Context Management', link: '/16-llm-systems/08-context-management' },
      { text: 'Harness Engineering', link: '/16-llm-systems/09-harness-engineering' },
    ]
  },
  {
    text: '17. GraphQL',
    collapsed: true,
    items: [
      { text: 'GraphQL Fundamentals', link: '/17-graphql/01-graphql-fundamentals' },
      { text: 'Schema Design', link: '/17-graphql/02-schema-design' },
      { text: 'Resolvers & Data Fetching', link: '/17-graphql/03-resolvers-data-fetching' },
      { text: 'Caching & Performance', link: '/17-graphql/04-caching-performance' },
      { text: 'Subscriptions & Real-Time', link: '/17-graphql/05-subscriptions-realtime' },
      { text: 'Federation', link: '/17-graphql/06-federation' },
    ]
  },
  {
    text: '18. Compound Engineering',
    collapsed: true,
    items: [
      { text: 'Fundamentals', link: '/18-compound-engineering/01-compound-engineering-fundamentals' },
      { text: 'Coding Agent Tool Design', link: '/18-compound-engineering/02-coding-agent-tool-design' },
      { text: 'Agent Context Engineering', link: '/18-compound-engineering/03-agent-context-engineering' },
      { text: 'AI-Native Architecture', link: '/18-compound-engineering/04-ai-native-software-architecture' },
      { text: 'Quality Engineering', link: '/18-compound-engineering/05-quality-engineering-with-ai-agents' },
      { text: 'Compound Workflows', link: '/18-compound-engineering/06-compound-development-workflows' },
    ]
  },
]

export default withMermaid({
  title: 'System Design Patterns',
  description: 'A comprehensive guide to distributed systems and system design patterns',
  
  // Base URL for the custom domain
  base: '/',
  
  // Ignore dead links in README.md (original repo content)
  ignoreDeadLinks: true,
  
  head: [
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Noto+Sans+JP:wght@400;500;700&display=swap' }],
    ['meta', { name: 'theme-color', content: '#0b1322' }],
    // favicon is served from the custom domain root
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/logo.svg' }],
    ['meta', { name: 'theme-color', content: '#2563EB' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'System Design Patterns' }],
    ['meta', { property: 'og:description', content: 'A comprehensive guide to distributed systems' }],
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
              { text: 'LLM Systems', link: '/16-llm-systems/01-agent-fundamentals' },
              { text: 'GraphQL', link: '/17-graphql/01-graphql-fundamentals' },
              { text: 'Compound Engineering', link: '/18-compound-engineering/01-compound-engineering-fundamentals' },
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
              ]
            },
            {
              text: '12. サービスメッシュ',
              collapsed: true,
              items: [
                { text: 'サービスディスカバリ', link: '/ja/12-service-mesh/01-service-discovery' },
                { text: 'APIゲートウェイ', link: '/ja/12-service-mesh/02-api-gateway' },
                { text: 'サイドカーパターン', link: '/ja/12-service-mesh/03-sidecar-pattern' },
              ]
            },
            {
              text: '13. データパイプライン',
              collapsed: true,
              items: [
                { text: 'バッチ処理', link: '/ja/13-data-pipelines/01-batch-processing' },
                { text: 'ストリーム処理', link: '/ja/13-data-pipelines/02-stream-processing' },
                { text: 'Lambda/Kappaアーキテクチャ', link: '/ja/13-data-pipelines/03-lambda-kappa-architecture' },
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
              ]
            },
            {
              text: '16. LLMシステム',
              collapsed: true,
              items: [
                { text: 'エージェント基礎', link: '/ja/16-llm-systems/01-agent-fundamentals' },
                { text: 'オーケストレーション', link: '/ja/16-llm-systems/02-orchestration-patterns' },
                { text: 'マルチエージェント', link: '/ja/16-llm-systems/03-multi-agent-systems' },
                { text: 'RAGパターン', link: '/ja/16-llm-systems/04-rag-patterns' },
                { text: 'LLMインフラ', link: '/ja/16-llm-systems/05-llm-infrastructure' },
                { text: 'プロンプトエンジニアリング', link: '/ja/16-llm-systems/06-prompt-engineering' },
                { text: 'ファインチューニング', link: '/ja/16-llm-systems/07-fine-tuning-patterns' },
                { text: 'コンテキスト管理', link: '/ja/16-llm-systems/08-context-management' },
                { text: 'ハーネスエンジニアリング', link: '/ja/16-llm-systems/09-harness-engineering' },
              ]
            },
            {
              text: '17. GraphQL',
              collapsed: true,
              items: [
                { text: 'GraphQL基礎', link: '/ja/17-graphql/01-graphql-fundamentals' },
                { text: 'スキーマ設計', link: '/ja/17-graphql/02-schema-design' },
                { text: 'リゾルバ', link: '/ja/17-graphql/03-resolvers-data-fetching' },
                { text: 'キャッシュとパフォーマンス', link: '/ja/17-graphql/04-caching-performance' },
                { text: 'サブスクリプション', link: '/ja/17-graphql/05-subscriptions-realtime' },
                { text: 'フェデレーション', link: '/ja/17-graphql/06-federation' },
              ]
            },
            {
              text: '18. コンパウンドエンジニアリング',
              collapsed: true,
              items: [
                { text: '基礎', link: '/ja/18-compound-engineering/01-compound-engineering-fundamentals' },
                { text: 'コーディングエージェントツール設計', link: '/ja/18-compound-engineering/02-coding-agent-tool-design' },
                { text: 'エージェントコンテキストエンジニアリング', link: '/ja/18-compound-engineering/03-agent-context-engineering' },
                { text: 'AIネイティブアーキテクチャ', link: '/ja/18-compound-engineering/04-ai-native-software-architecture' },
                { text: 'AIエージェント品質エンジニアリング', link: '/ja/18-compound-engineering/05-quality-engineering-with-ai-agents' },
                { text: 'コンパウンド開発ワークフロー', link: '/ja/18-compound-engineering/06-compound-development-workflows' },
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
    lineNumbers: true,
    theme: {
      light: 'github-light',
      dark: 'github-dark-dimmed'
    }
  },

  appearance: 'dark',
  lastUpdated: true,
  cleanUrls: true,
mermaid: {
  theme: 'base',
  themeVariables: {
    // diagrams render on light "schematic paper" cards in both modes
    primaryColor: '#0E7490',
    primaryTextColor: '#F5FBFF',
    primaryBorderColor: '#155E75',
    lineColor: '#46587A',
    secondaryColor: '#DBEDFB',
    tertiaryColor: '#EDF2F9',
    fontFamily: "'JetBrains Mono', monospace",
  },
},
mermaidPlugin: {
  class: 'mermaid',
},
})
