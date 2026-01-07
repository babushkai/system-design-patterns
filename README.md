# System Design Patterns

A hyper-detailed, framework-agnostic repository of system design patterns, concepts, and real-world case studies.

> "Most system design resources are unorganized and overly simple. This repository aims to change that."

## Philosophy

1. **Depth over breadth** - Each topic explored to its logical conclusion
2. **Framework-agnostic** - Patterns described independently of technologies  
3. **First-principles thinking** - Derive solutions from constraints
4. **Honest tradeoffs** - Every decision has costs; we make them explicit

## Table of Contents

### Part 1: Foundations
- [ACID Transactions](01-foundations/01-acid-transactions.md)
- [Isolation Levels](01-foundations/02-isolation-levels.md)
- [CAP Theorem](01-foundations/03-cap-theorem.md)
- [Consistency Models](01-foundations/04-consistency-models.md)
- [Distributed Time](01-foundations/05-distributed-time.md)
- [Failure Modes](01-foundations/06-failure-modes.md)
- [Network Partitions](01-foundations/07-network-partitions.md)
- [Idempotency](01-foundations/08-idempotency.md)

### Part 2: Distributed Databases
- [Single-Leader Replication](02-distributed-databases/01-single-leader-replication.md)
- [Multi-Leader Replication](02-distributed-databases/02-multi-leader-replication.md)
- [Leaderless Replication](02-distributed-databases/03-leaderless-replication.md)
- [Conflict Resolution](02-distributed-databases/04-conflict-resolution.md)
- [Partitioning Strategies](02-distributed-databases/05-partitioning-strategies.md)
- [Secondary Indexes](02-distributed-databases/06-secondary-indexes.md)
- [Distributed Transactions](02-distributed-databases/07-distributed-transactions.md)
- [Consensus Algorithms](02-distributed-databases/08-consensus-algorithms.md)
- [Leader Election](02-distributed-databases/09-leader-election.md)

### Part 3: Storage Engines
- [B-Trees](03-storage-engines/01-b-trees.md)
- [LSM Trees](03-storage-engines/02-lsm-trees.md)
- [SSTables and Compaction](03-storage-engines/03-sstables-compaction.md)
- [Write-Ahead Logging](03-storage-engines/04-write-ahead-logging.md)
- [Bloom Filters](03-storage-engines/05-bloom-filters.md)
- [Column-Oriented Storage](03-storage-engines/06-column-storage.md)
- [Data Encoding](03-storage-engines/07-data-encoding.md)

### Part 4: Caching
- [Cache Strategies](04-caching/01-cache-strategies.md)
- [Cache Invalidation](04-caching/02-cache-invalidation.md)
- [Distributed Caching](04-caching/03-distributed-caching.md)
- [Cache Stampede](04-caching/04-cache-stampede.md)
- [Multi-Tier Caching](04-caching/05-multi-tier-caching.md)
- [Cache Warming](04-caching/06-cache-warming.md)

### Part 5: Messaging
- [Message Queues](05-messaging/01-message-queues.md)
- [Pub/Sub Systems](05-messaging/02-pub-sub.md)
- [Message Ordering](05-messaging/03-message-ordering.md)
- [Delivery Guarantees](05-messaging/04-delivery-guarantees.md)
- [Event Sourcing](05-messaging/05-event-sourcing.md)
- [CQRS](05-messaging/06-cqrs.md)
- [Outbox Pattern](05-messaging/07-outbox-pattern.md)
- [Dead Letter Queues](05-messaging/08-dead-letter-queues.md)

### Part 6: Scaling
- [Load Balancing](06-scaling/01-load-balancing.md)
- [Horizontal vs Vertical](06-scaling/02-horizontal-vertical.md)
- [Database Sharding](06-scaling/03-database-sharding.md)
- [CDN Architecture](06-scaling/04-cdn-architecture.md)
- [Rate Limiting](06-scaling/05-rate-limiting.md)
- [Circuit Breakers](06-scaling/06-circuit-breakers.md)
- [Backpressure](06-scaling/07-backpressure.md)
- [Auto-Scaling](06-scaling/08-auto-scaling.md)

### Part 7: Real-Time Systems
- [Polling](07-real-time/01-polling.md)
- [Long Polling](07-real-time/02-long-polling.md)
- [Server-Sent Events](07-real-time/03-server-sent-events.md)
- [WebSockets](07-real-time/04-websockets.md)
- [WebRTC](07-real-time/05-webrtc.md)
- [Presence Systems](07-real-time/06-presence-systems.md)

### Part 8: Case Studies
- [Twitter Timeline](08-case-studies/01-twitter-timeline.md)
- [Instagram Feed](08-case-studies/02-instagram-feed.md)
- [Uber Ride Matching](08-case-studies/03-uber-ride-matching.md)
- [Netflix Streaming](08-case-studies/04-netflix-streaming.md)
- [Slack Messaging](08-case-studies/05-slack-messaging.md)
- [Stripe Payments](08-case-studies/06-stripe-payments.md)
- [Dropbox Sync](08-case-studies/07-dropbox-sync.md)
- [Discord Voice](08-case-studies/08-discord-voice.md)
- [Google Search](08-case-studies/09-google-search.md)
- [WhatsApp Messaging](08-case-studies/10-whatsapp-messaging.md)

### Part 9: Whitepapers
- [MapReduce](09-whitepapers/01-mapreduce.md) (2004)
- [Dynamo](09-whitepapers/02-dynamo.md) (2007)
- [BigTable](09-whitepapers/03-bigtable.md) (2006)
- [Spanner](09-whitepapers/04-spanner.md) (2012)
- [TAO](09-whitepapers/05-tao.md) (2013)
- [Kafka](09-whitepapers/06-kafka.md) (2011)
- [Raft](09-whitepapers/07-raft.md) (2014)
- [Chubby](09-whitepapers/08-chubby.md) (2006)
- [Aurora](09-whitepapers/09-aurora.md) (2017)
- [CockroachDB](09-whitepapers/10-cockroachdb.md) (2020)

## Notation

| Symbol | Meaning |
|--------|---------|
| N | Total nodes/replicas |
| W | Write quorum size |
| R | Read quorum size |
| f | Failures tolerated |

## License

MIT License
