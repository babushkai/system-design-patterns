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

## References

### Books

| Book | Author | Topics |
|------|--------|--------|
| [Designing Data-Intensive Applications](https://dataintensive.net/) | Martin Kleppmann | Replication, partitioning, transactions, distributed systems |
| [System Design Interview Vol. 1](https://www.amazon.com/System-Design-Interview-insiders-Second/dp/B08CMF2CQF) | Alex Xu | Rate limiting, consistent hashing, key-value stores |
| [System Design Interview Vol. 2](https://www.amazon.com/System-Design-Interview-Insiders-Guide/dp/1736049119) | Alex Xu | Real-world systems, proximity services, stock exchange |
| [Database Internals](https://www.databass.dev/) | Alex Petrov | B-trees, LSM trees, storage engines, distributed databases |
| [Understanding Distributed Systems](https://understandingdistributed.systems/) | Roberto Vitillo | Networking, coordination, scalability, resiliency |
| [Building Microservices](https://www.oreilly.com/library/view/building-microservices-2nd/9781492034018/) | Sam Newman | Service decomposition, integration, deployment |

### Original Papers

#### Distributed Systems Foundations
- [Time, Clocks, and the Ordering of Events](https://lamport.azurewebsites.net/pubs/time-clocks.pdf) - Lamport, 1978
- [Impossibility of Distributed Consensus with One Faulty Process (FLP)](https://groups.csail.mit.edu/tds/papers/Lynch/jacm85.pdf) - Fischer, Lynch, Paterson, 1985
- [The Part-Time Parliament (Paxos)](https://lamport.azurewebsites.net/pubs/lamport-paxos.pdf) - Lamport, 1998
- [Paxos Made Simple](https://lamport.azurewebsites.net/pubs/paxos-simple.pdf) - Lamport, 2001

#### Storage & Databases
- [The Google File System](https://static.googleusercontent.com/media/research.google.com/en//archive/gfs-sosp2003.pdf) - Ghemawat et al., 2003
- [MapReduce: Simplified Data Processing on Large Clusters](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf) - Dean & Ghemawat, 2004
- [Bigtable: A Distributed Storage System for Structured Data](https://static.googleusercontent.com/media/research.google.com/en//archive/bigtable-osdi06.pdf) - Chang et al., 2006
- [Dynamo: Amazon's Highly Available Key-value Store](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf) - DeCandia et al., 2007
- [Spanner: Google's Globally-Distributed Database](https://static.googleusercontent.com/media/research.google.com/en//archive/spanner-osdi2012.pdf) - Corbett et al., 2012
- [Amazon Aurora: Design Considerations for High Throughput Cloud-Native Relational Databases](https://web.stanford.edu/class/cs245/readings/aurora.pdf) - Verbitski et al., 2017

#### Consensus & Coordination
- [The Chubby Lock Service for Loosely-Coupled Distributed Systems](https://static.googleusercontent.com/media/research.google.com/en//archive/chubby-osdi06.pdf) - Burrows, 2006
- [ZooKeeper: Wait-free coordination for Internet-scale systems](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Hunt.pdf) - Hunt et al., 2010
- [In Search of an Understandable Consensus Algorithm (Raft)](https://raft.github.io/raft.pdf) - Ongaro & Ousterhout, 2014

#### Messaging & Streaming
- [Kafka: a Distributed Messaging System for Log Processing](http://notes.stephenholiday.com/Kafka.pdf) - Kreps et al., 2011
- [The Log: What every software engineer should know about real-time data's unifying abstraction](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying) - Jay Kreps, 2013

#### Social & Web Scale
- [TAO: Facebook's Distributed Data Store for the Social Graph](https://www.usenix.org/system/files/conference/atc13/atc13-bronson.pdf) - Bronson et al., 2013
- [Scaling Memcache at Facebook](https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final170_update.pdf) - Nishtala et al., 2013
- [F4: Facebook's Warm BLOB Storage System](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-muralidhar.pdf) - Muralidhar et al., 2014

#### NewSQL
- [CockroachDB: The Resilient Geo-Distributed SQL Database](https://dl.acm.org/doi/pdf/10.1145/3318464.3386134) - Taft et al., 2020
- [TiDB: A Raft-based HTAP Database](https://www.vldb.org/pvldb/vol13/p3072-huang.pdf) - Huang et al., 2020

### Engineering Blogs

| Company | Notable Posts |
|---------|---------------|
| [Netflix Tech Blog](https://netflixtechblog.com/) | Microservices, chaos engineering, streaming |
| [Uber Engineering](https://www.uber.com/blog/engineering/) | Real-time systems, geospatial, scaling |
| [Meta Engineering](https://engineering.fb.com/) | TAO, distributed systems, ML infrastructure |
| [Stripe Engineering](https://stripe.com/blog/engineering) | API design, idempotency, payments |
| [Cloudflare Blog](https://blog.cloudflare.com/) | Edge computing, DNS, DDoS mitigation |
| [Discord Engineering](https://discord.com/blog/engineering-posts) | Real-time messaging, voice, scaling |
| [Slack Engineering](https://slack.engineering/) | Messaging architecture, search, reliability |
| [Dropbox Tech Blog](https://dropbox.tech/) | Sync, storage, infrastructure |
| [Pinterest Engineering](https://medium.com/pinterest-engineering) | Recommendations, search, scaling |
| [LinkedIn Engineering](https://engineering.linkedin.com/blog) | Kafka, data infrastructure, ML |
| [Twitter Engineering](https://blog.twitter.com/engineering) | Timeline, real-time, graph processing |
| [Spotify Engineering](https://engineering.atspotify.com/) | Streaming, personalization, microservices |
| [GitHub Engineering](https://github.blog/category/engineering/) | Git internals, availability, scaling |
| [Shopify Engineering](https://shopify.engineering/) | E-commerce, flash sales, payments |

### Online Resources

| Resource | Description |
|----------|-------------|
| [High Scalability](http://highscalability.com/) | Architecture case studies |
| [The Morning Paper](https://blog.acolyer.org/) | CS paper summaries (archived) |
| [Papers We Love](https://paperswelove.org/) | Community for reading CS papers |
| [Distributed Systems Reading List](https://dancres.github.io/Pages/) | Curated paper collection |
| [System Design Primer](https://github.com/donnemartin/system-design-primer) | Popular GitHub resource |
| [ByteByteGo](https://bytebytego.com/) | Visual system design explanations |
| [Awesome Distributed Systems](https://github.com/theanalyst/awesome-distributed-systems) | Curated resources list |

### Video Lectures

| Course | Institution | Topics |
|--------|-------------|--------|
| [MIT 6.824: Distributed Systems](https://pdos.csail.mit.edu/6.824/) | MIT | MapReduce, Raft, Spanner, distributed transactions |
| [CMU 15-445: Database Systems](https://15445.courses.cs.cmu.edu/) | CMU | Storage, indexing, query processing, concurrency |
| [CMU 15-721: Advanced Database Systems](https://15721.courses.cs.cmu.edu/) | CMU | In-memory databases, query optimization |
| [Stanford CS244B: Distributed Systems](https://www.scs.stanford.edu/20sp-cs244b/) | Stanford | Consensus, replication, distributed storage |

### Tools & Technologies

| Category | Tools |
|----------|-------|
| Databases | PostgreSQL, MySQL, MongoDB, Cassandra, Redis, CockroachDB, TiDB |
| Message Queues | Kafka, RabbitMQ, Amazon SQS, Google Pub/Sub, NATS |
| Caching | Redis, Memcached, Hazelcast |
| Search | Elasticsearch, Apache Solr, Meilisearch |
| Monitoring | Prometheus, Grafana, Datadog, New Relic |
| Tracing | Jaeger, Zipkin, OpenTelemetry |
| Load Balancers | NGINX, HAProxy, Envoy, AWS ALB |
| Container Orchestration | Kubernetes, Docker Swarm, Nomad |

## License

MIT License
