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

### Additional Books

#### Architecture & Design Patterns
| Book | Author | Topics |
|------|--------|--------|
| [Clean Architecture](https://www.oreilly.com/library/view/clean-architecture-a/9780134494272/) | Robert C. Martin | Dependency rule, boundaries, components, frameworks |
| [Patterns of Enterprise Application Architecture](https://martinfowler.com/books/eaa.html) | Martin Fowler | Domain logic, data source, web presentation patterns |
| [Domain-Driven Design](https://www.domainlanguage.com/ddd/) | Eric Evans | Bounded contexts, aggregates, ubiquitous language |
| [Implementing Domain-Driven Design](https://www.oreilly.com/library/view/implementing-domain-driven-design/9780133039900/) | Vaughn Vernon | Practical DDD patterns and techniques |
| [Release It!](https://pragprog.com/titles/mnee2/release-it-second-edition/) | Michael Nygard | Stability patterns, capacity, networking |
| [Software Architecture: The Hard Parts](https://www.oreilly.com/library/view/software-architecture-the/9781492086888/) | Ford, Richards, Sadalage, Dehghani | Trade-off analysis, modularity, decomposition |
| [Fundamentals of Software Architecture](https://www.oreilly.com/library/view/fundamentals-of-software/9781492043447/) | Mark Richards, Neal Ford | Architecture styles, characteristics, decisions |
| [A Philosophy of Software Design](https://web.stanford.edu/~ouster/cgi-bin/book.php) | John Ousterhout | Complexity, modules, abstractions, comments |

#### Distributed Systems & Reliability
| Book | Author | Topics |
|------|--------|--------|
| [Site Reliability Engineering](https://sre.google/sre-book/table-of-contents/) | Google | SLOs, error budgets, toil, monitoring, on-call |
| [The Site Reliability Workbook](https://sre.google/workbook/table-of-contents/) | Google | Practical SRE implementation |
| [Distributed Systems](https://www.distributed-systems.net/index.php/books/ds4/) | Tanenbaum & Van Steen | Processes, communication, naming, coordination |
| [Designing Distributed Systems](https://www.oreilly.com/library/view/designing-distributed-systems/9781491983638/) | Brendan Burns | Patterns for scalable, reliable services |
| [Database Reliability Engineering](https://www.oreilly.com/library/view/database-reliability-engineering/9781491925935/) | Campbell & Majors | Database operations, infrastructure, recovery |
| [Web Scalability for Startup Engineers](https://www.oreilly.com/library/view/web-scalability-for/9780071843669/) | Artur Ejsmont | Practical scaling strategies |

#### Data & Streaming
| Book | Author | Topics |
|------|--------|--------|
| [Streaming Systems](https://www.oreilly.com/library/view/streaming-systems/9781491983867/) | Akidau, Chernyak, Lax | Watermarks, windows, triggers, exactly-once |
| [Kafka: The Definitive Guide](https://www.oreilly.com/library/view/kafka-the-definitive/9781492043072/) | Shapira, Palino, et al. | Kafka internals, producers, consumers, operations |
| [Making Sense of Stream Processing](https://www.oreilly.com/library/view/making-sense-of/9781492042563/) | Martin Kleppmann | Event sourcing, change capture, stream processing |
| [Data Mesh](https://www.oreilly.com/library/view/data-mesh/9781492092384/) | Zhamak Dehghani | Decentralized data architecture |
| [The Data Warehouse Toolkit](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/books/) | Ralph Kimball | Dimensional modeling, ETL, BI |

#### Performance & Optimization
| Book | Author | Topics |
|------|--------|--------|
| [Systems Performance](https://www.brendangregg.com/systems-performance-2nd-edition-book.html) | Brendan Gregg | Linux, observability, methodologies, tools |
| [BPF Performance Tools](https://www.brendangregg.com/bpf-performance-tools-book.html) | Brendan Gregg | Linux BPF observability and tracing |
| [High Performance MySQL](https://www.oreilly.com/library/view/high-performance-mysql/9781492080503/) | Silvia Botros, Jeremy Tinley | Query optimization, replication, scaling |
| [High Performance Browser Networking](https://hpbn.co/) | Ilya Grigorik | TCP, UDP, TLS, HTTP/2, WebSocket, WebRTC |

### More Papers

#### Consistency & Transactions
- [Linearizability: A Correctness Condition for Concurrent Objects](https://cs.brown.edu/~mph/HerlihyW90/p463-herlihy.pdf) - Herlihy & Wing, 1990
- [A Critique of ANSI SQL Isolation Levels](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-95-51.pdf) - Berenson et al., 1995
- [Large-scale Incremental Processing Using Distributed Transactions and Notifications (Percolator)](https://research.google/pubs/pub36726/) - Peng & Dabek, 2010
- [Calvin: Fast Distributed Transactions for Partitioned Database Systems](http://cs.yale.edu/homes/thomson/publications/calvin-sigmod12.pdf) - Thomson et al., 2012
- [Highly Available Transactions: Virtues and Limitations](https://www.vldb.org/pvldb/vol7/p181-bailis.pdf) - Bailis et al., 2013
- [Serializable Snapshot Isolation in PostgreSQL](https://drkp.net/papers/ssi-vldb12.pdf) - Ports & Grittner, 2012

#### Distributed Data Structures
- [Consistent Hashing and Random Trees](https://www.cs.princeton.edu/courses/archive/fall09/cos518/papers/chash.pdf) - Karger et al., 1997
- [CRUSH: Controlled, Scalable, Decentralized Placement of Replicated Data](https://ceph.io/assets/pdfs/weil-crush-sc06.pdf) - Weil et al., 2006
- [A Comprehensive Study of Convergent and Commutative Replicated Data Types (CRDTs)](https://hal.inria.fr/inria-00555588/document) - Shapiro et al., 2011
- [The Phi Accrual Failure Detector](https://www.cs.cornell.edu/projects/Quicksilver/public_pdfs/SRDS04.pdf) - Hayashibara et al., 2004

#### Search & Indexing
- [The Anatomy of a Large-Scale Hypertextual Web Search Engine](http://infolab.stanford.edu/~backrub/google.html) - Brin & Page, 1998
- [Web Search for a Planet: The Google Cluster Architecture](https://static.googleusercontent.com/media/research.google.com/en//archive/googlecluster-ieee.pdf) - Barroso et al., 2003
- [Elasticsearch: The Definitive Guide (online)](https://www.elastic.co/guide/en/elasticsearch/guide/current/index.html) - Clinton Gormley, Zachary Tong

#### Machine Learning Systems
- [Scaling Distributed Machine Learning with the Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf) - Li et al., 2014
- [TensorFlow: A System for Large-Scale Machine Learning](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf) - Abadi et al., 2016
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) - Sculley et al., 2015

#### Container & Orchestration
- [Borg, Omega, and Kubernetes](https://queue.acm.org/detail.cfm?id=2898444) - Burns et al., 2016
- [Large-scale cluster management at Google with Borg](https://research.google/pubs/pub43438/) - Verma et al., 2015
- [Mesos: A Platform for Fine-Grained Resource Sharing in the Data Center](https://people.eecs.berkeley.edu/~alig/papers/mesos.pdf) - Hindman et al., 2011

## License

MIT License
