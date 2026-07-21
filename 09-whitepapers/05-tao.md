# TAO (USENIX ATC 2013): Evidence-First Paper Analysis

TAO shows what happens when a company turns a dominant access pattern into a storage contract. Facebook did not build a general graph database. It built a geographically distributed, read-optimized service for **objects and time-ordered association lists**, then moved cache coherence out of thousands of application clients and into that service.

## Publication identity and scope

- **Paper:** *TAO: Facebook's Distributed Data Store for the Social Graph*
- **Authors:** Nathan Bronson, Zach Amsden, George Cabrera, Prasad Chakka, Peter Dimov, Hui Ding, Jack Ferris, Anthony Giardullo, Sachin Kulkarni, Harry Li, Mark Marchukov, Dmitri Petrov, Lovro Puzar, Yee Jiun Song, and Venkat Venkataramani
- **Venue and version:** USENIX Annual Technical Conference, 2013, pages 49–60
- **System described:** Facebook's production TAO deployment and workload measurements available to the authors in 2012–2013

The paper is not a specification for Meta's current graph infrastructure. It also does not describe arbitrary graph traversal, graph analytics, Cypher-like queries, or a globally serializable database. Its API is intentionally restrictive.

[Distributed Caching](../04-caching/03-distributed-caching.md) and [Cache Invalidation](../04-caching/02-cache-invalidation.md) cover the reusable mechanics. Scope here: why TAO could apply them safely to one graph workload.

## Workload that shaped the design

Facebook rendered content for a viewer by repeatedly fetching graph nodes and edges, then applying privacy and product logic at read time. Precomputing every viewer-specific feed object was infeasible. The original PHP/MySQL/memcache arrangement exposed three problems:

1. A key/value cache was a poor fit for incrementally changing edge lists; changing one edge could require reloading a whole list.
2. Cache-fill, invalidation, and thundering-herd control lived in independently deployed clients.
3. Asynchronous MySQL replication meant a cache miss in a remote region could refill from a stale replica just after a write.

TAO's goal in Section 2.2 is correspondingly narrow: serve a constantly changing social graph in many regions, optimize aggressively for reads, and prefer efficiency and availability over strong consistency. Applications should tolerate occasional stale data.

## Data model and semantic boundary

TAO models a node as an **object** and a directed, typed edge as an **association**.

An object has a globally unique 64-bit ID, type, and typed key/value fields. An association contains source ID `id1`, type `atype`, destination ID `id2`, a 32-bit timestamp, and optional data. The identity of an association is `(id1, atype, id2)`.

For each `(id1, atype)`, TAO maintains an association list ordered by descending timestamp. The core read API supports:

- lookup of selected destination IDs;
- a range of the newest associations;
- a timestamp-bounded range;
- a count.

That API matches “recent comments on this post” and “does this user like this object.” It does not expose multi-hop traversal. Applications compose calls and apply business logic themselves.

Association types may declare an inverse. TAO couples create, update, and delete with the inverse edge—for example, `AUTHORED` and `AUTHORED_BY`. This is a domain invariant built into the service, not a general graph constraint mechanism. An inverse can reside on another shard, so one logical association update may touch both source and destination shards. Section 4.2 is explicit that these two writes are **not atomic**: a failure can leave a forward edge without its inverse, and an asynchronous job repairs the hanging association.

## Persistence, sharding, and cache topology

Persistent state remains in sharded MySQL. Object IDs contain a shard identifier, making object placement direct. Associations are stored with the source object's shard; inverse associations place the reverse edge with the other endpoint. The database layer uses one master and asynchronous slaves, with the master region selectable per shard.

Above MySQL, each region has a **leader cache tier** and at least two **follower tiers**.

- Clients send normal reads to a nearby follower.
- On a hit, the follower responds without another hop.
- On a miss, the follower asks the leader responsible for that shard.
- The leader serves from its cache or fetches from the regional database.
- Writes travel follower → local leader → leader in the database-master region → MySQL master.

“Leader” here is a cache routing role, not a consensus leader. Many shards share a server, and a server may be master-region leader for some shards and slave-region leader for others.

This hierarchy separates capacity: followers absorb the enormous read volume; leaders serialize cache maintenance and shield databases from miss storms; MySQL supplies durable relational storage and replication.

Section 5.3 addresses skew rather than assuming hashing makes it disappear. Shards map to cache servers with consistent hashing, but TAO can clone a hot shard onto multiple followers and send consistency messages to every clone. For an individual item queried orders of magnitude more often than its peers, the client may retain the item and version in a small cache; later requests carry the version so a follower can omit unchanged data or throttle an extreme hot key. Cloning spreads load but increases the invalidation fan-out that correctness must cover.

## Cache state and consistency protocol

TAO caches objects, association counts, and bounded association-list ranges rather than an undifferentiated serialized blob. The service understands enough graph semantics to update or invalidate only the affected entries.

Writes commit synchronously at the MySQL master. The master-region leader returns a **changeset** to the originating follower path, which updates or invalidates the local cache before acknowledging the caller. In normal operation—defined by the paper as at most one failure encountered by a request—this supplies read-after-write consistency for clients that stay on one follower tier.

Separately, MySQL's asynchronous replication stream carries refill and invalidation messages to each slave region. The paper orders those messages after the corresponding database transaction reaches the slave. Sending an invalidation first would allow a subsequent miss to refill from a still-stale slave.

Association lists make invalidation subtle. A concurrent edge insertion can change list order, counts, and whether an item falls within a cached range. TAO uses versions and graph-specific rules to decide whether a changeset can be applied or whether the safer action is invalidation. The point is architectural: a semantic cache service can maintain list fragments more correctly than arbitrary application clients.

The guarantee remains bounded. Switching to a backup follower tier can violate read-after-write if that tier has not received the refill. A partial leader failure or permanently lost invalidation can leave stale data. Cross-region database state is eventually consistent; TAO does not claim causal or serializable consistency for the whole graph.

The paper also exposes an escape hatch rather than pretending eventual reads fit every operation: a request marked **critical** is proxied to the master region, where synchronous MySQL writes make the master database the consistent source of truth. This raises latency and centralizes the read path, so it is reserved for the small subset of operations—such as an authentication check in the paper's example—that cannot tolerate replica lag.

## Failure paths and repair

The paper treats failures at each layer differently.

- **Database master failure:** a slave is promoted. Writes that fail during the switch are returned as failures and are not silently retried, avoiding ambiguous duplicate effects.
- **Slave database failure:** regional cache misses route to the master region. A separate binlog tailer delivers consistency messages until the slave returns; replay may deliver them again, so invalidation is idempotent.
- **Leader cache failure:** followers route misses directly to the database and send writes through another leader. The substitute also records invalidations for the original leader.
- **Lost refill/invalidation:** messages for an unreachable follower are queued to disk. If permanent leader loss could have dropped them, TAO bulk-invalidates every cached object and association for the affected shard.
- **Follower failure:** clients use a backup tier. The substitute can serve the same shard but may not preserve session-local read-after-write.
- **Split inverse update:** the two shards of an association/inverse pair are not one transaction. If the second write fails after the first succeeds, an asynchronous repair job restores the missing edge.

These protocols reveal the chosen invariant order: preserve durable master writes, keep local reads available, repair cache freshness, and accept bounded periods of staleness. TAO explicitly does not permit disconnected regions to accept independent writes, avoiding application-side multi-master conflict resolution.

## Workload and performance evidence

The authors sampled 6.5 million requests randomly over 40 days and measured production behavior. The scope and sampling window belong with every percentage.

- Reads were 99.8% of calls; writes were 0.2% (Figure 3). Among reads, association range was 40.9%, object get 28.9%, association get 15.7%, association count 11.7%, and time-range 2.8%.
- Empty results were normal: only 19.6% of sampled `assoc_get`, 31.0% of `assoc_range`, and 1.9% of `assoc_time_range` calls returned edges. Forty-five percent of association-count calls returned zero, while 1% of nonzero counts exceeded 500,000. A useful system had to make both negative lookup and extreme fan-out efficient.
- The deployment continuously processed about one billion reads/s and millions of writes/s across regions at publication. This is a production scale statement, not a reproducible benchmark configuration.
- Overall read-cache hit rate was 96.4%. Figure 8 reports client-observed latency, including PHP client and network: median cache-hit latency was approximately 1.0–1.3 ms across read operations. The 99th percentile ranged from 24.8 to 32.8 ms for hits; misses varied more, with object-get averaging 75.3 ms and reaching 186.4 ms at the 99th percentile.
- The paper's follower-throughput curve used machines with 144 GB RAM, two 8-core 2.2 GHz Xeon E5-2660 CPUs with hyperthreading, and 10 Gbit Ethernet. Throughput rose with hit rate because misses and writes were more expensive; the figure is observational highest 15-minute average, not a controlled saturation benchmark.
- Average write latency was 12.1 ms in the master region and 74.4 ms from a region whose average round-trip was 58.1 ms. Geography accounts for most of the difference because writes synchronously reach the master.
- Slave replication lag was under one second for 85% of the trace, under three seconds for 99%, and under ten seconds for 99.8%. Those are empirical lag observations, not consistency bounds.
- Over 90 days, the web-server-observed failed-query fraction was `4.9 × 10^-6`; the authors warn that dependent queries and correlated systems complicate interpretation.

## Limits and assumptions

- The graph API supports only object and one-hop association-list access, not arbitrary traversal or analytics.
- The object API omits compare-and-set; eventual reads would make a client-visible CAS contract substantially less useful, and applications cannot infer conditional-update semantics that the paper never provides.
- Writes depend on one master region per shard; a disconnected region cannot continue independent mutation.
- Cross-region replication and caches are eventually consistent.
- Read-after-write is conditional on the normal path and tier affinity, not a global session guarantee.
- Association queries have a per-type result cap, typically 6,000 in the paper. Longer lists require pagination, and negative edge lookups on a high-degree object can reach MySQL because the uncached list tail might contain the destination.
- Hot shards can be cloned and extreme hot items can be version-cached at clients, but both mechanisms add coherence work rather than eliminating skew.
- MySQL, binlog delivery, cache versions, and shard-level invalidation all participate in correctness.
- Evaluation comes from one Facebook workload and author-operated infrastructure; there is no matched comparison with another graph store.

## Later evolution without back-projection

Meta's later engineering descriptions report that TAO continued to evolve operationally, but the 2013 paper's enduring pattern is narrower: **a domain-specific caching API can centralize coherence and make a relational backing store serve a read-dominant graph workload**. Modern graph stores may add traversal languages, stronger consistency, log-based storage, or different regional writes. Those additions change the proof boundary and should not be attributed to this paper.

The design also contrasts usefully with [Dynamo](./02-dynamo.md): TAO sacrifices disconnected write availability and uses one master to avoid application conflict merge; Dynamo accepts concurrent versions to keep writes available. Neither is “more distributed” in the abstract—they protect different invariants.

## Design review questions

1. Is the dominant API stable and narrow enough for the cache layer to understand mutations?
2. Which cached aggregates or ranges change for one write, and can updates be safely idempotent?
3. What exact session or tier boundary contains read-after-write?
4. Can a cache miss refill from storage before the corresponding mutation arrives there?
5. What repairs stale data after a permanently lost invalidation?
6. How are high-degree nodes, negative lookups, and long-tail object sizes represented?
7. Is regional write unavailability preferable to multi-master reconciliation for this domain?

## Primary sources

- [Bronson et al., *TAO: Facebook's Distributed Data Store for the Social Graph* (USENIX ATC 2013), official USENIX PDF](https://www.usenix.org/system/files/conference/atc13/atc13-bronson.pdf)
- [USENIX publication and presentation page](https://www.usenix.org/conference/atc13/technical-sessions/presentation/bronson)
- [Meta Engineering's publication-era TAO overview](https://engineering.fb.com/2013/06/25/core-infra/tao-the-power-of-the-graph/)
