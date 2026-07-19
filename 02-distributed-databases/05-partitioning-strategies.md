# Partitioning Strategies

Partitioning maps one logical key space onto independently placeable units. The choice determines which requests are single-partition, which become distributed operations, where hot spots form, and how much state must move when capacity changes. A good partition function is therefore part of the data model and query contract—not a storage detail added after launch.

This chapter owns **logical partition boundaries, key-to-partition routing, and replica-placement primitives**. [Database Sharding](../06-scaling/03-database-sharding.md) owns the application and operational workflow for introducing shards, moving live data, cutover, rollback, and organizational ownership. [Secondary Indexes](./06-secondary-indexes.md) owns index placement, while [Distributed Transactions](./07-distributed-transactions.md) owns atomic work that crosses the resulting boundaries.

## Workload contract

Start with the operations, not with hash versus range. For every high-volume request, identify:

- the routing key known before execution;
- equality, prefix, and range predicates;
- joins or constraints that touch another entity;
- per-key and per-tenant request distributions, including the hottest key;
- data size and growth distribution;
- ordering and locality requirements;
- the failure domains across which replicas must be placed.

A partition key is successful when the important operations can name a small, bounded set of partitions. Uniform bytes alone are insufficient. Hashing every row evenly can turn a tenant query into a cluster-wide scan; placing an entire tenant together can make one large tenant exceed a node.

The contract should state a fan-out budget. “Most requests are single-partition; reporting may scan all partitions” is actionable. “The database is sharded” says nothing about cost.

## State and invariants

A production partitioned system needs more state than `hash(key) mod N`:

- a **partition function** mapping a key to a logical partition or ordered range;
- a stable partition ID independent of its current machine;
- an authoritative map from partition ID to replica set and current write authority;
- a metadata epoch or generation;
- per-partition state such as `ACTIVE`, `SPLITTING`, `MOVING`, or `MERGING`;
- placement constraints describing region, zone, rack, hardware class, or tenant policy;
- a progress frontier for copy and catch-up during changes.

The central invariants are:

1. At one metadata epoch, every legal key maps to exactly one logical partition.
2. Range boundaries have neither gaps nor unintended overlap.
3. A request is evaluated by an owner authorized for the request's epoch or is rejected/redirected.
4. A placement satisfies the configured replica count and failure-domain constraints.
5. Publishing a new map does not make writes acknowledged by the old owner disappear.

Stable logical partitions separate two decisions: `key -> partition` and `partition -> machine`. This two-stage mapping lets operators move a bounded unit without changing every key's hash function and lets routing metadata name partitions rather than individual records.

## Partitioning primitives

### Hash partitioning into fixed logical buckets

Compute `bucket = H(key) mod P`, where `P` is a stable logical bucket count, then place buckets on machines through metadata. A suitable non-adversarial hash spreads independent keys and makes equality lookup direct.

The important word is **logical**. If `P` is the number of machines, adding a machine changes the modulus and relocates roughly `N/(N+1)` of uniformly hashed keys when growing from `N` to `N+1`; almost all traffic and storage participate. If `P` is a larger stable bucket count, growth moves selected buckets instead. The cost is metadata and granularity: too few buckets limit balancing; too many add per-bucket state, files, consensus groups, and scheduling work.

Hash partitioning destroys key order across buckets. A range predicate whose routing key is absent must scatter or use another index. It also cannot split one indivisible hot key; hashing distributes many keys, not traffic within a key.

### Consistent and rendezvous hashing

Consistent hashing places keys and node tokens in a circular hash space; a key belongs to the next eligible token. In an ideal balanced ring, adding one equal-capacity node moves approximately `1/(N+1)` of keys rather than remapping nearly everything. Virtual nodes create more placement samples per physical node and support heterogeneous capacity, but each token also increases metadata, movement concurrency, and operational surface.

Rendezvous hashing scores each `(key, node)` pair and chooses the highest-scoring eligible nodes. It avoids ring traversal and directly produces an ordered replica preference, but evaluating every node is expensive without hierarchy or candidate sets. Both methods need explicit failure-domain filtering: three top scores on one rack are not three independent replicas.

Consistent hashing minimizes expected movement under membership change; it does not guarantee balanced bytes or QPS when values and access frequencies are skewed.

### Range partitioning

Range partitioning stores adjacent keys together:

```text
[-infinity, g) -> range 17
[g, n)         -> range 42
[n, +infinity) -> range 63
```

It preserves ordered scans, prefix locality, and sequential prefetch. A large range can split at a chosen boundary, and neighboring small ranges can merge. Bigtable's tablets and CockroachDB's ranges are examples of this primitive.

Range partitioning exposes the write distribution. Monotonic timestamps or sequence numbers concentrate new writes at the rightmost range. Lexicographic prefixes can concentrate one tenant or popular domain. Splitting helps only if the hot workload spans separable keys; a single hot row remains one serialization point.

Boundary selection can be size-based, sample-based, or load-aware. Equal key-space widths are rarely equal in bytes or QPS. A split policy must consider write rate, read rate, storage bytes, and the ability to place the children independently.

### Directory partitioning

A directory explicitly maps an entity to a partition: `tenant_37 -> partition 912`. It supports exceptions, tenant moves, and heterogeneous placement without encoding every rule in the hash. It also creates a metadata service and cache-coherence problem. Directory availability, map versioning, stale-routing behavior, and bootstrap become part of the request path.

Directories are especially useful for tenant or cell placement, where policy matters more than perfect uniformity. They are less attractive for billions of individually mapped rows unless the directory itself is hierarchically partitioned.

### Composite and hybrid schemes

Many systems compose primitives:

- hash tenant ID, then sort by time inside the tenant bucket;
- directory-map a tenant to a cell, then range-partition within the cell;
- range-partition an ordered key space, then replicate each range with consensus;
- add a bounded write stripe `(entity_id, stripe)` and sort within each stripe.

The first key components choose placement; later components choose order within the partition. A write stripe relieves one aggregate hot spot only by making reads fan out across the stripes. That is a deliberate exchange, not a free salting trick.

## Routing and ownership changes

### Versioned routing

Clients, routers, or coordinator nodes cache the partition map. Every routed request should carry the map epoch or target partition ID. The receiver checks authority:

- if current, execute;
- if stale but safely forwardable, redirect with newer metadata;
- if ownership is ambiguous during transition, reject with a retryable error rather than accept under the wrong generation.

An unversioned cache can continue sending writes to an old owner indefinitely. A redirect loop is also possible when two routers alternately advertise stale maps, so responses should carry a monotonic epoch and clients must never downgrade it.

### Split and merge state machine

A safe range split separates data preparation from authority publication:

1. choose a boundary and durable split identity;
2. create child state from a consistent parent snapshot;
3. capture writes after that snapshot through a log or dual-application protocol;
4. bring children to the parent's cutover frontier;
5. atomically publish the new map/epoch and fence parent writes;
6. retain redirect and rollback metadata until old routers drain;
7. delete obsolete parent state only after no reader or rollback path references it.

The metadata operation must appear atomic even when copying bytes is not. Merge is the reverse problem and must preserve both children through one publication point. The end-to-end live migration runbook belongs in [Database Sharding](../06-scaling/03-database-sharding.md); the invariant here is that a map epoch names one authoritative partitioning of the key space.

### Replica placement

Partitioning determines the unit; placement chooses its replicas. A policy such as “three replicas” is incomplete without topology. State constraints in terms of failures to survive: one host, one rack, one zone, or one region. Then verify the selected replica set and its quorum can make progress after that correlated failure.

Placement also determines latency. Put the write authority near writers while ensuring the required acknowledgement set fits the durability policy. Moving a lease or leader without moving replicas changes latency but not storage placement; moving a replica changes data risk and network cost.

## Failure traces

### Stale router during a split

1. Range `[a,z)` splits into `[a,m)` and `[m,z)` at epoch 51.
2. A router cached epoch 50 and sends write `t` to the parent.
3. If the parent still accepts while a child also accepts, two histories form.
4. If the parent silently drops the write, acknowledged data is lost.

The parent must validate the epoch and either forward through a protocol that preserves ordering or reject with epoch 51. Child activation and parent fencing are one correctness transition.

### Hot key hidden by balanced bytes

1. One million keys distribute evenly over 100 partitions.
2. One key receives 40% of requests.
3. Storage dashboards show equal bytes, but its partition saturates CPU and queueing delay grows.
4. Adding nodes moves cold partitions and leaves the indivisible key hot.

The remedy is an application-level operation split, caching/replication for reads, or a deliberate striped representation—not another hash function.

### Correlated replica placement

1. Three replicas are placed on three hosts but in one rack.
2. The rack switch fails.
3. The partition loses every copy despite meeting replica count.

Placement validation must reason about shared risk, not host identity alone.

### Scatter-gather tail amplification

1. A request fans out to 40 partitions and waits for all results.
2. Thirty-nine finish quickly; one is in compaction or recovery.
3. End-to-end latency equals the slowest required branch plus aggregation.

If one branch latency has CDF `F(t)` and branches were independent, all 40 finish by `t` with probability `F(t)^40`. Real branches often share network and storage, making correlation worse than this model. Fan-out turns rare local tails into common request tails.

## Capacity and cost model

Let `D` be logical data bytes, `Q` request rate, `P` logical partitions, and `N` machines. `D/P` and `Q/P` are only means. Capacity planning must use high-percentile and maximum partition bytes/QPS, plus the hottest individual key.

For a request touching `f` partitions:

```text
network requests ~= f
intermediate bytes = sum(result bytes from each partition)
latency >= max(required branch latencies) + coordination
```

If each required partition is independently available with probability `a`, request availability is approximately `a^f`. Independence is an optimistic simplification, but it makes the design pressure clear: broad fan-out narrows the success window.

Moving a partition of size `S` has a lower-bound copy time:

```text
copy_time >= S / min(source_read_rate, network_rate, destination_write_rate)
```

Catch-up traffic, checksums, replication, foreground interference, and throttling make actual time longer. Growing from `N` to `N+1` equal-capacity nodes ideally moves about `D/(N+1)` bytes with balanced consistent hashing or stable buckets. Range skew and placement constraints can require more.

Choose `P` by balancing:

- enough units to distribute maximum bytes and QPS after failures;
- enough spare units to use future machines;
- small enough movement units to meet recovery time;
- not so many that metadata, files, consensus groups, heartbeats, and schedulers dominate.

There is no universal bucket count. Model the per-partition fixed cost from the actual engine and test the largest planned map.

## Operations, migration, and testing

Monitor bytes, reads, writes, CPU, queue time, compaction/recovery work, and replica lag **per partition**, with maxima and skew coefficients—not only node averages. Also track routing-epoch misses, redirects, split duration, copy backlog, unavailable placement constraints, and time to restore the desired replica set.

Before changing a partition function, build an offline mapping diff: how many keys and bytes move, which queries change fan-out, which partitions become hottest, and how rollback maps new writes. Shadow-route sampled production keys to compare old and new ownership without executing writes.

Property tests should prove range coverage with no gaps/overlaps, deterministic hash/rendezvous results across language implementations, replica diversity, and monotonic metadata epochs. Fault tests should pause old owners during cutover, lose map invalidations, duplicate copy-log entries, crash after metadata publication, and restore from every state-machine phase. Load tests need Zipfian keys and a single extreme hot key; uniform random traffic hides the failures partitioning is meant to manage.

## Decision framework

1. Which request fields are known before routing, and how many partitions does each critical operation touch?
2. Does order/range locality matter more than uniform placement?
3. What are the maximum key, tenant, and partition QPS and bytes—not only averages?
4. Can one hot key be subdivided without breaking its invariant or order?
5. What metadata epoch fences an old owner during split, move, and merge?
6. How much data moves for one node addition or failure, and how long can catch-up take?
7. Do replica placements survive the named correlated failure while retaining quorum?
8. Is operational online resharding mature enough for this logical scheme?

## Primary references

- [Karger et al., *Consistent Hashing and Random Trees* (STOC 1997)](https://www.akamai.com/site/en/documents/research-paper/consistent-hashing-and-random-trees-distributed-caching-protocols-for-relieving-hot-spots-on-the-world-wide-web.pdf)
- [DeCandia et al., *Dynamo: Amazon's Highly Available Key-value Store* (SOSP 2007)](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf)
- [Chang et al., *Bigtable: A Distributed Storage System for Structured Data* (OSDI 2006)](https://storage.googleapis.com/gweb-research2023-media/pubtools/4443.pdf)
- [Curino et al., *Schism: a Workload-Driven Approach to Database Replication and Partitioning* (VLDB 2010)](https://www.vldb.org/pvldb/vol3/R76.pdf)
- [Adya et al., *Slicer: Auto-Sharding for Datacenter Applications* (OSDI 2016)](https://www.usenix.org/system/files/conference/osdi16/osdi16-adya.pdf)
- [Taft et al., *CockroachDB: The Resilient Geo-Distributed SQL Database* (SIGMOD 2020)](https://www.cockroachlabs.com/pdf/cockroachdb-the-resilient-geo-distributed-sql-database-sigmod-2020.pdf)
