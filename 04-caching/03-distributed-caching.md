# Distributed Cache Internals

## TL;DR

A distributed cache is a sharded, replicated, memory-constrained data plane plus a routing and topology control plane. It scales only when keys distribute evenly, clients agree on ownership, one hot key cannot saturate a shard, and node loss does not send more misses to the origin than it can survive.

For ordinary caching, asynchronous replication and occasional lost cache writes are acceptable because the authoritative source can rebuild them. The same semantics are not automatically safe for sessions, locks, rate limits, or acknowledged write-behind data. Name the role before choosing the product.

Scope: partitioning, replication, topology changes, hot keys, network behavior, and cache-cluster failure containment. Policy and sizing of the cached working set are in [Cache Semantics and Economics](01-cache-strategies.md), invalidation ordering in [Cache Invalidation and Coherence](02-cache-invalidation.md), and cold refill control in [Stampede, Cold Start, and Warming](04-cache-stampede.md).

---

## 1. System Model

### 1.1 Components

~~~mermaid
flowchart LR
    C1["Client A\nrouting map epoch 42"] --> N1[("Shard 1")]
    C1 --> N2[("Shard 2")]
    C2["Client B\nrouting map epoch 42"] --> N2
    C2 --> N3[("Shard 3")]
    CP["Topology control plane"] -.-> C1
    CP -.-> C2
    N1 -.-> R1["Replica 1"]
    N2 -.-> R2["Replica 2"]
    N3 -.-> R3["Replica 3"]
    N1 --> O[("Authoritative origin")]
    N2 --> O
    N3 --> O
~~~

A production design has at least four concerns:

1. **Key placement:** deterministic mapping from a key to an owner.
2. **Topology:** membership, health, ownership epochs, and migration state.
3. **Data plane:** GET, SET, DELETE, expiry, admission, and eviction.
4. **Origin protection:** bounded fallback and refill when cache state is absent.

A topology is not correct merely because every server is healthy. All active clients must converge on compatible ownership, and transitions must preserve bounded miss load.

### 1.2 Cache role classification

| Role | May lose entries? | Eviction acceptable? | Required analysis |
|---|---:|---:|---|
| Derived object/query cache | Yes | Yes | Freshness, origin protection, rebuild |
| Negative cache | Yes | Yes | Short age, create invalidation |
| Session store | Often no | Usually no | Durability, revocation, failover |
| Rate-limit counter | Depends on abuse model | Usually no | Atomicity, partition behavior, fail-safe mode |
| Distributed lock | No silent split ownership | No | Lease, fencing, clock and pause assumptions |
| Write-behind buffer | No acknowledged loss | No | Durable log, replay, backpressure |

Do not use “Redis is already available” to collapse these roles into one cluster. A volatile cache eviction policy and an authoritative session or lock have conflicting failure requirements.

---

## 2. Partitioning and Routing

### 2.1 Stable hashing is a protocol

Never use a language runtime's default hash for distributed placement. Some runtimes randomize it between processes; implementations and versions may differ. Define:

- hash algorithm and exact byte encoding;
- normalization and namespace;
- seed or domain separator;
- placement algorithm;
- topology epoch;
- test vectors shared by every client language.

A routing change without a versioned protocol can make two clients store and read the same key on different nodes.

### 2.2 Modular hashing

~~~text
owner = stable_hash(key) mod N
~~~

It balances well when hashes are uniform, but changing $N$ remaps approximately $(N-1)/N$ of keys when one node is added. For an ephemeral cache that can be acceptable only when the origin can absorb an almost complete cold start.

### 2.3 Consistent, rendezvous, and jump hashing

| Algorithm | Lookup state/cost | Movement | Operational fit |
|---|---|---|---|
| Hash ring with virtual nodes | Sorted ring, $O(\log V)$ | Near proportional | Arbitrary weighted membership |
| Rendezvous / highest-random-weight | Score candidate nodes, $O(N)$ | Only keys won by changed node | Small node sets, simple membership |
| Jump consistent hash | $O(\log N)$, constant memory | Minimal when bucket count grows | Sequential bucket IDs |
| Fixed logical slots | Slot table plus stable key-to-slot hash | Move selected slots | Explicit control over resharding |

Virtual-node count and weights are control-plane parameters, not magic constants. Validate distribution with the real key trace and include object bytes, not only key counts.

### 2.4 Skew-aware placement

Uniform hash output does not imply uniform load:

- one key can receive a million requests;
- tenants can have different object sizes;
- commands have different CPU cost;
- multi-key requests co-locate work;
- a launch can change popularity faster than rebalancing.

Measure per-shard request rate, bytes, CPU, event-loop utilization, and top-key contribution. A balanced key count can hide a saturated shard.

### 2.5 Routing map continuity

A client should retain the last known good map during a transient control-plane outage. It must also:

- attach or log the map epoch used;
- refresh on explicit redirection or topology notification;
- cap refresh frequency to prevent a control-plane storm;
- fail boundedly when no owner is known;
- stop using removed nodes after a drain deadline;
- expose stale-map and redirection metrics.

Static stability matters: an unavailable topology service should not immediately make a healthy cache data plane unreachable.

---

## 3. Redis Cluster as a Concrete Slot System

### 3.1 Slots and hash tags

Redis Cluster maps keys into 16,384 logical hash slots and assigns slots to primary nodes. A hash tag—the non-empty substring inside braces—can co-locate related keys:

~~~text
{order:789}:summary
{order:789}:items
~~~

Co-location permits multi-key operations for those keys but can create a hot slot. Use it for an operation that genuinely needs atomic same-slot access, not merely because entities look related.

### 3.2 Client behavior

A cluster-aware client normally caches the slot ownership map:

- **MOVED** indicates the durable owner and should update routing.
- **ASK** is temporary during migration; the client sends **ASKING** to the target for that operation without treating it as permanent ownership.
- **CLUSTER SHARDS** is the current topology discovery interface; older clients may use the deprecated **CLUSTER SLOTS** response.

Redirections are part of normal resharding, but a persistent rise means stale clients, topology churn, or an incomplete migration.

### 3.3 Live resharding

For a slot moving from A to B:

~~~text
A marks slot MIGRATING to B
B marks slot IMPORTING from A
existing keys remain readable on A
missing/migrated keys receive temporary ASK routing to B
keys are migrated in bounded batches
all nodes converge on B as the slot owner
clients eventually receive MOVED and refresh maps
~~~

Large values make migration pauses and network usage unpredictable. Bound value and collection size before relying on live resharding.

### 3.4 Multi-key limitations

Multi-key commands, transactions, and scripts generally require all involved keys to share a slot. Designing hash tags around every possible transaction can destroy distribution. If an operation spans natural partitions, redesign it as an application workflow rather than forcing a cache cluster to behave like a distributed relational database.

---

## 4. Memcached as a Client-Partitioned System

Memcached servers do not coordinate a cluster membership or replicate values as part of the core server. Clients or a proxy choose an owner, commonly with consistent hashing.

Consequences:

- every client fleet needs the same server list and hash protocol;
- node loss means those entries are absent unless a proxy writes replicas;
- adding a node changes placement and creates misses for the moved range;
- there is no server-side ownership redirect to repair a stale client;
- simplicity and memory-efficient key/value caching are advantages when entries are truly disposable.

A routing proxy can centralize membership, replication, failover, request coalescing, and observability, but it becomes another data-plane dependency that needs horizontal capacity and static routing fallback.

---

## 5. Replication and Failover Semantics

### 5.1 Asynchronous replication

Many cache clusters acknowledge a primary write before replicas have applied it. On primary failure:

~~~text
SET version 12 acknowledged
primary fails before replica receives it
replica promoted with version 11 or no key
next read misses or returns older cache state
~~~

For a derived cache, the correct reaction is reload or invalidate against the authoritative source. For authoritative use, this is acknowledged data loss and requires a different design.

### 5.2 Redis Cluster write safety

Redis Cluster uses asynchronous replication and can lose acknowledged writes, especially around failures and minority partitions. It is designed for high performance and best-effort write safety, not linearizable consensus.

Settings that require a primary to observe a minimum number of sufficiently current replicas can reduce the risk of accepting isolated writes. They do **not** create a majority quorum protocol or fencing guarantee. On heal, one lineage wins and cache writes on the discarded lineage can disappear.

### 5.3 Failover preconditions

A cache failover design should state:

- how failure is suspected and confirmed;
- which nodes vote or authorize promotion;
- whether a sufficiently current replica exists;
- how clients discover the new owner;
- how long requests wait before bypass or rejection;
- what happens to in-flight writes;
- how the recovered old primary is prevented from serving as owner;
- how missing entries are refilled without overloading the origin.

Do not document a fixed “one-second failover” without measuring detection, election, client refresh, DNS/proxy behavior, and refill under the deployed configuration.

### 5.4 Replicas are not free read scale

Reading cache replicas can increase throughput but introduces:

- read-after-write lag;
- version regression across requests;
- extra network and memory copies;
- more refill paths after failover;
- hot-key load on every replica.

Use primary reads when a caller requires a just-written cache value, or attach a minimum source version and reject older entries.

---

## 6. Capacity and Failure Headroom

### 6.1 Size against three independent resources

Let:

- $M$ = logical cached bytes after admission;
- $F$ = allocator, metadata, growth, and failure factor;
- $C_m$ = usable dataset bytes per primary;
- $Q$ = peak operations per second;
- $C_q$ = safe operations per second per primary at target p99;
- $B$ = peak payload bytes per second;
- $C_b$ = safe network bytes per second per primary;
- $S$ = measured skew factor greater than or equal to 1.

A first-order primary count is:

$$
N \geq \max\left(
  \frac{MF}{C_m},
  \frac{QS}{C_q},
  \frac{BS}{C_b}
\right)
$$

Then add replication and enough spare primaries or slots to survive the declared failure. Measure $C_q$ with production command mix, payload sizes, persistence settings, network path, and pipelining. Generic “operations per second” tables are not capacity plans.

### 6.2 N-minus-one test

If load is evenly distributed, losing one of $N$ primaries multiplies surviving average load by approximately:

$$
\frac{N}{N-1}
$$

That says nothing about a hot key or a large migrating slot. Run the failure test with observed skew, while replicas promote and clients refresh maps.

Capacity gates should include:

- p99 latency within SLO after one failure;
- CPU/event-loop and network below safe saturation;
- memory below eviction cliff;
- origin miss QPS below its protected budget;
- enough room to reshard or rebuild a replica.

### 6.3 Memory cliffs

A memory cache does not degrade smoothly when full. It may:

- evict useful entries, increasing origin load;
- reject writes under no-eviction policy;
- fragment resident memory;
- consume buffers outside the configured dataset limit;
- pause on large expiry, deletion, persistence, or fork work;
- trigger host swapping, which destroys latency.

Separate logical dataset bytes from resident memory and from host memory. Leave capacity for client buffers, replication or persistence buffers, allocator fragmentation, the operating system, and failover.

---

## 7. Hot Keys and Large Objects

### 7.1 A hot key defeats horizontal sharding

If one key's request rate exceeds a node's safe capacity, adding ordinary shards does not move that key across CPUs. Detect top-key contribution with sampled access telemetry, frequency sketches, or product-level counters.

Mitigations:

1. **L1 replication:** copy a short-lived value to each application process.
2. **Read replicas:** spread reads when replica lag is acceptable.
3. **Application replication:** write $K$ deterministic copies and choose one on read.
4. **Compute or embed locally:** remove the network lookup for static configuration.
5. **Hierarchical shield:** one regional proxy collapses requests from many clients.

Application replication multiplies write and invalidation work by $K$. Use a stable suffix mapping, update all copies with source versions, and retain TTL safety.

### 7.2 Hot-key budget

For a key with rate $q_k$ and average command service time $s_k$, its direct server utilization contribution is approximately:

$$
u_k = q_k s_k
$$

When one key consumes a large fraction of a single execution resource, the shard has little queueing headroom even if aggregate cluster QPS looks low. Monitor p99 and queueing, not only average CPU.

### 7.3 Large objects

Large values cause:

- network head-of-line blocking;
- serialization and garbage-collection pressure;
- uneven memory and slot distribution;
- long migration and deletion operations;
- oversized pipeline reply buffers.

Set an item-size and collection-cardinality budget. Split objects only when callers can tolerate partial reads and versioned assembly. Compression should be chosen from measured CPU-versus-byte savings and must have decompression size limits.

---

## 8. Client and Network Design

### 8.1 Connections and deadlines

Clients should:

- reuse bounded connection pools;
- budget cache timeouts as a small fraction of the end-to-end deadline;
- cap in-flight operations and pending bytes;
- cancel work when the caller deadline expires;
- expose pool wait separately from server latency;
- apply retry budgets, not unbounded transparent retries.

A cache timeout followed by an origin read pays both costs. A long cache timeout can therefore make a failure slower than running without a cache.

### 8.2 Pipelining and batching

Pipelining amortizes round trips but the server and client must buffer replies. Use bounded batches and cap bytes, not only command count. Very large pipelines can increase tail latency and memory.

Prefer a multi-get or variadic command when it preserves semantics. In a sharded cluster, split a multi-get by owner and enforce an overall deadline; otherwise one slow shard holds the entire response.

Pipelining does not make a read-modify-write atomic. Use a server-side atomic primitive where appropriate and same-slot constraints permit it.

### 8.3 Retries

GET and DELETE are normally safe to retry within a deadline. A fill SET should include source version when reordering matters. Increment, append, lock acquisition, and rate-limit operations may not be safe to retry after an ambiguous timeout.

Retry at one layer. Client, proxy, and service retries multiplied together can saturate the cache during the incident they are intended to hide.

### 8.4 Serialization

Version payloads and treat unknown schema as a miss. Validate size before allocation, bound decompression, and never deserialize untrusted native object formats such as unrestricted language pickles. Measure encode/decode time as part of hit latency.

---

## 9. L1 and L2 Integration

An L1 cache is a replication layer in every process. It is valuable for hot, read-mostly values, but fleet size multiplies memory and invalidation endpoints.

A safe L1/L2 arrangement has:

- shorter L1 age than the object freshness budget;
- original source version and generation time copied from L2;
- invalidation continuity or flush on disconnect;
- an L2 and origin concurrency budget during process rollouts;
- per-instance hit rate so cold instances are visible;
- a size cap below garbage-collector or allocator pressure thresholds.

Redis server-assisted client tracking can send invalidations for keys a client has read, or broadcast invalidations by prefix. Client support and disconnect semantics are part of the design. A basic transient pub/sub subscriber that silently misses messages is not coherence.

Do not perform an L2 version lookup on every L1 hit unless measurement shows the remaining network hop is still worthwhile. If every L1 hit validates remotely, the tier may save serialization but not the network dependency.

---

## 10. Failure Containment

### 10.1 Cache outage does not imply unlimited origin fallback

~~~text
steady state:
100,000 reads/s * 2% miss = 2,000 origin reads/s

cache outage:
100,000 reads/s * 100% miss = 100,000 origin reads/s
~~~

If the origin safely handles 5,000 reads/s, “fail open to the database” creates an outage.

Use a protection stack:

- short cache timeout and circuit breaker to stop waiting on a failed cache;
- stale values with a hard age and version boundary;
- request coalescing per key;
- global and per-tenant origin concurrency budgets;
- admission control or feature degradation after the budget is exhausted;
- priority for critical reads;
- gradual recovery so fills do not compete with live misses.

### 10.2 Failure policy by data class

| Cache class | Cache unavailable | Origin unavailable |
|---|---|---|
| Public content | Protected origin fallback | Bounded stale or omit |
| User profile | Protected authoritative read | Bounded stale if product permits |
| Authorization | Validate authority or fail closed | Fail closed / explicit unavailable |
| Recommendation | Omit module | Bounded stale |
| Inventory display | Authoritative read with deadline | Unavailable rather than misleading |
| Negative cache | Bypass | Preserve upstream error, not “absent” |

### 10.3 Cache poisoning

Validate that a cached value came from the intended tenant, schema, and source version. Restrict write permissions by namespace. A compromised writer or key collision can amplify one bad value across the fleet. Provide targeted purge by key generation and retain an auditable way to trace who filled an entry.

---

## 11. Topology Change Runbooks

### 11.1 Planned scale-out or reshard

1. Verify N-minus-one and origin headroom.
2. Add nodes and establish replicas before assigning user load.
3. Publish a versioned topology or begin the product's native slot migration.
4. Rate-limit bytes and keys moved; large objects get a separate budget.
5. Watch redirections, client-map epochs, p99, replication lag, evictions, and origin misses.
6. Pause automatically when any safety gate is exceeded.
7. Warm or copy only the moved hot set; do not scan and refill everything blindly.
8. Confirm every client generation has converged before draining old owners.
9. Keep rollback ownership until moved-key and error audits pass.

### 11.2 Unplanned node failure

1. Stop retry amplification.
2. Determine whether a replica can promote or keys are simply absent.
3. Apply origin miss budgets before clients fan out.
4. Prefer stale or degraded responses where contracts allow.
5. Track promotion, routing-map convergence, and refill separately.
6. Rebuild redundancy before declaring the incident over.
7. Test failback; a recovered old node must not resume ownership from stale state.

### 11.3 Cross-region failure

A cold regional cache can be a larger risk than database replication itself. Maintain a warm-enough standby from a safe source, reserve inter-region and origin capacity, and include cache age in traffic-shift readiness. Do not copy entries across regions if residency, tenant, or encryption policy forbids it.

---

## 12. Observability and Verification

### 12.1 Metrics

Per node, shard, command, client version, and key class:

- useful conditional hit rate and miss reason;
- p50/p95/p99 server, network, pool-wait, and end-to-end latency;
- operations and bytes per second;
- memory dataset, resident memory, fragmentation, evictions, expirations, and rejected writes;
- connections, in-flight commands, pipeline depth, and reply bytes;
- replication offset/lag and promotion state;
- slot or ownership distribution, migration bytes, MOVED/ASK rate, and client-map epoch;
- top-key and top-tenant contribution;
- cache errors, bypasses, stale serves, and protected-origin QPS.

### 12.2 Verification matrix

Run these under production-shaped traffic:

| Experiment | Required assertion |
|---|---|
| Kill one primary | p99 and origin load stay within gate |
| Partition primary from replicas | documented write-loss and failover behavior occurs |
| Delay cache by more than timeout | circuit opens; requests do not queue without bound |
| Empty one shard | refill is coalesced and origin budget holds |
| Add/remove node | movement matches plan; clients converge |
| Reshard large keys | migration pause and network stay bounded |
| Send one hot key | mitigation prevents single-shard saturation |
| Fill memory | chosen admission/eviction behavior protects useful hit rate |
| Disconnect L1 tracking | local cache flushes or revalidates |
| Roll serializer version | old payload is read safely or treated as miss |

A benchmark on a warm, healthy, single node proves none of these properties.

### 12.3 Checklist

- [ ] Stable hash protocol and topology epoch are shared by every client.
- [ ] Partitioning was tested with real key bytes and popularity.
- [ ] One hot key and one large object cannot dominate a shard unnoticed.
- [ ] Capacity passes N-minus-one with measured command mix.
- [ ] Replication loss semantics match the cache's role.
- [ ] Redis replica constraints are not described as consensus or fencing.
- [ ] Client timeouts, retries, pools, and pipelines are bounded.
- [ ] Cache outage cannot create unlimited origin fallback.
- [ ] Reshard, failover, and failback have automated pause gates.
- [ ] Metrics expose topology convergence and user-visible miss load.

---

## Primary References

- [Redis Cluster specification](https://redis.io/docs/latest/operate/oss_and_stack/reference/cluster-spec/)
- [Redis pipelining](https://redis.io/docs/latest/develop/using-commands/pipelining/)
- [Redis latency diagnosis](https://redis.io/docs/latest/operate/oss_and_stack/management/optimization/latency/)
- [Redis key eviction](https://redis.io/docs/latest/develop/reference/eviction/)
- [Redis server-assisted client-side caching](https://redis.io/docs/latest/develop/reference/client-side-caching/)
- [Memcached performance and efficiency](https://docs.memcached.org/serverguide/performance/)
- [Consistent Hashing and Random Trees, Karger et al.](https://people.csail.mit.edu/karger/Papers/ConsistentHashing.pdf)
- [A Fast, Minimal Memory, Consistent Hash Algorithm](https://arxiv.org/abs/1406.2294)
- [Scaling Memcache at Facebook, NSDI 2013](https://www.usenix.org/conference/nsdi13/technical-sessions/presentation/nishtala)
