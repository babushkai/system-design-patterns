# Cache Invalidation and Coherence

## TL;DR

Invalidation is a distributed consistency protocol between an authoritative write and one or more disposable projections. A database commit and a cache operation are not atomic merely because they appear next to each other in application code.

Use a product-defined freshness contract, a durable change signal when propagation matters, monotonic source versions to reject stale work, and a TTL as a failure bound. Prefer idempotent deletion over blind value updates. For races where a reader can refill an old value after deletion, deletion alone is insufficient: use a version fence, generation, lease, or accept a measured TTL-bounded window.

This chapter owns freshness guarantees, invalidation races, dependency propagation, and coherence across L1, L2, CDN, and browser caches. Cache placement and economics live in [Cache Semantics and Economics](01-cache-strategies.md); cluster failover lives in [Distributed Cache Internals](03-distributed-caching.md); refill amplification lives in [Stampede, Cold Start, and Warming](04-cache-stampede.md).

---

## 1. Define the Consistency Target

### 1.1 “Eventually consistent” is not a requirement

Specify guarantees per operation and object class:

| Contract | Meaning | Typical mechanism |
|---|---|---|
| Maximum age | Never serve a value generated more than $\Delta$ ago | Origin timestamp plus hard fresh/stale deadlines |
| Propagation SLO | A committed change invalidates 99.9% of projections within $\Delta$ | Durable event plus lag measurement |
| Read-your-writes | A client that committed version $v$ never reads below $v$ | Version token, primary read, or cache bypass |
| Monotonic reads | One client does not move from version $v$ back to $v-1$ | Highest-seen version token |
| Revocation bound | Authorization removal takes effect within $\Delta$ | Authoritative version check or no response cache |
| Best effort | Stale data is acceptable until expiry | TTL with measured origin capacity |

Inventory, balances, permissions, and uniqueness checks often cannot tolerate the same cache contract as catalog descriptions or timelines. If serving a stale result can transfer money, oversell scarce stock, or restore revoked access, validate against an authoritative version or do not cache the decision.

### 1.2 Source version is the ordering primitive

Wall-clock timestamps are weak ordering evidence: clocks skew, transactions can commit out of timestamp order, and two writes can share a timestamp resolution. Prefer a monotonic value meaningful to the source:

- row or aggregate version;
- database log sequence or commit position;
- object-store generation or ETag;
- event-stream offset scoped to a partition;
- index or model generation.

A cache entry should expose both **source_version** and **generated_at**. Version orders updates; time enforces age.

### 1.3 Freshness budget

For an event-invalidated cache:

~~~text
typical stale window =
    commit-to-change-event
  + broker and consumer lag
  + invalidation execution
  + local/CDN propagation
~~~

This is a distribution, not a constant. Measure p50, p95, p99, and maximum observed lag.

If an invalidation is lost, the hard bound is normally:

~~~text
maximum served age = fresh TTL + explicitly allowed stale window
~~~

That bound holds only if downstream tiers retain the original generation time. Refilling an outer tier from a stale inner tier and resetting its TTL breaks the bound.

### 1.4 TTL-only staleness is quantifiable

Assume, only for a rough model, that updates arrive as a Poisson process at rate $\mu$ and a cache entry's observed age is uniformly distributed from zero to TTL $T$. The probability that an entry of age $a$ is stale is:

$$
P(\text{stale at age }a) = 1-e^{-\mu a}
$$

The mean stale probability over a TTL cycle is:

$$
1-\frac{1-e^{-\mu T}}{\mu T}
$$

Real traffic and writes are rarely Poisson, so use this to expose sensitivity, then validate with traces. The equation makes one point clear: choosing TTL from intuition rather than change rate and business impact is not engineering.

---

## 2. Canonical Invalidation Patterns

### 2.1 TTL as a safety net

TTL requires no change-delivery system and limits how long an entry is reused. It is appropriate when:

- stale data has a known, acceptable age;
- origin capacity supports periodic misses;
- invalidation precision is not worth another control plane.

TTL is not proof that a value is current. Expiry can also synchronize misses; always treat expiry as an origin-load event. Stale-while-revalidate extends availability but also extends the maximum age and must be in the contract.

### 2.2 Commit, then delete

For a database-owned value:

~~~text
BEGIN
  update authoritative rows
  insert outbox change record       # optional but durable
COMMIT
  delete cached projection          # direct or asynchronous
~~~

Deleting before commit can expose the old database value to a miss and repopulate it. Setting the cache before commit can expose a value that later rolls back. Committing first establishes authority; the remaining problem is a bounded stale window until invalidation succeeds.

Delete is usually safer than setting a new value because:

- it is idempotent;
- the invalidator does not need every projection's serializer;
- duplicated or reordered deletes converge to absence;
- the next loader reads the committed source.

Deletion still has a stale-refill race, covered in section 4.

### 2.3 Update after commit

Populate-after-commit can remove the next-read miss when the writer already has the exact committed representation. It needs a conditional write:

~~~text
set cache entry only if incoming source_version >= stored source_version
~~~

Without that comparison, a delayed update for version 41 can overwrite version 42. If the cached object is a derived query rather than the written row, the writer may not know the correct representation; delete it instead.

### 2.4 Generational keys

A generation makes old keys unreachable:

~~~text
catalog:t-7:product-generation-18:sku-123
~~~

Incrementing a generation is an O(1) logical invalidation for a namespace. Trade-offs:

- readers need a current generation;
- old entries consume memory until expiry;
- a missed generation update can split readers;
- changing a global generation can create a cold-cache event.

Use a generation for serializer changes, bulk content releases, or bounded groups—not as a reflex for every row write.

### 2.5 Tag or surrogate-key invalidation

A derived response may depend on many entities. Associate it with a bounded set of tags such as **product:123**, **seller:9**, or **template:7**. Invalidation can then bump a tag generation or enumerate tagged objects.

An explicit reverse index has its own atomicity and cleanup problem. A generation per tag avoids mass deletes but adds version lookups and orphaned entries. Bound tags per object and objects per tag; otherwise invalidation metadata becomes a second unbounded database.

---

## 3. Make the Change Signal Durable

### 3.1 The after-commit publish gap

This sequence loses invalidations:

~~~text
1. Commit database update.
2. Process crashes before publish.
3. Old cache entry remains until TTL.
~~~

Retrying step 2 in memory does not close the gap. Two common durable sources do.

### 3.2 Transactional outbox

Write the domain change and an outbox record in the same database transaction. A relay publishes outbox rows and marks progress.

Required properties:

- at-least-once publication is expected;
- consumers are idempotent;
- retention exceeds maximum relay outage;
- poison events are quarantined without blocking the partition;
- the outbox backlog has an SLO and alert;
- disaster recovery restores domain rows and their unconsumed outbox consistently.

Do not claim exactly-once delivery. Design repeated invalidation to be harmless.

### 3.3 Change data capture

CDC reads the database's commit log and catches application writes, migrations, and approved administrative writes that bypass service code. It is useful when many writers share a source.

Operational requirements include:

- a consistent snapshot-to-log handoff during bootstrap;
- durable source offsets and enough log retention;
- schema evolution and delete/tombstone handling;
- source failover and log-position translation;
- partitioning events by the entity whose order matters;
- monitoring source-to-consumer lag, not merely broker lag.

CDC does not automatically provide a single global order across shards. Preserve per-entity order and attach source versions so consumers can reject regression.

### 3.4 Event contract

Publish domain identity and source evidence, not only an implementation-specific cache key:

~~~json
{
  "event_id": "01J...",
  "entity_type": "product",
  "entity_id": "sku-123",
  "tenant_id": "t-7",
  "source_version": 9821,
  "committed_at": "2026-07-18T12:00:00Z",
  "changed_fields": ["price", "availability"],
  "schema_version": 3
}
~~~

The invalidator owns the mapping from a domain change to L2 keys, L1 namespaces, query tags, and CDN surrogate keys. This keeps cache topology out of every writer and permits a shadow invalidator during migrations.

---

## 4. Races That Survive “Delete on Write”

### 4.1 Stale fill after invalidation

~~~text
Reader R                         Writer W
--------                         --------
cache miss
read database -> version 7
                                 commit version 8
                                 delete cache key
set cache to version 7
return version 7
~~~

The final cache is stale even though the writer deleted it after commit.

Options, in increasing strength:

1. **Accept and bound it:** use a short TTL and measure the window.
2. **Read twice:** verify the source version after computing; expensive and still needs a conditional fill.
3. **Lease generation:** loader receives generation $g$; a write increments $g$; fill succeeds only if $g$ is unchanged.
4. **Version fence/tombstone:** invalidation records minimum acceptable version 8; a fill for version 7 is rejected atomically.
5. **Single authoritative materializer:** one ordered projection consumer owns cache updates.

A delayed second delete can reduce the probability of the race but is not a correctness proof.

### 4.2 Version-fenced fill

Conceptually, one atomic cache-side operation should perform:

~~~text
if incoming.source_version < version_floor[key]:
    reject fill
elif stored.source_version > incoming.source_version:
    reject fill
else:
    store incoming value
~~~

The invalidator advances **version_floor** even when no value is present. Its retention must exceed the longest possible in-flight load. This adds metadata and complexity, so reserve it for objects whose TTL-bounded race is unacceptable.

### 4.3 Reordered value updates

~~~text
commit v10 -> event delayed
commit v11 -> event applied, cache=v11
event v10 arrives -> blind SET, cache=v10
~~~

Delete events are naturally less sensitive to reordering. If consumers populate values, partition by entity and compare source versions atomically.

### 4.4 Read replica regression

Deleting a regional cache does not guarantee a fresh refill if the loader reads an asynchronous replica:

~~~text
primary commits v20
invalidation reaches region B
region B replica still at v19
miss refills v19
~~~

Carry the committed version in the event. The regional loader can wait until its replica reaches that version, read the primary, or leave the entry absent and serve an explicitly stale copy. Cache coherence cannot repair database replication semantics.

### 4.5 Concurrent writers

Two writers that both “update DB then set cache” can finish cache writes in the opposite order from database commits. Source versions and conditional fills are the remedy. A process-local mutex is not sufficient across instances.

---

## 5. Coherence Across Tiers

### 5.1 Treat each tier as a separate projection

~~~mermaid
flowchart LR
    DB[("Commit log / outbox")] --> BUS["Durable change stream"]
    BUS --> INV["Projection invalidator"]
    INV --> L2["Distributed L2"]
    INV --> EDGE["CDN / proxy"]
    INV --> FAN["L1 invalidation stream"]
    FAN --> A["Process A L1"]
    FAN --> B["Process B L1"]
    EDGE --> BR["Browser revalidation / new URL"]
~~~

Do not implement this as a synchronous chain of remote calls on the user write path. One slow CDN purge should not hold a database transaction open.

### 5.2 L1 coherence

Process-local caches are invisible to other writers. Options:

- very short TTL and accepted bounded staleness;
- replayable per-key or per-prefix invalidation stream;
- server-assisted client tracking;
- generation polling;
- no L1 for mutable objects.

Transient pub/sub is an optimization, not the only safety mechanism. A process that disconnects may miss messages; it must flush affected entries, resume from a durable offset, or rely on a short age bound.

### 5.3 CDN and reverse-proxy coherence

For immutable content, change the URL. For mutable content:

- use validators such as ETag;
- use shared-cache directives deliberately;
- purge by bounded surrogate key when the provider supports it;
- prefer soft purge when stale serving is allowed;
- measure purge completion across points of presence;
- retain an origin shield to collapse simultaneous revalidation.

A purge acknowledgement may mean “accepted,” not “every edge is empty.” Put that provider behavior into the propagation SLO.

### 5.4 Browser caches

A service cannot enumerate every browser and delete its entry. Use:

- content-addressed URLs for immutable assets;
- short freshness plus conditional requests for mutable responses;
- an application manifest or generation change for coordinated releases;
- no-store for sensitive data that must not persist.

A service worker is another cache implementation with its own upgrade and invalidation protocol; test old workers during rolling frontend deployments.

### 5.5 Do not reset age across tiers

Every fill copies the authoritative **generated_at** and **source_version**. An L1 populated from L2 receives L2's remaining freshness, not a new full TTL. HTTP caches use **Age** for the same reason.

---

## 6. Derived Queries and Dependency Graphs

### 6.1 Why query caches are harder

A product row can affect:

- product detail;
- category pages;
- search results;
- recommendations;
- seller summaries;
- feeds and emails.

Trying to maintain a recursive dependency graph for every query can cost more than recomputation and can fail partially.

Prefer, in order:

1. cache stable entity IDs and assemble volatile fields later;
2. use short TTL for broad or high-cardinality query results;
3. invalidate a bounded tag or generation;
4. materialize the projection from an ordered event stream;
5. maintain an explicit reverse index only when its size and atomicity are controlled.

### 6.2 Cascades need budgets

One change that invalidates one million keys is a workload, not a metadata operation. Rate-limit it, deduplicate keys, batch network calls, and protect the cache event loop. Consider a generation bump when physical deletion is unnecessary.

Deleting many hot keys simultaneously also creates a refill surge. Coordinate with chapter 04's soft invalidation and warming strategy.

### 6.3 Schema deployment

During a rolling deploy, old and new code may disagree about key or payload format. Safe choices:

- new key generation with old generation left to expire;
- backward-compatible reader and writer;
- dual-read with shadow comparison before cutover.

A global flush is simple but transforms a schema rollout into a cold-cache load test. Do not use it without an origin-capacity and warming plan.

---

## 7. Failure Analysis

| Failure | Observable consequence | Containment |
|---|---|---|
| Outbox relay stopped | Propagation lag rises; entries age | TTL bound, backlog alert, replay |
| CDC offset lost | Duplicate or missing change range | Restore checkpoint, resnapshot with verified handoff |
| Poison event | One partition stops advancing | Quarantine, alert, preserve order for later replay |
| Cache delete timeout | Old entry survives | Retry idempotently; durable work item |
| L1 subscriber disconnect | One process misses invalidations | Flush on continuity loss or replay from offset |
| CDN partial purge | Some regions remain stale | Measure regional age; version URL or re-purge |
| Reordered populate event | Cache regresses | Source-version conditional set |
| Replica lag after delete | Miss refills old source value | Minimum-version read fence |
| Clock jump | Age calculation regresses | Source version plus monotonic elapsed timers |
| Tag-index loss | Derived objects remain reachable | Generation fallback and sampled audit |
| Bad invalidation release | Mass miss and origin surge | Rate limit, canary invalidator, pause switch |

### 7.1 Fail open versus fail closed

The cache's business role decides failure behavior:

- Public catalog description: serve bounded stale.
- Authorization revocation: validate current authority or fail closed.
- Product inventory: return unavailable or authoritative value, not an old count.
- Expensive recommendation: omit the module or serve a stale result.
- Negative cache for existence: bypass on cache error; never turn an error into “not found.”

Document this per endpoint. “Cache unavailable, query the database” is unsafe unless a concurrency budget proves the database can absorb it.

---

## 8. Measuring Coherence

### 8.1 Required signals

- commit-to-event, event-to-consumer, and consumer-to-delete lag;
- age and source-version lag of served entries;
- stale-hit count by reason and allowed stale policy;
- invalidation attempts, retries, deduplications, and dead letters;
- L1 subscriber continuity and last applied offset;
- per-region and per-CDN-point purge completion;
- version-fence rejection count;
- sampled cache-versus-source mismatch rate;
- invalidations per source change and keys affected per invalidation.

Broker lag alone does not measure user-visible staleness. A consumer can be current while cache deletes fail.

### 8.2 Shadow verification

Sample requests and read cache plus authority without affecting the response. Compare:

- source version;
- normalized payload digest;
- absence versus presence;
- age against the endpoint budget.

Sampling must respect privacy and avoid doubling expensive origin load. Focus on new key generations, changed invalidation mappings, and high-impact objects.

### 8.3 Example SLOs

~~~text
99.9% of product invalidations remove all regional L2 copies within 2 seconds.
99.99% of served product entries are at most 30 seconds old.
No authorization response is served below the client's minimum policy version.
100% of invalidation consumers can replay from an offset within retention.
~~~

Each SLO needs a measurement source and a declared response when telemetry is missing.

---

## 9. Rollout and Recovery

### 9.1 New invalidation path

1. Define the event and source-version contract.
2. Publish to a shadow topic or consumer without deleting.
3. Compare predicted keys/tags with live cache accesses.
4. Enable deletion for a small tenant or key prefix.
5. Verify lag, mismatch rate, origin miss load, and duplicate handling.
6. Dual-run old and new invalidators during a bounded overlap.
7. Stop the old path only after new consumers have replay and recovery drills.
8. Let old key generations expire before deleting compatibility code.

### 9.2 Recovery after an invalidation outage

1. Stop any consumer that is applying changes out of order.
2. Preserve offsets and determine the first uncertain source version.
3. Decide whether replay, generation bump, or targeted purge is cheaper.
4. Reserve origin capacity before creating a mass miss.
5. Replay idempotently while measuring cache and origin saturation.
6. Sample-compare authority and cache before declaring recovery.
7. Retain incident evidence: lag, lost range, affected keys, and served age.

### 9.3 Checklist

- [ ] Every cacheable object has a named freshness or revocation contract.
- [ ] Source versions, not wall clocks alone, order updates.
- [ ] Database commit and invalidation intent have no unprotected publish gap.
- [ ] Consumers are idempotent and replayable.
- [ ] Stale-refill and reordered-update races have an explicit answer.
- [ ] L1 disconnect, CDN partial purge, and replica lag are tested.
- [ ] TTL bounds missed invalidations and retains original source age.
- [ ] Derived-query fan-out is bounded and rate-limited.
- [ ] A mass invalidation cannot exceed protected-origin capacity.
- [ ] User-visible staleness is measured directly.

---

## Primary References

- [RFC 9111: HTTP Caching](https://www.rfc-editor.org/rfc/rfc9111.html)
- [RFC 5861: stale-while-revalidate and stale-if-error](https://www.rfc-editor.org/rfc/rfc5861.html)
- [Debezium documentation](https://debezium.io/documentation/)
- [PostgreSQL logical decoding](https://www.postgresql.org/docs/current/logicaldecoding.html)
- [Redis server-assisted client-side caching](https://redis.io/docs/latest/develop/reference/client-side-caching/)
- [Scaling Memcache at Facebook, NSDI 2013](https://www.usenix.org/conference/nsdi13/technical-sessions/presentation/nishtala)
