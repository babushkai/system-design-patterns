# Cache Semantics and Economics

## TL;DR

A cache is a disposable, incomplete, and potentially stale projection of an authoritative system. The design is not complete when a team chooses Redis, a TTL, or an eviction policy. It is complete when the team can state:

- which system owns the truth;
- exactly which reads are eligible;
- how keys and values are versioned;
- the maximum acceptable age of a response;
- what happens on a miss, cache timeout, or stale hit;
- how much origin load a cold cache creates;
- which capacity, cost, and correctness measurements justify the cache.

Cache-aside and read-through describe who performs a load. Write-through, write-around, and write-behind describe where writes flow. None of them creates atomicity between an independent cache and database. Treat a cache hit as an optimization that must preserve the application's correctness contract.

This chapter owns cache policy, tier placement, sizing, and economics. Freshness and invalidation are covered in [Cache Invalidation and Coherence](02-cache-invalidation.md), cluster mechanics in [Distributed Cache Internals](03-distributed-caching.md), and refill protection in [Stampede, Cold Start, and Warming](04-cache-stampede.md).

---

## 1. Start With the Cache Contract

### 1.1 A cache entry is derived state

For an authoritative value $S(k)$, a cache entry is better modeled as metadata plus a value, not merely a byte string:

~~~text
CacheEntry {
    key_schema_version
    payload_schema_version
    source_version          # row version, event offset, object generation, or ETag
    generated_at            # when the value was read or computed
    fresh_until
    stale_until             # optional availability window
    value | NOT_FOUND
}
~~~

The useful invariants are:

1. **Authority:** loss of the entire cache does not lose authoritative data.
2. **Key determinism:** equivalent requests map to the same key; requests with different authorization or result semantics do not.
3. **Version safety:** old serializers, deployments, and invalidations cannot silently overwrite a newer source version.
4. **Bounded reuse:** a response is served only while its age and version satisfy the endpoint's freshness contract.
5. **Miss safety:** the origin remains protected when the cache is empty or unavailable.
6. **Observability:** every response can be classified as fresh hit, stale hit, negative hit, miss, bypass, or error.

If the first invariant is false—for example, acknowledged writes live only in Redis—the component is a database or durable write buffer. It needs durability, recovery, and consistency analysis beyond ordinary caching.

### 1.2 Contract worksheet

Record a contract per cacheable object or query class:

| Decision | Example question | Evidence required |
|---|---|---|
| Source of truth | Which committed version is authoritative? | Database transaction, object generation, or log offset |
| Eligibility | Which requests may share a value? | Tenant, locale, permissions, query shape |
| Freshness | How old may a successful response be? | Product or safety requirement, not a guessed TTL |
| Failure behavior | Serve stale, bypass, reject, or degrade? | Failure-mode review and origin capacity |
| Load ownership | Application, cache library, proxy, or CDN? | One named owner for miss and refresh logic |
| Admission | Which objects deserve memory? | Reuse-distance or request-trace measurement |
| Eviction | What may be discarded first? | Miss cost, size, frequency, and recency |
| Invalidation | TTL, delete, event, generation, or revalidation? | Race analysis from chapter 02 |
| Cold start | How much origin traffic appears at zero hit rate? | Load test and rollout budget |
| Privacy | Can two principals observe the same entry? | Cache-key and response-header review |

A statement such as “profiles use Redis for one hour” is not a contract. It omits source ownership, authorization, propagation, miss behavior, and the effect of a cache failure.

---

## 2. Read and Write Policies

### 2.1 Cache-aside versus read-through

Both policies normally have the same state transition:

~~~text
ABSENT --load authoritative value--> FRESH --expiry/invalidation--> ABSENT
~~~

They differ in ownership:

- **Cache-aside:** application code checks the cache, loads the origin on a miss, and fills the entry.
- **Read-through:** a cache library or service owns the loader and exposes one read interface.

Read-through centralizes timeouts, coalescing, serialization, and metrics. It does not make a stale value correct, and a remote read-through service can become an additional availability dependency.

A safe read path is explicit about outcomes:

~~~text
read(key, deadline):
    if request is not cache-eligible:
        return origin.read(deadline), BYPASS

    entry = cache.get(key, cache_budget)
    if entry is fresh and schema-compatible:
        return entry.value, FRESH_HIT
    if entry is NOT_FOUND and still fresh:
        return not_found, NEGATIVE_HIT

    value, source_version = protected_origin_load(key, remaining_deadline)
    best_effort_fill_if_newer(key, value, source_version)
    return value, MISS
~~~

The phrase **protected origin load** is deliberate. A raw fallback during a cache outage can turn a cache incident into a database incident. Chapter 04 covers coalescing, concurrency budgets, stale serving, and warming.

### 2.2 Write policy semantics

| Policy | Acknowledged authority | Cache action | Principal risk |
|---|---|---|---|
| Write-around | Database commit | Delete or let expire | Miss immediately after a write |
| Populate-after-commit | Database commit | Set committed value | Cache-set failure or reordered fills |
| Write-through adapter | Usually database commit | Adapter writes both systems | Still a dual write unless one transaction owns both |
| Write-behind | Durable cache or queue | Persist later | Data loss, replay, backlog, and ordering |
| Invalidate from log | Database commit log | Consumer deletes/refreshes | Propagation lag and consumer failure |

**Write-through is not a consistency guarantee.** If the database commits and the cache write fails, readers may see an older entry. If the cache is written first and the database transaction rolls back, readers may see a value that never committed. A correct design still needs ordering, retry, invalidation, or version validation.

For ordinary derived caches, the most auditable default is:

1. commit the database transaction;
2. durably record or derive an invalidation event;
3. delete the cached projection;
4. let a protected reader repopulate it.

Updating instead of deleting can avoid a miss, but only when the updater has the committed value and source version and the cache can reject an older write.

### 2.3 Write-behind changes the system boundary

Returning success before the authoritative database accepts a write means the buffer is now authoritative for some interval. The minimum design includes:

- a durable log before acknowledgement;
- idempotency keys and deterministic replay;
- per-key ordering or conflict rules;
- bounded backlog and producer backpressure;
- recovery point and recovery time objectives;
- reconciliation between buffered and persisted state;
- an explicit response when the buffer cannot accept durable writes.

An in-memory dirty-key set is not write-behind durability. It is acknowledged data-loss risk.

### 2.4 Negative caching

Caching an authoritative “not found” result protects the origin from repeated misses, typo traffic, and enumeration attacks. Use a distinct sentinel rather than overloading null.

The negative key must include the same tenant and authorization dimensions as the positive key. Its TTL is usually shorter because creation can make the result stale. Creation must invalidate the negative entry. Do not negative-cache transient upstream errors as absence.

---

## 3. Place Each Cache Tier Deliberately

### 3.1 A request may cross several coherence domains

~~~mermaid
flowchart LR
    B["Browser cache"] --> E["CDN / edge"]
    E --> P["Reverse proxy / origin shield"]
    P --> A["Application"]
    A --> L1["L1: process memory"]
    L1 --> L2["L2: distributed cache"]
    L2 --> O[("Authoritative origin")]
~~~

These are not interchangeable copies:

| Tier | Scope | Best fit | Main correctness hazard |
|---|---|---|---|
| Browser | One user agent | Immutable assets, private revalidation | Cannot centrally enumerate or purge every client |
| CDN/edge | Many users per point of presence | Public objects and responses | Cache-key mistakes can cross users or locales |
| Reverse proxy/origin shield | Region or service | Request collapsing, stale-on-error | Hidden reuse of personalized responses |
| L1 process cache | One process | Extremely hot, read-mostly objects | Every process is an independent stale copy |
| L2 distributed cache | Service or region | Shared hot working set | Node loss, hot shards, topology changes |
| Database buffer pool | Database instance | Pages and indexes | Managed by the engine, not application invalidation |

Add a tier only when a measured latency, bandwidth, or origin-load problem remains after the preceding tier. Every tier adds another failure mode, key definition, metric, and invalidation path.

### 3.2 HTTP cache semantics are part of system design

For HTTP responses:

- **no-store** tells caches not to store the response.
- **no-cache** permits storage but requires validation before reuse.
- **private** prevents storage by shared caches; a browser may still cache it.
- **public** explicitly permits shared caching when other rules allow it.
- **max-age** controls freshness for recipients; **s-maxage** can override it for shared caches.
- **ETag** and **Last-Modified** support conditional validation.
- **Vary** adds selected request headers to the cache key and can multiply variants.
- **immutable** is appropriate for content-addressed or versioned resources whose URL changes with content.

Treat authenticated and cookie-bearing responses as private unless a reviewed design explicitly makes them shareable. A CDN cache key must include every request property that can change the representation—tenant, locale, encoding, selected query parameters, and sometimes authorization class—while excluding tracking noise that only destroys hit rate.

Content-addressed assets avoid invalidation:

~~~text
/app.7f3a91.js
Cache-Control: public, max-age=31536000, immutable
~~~

Mutable APIs should usually use validation or short bounded freshness rather than pretending their URL is immutable.

### 3.3 L1 plus L2

L1 removes a network hop; L2 shares capacity across instances. Their contract should be asymmetric:

- L1 is small, short-lived, and expendable.
- L2 holds the regional working set.
- An L2 hit may populate L1 only with the original source timestamp and version.
- L1 must not reset the age of an already-old L2 value.
- L1 invalidation uses a replayable mechanism or a TTL safety bound; transient pub/sub alone is not sufficient.
- Process memory is multiplied by fleet size. A 500 MB L1 on 200 instances consumes 100 GB and increases garbage-collector or allocator pressure.

Server-assisted tracking, when supported by the client and cache, can target invalidations more precisely. Its disconnect behavior must be understood: local entries generally need flushing or revalidation when tracking continuity is lost.

### 3.4 Freshness does not reset at every tier

Suppose an edge stores a response for 60 seconds, refills from a proxy holding a 5-minute-old copy, and then resets its own timer. The user can see data older than the apparent edge TTL.

Carry **generated_at**, **source_version**, HTTP **Age**, or an equivalent origin timestamp through every tier. Compute remaining freshness from the authoritative generation time:

~~~text
remaining_freshness = freshness_budget - (now - generated_at)
~~~

Do not grant a fresh TTL to a value merely because it moved between caches. Chapter 02 derives the full coherence budget.

---

## 4. Hit Rate, Latency, and Origin Load

### 4.1 Conditional hit rates

In a hierarchy, report the hit rate at each tier **conditioned on reaching that tier**. If $h_i$ is the conditional hit rate at tier $i$, then:

$$
P(\text{origin}) = \prod_{i=1}^{n}(1-h_i)
$$

and:

$$
h_{\text{effective}} = 1 - P(\text{origin})
$$

Example: L1 hits 80% of application reads and L2 hits 90% of the reads that miss L1.

~~~text
effective hit rate = 1 - (1 - 0.80)(1 - 0.90) = 98%
20,000 requests/s -> 400 origin reads/s
~~~

Calling both tiers “90% hit rate” without saying whether rates are conditional leads to double-counting.

### 4.2 Expected latency

Let $l_i$ be the lookup cost at tier $i$, $m_i$ the probability of missing every preceding tier, and $L_o$ the origin service time. A first-order model is:

$$
E[L] = \sum_{i=1}^{n} m_i l_i + P(\text{origin})L_o
$$

Averages are insufficient when a miss consumes a database connection, fans out to several services, or creates a large payload. Model p95/p99 hit and miss latency separately and include deadline propagation.

A cache can reduce average latency while worsening tail latency if cache timeouts are long enough that requests pay both the failed cache lookup and the full origin lookup.

### 4.3 Useful hit rate

Count a hit as useful only if it:

- returns a schema-compatible value;
- meets the freshness and authorization contract;
- avoids work at the protected origin;
- finishes within the cache latency budget.

Track these separately:

~~~text
fresh_hit
stale_hit
negative_hit
miss_absent
miss_expired
miss_evicted
miss_schema
miss_error
bypass
~~~

A high raw hit rate can hide stale responses, oversized objects, or a cache that is slower than the origin for cheap queries. Byte hit rate and avoided-origin-work rate are often more informative than object hit rate.

### 4.4 Cold-cache envelope

For request rate $\lambda$ and effective hit rate $h$:

$$
\lambda_{\text{origin}} = \lambda(1-h) + \lambda_{\text{refresh}}
$$

The design must be safe at the lowest hit rate expected during restart, failover, resharding, or deploy—not only at steady state. If zero-hit traffic exceeds origin capacity, the service needs admission control, staged traffic, pre-warming, stale serving, or a smaller cacheable surface.

---

## 5. Economic Model

### 5.1 Break-even equation

Evaluate a cache against the work it avoids:

~~~text
monthly benefit =
    avoided origin CPU and I/O
  + avoided database replicas or provisioned headroom
  + avoided network/egress work
  + latency or conversion value
  - cache compute and memory
  - replication and cross-zone traffic
  - invalidation and telemetry infrastructure
  - operational and incident cost
~~~

Do not paste a cloud instance price into a timeless design rule. Measure the marginal cost curve of the actual origin and cache in the deployment region.

A cache is justified when the cache's total operating cost is lower than the value of avoided origin work **and** its failure modes fit the service objective.

### 5.2 Miss-ratio curve, not folklore

The working set is not “20% of the database.” Derive a miss-ratio curve from representative access traces:

1. canonicalize requests into proposed cache keys;
2. replay at least a full business cycle, including peaks;
3. compute reuse distance or simulate candidate admission/eviction policies;
4. plot miss rate against bytes, not only item count;
5. repeat for tenant, region, and endpoint because one aggregate hides skew;
6. test scans, launches, and failover traffic separately.

Choose the knee of the measured curve. A power-law workload may have a small valuable hot set; a uniform or scan-heavy workload may not.

### 5.3 Capacity equation

A practical memory estimate is:

$$
M = N \times (B_k + B_v + B_m) \times F_a \times F_h \times R
$$

where:

- $N$ is admitted entries in the target working set;
- $B_k$, $B_v$, and $B_m$ are measured key, value, and metadata bytes;
- $F_a$ covers allocator and fragmentation overhead;
- $F_h$ is growth and failure headroom;
- $R$ is the physical copy count, including replicas.

Example:

~~~text
8,000,000 entries
48 B key + 512 B value + 80 B measured metadata = 640 B/entry
logical data = 5.12 GB
allocator factor 1.25, headroom 1.30, two physical copies

required memory = 5.12 * 1.25 * 1.30 * 2 = 16.64 GB
~~~

Validate with production-shaped serialized values and the cache's own memory-accounting command. Include replication/AOF buffers, fork copy-on-write exposure, client buffers, operating-system memory, and failover headroom where applicable.

### 5.4 Admission and eviction are different decisions

- **Admission** asks whether a new object deserves cache space.
- **Eviction** chooses what to discard when space is needed.
- **Expiration** decides when freshness ends.

| Workload | Useful policy direction | Failure to test |
|---|---|---|
| Recency-heavy | LRU or segmented LRU | One scan evicts the hot set |
| Stable popularity | LFU or frequency-aware admission | Old popularity survives a regime change |
| Mixed recency/frequency | TinyLFU-style admission plus recency window | Sketch aging and burst behavior |
| Uniform random | Small or no cache | Memory buys little reuse |
| Objects with different costs | Cost/size-aware admission | Large cheap objects evict small expensive ones |

Redis's LRU and LFU policies are approximations. A **volatile** eviction policy considers only keys with expiration; non-expiring keys can consume the memory and leave no useful eviction candidates. For a pure derived cache, an all-keys policy is usually easier to reason about. For mixed durable and cache data, split workloads rather than depending on TTL discipline to create safety.

---

## 6. Key and Value Design

### 6.1 Key schema

A key should encode all semantic dimensions and no accidental ones:

~~~text
service:tenant:entity:key-schema-version:identity-or-query-hash
catalog:t-42:product:v3:sku-123
search:t-42:results:v7:sha256(canonical-query)
~~~

Requirements:

- use a stable, language-independent hash when hashing composite keys;
- canonicalize query parameters before hashing;
- include tenant, locale, permission class, model/index generation, and feature variant when they affect the result;
- never put secrets or unnecessary personal data in observable keys;
- bound key length and reject unbounded user-controlled cardinality;
- document ownership so two services do not mutate the same namespace.

A schema generation changes reachability without scanning and deleting old keys. Old generations still consume memory until expiry, so set a TTL and monitor orphaned bytes.

### 6.2 Payload schema

Store enough metadata to reject unsafe reuse:

~~~text
{
  "payload_schema": 4,
  "source_version": 981274,
  "generated_at_ms": 1784389200123,
  "fresh_until_ms": 1784389230123,
  "stale_until_ms": 1784389530123,
  "value": ...
}
~~~

During rolling deploys, readers must either understand both old and new payloads or use different key generations. Deserialization failures are misses, not request crashes. Compression saves memory and network bytes but consumes CPU and can make small objects slower; choose it from measured payload distributions.

### 6.3 Do not cache authorization decisions blindly

A response cache and an authorization system have different failure consequences. If a permission revocation must take effect immediately, validate a current authorization version or bypass the cache. Never share a personalized object solely because the URL matches. Include the principal or a reviewed authorization cohort in the key, and avoid caching sensitive responses in shared HTTP caches.

---

## 7. Rollout and Verification

### 7.1 Safe introduction

1. **Baseline:** measure origin QPS, service time, dependency fan-out, and capacity without the cache.
2. **Shadow keying:** compute proposed keys and sizes without serving cached values; detect cardinality and privacy mistakes.
3. **Dark fill:** populate a small isolated cache and derive the miss-ratio curve.
4. **Canary reads:** serve a small cohort while shadow-reading the origin and comparing source versions or payload digests.
5. **Failure tests:** inject timeout, empty cache, eviction pressure, malformed payload, and unavailable cache.
6. **Traffic ramp:** increase only while useful hit rate, origin load, freshness, and tail latency meet gates.
7. **Kill switch:** retain a bounded, protected bypass path and a way to disable fills independently of reads.

Never validate a cache only with a warm benchmark. Run the miss path and recovery path at production concurrency.

### 7.2 Required telemetry

Per tier, key class, tenant, and region:

- request rate and conditional useful hit rate;
- hit, miss, bypass, stale, and error latency distributions;
- miss reason and origin work caused;
- entry age and source-version lag at serve time;
- logical bytes, resident bytes, fragmentation, and allocator pressure;
- admissions, expirations, evictions, rejected writes, and oversized items;
- hot-key and hot-shard contribution;
- serialization errors and key-generation distribution;
- cache spend and avoided-origin cost.

Use sampling for per-key telemetry; logging every key can create a new cost and privacy incident.

### 7.3 Decision checklist

- [ ] Cache is derived from a named authoritative source.
- [ ] Freshness, stale serving, and negative caching are product decisions.
- [ ] Key semantics include tenant and authorization boundaries.
- [ ] Read and write races have version-aware handling.
- [ ] Effective hit rate and origin load are calculated across all tiers.
- [ ] Capacity comes from measured values and a miss-ratio curve.
- [ ] The cold-cache envelope fits origin or admission-control capacity.
- [ ] Cache timeouts consume only a small part of the request deadline.
- [ ] Rollout includes shadow comparison and an independently tested kill switch.
- [ ] Metrics distinguish useful hits from stale, incompatible, or slow hits.

---

## Primary References

- [RFC 9111: HTTP Caching](https://www.rfc-editor.org/rfc/rfc9111.html)
- [RFC 5861: stale-while-revalidate and stale-if-error](https://www.rfc-editor.org/rfc/rfc5861.html)
- [Redis key eviction](https://redis.io/docs/latest/develop/reference/eviction/)
- [Redis server-assisted client-side caching](https://redis.io/docs/latest/develop/reference/client-side-caching/)
- [Caffeine: eviction and admission efficiency](https://github.com/ben-manes/caffeine/wiki/Efficiency)
- [Scaling Memcache at Facebook, NSDI 2013](https://www.usenix.org/conference/nsdi13/technical-sessions/presentation/nishtala)
