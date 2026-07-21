# Cache Stampede, Cold Start, and Warming

## TL;DR

A stampede is overload caused by many requests discovering the same missing or unusable cache state before one protected refill completes. A cold start is the same mechanism across many keys. Warming is controlled refill before or during traffic.

The safety objective is not “restore hit rate quickly.” It is:

1. keep live origin work below a measured safe budget;
2. allow at most a bounded number of recomputations per key;
3. preserve request deadlines and fairness;
4. never let an older refill overwrite a newer source version;
5. serve stale data only inside an explicit hard-age contract;
6. ramp traffic only when weighted cache coverage proves the origin can carry the remaining misses.

Use local request coalescing, a distributed lease or server-supported recache token where needed, stale-while-revalidate, probabilistic early recomputation, TTL jitter, and admission control as complementary layers. Warming without rate limits is a denial-of-service job pointed at your own origin.

Scope: refill amplification, stampede control, cold-start recovery, and warming rollouts. Cache policy and capacity are in [Cache Semantics and Economics](01-cache-strategies.md), invalidation races in [Cache Invalidation and Coherence](02-cache-invalidation.md), and node topology in [Distributed Cache Internals](03-distributed-caching.md).

---

## 1. Model the Miss Amplification

### 1.1 One hot key

Let a hot key receive $\lambda_k$ requests per second and take $C_k$ seconds to recompute. With no coalescing, approximately:

$$
D_k \approx \lambda_k C_k
$$

duplicate computations can begin before the first one fills the cache.

~~~text
20,000 requests/s for one key
250 ms authoritative computation

duplicates during one fill window = 20,000 * 0.25 = 5,000
~~~

Those 5,000 operations may each hold a database connection, allocate a large object, and call downstream services. The cache outage is only the trigger; queueing at the origin is the destructive feedback loop.

### 1.2 A cold working set

For total request rate $\lambda$ and effective useful hit rate $h$:

$$
\lambda_{\text{miss}} = \lambda(1-h)
$$

By Little's Law, approximate concurrent origin operations as:

$$
L_{\text{origin}} = \lambda_{\text{miss}} W_{\text{origin}}
$$

where $W_{\text{origin}}$ is origin service time. As origin utilization rises, $W_{\text{origin}}$ rises; more requests overlap, deadlines expire, retries arrive, and the system can enter a self-reinforcing overload state.

The recovery condition is therefore not simply “cache fills faster than before.” Live misses, foreground refreshes, retries, and warming work together must remain below the origin's safe work budget.

### 1.3 Distinguish miss causes

Track at least:

- first request for a legitimate key;
- TTL expiry;
- capacity eviction;
- explicit or bulk invalidation;
- cache node loss or reshard;
- process-local L1 cold start;
- key or payload generation rollout;
- negative-cache expiry;
- cache timeout or bypass;
- corrupted or incompatible entry.

Each cause has a different fix. Increasing TTL does not repair a node-loss surge; adding a lock does not repair uniform scan pollution.

---

## 2. Layered Protection

~~~mermaid
flowchart TD
    R["Request"] --> H{"Fresh hit?"}
    H -->|yes| RET["Return"]
    H -->|no| S{"Usable stale value?"}
    S --> C["Join per-key coalescer"]
    C --> L{"Leader / valid lease?"}
    L -->|yes| B{"Origin budget available?"}
    B -->|yes| O["Load authoritative version"]
    O --> F["Conditional fill by source version"]
    F --> RET
    B -->|no| D["Serve stale, degrade, or reject"]
    L -->|no| W["Wait with deadline or serve stale"]
    S -->|no stale| C
    D --> RET
    W --> RET
~~~

No single mechanism covers every failure:

| Mechanism | Scope | What it prevents | What it does not prevent |
|---|---|---|---|
| Local singleflight | One process | Duplicate local loads | Duplicates across processes |
| Distributed lease | Cache-sharing fleet | Most cross-process duplicate loads | Lease expiry, pauses, split ownership |
| Stale-while-revalidate | Request path | Waiter latency and origin failure | Unauthorized or excessively old reuse |
| Probabilistic early recompute | Before expiry | Synchronized hot-key expiry | First load or whole-cluster loss |
| TTL jitter | Many entries | Batch-aligned expiry | One hot key's expiry |
| Origin admission budget | Whole service | Cascading overload | It must reject or degrade some work |
| Warming | Planned transition | Broad cold start | Surprise failure unless pre-positioned |

---

## 3. Request Coalescing

### 3.1 Process-local singleflight

For each key, a coalescer maintains one in-flight future:

~~~text
ABSENT -> LEADER_LOADING
              | joiners wait on same future
              v
        SUCCESS(value, version) or FAILURE
              |
              v
           removed
~~~

Required semantics:

- perform a second cache check after becoming leader;
- attach waiter cancellation to its own deadline without cancelling work needed by others;
- cap waiters per key and total in-flight keys;
- remove failed futures so the key can recover;
- add backoff before a failed key is retried;
- propagate one normalized result to all waiters;
- record leader, joiner, wait, timeout, and error counts.

Do not hold a global mutex while running the loader. A coalescer prevents duplicate work; it does not make the origin call safe at unlimited distinct-key concurrency.

### 3.2 Distributed lease

When many processes can miss the same expensive key, use a lease or a cache primitive that elects one recacher:

~~~text
token = random 128-bit value
acquire lock:key with token, only-if-absent, bounded lease
if acquired:
    read authoritative value and source version
    conditionally fill cache if version is not older
    release only if lock still contains token
else:
    wait with deadline or use stale value
~~~

The ownership token prevents an old holder from deleting a new holder's lease. A compare-and-delete operation should be atomic.

Lease duration trades two failures:

- too short: a slow or paused leader outlives the lease and a second leader starts;
- too long: a crashed leader delays recovery.

Use a bounded renewal protocol only while the loader is healthy and the request remains inside a system deadline. Source-version-conditional fill makes duplicate leaders tolerable for derived data.

A cache lease is not a fencing mechanism for external side effects. If the leader charges a card, sends an email, or mutates another system, use that system's idempotency or a durable workflow.

### 3.3 Waiter behavior

Waiters should not spin on a fixed interval. Prefer:

1. notification/future completion when available;
2. stale value inside the hard stale window;
3. bounded exponential backoff with jitter;
4. deadline-aware origin attempt only if a separate budget permits it;
5. explicit degraded or unavailable response.

A lock that moves 10,000 duplicate database queries into 10,000 synchronized polling requests merely relocates the stampede.

### 3.4 Failure caching

If an origin repeatedly returns a deterministic absence, negative-cache it briefly. If it returns a transient error, do not turn that error into “not found.” A very short failure backoff marker can prevent immediate retries, but preserve the error class and never serve it as successful data.

---

## 4. Serve Stale Deliberately

### 4.1 Two deadlines

Store separate freshness and availability boundaries:

~~~text
generated_at ---- fresh_until -------- stale_until
       fresh          stale allowed         unusable
~~~

- Before **fresh_until**, return normally.
- Between **fresh_until** and **stale_until**, one leader refreshes while others may receive stale.
- After **stale_until**, stale data is forbidden; apply protected origin load, degradation, or failure.

Do not reset these times when copying between CDN, proxy, L2, and L1. Preserve the authoritative generation time.

### 4.2 When stale is safe

| Data | Stale policy |
|---|---|
| Immutable asset | Content-address it; no stale correctness issue |
| Product description | Often bounded stale is acceptable |
| Recommendation | Bounded stale or omit |
| Price display | Product/legal decision; often short bound |
| Inventory | Prefer authoritative or unavailable to misleading stock |
| Authorization revocation | Do not serve below required policy version |
| Account balance | Usually authoritative/version-validated |
| Negative existence result | Short stale bound and invalidate on creation |

A stale response should carry internal telemetry such as age, source version, and stale reason even if those fields are not exposed publicly.

### 4.3 stale-if-error

Stale-on-error can preserve availability when the origin times out or returns a server error. It must not hide:

- authentication or authorization failures;
- a successful “deleted” or “revoked” state;
- schema corruption;
- an entry older than the hard stale deadline.

HTTP defines **stale-while-revalidate** and **stale-if-error** controls. Reverse proxies and CDNs may implement request collapsing and grace behavior differently; verify the actual product semantics.

---

## 5. Probabilistic Early Recompute

### 5.1 Why fixed refresh thresholds synchronize

If every request refreshes when 10 seconds remain, many concurrent requests can cross that threshold together. A lock can elect one, but probabilistic early recompute spreads election attempts over time without a separate scheduler.

### 5.2 XFetch decision

The XFetch family uses the previous recomputation duration $\Delta$, a tuning factor $\beta$, and a random $U$ uniformly distributed in $(0,1)$:

$$
\text{recompute if } now - \beta\Delta\ln(U) \geq expiry
$$

Because $\ln(U)$ is negative, expensive recomputations receive a wider early-refresh window. Equivalent pseudocode is:

~~~text
early_window = beta * last_compute_seconds * (-ln(random_0_to_1))
if now + early_window >= fresh_until:
    attempt recomputation
~~~

Store the measured compute duration with the value. A formula that compares a dimensionless random value directly with seconds but omits recomputation duration is dimensionally wrong.

### 5.3 Operational properties

- XFetch greatly reduces synchronized expiry but does not guarantee one leader.
- Combine it with local coalescing or a distributed lease for very hot keys.
- A rarely read key may receive no early request; its next access still performs a normal miss.
- $\beta$ is a workload parameter. Tune it from recomputation cost, request rate, and acceptable early work rather than fixed folklore.
- Reject an early result if its source version is older than the cache's current version floor.

### 5.4 TTL jitter

For groups written together, choose:

~~~text
actual_ttl = base_ttl * (1 + uniform(-j, +j))
~~~

Use a random source independent across entries and bound the result above zero. Jitter spreads batch expiry; it does not protect a single hot key. Do not jitter a hard regulatory or security freshness deadline beyond its maximum.

---

## 6. Admission Control Is the Last Safety Boundary

### 6.1 Origin work budget

Express origin capacity in work units when queries have different costs. Let:

- $W_{\text{safe}}$ = measured safe origin work per second at target p99;
- $W_{\text{live}}$ = current foreground work;
- $W_{\text{reserve}}$ = failure and critical-traffic reserve.

Then background refresh and warming must satisfy:

$$
W_{\text{refresh}} + W_{\text{warm}}
\leq
W_{\text{safe}} - W_{\text{live}} - W_{\text{reserve}}
$$

When the right-hand side is zero, warming pauses. Live critical traffic receives priority.

### 6.2 Budgets to enforce

- per-key loader concurrency;
- global origin loader concurrency;
- per-tenant concurrency and rate;
- maximum queued waiters and bytes;
- database connection and downstream RPC budgets;
- refresh and warming token buckets;
- request retry budget;
- maximum stale-serving rate and age.

A circuit breaker that bypasses a failed cache but has no origin limiter is incomplete.

### 6.3 Overload response

After the budget is exhausted, choose deliberately:

- serve a valid stale entry;
- omit a nonessential module;
- return a partial response;
- reject low-priority traffic with retry guidance;
- fail closed for security-sensitive decisions;
- shed warming and refresh work before live requests.

Queueing without a bound is not graceful degradation; it converts overload into timeout and memory pressure.

---

## 7. Failure Traces

### 7.1 Lease holder pauses

~~~text
t0 leader A acquires 2 s lease
t1 A pauses for 3 s
t2 lease expires; leader B loads version 12 and fills
t3 A resumes with version 11
~~~

If A blindly sets, the cache regresses. Conditional fill by source version rejects version 11. Token-checked release prevents A from deleting B's new lease.

### 7.2 Leader fails and waiters retry

~~~text
leader origin call fails
all 5,000 waiters receive failure
all retry immediately
~~~

Coalescing reduced the first wave but can synchronize the second. Add failure backoff with jitter, retry budgets, and a stale/degraded response.

### 7.3 Cache cluster is unreachable

Local coalescing still protects each process, but 200 processes can each elect a leader for the same key. A service-wide origin budget and stale store remain necessary even when the distributed coordination layer is gone.

### 7.4 Invalidation of a hot namespace

A generation bump makes millions of old entries unreachable at once. The event is operationally equivalent to a cold cluster. Use soft invalidation, staged generation rollout, or pre-warm the new generation within the origin budget.

### 7.5 Refill from a lagging replica

An invalidation for source version 50 arrives before a regional read replica reaches 50. The refiller must wait, read a sufficiently current source, or decline the fill. Otherwise a “successful” rebuild restores version 49.

### 7.6 Warming job competes with users

A fixed-rate warmer keeps issuing 5,000 queries/s while a traffic spike consumes all origin headroom. Live latency grows and retries begin. The warmer must use feedback from current origin saturation and pause automatically.

---

## 8. Cold-Start Taxonomy

| Event | Cold scope | Preferred first response |
|---|---|---|
| New application process | Its L1 | Small critical manifest, then gradual traffic |
| Rolling deploy | Growing fraction of L1 fleet | Stagger rollout; shared L2 absorbs misses |
| New cache node | Moved slots/ranges | Native copy or peer transfer, rate-limited |
| Whole cache restart | Regional working set | Stale standby, priority warm, admission control |
| Key schema generation | New namespace | Shadow fill and dual-read comparison |
| Bulk invalidation | A tag or namespace | Soft purge, coalesced refill |
| New feature | Unknown key distribution | Shadow keying and dark fill |
| Region failover | Region-wide cache and read model | Catch-up gate plus traffic ramp |
| CDN purge | Many points of presence | Soft purge and origin shield |

“Warm the cache” is too vague. State the cold scope, authoritative source, allowed copy method, and work budget.

---

## 9. Warming Sources

### 9.1 Heavy-hitter manifest

Build a manifest from sampled access logs or telemetry:

~~~text
key
key-generation
payload-schema
estimated request rate
entry bytes
origin work per miss
source version if known
tenant / residency domain
priority and expiry
~~~

Use a streaming heavy-hitter algorithm or trace analysis rather than retaining every raw key indefinitely. Protect personal data in logs and manifests.

### 9.2 Access-log replay

Replay canonical cache keys, not raw HTTP requests with side effects. Deduplicate, remove expired tenants, respect authorization boundaries, and sample by expected future traffic. Historical logs can be wrong for launches or seasonal changes.

### 9.3 Predictive/event-driven manifest

Known releases, scheduled events, and product launches can add keys before historical traffic exists. A forecast should remain a hint: low-value predictions must not evict proven hot entries.

For event-sourced read models, stream catch-up may build the projection more correctly than point-reading every key. Consumer lag is one readiness input, but lag zero alone does not prove that writes to the cache succeeded or that payloads are valid.

### 9.4 Peer or snapshot copy

Copying entries from a healthy peer avoids origin recomputation. Preserve:

- remaining TTL and original generation time;
- source version;
- key and payload generation;
- tenant, region, and encryption boundaries.

Do not restore an unrestricted serialized snapshot into a new code version. Native Redis migration or dump/restore preserves cache bytes, not application schema compatibility. Validate a sample and reject incompatible entries.

### 9.5 Shadow traffic

A new cache can observe a copy of read-only traffic and fill without serving responses. Ensure shadow requests:

- cannot perform writes or duplicate side effects;
- have lower priority than live work;
- carry only authorized data to the target region;
- are rate-limited independently;
- are excluded from product analytics.

---

## 10. Warming Capacity and Priority

### 10.1 Maximum warm rate

For approximately uniform warm operations:

$$
R_{\text{warm}}
\leq
\min\left(
  \frac{W_{\text{available origin}}}{\bar{c}_{\text{origin}}},
  C_{\text{cache writes}},
  \frac{B_{\text{network}}}{\bar{b}_{\text{entry}}}
\right)
$$

where $\bar{c}_{\text{origin}}$ is mean origin work per warmed key and $\bar{b}_{\text{entry}}$ is mean transferred bytes. Use percentiles or separate classes when costs are skewed.

Estimated completion time is:

$$
T_{\text{warm}} = \frac{N_{\text{selected}}}{R_{\text{warm}}}
$$

Batching and pipelining reduce round trips but do not reduce database work. Bound batch bytes and transaction duration.

### 10.2 Prioritize saved work per byte

A useful first-order score is:

$$
score_k =
\frac{
  request\_rate_k \times origin\_cost_k \times business\_criticality_k
}{
  entry\_bytes_k
}
$$

This favors small, frequently used, expensive-to-recompute, important entries. Do not assume a universal 80/20 distribution; calculate weighted coverage from the actual trace.

### 10.3 Partial warming

Full warming is often wasteful because long-tail entries expire before reuse. Warm until projected origin load fits the safe budget:

~~~text
choose the smallest priority prefix such that
projected live misses + refresh + reserve <= safe origin capacity
~~~

Lazy-fill the remaining tail under normal admission control.

### 10.4 Feedback-controlled warmer

The coordinator should:

1. acquire tokens from the current origin-work budget;
2. select the highest remaining priority;
3. load with a deadline and no user-visible retry loop;
4. conditionally fill by generation and source version;
5. record bytes, work, latency, and result;
6. reduce rate on origin p99, error, replica-lag, or cache saturation;
7. pause on live-traffic demand;
8. resume from a durable manifest checkpoint.

Fixed worker count is not sufficient because origin capacity changes with live traffic.

---

## 11. Deployment and Topology Rollouts

### 11.1 New application instances

For an L1 cache:

1. start process and establish L2/invalidation continuity;
2. load only critical small entries within a startup budget;
3. mark technically ready without claiming fully warm;
4. send a small traffic weight;
5. measure per-instance weighted hit rate and L2/origin misses;
6. ramp while origin headroom and p99 remain safe;
7. stop or remove the instance if convergence stalls.

Blocking readiness until every possible key is loaded can make deployment impossible. Joining fully cold at full weight is the opposite error.

### 11.2 Blue-green and key-generation rollout

1. Dark-fill the new generation from authoritative reads or validated old entries.
2. Shadow-read old and new generations and compare source versions/digests.
3. Route a small cohort to the new generation.
4. Keep old generation readable for rollback.
5. Ramp under origin and cache-write gates.
6. stop filling old generation;
7. let old keys expire before removing old serializers.

Do not globally flush the old generation at cutover.

### 11.3 Cache-node addition and reshard

Prefer the cache product's native key migration or a peer copy for moved ranges. Warming from the database should be the fallback because it consumes origin work and can read a different version.

Rate-limit migration and warming together; both consume destination network, memory, and event-loop time. Verify moved-key hit rate and routing convergence before removing the old owner.

### 11.4 Regional failover

A region is ready when:

- authoritative/read-model replication meets the required version;
- invalidation or event consumers are caught up;
- cache weighted coverage keeps projected misses inside origin capacity;
- stale entries satisfy residency and age policy;
- traffic ramp and rollback are tested.

Fleet-wide average hit rate can hide a cold region or instance. Gate per failure domain.

### 11.5 CDN purge

Prefer content-addressed URLs for assets. For mutable content, soft purge so edges can serve bounded stale while one request revalidates. An origin shield collapses simultaneous point-of-presence misses. Rate-limit broad purges and observe origin requests before expanding them.

---

## 12. Readiness and Temperature

### 12.1 Weighted coverage

Raw “keys present / keys planned” treats a once-a-day object like a 10,000-QPS object. Use request-weighted valid coverage:

$$
coverage =
\frac{
  \sum_k request\_weight_k \cdot I(\text{entry k is valid})
}{
  \sum_k request\_weight_k
}
$$

Validity includes key generation, payload schema, source version, and remaining hard age.

### 12.2 Projected origin load

Replay a representative trace against the warmed cache or manifest:

~~~text
projected_origin_qps
projected_origin_work
largest missing hot key
misses by tenant and endpoint
p95/p99 miss fan-out
~~~

Traffic may ramp only if projected plus observed origin work remains inside the declared budget after a failure reserve.

### 12.3 Readiness gates

- weighted valid coverage;
- origin QPS/work and p99 under canary traffic;
- cache write latency, memory, evictions, and errors;
- warm errors by class;
- source-version and age distribution;
- invalidation consumer continuity;
- hottest absent keys;
- time-to-convergence estimate;
- no cross-tenant, residency, or schema violation.

A synthetic GET hit rate after writing the manifest proves only that writes returned success. Sample actual decode and source-version correctness.

---

## 13. Observability

### 13.1 Stampede signals

- misses per key and miss-rate derivative;
- leaders, coalesced joiners, waiters, and waiter timeouts;
- distributed lease attempts, wins, expiries, renewals, and duplicate leaders;
- origin calls avoided and origin calls admitted;
- refresh duration stored for probabilistic recompute;
- stale serves by age and reason;
- conditional-fill rejection by source version;
- retry and load-shed counts;
- database connections, downstream fan-out, and p99 correlated with misses.

Aggregate hit rate can remain healthy while one hot key is stampeding. Preserve sampled key-class and top-key visibility.

### 13.2 Warming signals

- manifest entries and weighted request coverage;
- keys/bytes/work warmed per second;
- live versus background origin budget;
- errors, retries, and skipped obsolete entries;
- cache memory, evictions, and overwritten newer versions;
- projected and observed origin miss load;
- per-instance/region temperature and convergence time.

### 13.3 Alerts should express coupling

A useful alert detects cache miss growth **and** protected-origin stress, or a rising lease-waiter count with leader failure. A fixed global hit-rate threshold is workload-dependent and can page on harmless traffic changes while missing a single-key incident.

---

## 14. Verification Matrix

| Test | Required assertion |
|---|---|
| Expire one hottest key | Bounded leaders; origin work stays within budget |
| Pause lease holder past expiry | Duplicate result cannot overwrite newer version |
| Fail leader origin request | Waiters do not retry synchronously |
| Remove distributed cache | Global origin limiter prevents fleet fan-out |
| Flush one shard | Refill rate and p99 remain bounded |
| Invalidate a hot namespace | Soft/staged path avoids origin cliff |
| Start 25% cold application instances | Per-instance ramp protects L2 and origin |
| Warm with live traffic spike | Warmer yields capacity automatically |
| Replay obsolete manifest | Generation/version checks reject entries |
| Restore peer snapshot | TTL, schema, tenant, and source-version checks hold |
| Region failover | Weighted coverage and source catch-up gates block unsafe ramp |
| Origin unavailable | Stale/degraded/fail-closed policy matches data class |

### Incident runbook

1. Stop warming, background refresh, and nonessential retries.
2. Identify scope: one key, one shard, one generation, or whole region.
3. Protect origin with concurrency and tenant budgets.
4. Enable only policy-approved stale or degraded paths.
5. Coalesce the hottest misses and inspect leader failures.
6. Restore cache topology or refill priority keys at feedback-controlled rate.
7. Ramp normal traffic by weighted coverage, not elapsed time.
8. Verify source versions and stale-age distribution.
9. Re-enable refresh and warming last.
10. Preserve miss, origin, lease, and topology timelines for review.

### Checklist

- [ ] Duplicate-fill concurrency is modeled from key rate and compute time.
- [ ] Coalescing has bounded waiters, deadlines, and failure backoff.
- [ ] Distributed leases use ownership tokens and conditional release.
- [ ] Older refill results cannot overwrite newer source versions.
- [ ] Fresh and hard-stale deadlines are distinct and preserved across tiers.
- [ ] Probabilistic recompute uses measured computation duration.
- [ ] TTL jitter is not mistaken for hot-key protection.
- [ ] Origin admission protects live traffic before refresh and warming.
- [ ] Warming priority is based on saved work per byte and real traces.
- [ ] Readiness uses weighted valid coverage and projected origin work.
- [ ] Deploy, reshard, flush, and region failover are exercised under load.

---

## Primary References

- [Optimal Probabilistic Cache Stampede Prevention, Vattani et al., PVLDB 2015](https://www.vldb.org/pvldb/vol8/p886-vattani.pdf)
- [RFC 5861: stale-while-revalidate and stale-if-error](https://www.rfc-editor.org/rfc/rfc5861.html)
- [Go singleflight](https://pkg.go.dev/golang.org/x/sync/singleflight)
- [Memcached Meta Text Protocol: serve stale and recache tokens](https://docs.memcached.org/protocols/meta/)
- [Scaling Memcache at Facebook, NSDI 2013](https://www.usenix.org/conference/nsdi13/technical-sessions/presentation/nishtala)
- [Redis pipelining](https://redis.io/docs/latest/develop/using-commands/pipelining/)
