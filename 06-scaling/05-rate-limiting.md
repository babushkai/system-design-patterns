# Rate Limiting: Admission Policy and Distributed Budgets

## TL;DR

Rate limiting decides whether new work may enter a protected scope. A complete design names the **subject**, **resource**, **cost unit**, **sustained rate**, **burst allowance**, **decision scope**, and **failure behavior**. “100 requests per second” is incomplete if one request costs a thousand times another, ten gateway replicas each enforce their own 100, or an unavailable counter silently changes the policy.

Token buckets are the common admission primitive because they express both rate and burst. The distributed challenge is accounting: a linearizable global decision is accurate but adds a dependency to every request; local decisions are available and fast but overshoot unless they spend bounded leases allocated by a global authority. Put cheap local protection in front of shared policy, expose honest retry guidance, and treat policy rollout and counter recovery as production migrations.

Rate limiting governs admission and quota accounting. [Circuit Breakers](./06-circuit-breakers.md) own dependency health and in-flight concurrency, [Backpressure](./07-backpressure.md) owns bounded queues and producer signaling, and [Retries, Timeouts, and Hedging](./10-retries-timeouts-hedging.md) owns later attempts.

---

## 1. Admission Contract

Define the decision before choosing an algorithm:

| Field | Required answer |
|---|---|
| **Subject** | User, tenant, credential, IP prefix, device, workload, or a hierarchy of them. |
| **Resource** | Route, operation, model, bytes, database partition, global service, or daily entitlement. |
| **Cost unit** | Request, byte, row, CPU estimate, token, recipient, or another stable weighted unit. |
| **Policy** | Sustained rate, burst, quota period, priority, and whether unused entitlement carries forward. |
| **Scope** | Process, host, zone, region, or global; exact or bounded-error enforcement. |
| **Decision** | Reject, degrade, redirect, or shape. Queueing belongs to backpressure, not an implicit limiter buffer. |
| **Response** | Machine-readable reason, policy identity, retry guidance, and remaining budget if safe to expose. |
| **Failure behavior** | Fail open, fail closed, spend a cached lease, or enter a restricted emergency policy. |
| **Change semantics** | When a new policy becomes effective and how already leased capacity is handled. |

### Invariants

1. Every admitted unit is charged to all mandatory policy dimensions exactly once at the declared accounting boundary.
2. Distributed overshoot is bounded and derived from lease or replica configuration.
3. A client cannot select another gateway, identity form, or region to multiply entitlement.
4. Clock rollback cannot mint credit or extend a quota window.
5. Policy and counter state are versioned so a stale data plane cannot enforce an incompatible rule indefinitely.
6. Cardinality and stored state remain bounded under attacker-controlled identifiers.
7. A rejection is cheaper than the work it protects.

Rate limits are policy, not capacity discovery. Derive them from downstream safe goodput, fairness, commercial entitlement, and recovery headroom; do not use a limiter to guess where saturation begins.

---

## 2. Data Plane and Control Plane

~~~mermaid
flowchart LR
    subgraph DP["Data plane"]
        R["Request + authenticated subject"]
        L["Local guard<br/>cached policy + lease"]
        G["Global decision<br/>only when required"]
        S["Protected service"]
        X["Reject / degrade<br/>reason + retry guidance"]
        R --> L
        L -->|local credit| S
        L -->|refresh or exact check| G
        G -->|allow| S
        L -->|deny| X
        G -->|deny| X
    end

    subgraph CP["Control plane"]
        P[("Versioned policies")]
        A["Allocator<br/>issues bounded leases"]
        M["Audit, usage, billing,<br/>rollout and revocation"]
        P --> A
        P --> M
    end

    A -.leases.-> L
    L -.usage reports.-> A
    P -.policy snapshots.-> L
    G -.durable decisions.-> M
~~~

The data plane must continue with a deliberate degraded policy when the control plane is impaired. If every request synchronously fetches policy, a policy-store incident becomes a full application outage. If cached policy never expires or carries no version, revocation and emergency reduction may not take effect.

---

## 3. Rate, Burst, Quota, and Concurrency Are Different

- **Rate** bounds admitted work per unit time.
- **Burst** permits temporary accumulation of unused rate credit.
- **Quota** bounds total entitlement over a longer business interval.
- **Concurrency** bounds simultaneous in-flight work and adapts to service time; [Circuit Breakers](./06-circuit-breakers.md) governs it.
- **Backpressure** slows producers or bounds queued work; [Backpressure](./07-backpressure.md) covers its end-to-end propagation.

A service may need all four. A rate limiter alone can overload a slow dependency: under Little’s Law, admitted concurrency is approximately arrival rate × service time. If service time grows tenfold while rate stays fixed, in-flight work grows tenfold. Pair an entitlement limiter with a concurrency guard at the dependency boundary.

### Multi-dimensional policy

A request may consume:

- one global service budget;
- one tenant budget;
- one route budget;
- a weighted compute budget;
- a security/abuse budget.

Define whether all dimensions must succeed atomically. Sequentially consuming global credit and then discovering the tenant is empty leaks global credit unless the first reservation can be rolled back safely. Options are:

- one atomic script/transaction for co-located counters;
- reserve all dimensions under one decision ID and commit/expire them;
- order checks from cheapest/coarsest to most specific and accept documented conservative under-utilization;
- use independent hierarchical leases whose parent already bounds their sum.

Do not reveal another tenant’s remaining capacity through response headers or timing.

---

## 4. Canonical Algorithms

### Token bucket

Let:

- <code>r</code> be credit added per second;
- <code>B</code> be maximum stored credit;
- <code>x</code> be the request’s weighted cost;
- <code>t_last</code> and <code>b_last</code> be the prior update.

At monotonic time <code>t</code>:

> b_now = min(B, b_last + r × max(0, t − t_last))

Admit if <code>b_now ≥ x</code>, then store <code>b_now − x</code>. The state update and decision must be atomic.

The bucket permits at most <code>B + rT</code> units over any interval of length <code>T</code>, subject to initialization policy. A full bucket’s burst duration at rate <code>r</code> is <code>B/r</code>; choose <code>B</code> from downstream queue/concurrency headroom, not convenience.

Use weighted tokens when work varies, but validate estimates against actual resource consumption. If clients declare their own cost, the server must constrain or recompute it.

### Leaky-bucket shaping and GCRA

A shaper schedules admitted units at a controlled departure rate rather than rejecting them immediately. This is appropriate only when a bounded delay still meets the caller deadline. Its queued bytes and wait time are backpressure state and must be capped.

The Generic Cell Rate Algorithm represents a conforming schedule with a theoretical arrival time. Each unit advances that time by an emission interval; burst tolerance allows arrivals a bounded distance before it. It stores compact state and avoids a log of timestamps, but weighted work and distributed updates still require atomic accounting.

### Fixed and sliding windows

A fixed-window counter is simple but permits a boundary burst: nearly a full allowance just before reset and another just after. A rolling log is exact for recorded events but costs memory and deletion work proportional to activity. A sliding-window counter interpolates adjacent buckets, reducing boundary error without storing every arrival.

Use window counters for contractual calendar quotas, reporting, or when their error is explicitly acceptable. Do not present one algorithm as universally “best”; state the maximum burst and approximation error the product accepts.

### Avoid client-clock authority

Use a monotonic server clock for replenishment. For distributed durable state, server-side store time or logical expiries are safer than a caller timestamp. Civil-time quota boundaries require explicit timezone and repeated/missing-hour behavior; a daily entitlement is not the same mechanism as a per-second traffic shaper.

---

## 5. Distributed Accounting

### Linearizable central decision

Every request atomically updates one authoritative counter. This gives the clearest global bound and easy revocation, but adds network latency, store throughput, and a new availability dependency. Hot tenants create hot keys even if the counter store is horizontally scalable.

Use it when the entitlement is financially or operationally strict and the decision rate fits the authority. Partition by policy key while preserving atomicity across mandatory dimensions.

### Independent local buckets

Each of <code>N</code> replicas enforces a local rate <code>r_local</code>. If every replica uses the full global rate, total admission can reach <code>N × r_global</code>, and autoscaling silently raises the limit. Dividing by an expected replica count fails during rollout, skew, and partial outage.

Independent buckets are appropriate for **per-instance self-protection**, not an exact tenant-global entitlement.

### Leased credit

A global allocator grants each enforcement point a bounded amount of spend:

1. Data plane requests a lease containing policy version, subject/resource, credit, epoch, and expiry.
2. Allocator atomically subtracts that credit from the parent budget.
3. Data plane admits locally until credit or lease time is exhausted.
4. It reports usage and requests more before depletion.
5. Expired or revoked leases cannot be reused; unused credit is reclaimed only by a protocol that prevents double spend.

If at most <code>q_i</code> unreported credit exists at enforcer <code>i</code>, crash/failover overshoot or stranded-credit error is bounded by:

> distributed error bound ≤ sum of outstanding lease credit + in-flight decision race

Smaller leases improve accuracy and revocation speed but increase allocator QPS and sensitivity to latency. Larger leases improve availability and locality but reserve more unused capacity and enlarge the error bound.

### Regional hierarchy

For global traffic, allocate global → region → cell/process. Enforce cheap local safety before spending regional/global entitlement. Rebalance using measured demand, but keep emergency reserve rather than allocating the entire parent budget.

During a partition choose explicitly:

- **fail closed:** preserve strict quota, sacrifice availability;
- **cached lease:** preserve bounded service until credit expires;
- **fail open:** preserve availability with unbounded entitlement risk;
- **restricted policy:** allow critical operations, deny optional or expensive work.

This is a product and security choice, not an implementation default.

---

## 6. Fairness, Identity, and Abuse

Rate-limit keys must follow authenticated identity. IP-only limits combine many users behind NAT, allow address rotation, and can let an attacker exhaust a victim’s shared prefix. Use IP/network reputation as one abuse signal, not the commercial tenant identity.

Hierarchical fairness prevents one subject from taking every resource:

- reserve a minimum or weighted share per class;
- permit borrowing from unused shared capacity;
- revoke borrowed credit when owners become active;
- cap expensive routes with weighted cost;
- isolate administrative and recovery traffic from public traffic.

High-cardinality keys are a denial-of-service vector against the limiter itself. Validate key length, canonicalize identity, bound inactive-state retention, aggregate unauthenticated traffic, and cap dynamic policy descriptors.

Quota enforcement is not authorization. A valid token balance never grants access to the underlying resource.

---

## 7. Response and Client Contract

For HTTP, 429 means the client exceeded a rate policy. A temporarily overloaded service may instead use a service-unavailable response according to its API contract. Include:

- a stable reason/policy code;
- whether retry is permitted;
- <code>Retry-After</code> when the server can provide meaningful guidance;
- standardized RateLimit fields where deployed and safe;
- correlation and decision IDs for support.

Remaining-credit values are observations, not reservations. Concurrent requests can spend them before the next call.

Clients must obey the attempt policy in [Retries, Timeouts, and Hedging](./10-retries-timeouts-hedging.md). A retry at exactly the reset instant can synchronize a herd; retry guidance should still be combined with client jitter and deadline checks.

---

## 8. Concrete Failure Trace: Scaling Multiplies the Limit

1. A gateway process has a local bucket allowing <code>R</code> requests/s for tenant T.
2. The fleet runs four replicas, so T can reach roughly <code>4R</code> by spreading connections.
3. A burst increases CPU and [Auto-Scaling](./08-auto-scaling.md) adds eight replicas.
4. Aggregate entitlement becomes roughly <code>12R</code> exactly while the downstream is stressed.
5. Requests slow; in-flight work rises; retries add more attempts.
6. The downstream fails despite every gateway reporting that its limiter is healthy.

The bug is a mismatch between declared global scope and process-local state. Fix it with a global authority or bounded regional/process leases. Keep a separate local self-protection bucket so a global-accounting outage cannot flood one instance.

---

## 9. Capacity and Cost Model

Let:

- <code>lambda</code>: offered requests/s;
- <code>r</code>: admitted weighted units/s;
- <code>B</code>: burst units;
- <code>S</code>: mean admitted service time;
- <code>K</code>: active subject/resource counter cardinality;
- <code>u</code>: atomic store operations per exact decision;
- <code>q</code>: average lease size;
- <code>N</code>: enforcement points.

Expected admitted concurrency is approximately <code>r × S</code> for stable traffic. A burst can add up to <code>B</code> near-simultaneous units, so downstream concurrency or queue headroom must absorb the chosen burst.

An exact central limiter needs approximately <code>r × u</code> store operations/s plus rejected-decision traffic if denials also read/update state. With leases, allocator request rate is approximately:

> allocator QPS ≈ admitted weighted units/s ÷ average lease size

but hot-key skew, unused leases, refresh-before-empty, and failover increase it.

State memory is roughly <code>K × bytes per counter</code> plus indexes, expiry structures, policy cache, and replicas. Estimate attacker-created inactive keys and cleanup cost, not only paying tenants.

Cost includes decision latency, counter storage/replication, cross-region traffic, audit retention, unused reserved credit, rejected-request CPU/TLS, and engineering for reconciliation. If the limiter call costs more than the request it rejects, add an earlier local guard.

---

## 10. Operations and Migration

### Safe policy rollout

1. Publish a versioned policy with effective time and compatibility metadata.
2. Evaluate in shadow mode and record would-allow/would-deny decisions.
3. Compare impact by tenant, route, cost, and region.
4. Enforce for a controlled cohort while retaining a kill switch.
5. Reconcile leased credit across old/new versions.
6. Expand only when rejection and downstream-goodput effects match the model.

Changing a key, cost unit, or quota window is a state migration. Dual-account old and new keys before cutover; otherwise subjects can reset usage simply by crossing the version boundary.

### Recovery

- Counter-store loss: restore durable entitlement state, then reconcile audit/usage; do not invent remaining paid quota.
- Allocator partition: use only valid cached leases and explicit emergency behavior.
- Clock anomaly: clamp negative elapsed time and alert; never refill from rollback.
- Policy-store outage: serve last verified snapshot until its safety expiry.
- Hot subject: isolate shard/key and reduce lease size or move to a dedicated authority.

---

## 11. Security and Governance

- Authenticate before charging a privileged identity; apply a cheap unauthenticated guard before expensive authentication.
- Sign or mutually authenticate lease and policy distribution.
- Prevent tenants from choosing policy descriptors or weighted cost.
- Encrypt counter, lease, policy, and audit traffic; usage reveals customer behavior.
- Separate policy authorship, emergency override, and audit permissions.
- Audit policy changes, manual credit grants, fail-open activation, key rewrites, and counter resets.
- Retain decision evidence according to billing, fraud, privacy, and dispute requirements.

Never expose a global limiter service directly to untrusted callers.

---

## 12. Observability

Measure:

- offered, admitted, rejected, degraded, and shaped weighted units;
- rejection by policy version, subject class, resource, region, and reason;
- bucket/lease utilization and outstanding credit;
- overshoot, stranded credit, reconciliation difference, and stale-policy age;
- decision latency and error by local/global path;
- counter-store QPS, hot keys, conflicts, expiry backlog, and cardinality;
- downstream goodput, latency, concurrency, and saturation alongside admission.

Avoid unbounded subject IDs in metrics labels. Put high-cardinality detail in sampled logs or an audit store.

A rising rejection rate may mean policy is working. The alert condition is a contract violation: unexpected protected-traffic rejection, error bound exceeded, stale policy, allocator exhaustion, or downstream saturation despite admission.

---

## 13. Verification

- prove the token-bucket interval bound with deterministic and randomized arrival sequences;
- race concurrent spends on the same key;
- test monotonic-clock rollback and civil-time quota transitions;
- distribute one subject across every replica and region;
- add/remove replicas during a burst and verify entitlement is unchanged;
- partition enforcers from the allocator and measure the declared error bound;
- crash an enforcer with unused and just-spent lease credit;
- roll policy/key/cost versions while traffic continues;
- generate attacker-controlled unique identities and verify bounded limiter state;
- overload the counter store and exercise fail-open/closed/restricted behavior;
- compare audit usage with authoritative billing/resource totals;
- verify rejected work is materially cheaper than admitted work.

Test the full composition with retry budgets and downstream concurrency limits. A mathematically correct bucket can still participate in an overload loop if clients ignore rejection guidance.

---

## 14. Decision Framework

| Requirement | Preferred mechanism |
|---|---|
| Per-process self-protection | Local token bucket |
| Strict global commercial quota | Linearizable authoritative counter |
| High-rate global entitlement with bounded error | Hierarchical leased credit |
| Smooth egress into a batchable dependency | Bounded shaper plus backpressure |
| Calendar usage entitlement | Versioned window/quota ledger |
| Variable request cost | Weighted units validated by server |
| Limit must follow changing service latency | Concurrency control, not a fixed rate |
| Control plane may be unavailable | Cached policy and bounded leases with declared emergency behavior |

Select the weakest coordination that still satisfies the entitlement and error contract. Add an exact global decision only where its precision is worth the latency and availability dependency.

---

## Primary References

- IETF, [RFC 1363: A Proposed Flow Specification](https://www.rfc-editor.org/rfc/rfc1363.html), for the token-bucket interval bound.
- IETF, [RFC 2697: A Single Rate Three Color Marker](https://www.rfc-editor.org/rfc/rfc2697.html).
- IETF, [RFC 6585: Additional HTTP Status Codes](https://www.rfc-editor.org/rfc/rfc6585.html), including 429.
- IETF, [RateLimit header fields for HTTP (draft-ietf-httpapi-ratelimit-headers)](https://datatracker.ietf.org/doc/draft-ietf-httpapi-ratelimit-headers/), an active Internet-Draft, not yet an RFC.
- Envoy, [Local Rate Limiting](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/other_features/local_rate_limiting).
- Google SRE, [Handling Overload](https://sre.google/sre-book/handling-overload/).
- Google SRE, [Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/).
