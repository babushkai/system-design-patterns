# Multi-Region Architecture: Authority, Routing, and Failover

## TL;DR

Multi-region design is a set of **authority and data-consistency decisions**, not a map with two identical boxes. For each data class, choose who may write, how replicas acknowledge and order changes, which reads may be stale, how users route to their data, and what happens to the unreplicated tail during failure.

Failover is a distributed state transition: detect the fault, serialize the decision, fence old authority, prove/promote a usable replica, shift traffic, operate within pre-existing capacity, and reconcile divergent or lost work. If the surviving region needs a successful provisioning or configuration mutation during the incident, the design is not statically stable.

Do not hard-code “intercontinental latency” or symmetric utilization folklore. Measure the actual paths and derive commit latency, RPO, RTO, and capacity headroom from the chosen quorum, replication lag, failure scope, and traffic distribution.

---

## 1. Regional Contract per Data Class

One product commonly uses several postures. Define this table for user records, ledger entries, sessions, blobs, search, queues, configuration, analytics, and derived caches separately:

| Field | Required answer |
|---|---|
| **Driver** | Regional survivability, user latency, residency/sovereignty, capacity, or business continuity. |
| **Authority** | Single primary, home region per key, quorum leader, or true multi-writer merge. |
| **Replication** | Synchronous quorum, asynchronous log, snapshot/backup, or derived rebuild. |
| **Consistency** | Linearizable/external, causal/session, monotonic, bounded-staleness, or eventual. |
| **Routing** | How edge, application, and storage locate authority; behavior for stale maps. |
| **Failure scope** | Zone, region, network partition, control-plane outage, corruption, bad deploy, or credential loss. |
| **RPO** | Maximum lost committed business work, including derived/event paths. |
| **RTO** | Maximum time until the required service level, not merely DNS change. |
| **Capacity** | Normal, failure, rollout, repair, and backlog-drain headroom. |
| **Residency** | What may be stored, processed, logged, backed up, or administered in each jurisdiction. |
| **Reconciliation** | Lost-tail, duplicate, conflict, and failback procedure. |

### Invariants

1. At most one write authority exists for single-writer data in any reachable epoch.
2. Stale routing cannot grant write authority; storage validates epoch/fencing state.
3. Reads meet their declared consistency even during routing changes.
4. Surviving regions serve the admitted failure load without just-in-time control-plane success.
5. RPO includes every accepted write path, queue, cache, index, and event—not only the primary database.
6. Promotion and failback are idempotent, audited state machines with explicit abort points.
7. Residency and access policy remain valid during failover and operator access.
8. Returned regions cannot rejoin or serve divergent state until fenced and reconciled.

---

## 2. Data Plane and Control Plane

~~~mermaid
flowchart TB
    U["Users / clients"]
    GR["Global routing<br/>coarse region choice"]

    subgraph RA["Region A data plane"]
        EA["Edge / stateless service"]
        DA[("Data replica<br/>authority epoch E")]
        QA[("Queue / derived views")]
        EA --> DA
        EA --> QA
    end

    subgraph RB["Region B data plane"]
        EB["Edge / stateless service"]
        DB[("Data replica")]
        QB[("Queue / derived views")]
        EB --> DB
        EB --> QB
    end

    subgraph CP["Failure-isolated control plane"]
        PM[("Partition/home map")]
        AUTH["Promotion authority<br/>quorum or governed workflow"]
        CFG[("Versioned config, keys,<br/>policy and routing intent")]
        AUD["Audit + reconciliation"]
    end

    U --> GR
    GR --> EA
    GR --> EB
    DA <-->|replication protocol| DB
    PM -.route/proxy.-> EA
    PM -.route/proxy.-> EB
    AUTH -.epoch/fence.-> DA
    AUTH -.epoch/fence.-> DB
    CFG -.cached snapshots.-> RA
    CFG -.cached snapshots.-> RB
    RA -.evidence.-> AUD
    RB -.evidence.-> AUD
~~~

The data plane should operate from locally cached, versioned maps/configuration through a control-plane impairment. Promotion is different: it changes authority and must use a serialization mechanism that survives the declared failure domains.

---

## 3. Topology Choices

### Active-passive

One region serves writes and usually reads; another receives replication or backups. This minimizes concurrent-write complexity but standby readiness decays unless exercised.

Decide whether passive means:

- hot and receiving a small safe traffic slice;
- warm with provisioned capacity but cold processes/caches;
- cold infrastructure/data requiring restore.

Those modes have different RTO and evidence. A backup that restores successfully is not a hot replica; a replica that has data but insufficient quota/configuration is not failover capacity.

### Active stateless edge, single data authority

Serve TLS, static content, and cacheable work regionally; proxy authoritative operations to one data region. This improves edge latency but cross-region application chatter can dominate. Collapse request fan-out so a user operation pays as few wide-area round trips as possible.

### Read-local, write-to-home

Each region serves a replica; writes route to a primary/home. The contract must address:

- read-your-writes after an asynchronous update;
- monotonic reads as the user changes region;
- maximum acceptable replica lag;
- behavior when local replica is stale or unavailable.

Solutions include a session/causal token, minimum source position, temporary routing to authority, or explicit stale reads. Sleeping “for replication” is not a consistency protocol.

### Partitioned active-active

Assign each key/tenant/jurisdiction one write home. Regions can write different partitions concurrently while each key remains single-writer. The partition map is critical control state.

Re-homing is a migration:

1. Stop or version-gate old-home writes.
2. Establish a source cut and copy/catch up state.
3. Transfer authority with a higher epoch.
4. Update routing.
5. Retain redirects/fences for stale clients.
6. Reconcile and retire the old copy.

Cross-home transactions become distributed workflows or require a global transaction system.

### Concurrent multi-writer

The same logical key accepts independent regional writes. This requires a merge algebra, causality/version tracking, and product behavior for conflicts. Last-writer-wins can discard a valid concurrent change; clock timestamps alone do not establish causality.

Use multi-writer only where conflicts are impossible by invariant or merge naturally. See [Multi-Leader Replication](../02-distributed-databases/02-multi-leader-replication.md) and [Conflict Resolution](../02-distributed-databases/04-conflict-resolution.md) for canonical algorithms.

---

## 4. Replication and Consistency

### Synchronous quorum

A write is acknowledged after the protocol reaches the required replicas/quorum. Its latency lower bound includes network delay to the slowest replica required on the critical path plus local processing, serialization, queueing, and commit work.

Measure actual region-to-region latency and loss under normal and impaired routes. Replica placement determines which failures preserve a quorum and which regions pay wide-area latency.

Synchronous replication can provide RPO zero within its acknowledged-write model, but not protection from:

- application-level dual writes;
- correlated logical corruption;
- compromised credentials;
- missing derived events;
- clients that saw an ambiguous timeout;
- backups with different retention.

### Asynchronous replication

Authority acknowledges before the remote replica durably applies the change. This lowers write latency and permits regional operation but creates a non-zero failure window.

Track:

- source commit coordinate;
- remote received/durable/applied coordinates;
- bytes/transactions and wall-clock lag;
- log retention headroom;
- large transaction and schema effects.

Wall-clock lag is not a precise count of lost business operations. RPO on promotion is the authoritative committed set minus the safely promotable set, evaluated in source coordinates and reconciled with accepted responses.

### Derived and asynchronous state

Inventory queues, outboxes, caches, search, object storage, feature flags, secrets, scheduled jobs, analytics, and CDC. A database may fail over correctly while:

- a regional queue replays duplicate commands;
- an outbox tail did not replicate;
- cache invalidation routes to the old region;
- a singleton scheduler runs in both;
- encryption keys/config are stale;
- derived indexes lag behind the promoted primary.

Give each a source-of-truth and rebuild/replay boundary.

---

## 5. Authority, Epochs, and Fencing

Failure detectors cannot distinguish a dead region from an isolated but running region. Therefore “cannot reach primary” is insufficient authority to promote.

A promotion authority can be:

- a consensus quorum across independent failure domains;
- a storage/database protocol with built-in leader election and quorum;
- a governed human workflow backed by an atomic authority record.

It issues a monotonically increasing epoch/term. Every write reaches storage with the current epoch; storage rejects stale epochs. Revoking DNS, load-balancer membership, or application credentials helps, but fencing at the authoritative write boundary closes the split-brain race.

### Promotion state

Record:

- incident/failure evidence;
- source and candidate replica coordinates;
- current and proposed epoch;
- fence acknowledgement;
- expected RPO and lost-tail procedure;
- approver/quorum;
- routing and capacity status;
- rollback/abort condition.

If the outcome of an authority write is ambiguous, read/resolve current epoch before retrying. Two independent automation systems must not both “helpfully” promote.

---

## 6. Routing: Reachability Is Not Authority

### DNS and global load balancing

DNS-based steering is affected by authoritative TTL, recursive resolver behavior, client/OS/application caches, connection reuse, and health-evaluation delay. Measure end-to-end convergence; changing one record is not the RTO.

Anycast/BGP can route to a topologically reachable edge quickly, but the nearest edge may not hold the user’s authority or data. Network convergence is not application/data failover.

A global load balancer can shift new traffic, while existing TCP/QUIC sessions and long streams remain attached. Define drain/reset behavior and client reconnection.

### Application routing

Carry a stable tenant/key → home mapping with version/epoch. A request landing in the wrong region can:

- proxy once to the home;
- redirect a trusted client;
- serve a permitted local replica;
- fail explicitly if residency/consistency forbids crossing.

Avoid repeated cross-region hops caused by service-by-service “nearest” routing. Trace region transitions.

### Health signals

Health must represent the user path:

- regional edge reachability;
- application goodput/latency;
- dependency and data-authority readiness;
- replication/promotability;
- capacity under shifted traffic.

A shallow endpoint can route traffic into a region whose database is read-only or whose config is stale. Use multiple signals and damp gray-failure routing changes to avoid oscillating the world between partially degraded regions.

---

## 7. Failover State Machine

1. **Detect and scope.** Distinguish instance/AZ failure, region isolation, dependency failure, bad deployment, and data corruption.
2. **Freeze unsafe automation.** Ensure one failover authority and stop conflicting deploy/scale/routing changes.
3. **Check candidate.** Data coordinate, schema, keys, config, capacity, queues, and dependency health.
4. **Fence old authority.** Advance epoch/revoke write path at storage; record evidence.
5. **Promote.** Make candidate writable or activate the correct regional home map.
6. **Shift gradually when possible.** Verify goodput, tail latency, error, and downstream capacity.
7. **Shed/degrade.** Apply failure-mode admission if surviving capacity is below full demand.
8. **Reconcile.** Resolve ambiguous/lost writes, duplicate commands, derived lag, and client sessions.
9. **Stabilize.** Protect the surviving region from repair/replay traffic.
10. **Fail back as a separate migration.** Resynchronize, fence, transfer authority, shift, and observe.

### RTO decomposition

> RTO = detection + decision + fencing + data readiness/promotion + routing convergence + application warm-up + validation

These phases can overlap only when their safety dependencies permit. Measure each distribution during drills. “DNS changed in one minute” does not show service recovery if data promotion or cache/config warm-up took longer.

### RPO evidence

For asynchronous replication:

> unprotected tail = source committed coordinate − candidate durable/applied coordinate

Translate that native difference into business operations through commit/audit logs. Client timeouts create ambiguous outcomes even if storage RPO is zero; idempotency and reconciliation remain necessary.

---

## 8. Static Stability and Capacity Headroom

Let:

- <code>L_i</code>: normal admitted load served by region <code>i</code>;
- <code>C_j</code>: tested safe useful capacity of surviving region <code>j</code>;
- <code>F</code>: failed-region set;
- <code>h</code>: retained operational headroom for variance/repair;
- <code>L_shed</code>: load intentionally rejected/degraded during failure.

For a scenario:

> sum of surviving capacity × (1 − h) ≥ total offered load − L_shed

Apply placement and dependency constraints: total compute is irrelevant if data authority, egress, queue partitions, database connections, or a third-party regional quota caps goodput.

Symmetric two-region active-active systems often imply substantial idle headroom for loss of either region, but the exact utilization follows the traffic/capacity distribution and shedding contract. Derive it; do not copy a fixed percentage.

### Backlog and replay

If failure creates backlog <code>B</code>, live admitted arrival <code>lambda</code>, and safe completion rate <code>mu</code>:

> drain time = B / (mu − lambda), requiring mu > lambda

Reserve bandwidth/concurrency for live traffic and throttle replication repair, cache warm, reconciliation, and replay. Recovery jobs can re-outage the promoted region.

### Cost

Include:

- standby/active compute and database replicas;
- synchronous/quorum overhead;
- cross-region replication and request egress;
- global routing/control services;
- reserved quota and unused failure headroom;
- duplicate regional state and derived systems;
- drills, reconciliation, and operator complexity.

Compare against a multi-zone single-region design and a stated business loss. “Multi-region” is not automatically the optimal availability investment.

---

## 9. Concrete Failure Trace: Automated Split Brain

1. A fiber/control-plane partition isolates regions A and B; both data planes still serve local users.
2. B’s automation cannot reach A and declares it failed.
3. A separate automation in A sees B fail and reaches the same conclusion.
4. Both promote local replicas because reachability is treated as authority.
5. Global routing is inconsistent across resolvers/edges; both accept writes for the same keys.
6. Connectivity returns with two committed histories and no deterministic business merge.

The trigger was a network partition; corruption came from the authority protocol. Fix with a quorum/governed serialization point, monotonically increasing epochs enforced by storage, a failure mode that sacrifices writes when authority cannot be proven, and one audited promotion workflow.

---

## 10. Residency, Privacy, and Security

Residency applies to:

- primary and replicas;
- backups, logs, queues, CDC, search and analytics;
- caches and temporary spill;
- observability payloads and support tooling;
- encryption keys and operator access locations;
- disaster-recovery copies and repair exports.

Treat jurisdiction/home as a partition attribute enforced by routing and storage policy. During failover, do not route protected data to an otherwise healthy forbidden region. Define a compliant degraded mode.

Security requirements:

- independent regional credentials and blast radii;
- signed/versioned routing and home maps;
- least privilege for promotion/fencing;
- multi-party approval where business risk requires it;
- key availability without unsafe global key replication;
- tamper-evident audit of authority changes;
- protection against routing/BGP/DNS/config compromise;
- break-glass access that expires and is reviewed.

Availability design must include malicious and accidental control-plane actions, not only power loss.

---

## 11. Operations, Migration, and Failback

### Introducing a region

1. Classify data and choose posture/authority per class.
2. Provision capacity, quota, config, keys, observability, and dependencies.
3. Seed data from a consistent snapshot and catch up replication.
4. Verify source/candidate coordinates and derived systems.
5. Run shadow reads and compare.
6. Send a small traffic slice without changing write authority.
7. Enable regional reads or partitioned writes by cohort.
8. Drill isolation and evacuation before declaring the region protective.

### Config and software rollout

Use cells/cohorts so one bad global configuration does not fail every region. A multi-region deployment with simultaneous global rollout has correlated software risk. Preserve a known-good regional control/data-plane version through high-risk changes.

### Returned region

Do not let it rejoin automatically. It may contain:

- stale authority epoch;
- accepted local writes not in the promoted history;
- old queue leases/messages;
- cold or divergent caches/indexes;
- outdated credentials/config.

Fence, snapshot evidence, reconcile/reseed, catch up, validate, then reintroduce as a follower before any authority transfer.

---

## 12. Observability

By region and data class:

- original/admitted/completed traffic and goodput;
- client→edge and every cross-region hop latency;
- routing decision, home-map version, wrong-region/proxy rate;
- data authority, epoch/term, quorum and fence state;
- source/received/durable/applied replication coordinates and lag;
- read staleness/session-consistency fallbacks;
- ready capacity, failure headroom, quota, and downstream constraints;
- queue age/backlog, repair/replay/drain rates;
- config/key/schema drift;
- failover phase duration and RPO reconciliation;
- residency-policy denies and cross-border flow.

Use external probes from multiple networks, but correlate them with data-plane goodput. A region can be reachable yet unable to serve correct data.

---

## 13. Verification and Game Days

- isolate one region’s data plane while its control plane remains reachable, and vice versa;
- partition regions without powering either off;
- inject gray latency/loss on one direction only;
- fail authority/quorum components and make promotion outcome ambiguous;
- prove stale epochs are rejected at storage;
- shift traffic with stale DNS and long-lived connections still present;
- fail over with maximum measured replication lag and reconcile the exact tail;
- run at failure-load capacity while a deployment and repair job compete;
- test local reads immediately after writes and after region movement;
- fail every derived system: queue, outbox, cache, index, scheduler, secrets;
- return the old region with divergent writes and execute rejoin/failback;
- exercise residency restrictions during evacuation;
- restore from backup when replicas contain logical corruption;
- measure RTO phases and RPO in business operations.

A diagram review is not evidence. Run the state machine with real traffic and constrained operator access.

---

## 14. Decision Framework

| Requirement | Direction |
|---|---|
| Survive machine/AZ failure only | Multi-zone single region may be sufficient |
| Region outage with simplest write semantics | Active-passive with drilled promotion |
| Global low-latency reads, centralized writes | Read-local/write-to-home plus session consistency |
| Regional writes with no same-key conflict | Home-region partitioning |
| Same key must accept disconnected writes | Explicit multi-writer merge/CRDT semantics |
| RPO zero across region loss | Synchronous quorum whose placement survives that loss |
| Lowest local write latency | Regional authority plus async replication and explicit non-zero RPO |
| Residency forbids replication | Jurisdiction sharding and compliant degraded mode |
| No budget for standby/headroom/drills | Do not claim region-failure availability |

Choose posture per data class. Start from authority and failure semantics, then select routing and infrastructure that implement them.

---

## Primary References

- James C. Corbett et al., [Spanner: Google’s Globally-Distributed Database](https://research.google/pubs/spanner-googles-globally-distributed-database-2/).
- Giuseppe DeCandia et al., [Dynamo: Amazon’s Highly Available Key-value Store](https://www.amazon.science/publications/dynamo-amazons-highly-available-key-value-store).
- AWS Builders’ Library, [Static Stability Using Availability Zones](https://aws.amazon.com/builders-library/static-stability-using-availability-zones/).
- AWS Builders’ Library, [Minimizing Correlated Failures in Distributed Systems](https://aws.amazon.com/builders-library/minimizing-correlated-failures-in-distributed-systems/).
- Google Cloud Architecture Center, [Architecting Disaster Recovery for Cloud Infrastructure Outages](https://cloud.google.com/architecture/disaster-recovery).
- Martin Kleppmann, [Designing Data-Intensive Applications](https://dataintensive.net/), chapters on replication, partitioning, and consistency.
