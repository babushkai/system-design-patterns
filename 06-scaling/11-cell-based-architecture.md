# Cell-Based Architecture and Shuffle Sharding

Cell-based architecture scales a service by repeating a bounded, independently operable slice of the stack and assigning each workload unit to one cell. Its value is failure containment: a poison request, hot tenant, bad deployment, exhausted quota, or corrupt local state should affect one bounded population instead of the whole service. Cells work only when placement, routing, global dependencies, and evacuation preserve that boundary.

This chapter owns **cell boundaries, placement and routing, tested cell limits, fleet lifecycle, cell deployment rings, evacuation, and shuffle-shard overlap mathematics**. [Multi-Tenant Isolation](./12-multi-tenancy.md) owns tenant identity, authorization, quotas, noisy-neighbor policy, and tenant lifecycle. [Multi-Region Architecture](./09-multi-region-architecture.md) owns regional authority, replication, and regional failover. [Database Sharding](./03-database-sharding.md) owns live data movement and shard fencing.

## Primary Evidence and Scope

| Primary evidence | What it establishes | Boundary |
|---|---|---|
| AWS Architecture Blog, April 2014 | Shuffle sharding maps a workload to a small overlapping subset, creating many possible isolation groups | Historical Route 53/AWS explanation; published examples are not universal retry settings |
| AWS Well-Architected, cell architecture guidance (accessed July 2026) | Cells are bounded, self-contained scale/fault units with explicit control/data planes, placement, routing, sizing, migration, deployment, and observability | AWS guidance, not proof that every service needs cells |
| Azure Architecture Center, Deployment Stamps (accessed July 2026) | Repeated independent stacks can serve assigned tenant subsets and scale by adding stamps | Provider-neutral pattern expressed through Azure terminology |

Cells, deployment stamps, scale units, and bulkheads overlap in vocabulary. Here a **cell** means a full serving unit with a declared capacity ceiling and no required synchronous dependency on another peer cell for its normal request path.

## Workload and isolation contract

Define:

| Field | Required answer |
|---|---|
| **Placement unit** | Tenant, account, workspace, user, key range, queue, or another independently movable unit? |
| **Cell boundary** | Which compute, storage, cache, queue, control, and observability components are dedicated to one cell? |
| **Authority** | Which cell owns reads/writes for one placement unit and operation class? |
| **Maximum tested envelope** | Peak requests, writes, bytes, connections, tenants, hot-key/tenant size, and recovery work? |
| **Failure containment** | Which failures must remain within one cell, and which global dependencies remain shared? |
| **Placement constraints** | Region, residency, hardware, version, isolation tier, data size, and anti-affinity? |
| **Evacuation** | Is data already available elsewhere; what can move, degrade, queue, or remain unavailable? |
| **Cross-cell behavior** | Which operations span cells, and what consistency/partial-failure semantics apply? |
| **Lifecycle** | How are cells provisioned, certified, opened, filled, drained, upgraded, and retired? |

A cell does not automatically equal one tenant. A cell may host many tenants, and a large tenant may require a dedicated cell or several explicitly partitioned placement units. That policy belongs to the tenant isolation contract; this chapter ensures the resulting placement has bounded blast radius.

## State, authority, and invariants

The global cell directory stores:

```text
placement unit -> home cell and placement generation
eligible region/residency/isolation/version constraints
cell identity, lifecycle state, software/config generation
tested capacity vector and current reserved load
migration/evacuation state and target
directory revision and audit evidence
```

Each cell owns local canonical data and serving state for its assigned units. Fleet-wide analytics, billing, search, or audit may consume asynchronous events, but a peer cell is not a hidden synchronous database replica for normal serving.

**Reference-design invariants:**

1. One placement generation names exactly one authoritative home cell for a unit and operation class.
2. A stale router or old cell cannot accept writes after authority moves to a newer generation.
3. A cell can serve its assigned workload through loss of the fleet control plane for a declared stale-state interval.
4. Peer-cell failure cannot consume another cell's reserved CPU, storage, connections, queues, or dependency quota without an explicit evacuation admission decision.
5. A cell stays below its tested workload and recovery envelope, including its largest assigned unit.
6. Global services are either outside the request's critical path, independently partitioned, or sized/failure-tested so they do not recreate global shared fate.
7. Placement respects residency, isolation, and version constraints through migration and rollback.
8. Cross-cell operations expose partial failure; they do not silently turn every cell into one distributed transaction domain.

## Data plane and control plane

~~~mermaid
flowchart LR
    C[Client]
    R[Thin cell router<br/>versioned placement cache]
    A[Cell A<br/>complete serving stack]
    B[Cell B<br/>complete serving stack]
    D[Cell C<br/>complete serving stack]
    P[(Global placement directory)]
    F[Cell fleet controller]
    E[Asynchronous fleet events]

    C --> R
    R --> A
    R --> B
    R --> D
    P --> R
    F --> P
    F --> A
    F --> B
    F --> D
    A --> E
    B --> E
    D --> E
~~~

The **data plane** resolves a placement unit, carries the placement generation, and serves entirely inside the selected cell. The **control plane** provisions cells, certifies versions/capacity, places units, orchestrates moves, and distributes directory revisions.

The router must be thinner and more reliable than the cells it protects. It authenticates the request, extracts a stable placement key, reads a versioned local directory snapshot, and forwards to the named cell. It does not fan out to “find” the tenant, perform cross-cell business logic, or select a peer by transient CPU.

If the global directory is unavailable, existing placements continue from last known good state while new onboarding/moves stop. Security revocation and completed authority transfer may require a shorter stale window than ordinary routing. An empty or unknown placement is not permission to broadcast.

## Designing a real cell boundary

A useful cell contains every resource whose failure or saturation would otherwise propagate:

- request/worker compute and local load balancing;
- authoritative database partitions or dedicated database group;
- local cache and queue capacity;
- tenant-scoped rate/admission budgets;
- cell-local configuration and secrets projection;
- cell-level metrics, logs, traces, health, and deployment controls.

Some services remain global: identity roots, placement, artifact distribution, billing ledger, fleet analytics, or public DNS. For each, ask:

1. Is it synchronous on every request?
2. Can one bad cell overload or corrupt it?
3. Does it share one connection/thread/quota pool across cells?
4. Can it return data from the wrong cell or tenant?
5. Can a control-plane outage stop existing cell traffic?

**Inference:** a “cellular” frontend backed by one globally shared write database has compute blast-radius isolation but not data or dependency isolation. Name the partial boundary honestly and plan the shared component separately.

## Placement and routing

### Explicit placement directory

Prefer stable placement IDs independent of machines and a directory mapping to cell identity. Naive `hash(unit) mod cell_count` remaps most units when cell count changes and cannot express residency, dedicated cells, version rings, or whale tenants. Stable buckets or consistent hashing reduce movement, but policy-rich cell assignment normally benefits from an explicit directory. The canonical hashing primitives are in [Partitioning Strategies](../02-distributed-databases/05-partitioning-strategies.md).

Placement is constrained bin packing across several dimensions:

```text
unit demand vector = peak CPU, writes, storage, connections, queue, recovery
cell usable vector = tested capacity - reserved failure/rollout/movement margin
```

A placement fits only if every dimension and policy constraint fits after correlated peaks. Average tenant CPU cannot offset one tenant larger than the cell's write or connection ceiling.

### Placement generations

Every route carries the directory generation or unit placement generation. During a move, the source rejects stale-generation writes after fencing; routers refresh and retry within the operation's idempotency/deadline budget. Physical copies in two cells do not mean two write authorities.

Detailed copy/catch-up/validation belongs to [Database Sharding](./03-database-sharding.md) and [Service Migration](../15-deployment/06-migration-strategies.md). The cell-specific requirement is that placement publication, source fencing, target capacity, and rollback state remain consistent.

### Cross-cell operations

Avoid request-time fan-out across every cell. Fleet search, reports, and billing usually consume asynchronous projections. If an interactive request must span cells, bound fan-out, deadline each branch, define partial results, and never hold cell-local locks while waiting on peers.

A business transaction spanning placement units may require co-placement, a saga, or a distributed transaction; cells do not make that invariant disappear.

## Cell sizing and capacity

A cell ceiling is a tested multi-resource envelope, not “up to 100 tenants.” Measure:

- per-route and hottest-unit request/write demand;
- storage occupancy, growth, compaction, backup, and repair;
- cache/queue working set and cold-start cost;
- connections, downstream pools, quotas, and control-plane cardinality;
- one-host/zone/replica loss plus deployment and migration work;
- recovery time from empty or restored cell.

Let cell workload demand vector be $D_r$, planned useful per-unit capacity be $C_r$, and required unavailable units in the scenario be $f$. For homogeneous units:

$$
N_r \ge \left\lceil \frac{D_r}{C_r} \right\rceil + f
$$

The maximum over resources binds, subject to placement. See [Capacity Planning](../01-foundations/10-capacity-planning.md) for the full model.

### Illustrative fleet calculation

Assume:

- projected peak demand is `3.6 million requests/s`;
- one certified cell serves `180,000 requests/s` while surviving its named internal failure;
- fleet keeps one empty certified cell for urgent placement/migration;
- a rollout may make one additional cell unavailable;
- existing cells are capped at 85% of the certified workload envelope to retain placement skew margin.

```text
planned usable/cell = 180,000 × 0.85 = 153,000 requests/s
serving cells = ceil(3,600,000 / 153,000) = 24
total cells with empty reserve + rollout overlap = 26
```

These are illustrative assumptions, not recommended percentages. Repeat for storage, writes, connections, and the hottest tenant; the largest requirement wins.

### Evacuation capacity

If cell $A$ has admitted demand $D_A$, evacuation destinations $i$ have safe spare capacities $s_i$, and routing/data constraints make only fraction $e_i$ usable:

$$
\sum_i s_i e_i \ge D_A
$$

Even then, data copy, cache cold start, connection reconnects, and dependency quotas may prevent immediate transfer. If authoritative data is not replicated to a target, regional or cell failure may require restore and bounded unavailability rather than fictional instant failover.

An empty “swing cell” helps only if it is continuously patched, warmed enough, quota-complete, security-certified, and tested with real placement traffic.

## Cell lifecycle

~~~mermaid
stateDiagram-v2
    [*] --> Provisioning
    Provisioning --> Certifying
    Certifying --> Open: failure and capacity gates pass
    Open --> Filling: placement admitted
    Filling --> Full: placement ceiling reached
    Open --> Draining
    Filling --> Draining
    Full --> Draining
    Draining --> Empty: units moved or retired
    Empty --> Retired
    Certifying --> Quarantined: validation fails
    Open --> Quarantined: integrity/security fault
    Quarantined --> Draining
~~~

### Provision and certify

Create a cell from one versioned infrastructure/application manifest. Certification verifies network, identity, keys, quotas, storage durability, backups, observability, synthetic transactions, capacity, and declared failures. “Resources created” is not ready.

### Fill

Place small canary units first. Admit new units while projected peak plus failure/migration margin fits every resource. Maintain a placement watermark so concurrent controllers cannot overfill from stale capacity snapshots.

### Drain and retire

Close the cell to new placements, move units through persisted workflows, fence source authority, drain connections/jobs, verify no routes/data/keys remain, and only then delete infrastructure. Emergency quarantine stops new work and chooses per-unit failover, queue, degrade, or unavailable semantics.

## Shuffle sharding

Shuffle sharding assigns each workload unit a small subset of a shared resource pool. Unlike cells, subsets overlap; unlike ordinary sharding, the number of possible isolation groups can greatly exceed the resource count.

For $N$ resources and subset size $k$, the number of distinct subsets is:

$$
\binom{N}{k} = \frac{N!}{k!(N-k)!}
$$

With $N=20$ and $k=3$, there are 1,140 distinct subsets. Two independently and uniformly assigned tenants receive the identical subset with probability $1/1{,}140$.

The overlap distribution for two subsets is hypergeometric:

$$
\Pr(J=j)=\frac{\binom{k}{j}\binom{N-k}{k-j}}{\binom{N}{k}},
\qquad \mathbb{E}[J]=\frac{k^2}{N}
$$

For $N=20$ and $k=3$, expected overlap is $0.45$ resources. This is an isolation model, not an availability proof: correlated rack/zone placement, uneven capacity, retries outside the assigned subset, and a poisoned shared dependency change the result.

**Documented, AWS 2014:** the published Route 53 explanation used card-hand combinations to show that overlapping subsets can create many more isolation groups than disjoint shards and described stateless hashing and stateful searching variants.

### Where shuffle sharding fits

Use it for stateless workers, queues, caches, rate-limit partitions, or other pools where a tenant can safely use a small subset. Choose `k` from redundancy and overload containment together:

- larger `k` gives a tenant more failure options and spreads its load wider;
- larger `k` also increases overlap and the number of resources one poison tenant can affect;
- smaller `k` improves containment but must survive resource failure and tenant peak.

Keep retries within the assigned subset; falling back to the full pool destroys isolation during the exact failure shuffle sharding targets. Place subset members across intended failure domains.

Stateful resources are harder: overlapping subsets do not define one write authority or data location. Use cells or explicit partition placement for canonical data; shuffle sharding may still isolate stateless frontends inside each cell.

## Deployment and configuration

Cells are natural deployment rings:

1. validate artifacts and schema compatibility outside serving;
2. deploy to a noncritical/canary cell;
3. observe production-shaped traffic and inject local failure;
4. advance across cells in bounded waves;
5. stop on cell-local SLO, integrity, cost, or control-plane regression;
6. keep prior version and schema compatibility until the rollback wave completes.

Do not deploy every cell from one unbounded global job or make cell startup require a newly broken control plane. Configuration is versioned per cell; fleet policy declares allowed skew. Emergency rollback remains cell-scoped.

A schema or protocol change spanning interacting cells needs compatibility across the entire mixed-version window. Cells reduce rollout blast radius; they do not remove compatibility requirements.

## Specialized failure traces

### Shared global dependency defeats containment

1. Each frontend/database stack is labeled a cell.
2. Every request synchronously calls one global entitlement service with one pool/quota.
3. One hot cell exhausts the shared service; all cells fail.

Partition or cache the dependency safely, allocate per-cell budgets, or acknowledge that the end-to-end blast radius remains global.

### Stale placement sends writes to two cells

1. Tenant 42 moves from A generation 17 to B generation 18.
2. One router misses the directory update and writes to A.
3. A still accepts because routing removal was treated as the fence.

Source and target validate placement generation at the write boundary. Directory consistency alone cannot stop a disconnected router.

### Evacuation creates a fleet cascade

1. Cell A is degraded but still serving 150k requests/s.
2. Control plane transfers all traffic immediately to B and C, each with only 40k spare.
3. Reconnects, cold caches, and retries push both beyond their envelope.
4. Health logic evacuates them in turn.

Evacuation is admission-controlled. Shed/degrade noncritical work, move bounded cohorts, and stop when destination headroom or data readiness is exhausted.

### Placement averages hide a whale tenant

Tenants are balanced by count, but one tenant consumes half a cell's write IOPS. A normal event drives it beyond the cell ceiling and affects every co-tenant. Placement uses multidimensional peak demand; dedicated or subpartitioned treatment belongs to the [multi-tenancy](./12-multi-tenancy.md) policy.

### Fleet-wide configuration bypasses cells

One malformed configuration is atomically published to every cell. Data and compute are isolated, yet every cell rejects startup or traffic simultaneously. Roll out configuration by cell ring, validate complete snapshots locally, and retain last known good.

### Shuffle fallback dissolves isolation

A tenant's three assigned workers fail under poison input. Client library retries against the full pool “for availability,” spreading the poison to all workers. Failure behavior must stay inside the subset or reject the operation.

## Security and abuse boundaries

Cell placement is sensitive security state. Authenticate and authorize placement/move operations, enforce tenant identity independently of routing keys, sign or integrity-check directory snapshots, and prevent callers from selecting arbitrary cell addresses.

Cells can support residency and dedicated-isolation policy, but infrastructure separation does not replace [tenant authorization](./12-multi-tenancy.md). Temporary copies and event streams during moves preserve encryption, retention, deletion, audit, and key-domain requirements.

Compromise of a global artifact, identity root, placement service, or fleet administrator can cross every cell. Minimize global privileges, use staged deployment, isolate credentials per cell, and test revocation/control-plane loss. Logs and metrics include cell identity without exposing tenant placement broadly.

Protect placement and shuffle algorithms from adversarial key choice. Use authenticated tenant IDs and a keyed, versioned hash where predictable subsets would enable deliberate co-location attacks.

## Observability and verification

Observe both cell and fleet:

- directory and per-unit placement generation, stale-route rejection, and unknown placements;
- demand/capacity vector per cell, hottest unit, skew, and projected exhaustion;
- cell SLO/goodput/rejection independently, not only fleet aggregate;
- shared global dependency use and per-cell budget enforcement;
- lifecycle state, certification evidence, fill/drain/move progress, and empty reserve readiness;
- evacuation feasible capacity, reconnect/cache warm-up, and admission decisions;
- software/config/schema version by cell and rollout wave;
- shuffle subset distribution, overlap, member failure, and fallback violations;
- cross-cell fan-out and asynchronous projection lag.

Tests include deterministic placement across implementations, directory snapshot gaps, stale routers, source fencing, one-cell overload/failure, poison tenant, shared-dependency saturation, empty-cell recovery, evacuation at peak, controller outage, mixed-version cells, restore, and shuffle overlap under correlated failures.

Load-test one cell past every resource knee with the largest supported unit and recovery work. Then repeat across several cells to expose supposedly global limits. The value of a cell ceiling is that it can be tested regularly; an untested number is only a placement guess.

## Decision framework

1. Which placement unit can move and fail independently without cross-cell transactions?
2. What complete resource set forms one cell, and which global dependencies still create shared fate?
3. What measured multidimensional envelope and largest-unit limit define a full cell?
4. How is placement versioned and fenced at the authoritative write path?
5. Can existing traffic continue through control-plane loss without defeating revocation or moves?
6. Where does reserve capacity live, and is cell evacuation feasible with data, caches, connections, and quotas?
7. Which cross-cell features can use asynchronous projections or explicit partial results?
8. Does shuffle subset size balance redundancy against overlap and poison spread?
9. Are deployments, configurations, identities, and credentials truly cell-scoped?
10. Can the team provision, certify, load-test, drain, restore, and retire cells repeatedly?

## Primary references

- [AWS Well-Architected, *Reducing the Scope of Impact with Cell-Based Architecture*](https://docs.aws.amazon.com/wellarchitected/latest/reducing-scope-of-impact-with-cell-based-architecture/welcome.html)
- [AWS Architecture Blog, *Shuffle Sharding: Massive and Magical Fault Isolation* (April 2014)](https://aws.amazon.com/blogs/architecture/shuffle-sharding-massive-and-magical-fault-isolation/)
- [AWS Builders' Library, *Workload isolation using shuffle-sharding*](https://aws.amazon.com/builders-library/workload-isolation-using-shuffle-sharding/)
- [Microsoft Azure Architecture Center, *Deployment Stamps pattern*](https://learn.microsoft.com/en-us/azure/architecture/patterns/deployment-stamp)
- [Microsoft Azure Architecture Center, *Multitenant control-plane considerations*](https://learn.microsoft.com/en-us/azure/architecture/guide/multitenant/considerations/control-planes)
