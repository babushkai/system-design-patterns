# Horizontal vs Vertical Scaling

Scaling is the deliberate change of a system's useful-capacity envelope. “Scale up” and “scale out” describe actuators, not outcomes: a larger machine can remain blocked on a serial lock, and ten more application replicas can overload a database, connection limit, or shared cache. Start with the workload, identify the limiting resource and coordination path, then design a reversible transition that preserves authority while capacity changes.

A scaling plan must define **scaling dimensions, bottleneck models, and safe manual or planned scale transitions**. [Auto-Scaling](./08-auto-scaling.md) owns metric selection, delayed feedback, hysteresis, stabilization, and controller mechanics. [Database Sharding](./03-database-sharding.md) owns the operational lifecycle for distributing application data, while [Partitioning Strategies](../02-distributed-databases/05-partitioning-strategies.md) owns the underlying key-to-partition primitives.

## Primary Evidence and Scope

| Primary evidence | What it establishes | Boundary |
|---|---|---|
| Amdahl, AFIPS 1967 | For fixed work, the serial fraction bounds speedup from parallel resources | A model, not a capacity forecast without measured fractions |
| Little, *Operations Research* 1961 | Under stated steady-state assumptions, mean in-system work equals throughput times mean residence time | Means do not describe bursts or tail latency |
| Gunther, 2008 | A rational scalability model can represent contention and coordination/coherency penalties that eventually reduce throughput | Fit to controlled measurements; coefficients are not architectural constants |
| Dean and Barroso, *The Tail at Scale* (2013) | Fan-out makes rare component tails common at request level; large services need tail-tolerant techniques | Historical Google examples, not universal latency numbers |
| Verma et al., Borg paper (EuroSys 2015) | Large-scale scheduling uses admission, resource estimation, priorities, and overcommit rather than equating requested replicas with useful capacity | Historical Borg design |
| Google SRE book (2016) | Overload, retries, load shifts, and simultaneous change can produce cascading failure | Operational guidance, not a copied threshold policy |

## Scaling contract

A scaling proposal is incomplete until it states:

| Field | Required answer |
|---|---|
| **Objective** | Which throughput, latency, queue-age, availability, recovery, or cost target must improve? |
| **Workload unit** | Request, event, byte, query, model token, connection, tenant, or batch job? |
| **Cost distribution** | Mean, high percentile, hottest key/tenant, read/write mix, fan-out, and burst envelope? |
| **Limiting resource** | CPU time, memory capacity/bandwidth, storage IOPS/throughput, network, lock, quota, or dependency? |
| **Scalable unit** | Thread, process, container, VM, node, replica, partition, cell, region, or algorithm? |
| **State class** | Durable authority, replicated state, rebuildable projection, cache, local session, or transient in-flight work? |
| **Transition** | Provision, copy/warm, validate, admit, transfer/fence, drain, retire, and rollback phases? |
| **Failure target** | Which host, zone, rollout, dependency, or region failure must the new shape survive? |
| **Economic boundary** | Full cost including headroom, licenses, transfer, storage duplication, and operator effort? |

Scaling should improve **goodput** (valid work completed within its contract), not merely requests accepted, threads created, or CPU consumed.

## State, authority, and invariants

Capacity may be fungible; authority rarely is. A stateless replica still holds connections and in-flight work. A stateful replica may own a lease, log position, partition, or local durable bytes. Separate:

- **canonical state:** the durable record whose acknowledged updates must survive;
- **authority metadata:** which instance or partition may accept a write at an epoch;
- **replicated state:** copies with explicit lag and promotion semantics;
- **derived state:** indexes and materialized views that can be rebuilt from authority;
- **warm state:** caches, JIT code, connection pools, and page cache that affect useful capacity;
- **transient state:** requests, locks, sessions, and work leases that need drain or recovery.

**Reference-design invariants:**

1. Adding capacity does not create a second unfenced writer for the same authority domain.
2. Removing capacity stops new admission before terminating owned or in-flight work.
3. Only ready, warmed, routed, and dependency-connected units count as useful capacity.
4. Fleet expansion cannot silently multiply a global connection, retry, poll, or rate budget.
5. The remaining fleet meets admitted demand plus the named failure and rollout margin.
6. A failed transition has a defined data and routing rollback path, not only an infrastructure rollback.
7. Metrics remain comparable across old and new shapes; otherwise the transition cannot be judged.

## Scaling dimensions

Horizontal and vertical are two axes in a larger design space.

| Dimension | Changes | Gains | New limit or risk |
|---|---|---|---|
| **Vertical** | CPU, RAM, memory bandwidth, local storage, NIC, or accelerator per unit | More capacity without distributing authority | Hardware ceiling, restart/migration, larger failure domain, NUMA and contention |
| **Horizontal replication** | More interchangeable compute/read replicas | Parallel independent work and failure redundancy | Load distribution, shared dependencies, replicated caches/connections |
| **Horizontal partitioning** | More independent authority domains | Write/storage scale when work names a partition | Routing metadata, skew, cross-partition work, resharding |
| **Functional decomposition** | Separate workload classes or services | Scale expensive paths independently and isolate overload | Network hops, contracts, distributed transactions, operator surface |
| **Geographic placement** | More regions/edges | Latency, jurisdiction, and regional resilience | Replication delay, failover authority, data-transfer cost |
| **Temporal smoothing** | Queue, batch, cache, or precompute | Absorb bursts and improve utilization | Staleness, queue age, storage, delayed failure |
| **Algorithmic/data-model change** | Less work per operation | Can dominate infrastructure scaling | Product semantics, migration, verification complexity |

“Stateless” means no durable request authority is local; it does not mean the instance is instantly fungible. Cold caches, connection pools, JIT compilation, model weights, and in-flight work can make a nominally ready replica harmful during its first minutes.

## Bottleneck model before actuator choice

For each request class `j`, measure service demand on resource `r`, `d[j,r]`, and arrival rate `λ[j]`. The mean demand on that resource is:

```text
D[r] = sum over j of λ[j] × d[j,r]
```

With `N` equivalent units, per-unit capacity `C[r]`, and planned utilization `u[r]`, a necessary resource bound is:

```text
N >= D[r] / (C[r] × u[r])
```

Compute it for CPU, memory, disk IOPS, disk throughput, network packets and bytes, accelerators, connections, and downstream quotas; the largest bound wins. This is necessary, not sufficient: skew, correlated failure, queues, and serial work reduce effective capacity.

### Concurrency and queueing

**Documented model, Little 1961.** Under the paper's steady-state assumptions:

```text
mean in-flight work L = throughput λ × mean residence time W
```

At `20,000 requests/s` and `150 ms` mean end-to-end residence, roughly `3,000` requests are in the system. If a dependency slowdown doubles residence time while arrival rate remains admitted, concurrency doubles even before throughput grows. That consumes memory, threads, sockets, and pool slots; it is why latency is an early capacity signal and an unbounded queue is not capacity.

Use [Capacity Planning](../01-foundations/10-capacity-planning.md) for utilization and tail models, and [Backpressure](./07-backpressure.md) for keeping offered work bounded while a transition catches up.

### Serial work and coordination

**Documented model, Amdahl 1967.** For a fixed task with parallel fraction `p`, the model gives idealized speedup on `N` processors:

```text
S(N) = 1 / ((1 - p) + p/N)
```

If 5% is serial, unlimited processors cannot exceed 20× speedup. In services, the serial component may be a lock, one leader, one ordered log, a hot key, schema coordination, or a dependency with fixed quota. Measure it rather than inferring it from low average CPU.

**Documented model, Gunther 2008.** The normalized Universal Scalability Law adds empirical contention and coherency terms:

```text
C(N) = N / (1 + α(N - 1) + βN(N - 1))
```

`α` captures contention/serialization and `β` captures pairwise coordination cost in the model. A fitted curve that peaks and falls is evidence that more units increase shared work. It does not identify the mechanism; profiles, lock metrics, traces, and network/storage evidence must do that.

### Fan-out and tails

**Inference from the probability model.** If a request requires all `k` independent branches whose latency CDF is `F(t)`, the optimistic probability that all finish by `t` is `F(t)^k`. With 100 branches each independently within a deadline 99.9% of the time, the request succeeds within it only about `0.999^100 ≈ 90.5%` of the time. Real branches share queues and networks, so independence can be optimistic. Scaling by increasing fan-out may raise throughput while destroying request-tail latency.

## Vertical scaling

**Reference design.** Vertical scaling changes capacity within one authority or failure unit. It is often the best first move when:

- the software cannot yet partition safely;
- the working set benefits directly from a larger memory/page cache;
- a single-threaded or latency-sensitive path benefits from faster cores;
- local joins and transactions are more valuable than distributed parallelism;
- the larger shape buys enough runway for a planned architectural migration.

Its simplicity is conditional. A 4× larger machine is not a 4× faster system if storage, memory bandwidth, a lock, or downstream quota remains fixed. More cores can expose NUMA distance and synchronization. More RAM lengthens restart, checkpoint, backup, and failover. A larger node also makes one failure remove a larger fraction of fleet capacity.

**Reference design: replica-first vertical transition for durable state:**

1. provision the larger shape as a new replica rather than resize the sole authority in place;
2. copy from a consistent checkpoint and catch up the change log;
3. run checksums, query shadows, and performance tests while non-authoritative;
4. transfer authority through a fenced lease/term change;
5. retain the old node as rollback capacity until new backups and recovery evidence exist;
6. drain and retire only after the rollback horizon.

When no replica path exists, the resize is a maintenance event with explicit downtime, backup/restore time, and abort points, not an instantaneous capacity toggle.

## Horizontal scaling

### Independent compute and replicated reads

**Reference design.** Horizontal replication works when an operation can execute on any member without violating authority. The data plane must distribute the actual cost, not only request count; see [Load Balancing](./01-load-balancing.md).

Shared state does not disappear when application servers are replicated. Externalizing sessions moves availability and capacity to the session store. Adding read replicas shifts read load but does not increase one leader's write/commit capacity, and stale reads need an explicit consistency contract. Adding consumers helps only if work can be partitioned and ordering is not global.

Every replica can multiply hidden budgets:

```text
total possible database connections = replicas × pool size per replica
total polling rate = replicas × poll rate per replica
total retry rate = original attempts × attempts allowed by each layer
```

Define global budgets and allocate shares; do not let instance-local defaults become fleet-wide policy.

### Partitioned authority

Write and storage scale usually require more independent authority domains, not merely more copies. That introduces routing, skew, cross-partition operations, and online movement. Partition addition is one scaling dimension; [Database Sharding](./03-database-sharding.md) covers tenant moves, dual-routing, backfill, cutover, and rollback, while [Partitioning Strategies](../02-distributed-databases/05-partitioning-strategies.md) covers hash, range, and directory primitives.

### Cells and failure containment

One ever-larger homogeneous fleet can make every controller, dependency, and rollout global. [Cell-Based Architecture](./11-cell-based-architecture.md) repeats a bounded stack and assigns tenants to cells, trading spare-capacity overhead and cross-cell complexity for a capped blast radius. Cells are horizontal scale units only when the placement service and global dependencies remain simpler and safer than the cells they coordinate.

## Safe scale transitions

Treat capacity change as a versioned state machine:

~~~mermaid
stateDiagram-v2
    [*] --> Planned
    Planned --> Provisioned
    Provisioned --> Warmed
    Warmed --> Admitted
    Admitted --> Stable: SLO and dependency gates pass
    Admitted --> RolledBack: gates fail
    Stable --> Draining: scale in / replacement
    Draining --> Transferred: work and authority fenced
    Transferred --> Retired
    RolledBack --> Retired
~~~

### Scale-out protocol

1. **Plan:** recalculate every resource and dependency budget at the future fleet size, including rollout overlap.
2. **Provision:** create instances with the intended binary, configuration, identity, placement, and quotas.
3. **Warm:** load code/data, establish pools, and validate representative operations without full traffic.
4. **Admit gradually:** increase routing weight while comparing old and new cohorts by workload class.
5. **Stabilize:** observe a full demand cycle and the named failure; do not call creation success.
6. **Rollback:** stop new admission, drain, and remove the cohort without leaving orphan state or multiplied budgets.

### Scale-in protocol

1. prove the remaining fleet has capacity for demand plus failure and rollout headroom;
2. stop new requests, connections, jobs, and leases to the target;
3. transfer durable ownership with a new epoch and fence the old owner;
4. drain or recover in-flight work within a bounded deadline;
5. verify no directory, queue, replica set, or client still references the target;
6. retire resources and reclaim credentials, storage, addresses, and reservations.

The transition invariants remain the actuator's safety contract; [Auto-Scaling](./08-auto-scaling.md) adds delayed metrics, stabilization, and controller-conflict handling.

## Capacity and cost example

Consider an API with an illustrative peak of `48,000 requests/s` and measured CPU demand of `3 ms/request` at the production request mix:

```text
CPU demand = 48,000 × 0.003 = 144 CPU-seconds/second
useful capacity of one 16-core node at 60% planned CPU = 9.6 CPU-seconds/second
healthy-fleet minimum = ceil(144 / 9.6) = 15 nodes
one-node-failure minimum: (N - 1) × 9.6 >= 144, so N = 16
```

Now include the dependency. If each request makes `0.55` database queries on average, offered database load is `26,400 queries/s`. If the database's tested safe capacity is `20,000 queries/s`, sixteen or sixty application nodes cannot meet the contract. Reduce queries per request, cache safely, change the data model, or scale the database authority before admitting that peak.

For state movement, copying `2 TiB` at a foreground-safe effective rate of `80 MiB/s` has a lower bound of about `7.3 hours`:

```text
copy lower bound = 2,097,152 MiB / 80 MiB/s = 26,214 s ≈ 7.3 h
```

The real transition also catches writes, verifies checksums, and may be throttled during peaks. Provision duplicate storage and network for that entire interval plus rollback.

Compare alternatives with a workload-normalized cost model:

```text
cost per useful request =
  (compute + storage + transfer + licenses + observability
   + failure/rollout headroom + migration amortization)
  / good requests completed within SLO
```

Vertical price curves, spot availability, and transfer tariffs are volatile. Insert current provider quotes only in a decision record with region, date, commitment, and failure assumptions; do not bake them into an architectural rule.

## Failure traces

### Application scale-out causes a database connection storm

1. Forty application replicas each permit 100 database connections: a possible 4,000 connections.
2. A rollout temporarily doubles replicas before old ones drain.
3. New processes eagerly fill pools, exceeding the database's connection and memory budget.
4. Queries slow, request concurrency rises, health checks fail, and the load balancer shifts traffic into a cascade.

Allocate a global connection budget, start pools lazily, cap rollout overlap, and gate readiness on dependency capacity, not merely process startup.

### More workers reduce throughput

1. A job fleet doubles workers against one metadata lock or ordered commit path.
2. Lock wait and invalidation traffic rise faster than useful work.
3. CPU stays busy, but completion throughput falls and retries add contention.

Sweep throughput over worker count, fit only as a diagnostic, and profile the serialized path. The remedy may be batching or authority partitioning, not another worker increase.

### Cold replicas steal traffic from warm replicas

1. A scale-out adds nominally ready instances.
2. Round robin gives them a full share before page cache, JIT, model, or pools are warm.
3. Their latency triggers retries and passive ejection.
4. Traffic returns to the old fleet plus retry load, leaving less capacity than before scale-out.

Separate process health from useful readiness, warm representative state, and ramp weight with rollback gates.

### Scale-in creates two owners

1. A stateful worker is selected for removal and its partition is copied.
2. Routing changes, but the old worker's lease remains valid during a network partition.
3. Old and new workers both acknowledge writes.

Authority transfer needs a monotonic term/epoch checked by the storage or commit path. “Removed from discovery” is not a write fence.

### Rebalancing consumes the headroom it is meant to create

1. A storage tier is near its I/O limit, so operators add nodes.
2. Backfill and replica repair use the same disks and network as foreground work.
3. Latency rises, retries increase, and catch-up falls further behind.

Reserve and enforce a movement budget, prioritize foreground work, and estimate catch-up under continuous writes before starting. Scaling from zero spare capacity may require temporary vertical or admission relief first.

## Overload, operations, and migration

Capacity changes take time. Admission control, [rate limiting](./05-rate-limiting.md), bounded queues, and degraded modes keep the system inside its current envelope while new capacity becomes useful. Retries must use budgets because a failing scale transition is exactly when unbounded extra attempts are most destructive.

For a large migration:

1. establish workload and resource baselines by route/tenant/partition;
2. benchmark one new unit cold, warm, healthy, and under dependency degradation;
3. shadow requests or reads to compare correctness and cost without authority;
4. canary one failure domain and explicitly inject unit loss;
5. ramp with abort gates on goodput, latency, error, queue, dependency saturation, and cost;
6. preserve rollback state until the longest state-copy, backup, and client-connection horizon closes;
7. update capacity models from measured service demand after the migration.

Avoid changing instance shape, runtime version, partition function, and traffic policy in one step. If the result changes, those coupled variables make causality and rollback ambiguous.

Runbooks need current answers for quota exhaustion, placement failure, image/secret startup failure, cold-capacity rejection, backfill throttling, stuck drains, duplicate authority, dependency saturation, and a scale action that increased cost without goodput.

## Security and governance

Scaling control can create machines, move data, and multiply spend. Treat it as privileged production authority:

- separate permission to propose capacity from permission to transfer data authority;
- constrain maximum fleet size, region, instance/accelerator class, and cost through reviewed policy;
- authenticate images, bootstrap configuration, workload identity, and controller updates;
- revoke credentials and erase local data when units retire;
- preserve tenant residency and isolation during placement or movement;
- defend expensive endpoints from cost-amplification and denial-of-wallet attacks;
- audit who changed limits, desired capacity, movement throttle, and failure headroom.

Horizontal scale also widens the supply-chain and secret-distribution surface. A compromised image deployed to 500 replicas is not safer than one large server.

## Observability and verification

Observe offered demand, admitted demand, goodput, and rejected work separately. Then correlate:

- per-operation CPU time, allocations, disk/network bytes, IOPS, accelerator time, and downstream calls;
- concurrency, queue depth and age, residence time, and timeout/retry volume;
- per-unit and per-partition distribution, maximum/median skew, hot tenant/key, and serial lock/leader utilization;
- desired, provisioned, initialized, ready, routed, and actually busy capacity;
- cache/page temperature, pool establishment, replication/backfill lag, and drain progress;
- cost per good request/job/byte and unused failure/rollout headroom;
- configuration/authority epoch and stale participants.

Verification should include:

- capacity sweeps across unit count and unit size, long enough to expose garbage collection, throttling, compaction, and leaks;
- production-shaped and adversarial skew, not only uniform requests;
- cold start, one-unit and one-zone loss, dependency slowdown, and quota denial;
- scale-out, aborted scale-out, scale-in, authority-transfer crash at every phase, and rollback;
- multiplied-budget checks for connections, retries, polling, leases, and rate limits;
- correctness comparison across old and new shapes, including tail and degraded responses.

## Decision framework

1. Which user outcome must improve, and is throughput currently limited by capacity or by correctness/admission policy?
2. What resource or serial path saturates first at the production workload distribution?
3. Will a larger unit relieve that resource, or merely enlarge a different idle resource?
4. Can work execute independently across units, and what state or coordination prevents it?
5. Does horizontal replication improve writes, reads, compute, or only availability?
6. What fleet-wide budget multiplies when one more replica starts?
7. How much warm-up, data movement, and catch-up time precedes useful capacity?
8. What epoch fences old authority during scale-in or repartitioning?
9. Does the target shape survive a unit/zone failure and a rollout simultaneously?
10. Is an algorithmic, caching, batching, queueing, or workload-isolation change cheaper than more infrastructure?
11. What evidence triggers rollback, and how long must duplicate capacity/state be retained?
12. Which metrics prove lower cost per good result rather than higher resource consumption?

## Primary references

- [Amdahl, *Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities* (AFIPS 1967)](https://dl.acm.org/doi/10.1145/1465482.1465560)
- [Little, *A Proof for the Queuing Formula: L = λW* (Operations Research, 1961)](https://pubsonline.informs.org/doi/10.1287/opre.9.3.383)
- [Gunther, *A General Theory of Computational Scalability Based on Rational Functions* (2008)](https://arxiv.org/abs/0808.1431)
- [Dean and Barroso, *The Tail at Scale* (Communications of the ACM, 2013)](https://research.google/pubs/the-tail-at-scale/)
- [Verma et al., *Large-scale cluster management at Google with Borg* (EuroSys 2015)](https://research.google/pubs/large-scale-cluster-management-at-google-with-borg/)
- [Google SRE, *Handling Overload* (2016)](https://sre.google/sre-book/handling-overload/)
- [Google SRE, *Addressing Cascading Failures* (2016)](https://sre.google/sre-book/addressing-cascading-failures/)
- [Kubernetes, *Horizontal Pod Autoscaling*](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Kubernetes Autoscaler, *Vertical Pod Autoscaler*](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler)
