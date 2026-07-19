# Capacity Planning and Back-of-the-Envelope Estimation

Capacity planning turns a workload contract into a resource, failure-headroom, and lead-time plan. The useful result is not one server count. It is a versioned model that explains which resource binds, which assumptions dominate, what happens during a named failure or rollout, when a hard limit will be reached, and which measurements will recalibrate the plan.

This chapter owns **technical demand estimation, resource service-demand models, queueing/headroom reasoning, workload forecasts, failure capacity, benchmark calibration, and exhaustion lead time**. [Auto-Scaling](../06-scaling/08-auto-scaling.md) owns delayed feedback, metrics, controller stability, and safe scale-down. [FinOps and Cost Engineering](../11-observability/06-finops-cost-engineering.md) owns prices, allocation, commitments, showback, and unit economics. [Horizontal vs Vertical Scaling](../06-scaling/02-horizontal-vertical.md) owns capacity-change actuators and transition protocols.

## Primary Evidence and Scope

| Primary evidence | What it establishes | Boundary |
|---|---|---|
| Little, *Operations Research* 1961 | Under stated stationarity/finite-mean conditions, average in-system work is arrival rate times average residence time | An average identity, not a percentile or burst model |
| Kingman, 1961 | A single-server heavy-traffic approximation relates wait to utilization and arrival/service variability | Approximation with explicit queue assumptions |
| Schroeder, Wierman, Harchol-Balter, NSDI 2006 | Open and closed workload models can produce radically different response-time results | Study/model results, not a claim that every test is wrong |
| Dean and Barroso, 2013 | Large fan-out amplifies component latency tails into user-visible request tails | Historical large-service evidence |
| Google SRE Workbook, NALSD (2018) | Concrete resource arithmetic, assumptions, reliability constraints, and iterative design are part of large-system planning | One published engineering method, not a provider sizing table |

Static “latency numbers every programmer should know” and generic QPS-per-node tables age quickly and omit workload shape. Use order-of-magnitude arithmetic to find dominant terms, then replace assumptions with measurements from the real software, data, protocol, and failure mode.

## Capacity contract

Before calculating, define:

| Field | Required answer |
|---|---|
| **Useful work** | Successful request, committed event, decoded media minute, token, query, byte, or completed job? |
| **Demand** | Offered, admitted, retried, rejected, and completed rates, with burst and seasonality? |
| **Cost distribution** | Mean, percentiles, heavy operations, hottest key/tenant, and payload sizes? |
| **SLO** | Latency, freshness, loss, recovery, and availability target at which capacity is useful? |
| **Topology** | Host, zone, region, cell, shard, and shared-dependency constraints? |
| **Failure/rollout** | Which concurrent unit losses and deployment overlap must be tolerated? |
| **Growth horizon** | Forecast interval, uncertainty, procurement/quota/migration lead time? |
| **Degradation** | Which work may queue, shed, approximate, serve stale, or never be dropped? |
| **Model owner** | Who updates unit costs, forecast inputs, limits, and decisions? |

Separate these rates:

```text
offered demand >= admitted demand >= attempted work >= good completed work
```

Retries can make attempted work exceed admitted logical operations, while errors make goodput smaller. Planning from request attempts alone can reward retry amplification with more machines; planning from goodput alone can hide the offered load that admission must reject.

## Model state, control path, and serving path

**Reference design:** store a capacity model as reviewed, machine-readable state:

```text
model revision and owner
workload classes and useful-work definitions
demand history, forecast scenarios, and confidence interval
per-class resource service-demand distributions
usable unit capacity and benchmark provenance
topology, quota, and placement constraints
failure and rollout scenarios
headroom policy by resource
storage retention/replication/index/compaction factors
lead times and irreversible limits
observed-vs-predicted error
```

The **serving data path** admits and completes work with current capacity. The **capacity control path** collects workload/resource evidence, forecasts, reserves or provisions units, validates readiness, and updates limits. A control-plane forecast failure must not unbound the serving path; admission and backpressure protect the current envelope.

~~~mermaid
flowchart LR
    B[Business and product drivers] --> F[Demand scenarios]
    T[Production telemetry] --> U[Measured unit costs]
    L[Load/failure tests] --> U
    F --> M[Versioned capacity model]
    U --> M
    Q[Quotas, topology, lead time] --> M
    M --> P[Procure / provision / migrate]
    P --> V[Warm and validate usable capacity]
    V --> S[Serving fleet]
    S --> T
~~~

## Unit discipline and first-pass arithmetic

Track dimensions on every quantity. Network rates are commonly bits/s; payload and storage are bytes. Decimal provider units and binary memory/storage units differ. Compression changes bytes but consumes CPU. Replication and index amplification apply at different stages.

Useful calendar conversions:

```text
one day = 86,400 seconds
average rate = events per interval / interval seconds
daily bytes = events/s × bytes/event × 86,400
```

Average is a conservation check, not a peak plan. Preserve at least hourly/minute peaks and burst windows. If 864 million operations/day all arrive in one busy hour, the peak is 240,000/s, not the daily average of 10,000/s.

For each workload class $j$ and resource $r$, let $\lambda_j$ be admitted rate and $d_{j,r}$ measured resource demand per completed operation:

$$
D_r = \sum_j \lambda_j d_{j,r}
$$

For $N$ equivalent units, tested resource capacity $C_r$, and planned usable fraction $u_r$:

$$
N_r \ge \left\lceil \frac{D_r}{C_r u_r} \right\rceil
$$

Compute CPU time, memory capacity and bandwidth, storage IOPS and throughput, network packets and bytes, accelerator time/memory, open connections, file descriptors, and every downstream quota. The largest topology-feasible result binds. `u[r]` is not a universal percentage; derive it from the measured latency/recovery knee and the required failure margin.

## Concurrency, queues, and variability

### Little's Law

**Documented model, Little 1961:** under its steady-state assumptions,

$$
L = \lambda W
$$

If a service completes 12,000 requests/s and mean end-to-end residence time is 250 ms, it contains about 3,000 requests on average. That concurrency consumes request memory, sockets, thread/async state, and downstream pool slots.

Little's Law does not predict latency from utilization and does not describe p99. It can still diagnose coupling: if throughput is flat while in-flight work doubles, residence time doubled somewhere in the boundary.

For a stable queue with arrival rate $\lambda$ and completion rate $\mu>\lambda$, backlog $B$ has an optimistic drain time:

$$
T_{\mathrm{drain}} \ge \frac{B}{\mu-\lambda}
$$

Using `B/μ` ignores continuing arrivals. If `λ >= μ`, no finite drain time exists; add useful capacity, reduce admitted work, or change service demand.

### Utilization is nonlinear

For one illustrative M/M/1 queue with mean service time $S$ and utilization $\rho=\lambda S$:

$$
W = \frac{S}{1-\rho}
$$

This curve demonstrates the asymptote near saturation; it is not a universal server formula. Real systems have parallel servers, scheduling, batching, finite queues, bursty arrivals, and heavy-tailed work.

Kingman's G/G/1 heavy-traffic approximation makes variability visible:

$$
W_q \approx \frac{c_a^2+c_s^2}{2}\,\frac{\rho}{1-\rho}\,S
$$

`ca` and `cs` are coefficients of variation for interarrival and service times. Two systems with equal mean demand and utilization can have very different waits when one receives bursts or highly variable jobs. Measure distributions and isolate expensive classes rather than selecting a target utilization from folklore.

### Fan-out consumes a probability budget

If a request requires all `k` branches and each independently meets a deadline with probability `p`, the optimistic request probability is `p^k`. At `p = 99.9%` and `k = 100`, it is about `90.5%`. Shared racks, networks, and queues make independence optimistic.

Capacity plans include fan-out width, duplicated/hedged requests, cancellation effectiveness, intermediate bytes, and the slowest required branch. [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md) owns attempt budgets and tail mitigation.

## Demand models: distribution, not one multiplier

Build scenarios from drivers such as active tenants, devices, orders, content bytes, model tokens, or scheduled jobs:

```text
demand = driver count × actions per driver × peak-shape factor × retry/fanout amplification
```

Keep low/base/high scenarios and the assumptions that distinguish them. Segment by route, tenant cohort, geography, payload class, and day/event type. Report:

- average and chosen peak-window rates;
- peak duration and ramp slope;
- hottest tenant/key and correlated groups;
- launch/event scenarios not present in history;
- confidence interval and forecast error;
- demand that will be shed or queued by policy.

A global percentile can hide a deterministic regional opening hour or one customer larger than an entire average cell. Use traces and product calendars alongside statistical forecasting.

**Reference design:** compare forecast to actual at every horizon. Track signed model error `(actual - forecast) / forecast` and recalibrate unit costs separately from business-driver forecast. Otherwise a code regression and user growth are indistinguishable.

## Failure, rollout, and topology headroom

Capacity must fit the surviving topology, not only the healthy total. If each ready unit provides planned useful capacity $c$, peak admitted demand is $D$, the named scenario removes $f$ units, and a rollout removes $r$ more:

$$
(N-f-r)c \ge D
$$

This formula assumes demand and units are fungible. Zone affinity, tenant placement, shards, caches, and dependency quotas can make aggregate spare capacity unusable. Validate per failure domain and the hottest placement group.

Examples of explicit scenarios:

- one host and one rollout batch unavailable;
- one zone lost while another zone is receiving a canary;
- one cell drained into named destinations;
- one shard replica rebuilding while compaction runs;
- one region evacuated only for traffic classes the data plane can safely serve.

[Multi-Region Architecture](../06-scaling/09-multi-region-architecture.md) owns cross-region authority/failover. Here the plan records the resulting surviving-capacity constraint.

## Storage and data-movement model

Size storage in stages:

```text
logical retained bytes
× compression/encryption expansion
+ indexes and derived structures
+ metadata, tombstones, and transaction log
× replica count
+ backup/snapshot overlap
+ compaction/rebuild/migration temporary space
all divided by planned maximum occupancy
```

Capacity is invalid if bytes fit but foreground plus compaction, repair, backup, and migration exceed IOPS or throughput. Model read/write amplification and the busiest disk/range, not cluster averages.

For $D$ bytes moved with effective source read $R_s$, network $R_n$, destination write $R_d$, and foreground-safe duty factor $q$:

$$
T_{\mathrm{copy}} \ge \frac{D}{q\,\min(R_s,R_n,R_d)}
$$

Catch-up, checksums, indexes, replicas, and continuous writes extend the time. Recovery-point and recovery-time targets therefore consume steady spare bandwidth and temporary storage.

## Network and connection model

For each edge in the request/dataflow graph:

```text
bytes/s = operations/s × messages/operation × wire bytes/message
packets/s ≈ bytes/s / measured mean wire packet size
```

Include protocol headers, TLS, acknowledgements, replication, retries, compression ratio, and cross-zone/region copies. Packets/s can bind before bits/s for small messages.

New-connection rate and mean connection lifetime produce mean active connections through Little's Law. Then size file descriptors, socket memory, TLS handshakes, NAT/conntrack, and load-balancer tables through failure/reconnect bursts. [DNS and Connection Management](../06-scaling/13-dns-and-connection-management.md) owns those lifecycle mechanics.

## Worked example with labeled assumptions

Consider an illustrative regional event-ingestion service.

**Assumptions:**

- peak admitted rate: `75,000 events/s` for four hours;
- mean payload: `1.4 KiB` after protocol framing;
- validation CPU demand: `0.35 ms/event` measured at the production mix;
- 16-core workers, planned at 55% CPU because the measured latency/recovery gate fails above it;
- tolerate one worker loss plus one worker unavailable for rollout;
- 30-day retention; data compression factor `0.60`;
- indexes/metadata add 35% of compressed data;
- three storage replicas and 25% free-space/repair reserve.

### Compute

```text
CPU demand = 75,000 × 0.00035 = 26.25 CPU-seconds/second
useful CPU/worker = 16 × 0.55 = 8.8 CPU-seconds/second
(N - 2) × 8.8 >= 26.25  =>  N >= 5 workers
```

This is only the CPU bound. Five workers are insufficient if one hot tenant pins more than 8.8 CPU-seconds/s to one partition or a downstream quota binds first.

### Ingest network

```text
payload rate = 75,000 × 1.4 KiB = 102.5 MiB/s ≈ 0.86 Gbit/s
```

Add replication traffic, acknowledgements, encryption overhead, retries, and failover routing before selecting network capacity.

### Storage

```text
logical/day ≈ 102.5 MiB/s × 86,400 = 8.45 TiB/day
30-day logical = 253.5 TiB
compressed + indexes = 253.5 × 0.60 × 1.35 = 205.3 TiB
three replicas = 615.9 TiB
with 25% free reserve = 615.9 / 0.75 = 821.2 TiB provisioned
```

This estimate deliberately exposes the dominant storage term. A real design would decide whether the entire 30-day set needs three hot replicas, whether colder tiers change the model, and whether indexes share the same retention.

### Queue recovery

If a 20-minute dependency outage admits events into a durable queue:

```text
backlog = 75,000 × 1,200 = 90 million events
```

At post-recovery completion capacity `105,000/s` while `75,000/s` continues to arrive:

```text
drain time >= 90,000,000 / (105,000 - 75,000) = 3,000 s = 50 min
```

The plan must prove storage, consumers, and downstream systems tolerate that recovery rate; “the queue can hold it” is not end-to-end capacity.

## Benchmark and load-test methodology

**Documented, Schroeder et al. 2006:** a closed workload waits for completions before generating more work, while an open workload generates arrivals independently. Under overload, a closed test self-throttles and can substantially understate open-system queues and response times.

A capacity test should:

1. reproduce production route, key/tenant, payload, cache, and dependency distributions;
2. use open-loop arrivals where users/events do not wait for prior completion;
3. preserve intended bursts and correct coordinated omission in latency recording;
4. sweep past the knee to find saturation and the failure mode;
5. hold long enough to expose garbage collection, compaction, thermal/credit throttling, and leaks;
6. drop load and verify recovery rather than metastable degradation;
7. repeat cold/warm, normal/failure, and rollout-overlap states;
8. record per-operation resource service demands, not only maximum throughput.

A benchmark result includes binary/configuration, hardware, data shape, concurrency, protocol, cache temperature, duration, failure state, and confidence range. “Database does 50k QPS” is not portable evidence.

## Forecast, limits, and lead time

Maintain an exhaustion date for every hard or slow-moving limit:

$$
T_{\mathrm{limit}} \approx \frac{\text{usable limit}-\text{current peak demand}}{\text{growth rate}}
$$

Linear extrapolation is only a first alarm. Use scenario forecasts for launches, contracts, regional growth, retention changes, and step-function migrations. Compare exhaustion horizon with:

- quota approval and hardware/accelerator procurement;
- shard split or tenant-move duration;
- data copy and index build time;
- region/cell construction and certification;
- contract/reservation decision windows;
- engineering and verification lead time.

Autoscaling handles demand after capacity is available to the actuator. It cannot invent provider quota, split a hot key, shorten a seven-hour state copy, or make a dependency accept more connections. See [Auto-Scaling](../06-scaling/08-auto-scaling.md).

## Specialized failure traces

### Average demand hides a deterministic peak

Daily average needs eight nodes; regional business opening needs twenty-four for 45 minutes. A plan based on daily totals repeatedly overloads at the same time. Preserve time-window and regional distributions, then test the ramp slope and queue recovery.

### Per-node benchmark ignores a shared dependency

One worker sustains 2,000 requests/s in isolation. Fifty workers appear to promise 100,000/s, but each request uses one database query and the database's tested safe limit is 40,000/s. Adding workers multiplies connection pressure and lowers goodput. Model the whole demand graph and cap fleet budgets.

### Closed-loop test reports a false plateau

Virtual users wait for slow responses, so offered rate falls exactly when the service saturates. The chart shows stable errors and modest queues; production's independent arrivals produce an unbounded backlog. Use an open arrival model and report offered versus achieved rate.

### Failover headroom exists in the wrong place

The fleet has 30% global spare CPU, but tenants pinned to zone A cannot be routed to spare zone C because their data is absent. Zone A fails and aggregate dashboards claim capacity remains. Model placement-feasible capacity per authority domain, not global free resources.

### Recovery work causes a second outage

After storage failure, replica rebuild and backlog drain consume the same disks/network as foreground traffic. Retries raise offered work and latency crosses the knee. Reserve separate recovery budgets, throttle maintenance, and verify recovery under continued peak arrivals.

### Forecast is right but quota arrives late

Demand stays inside forecast, yet a required accelerator or address quota has a six-week approval lead and exhaustion is three weeks away. Capacity planning includes supply and organizational lead time; a perfect demand model submitted too late still fails.

## Security and abuse boundaries

Capacity telemetry can reveal customer size, traffic patterns, regions, and commercial events. Apply least privilege, aggregation, retention, and redaction to raw dimensions. Authenticate forecast and limit changes; an attacker or mistaken automation that raises concurrency or fleet maximum can create denial of wallet or overload dependencies.

Plan adversarial demand separately from organic peaks: handshake floods, expensive queries, decompression bombs, hot-key attacks, tenant-created fan-out, and quota probing. Security controls need their own CPU/memory budget and must remain effective when the application is overloaded.

Resource quotas are safety policy, not a substitute for authorization. Audit overrides and give emergency capacity actions bounded scope and expiry.

## Operations, rollout, and reconciliation

Introduce a new model in shadow mode. Compute old and new forecasts for the same history, explain deltas, then use the new model for one reversible capacity decision. Reconcile predicted versus observed demand, resource use, ready capacity, and user SLO after every launch, incident, and architecture change.

Dashboards should show:

- offered/admitted/goodput/rejected rates by workload class;
- per-operation service demand and mix;
- peak-window forecast, confidence, and error;
- ready versus provisioned versus warming capacity;
- headroom after named host/zone/cell/rollout failures;
- hottest tenant/key/partition and placement skew;
- queue age plus modeled drain time;
- storage occupancy, growth, amplification, repair/migration reserve;
- hard limits, quotas, and time-to-exhaustion versus lead time;
- model revision and stale/missing inputs.

Rollback of a capacity-model change means restoring decision authority to the prior model while preserving collected evidence. It does not automatically remove already provisioned stateful capacity; that follows the safe scale-in protocol.

## Verification matrix

| Test | Evidence required |
|---|---|
| Unit/dimensional checks | No bits/bytes, seconds/milliseconds, decimal/binary, or replica-factor errors |
| Conservation | Daily totals reconcile with rate integration and storage growth |
| Sensitivity | Identify assumptions whose plausible range changes architecture |
| Workload realism | Route/tenant/key/payload and burst distributions match measured production |
| Saturation | Knee, rejection mode, queue bound, and recovery are observed |
| Failure topology | Named host/zone/cell/region loss fits surviving eligible capacity |
| Stateful recovery | Rebuild/copy/backlog traffic coexists with foreground SLO |
| Forecast backtest | Error and interval coverage reported by horizon |
| Supply | Quota/procurement/migration lead time precedes exhaustion with margin |

## Decision framework

1. What is the useful-work unit, and how do offered demand, attempts, and goodput differ?
2. Which workload classes, peaks, hot tenants, and failure-correlated bursts must be represented?
3. What measured service demand does each class place on each resource and dependency?
4. Which resource binds first in healthy, rollout, failure, and recovery states?
5. What queue model is defensible, and how do arrival/service variability change the knee?
6. Is spare capacity usable by the affected placement/authority domain?
7. How much temporary storage and bandwidth do compaction, repair, backup, and migration require?
8. Which assumptions dominate the result, and what measurement will replace each?
9. When does every quota/hard limit exhaust under low/base/high scenarios versus its lead time?
10. Does the load test preserve open arrivals, production skew, saturation, and recovery?

## Primary references

- [Little, *A Proof for the Queuing Formula: L = λW* (Operations Research, 1961)](https://pubsonline.informs.org/doi/10.1287/opre.9.3.383)
- [Kingman, *The Single Server Queue in Heavy Traffic* (Mathematical Proceedings of the Cambridge Philosophical Society, 1961)](https://doi.org/10.1017/S0305004100036094)
- [Schroeder, Wierman, and Harchol-Balter, *Open Versus Closed: A Cautionary Tale* (NSDI 2006)](https://www.usenix.org/legacy/event/nsdi06/tech/full_papers/schroeder/schroeder.pdf)
- [Dean and Barroso, *The Tail at Scale* (Communications of the ACM, 2013)](https://research.google/pubs/the-tail-at-scale/)
- [Google SRE Workbook, *Introducing Non-Abstract Large System Design* (2018)](https://sre.google/workbook/non-abstract-design/)
- [Gil Tene, *How NOT to Measure Latency*](https://www.infoq.com/presentations/latency-response-time/)
- [HdrHistogram, coordinated-omission background and implementation](https://github.com/HdrHistogram/HdrHistogram)
