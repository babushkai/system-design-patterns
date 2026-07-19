# Metrics Systems and Monitoring

## TL;DR

Metrics turn many observations into bounded numeric time series. Their power comes from aggregation; their failure mode is accidental dimensionality. A useful metric contract names an instrument, unit, monotonicity, aggregation, temporality, attribute schema, source identity, and reset/gap behavior. Without those semantics, a number that looks queryable can yield a wrong rate, percentile, or SLO.

Counters represent additive change, gauges represent a sampled state, and histograms preserve distributions so they can be aggregated before quantiles are estimated. Never average per-instance percentiles. Cumulative and delta streams can both work, but collectors must preserve single-writer identity, start time, resets, and gaps when converting between them.

Scrape, push/OTLP, and remote write solve different transport boundaries. All require bounded queues, explicit timestamps, authentication, tenant isolation, and loss telemetry. Capacity is driven by active series, churn, samples, histogram buckets, retention, replication, and query scan—not the number of metric names.

The shared instrumentation/collector/control-plane architecture is defined in [Distributed Tracing and Telemetry Pipelines](./01-distributed-tracing.md). This chapter owns metric instruments, aggregation, transport, storage, and query capacity. [Alerting](./04-alerting.md) owns notification state and [SLOs](./05-slos-error-budgets.md) owns reliability interpretation.

---

## Metric Workload and Contract

A time series is identified by:

~~~text
tenant
resource identity
metric name and unit
instrument/aggregation semantics
attribute key-value set
temporality and start-time semantics
schema version
~~~

A point includes value/distribution, observation interval or timestamp, flags for missing/no-recorded-value where supported, and optional exemplar links.

### Metric invariants

1. A metric name and unit do not change meaning in place.
2. Attribute keys and allowed value domains are part of a reviewed schema.
3. Additive streams have one logical writer per series identity and interval.
4. Counter resets and collection gaps are distinguishable from negative work.
5. Histograms aggregate counts in compatible bucket schemas before quantiles are calculated.
6. Gauge aggregation is explicitly defined; summing, averaging, maxing, or taking last are not interchangeable.
7. Producer timestamps and collector timestamps are not silently mixed.
8. Missing data is represented as unknown/stale, not automatically zero.
9. Tenant and resource labels are authenticated/enriched, not accepted from an untrusted metric payload.
10. Collection and query loss, lag, and policy revision are observable.

## Instrument Model

### Counter

A counter is a non-negative additive quantity over time:

- requests started;
- bytes sent;
- failures observed;
- jobs completed; or
- CPU seconds consumed.

The useful query is change or rate over an interval, not the raw cumulative value. A process restart can reset a cumulative counter, so the time series needs start time/resource incarnation. A negative delta is a reset or data error unless the instrument explicitly supports up/down change.

For cumulative samples $v_0$ and $v_1$ over elapsed time $\Delta t$ with no reset:

$$
\text{rate} = \frac{v_1 - v_0}{\Delta t}.
$$

Production rate functions also handle extrapolation, sparse points, reset detection, and gaps. Do not reimplement them casually in dashboards.

### Up/down counter

An up/down counter records additive changes that may be positive or negative, such as in-flight work acquired/released. It is useful when each producer observes transitions. If a process crashes before decrementing, its resource identity must disappear or reset; otherwise a global in-flight total remains inflated.

### Gauge

A gauge records a sampled state:

- queue depth;
- temperature;
- memory in use;
- desired replicas;
- last completed sequence.

The aggregation question is part of the definition:

| Gauge | Valid fleet aggregation |
|---|---|
| bytes currently allocated per process | sum |
| node temperature | max or distribution, not sum |
| configuration generation | min/max and mismatch count |
| queue age | max or histogram |
| utilization ratio | weighted ratio from numerator/denominator, not average ratios |

Last-value gauges are vulnerable to stale producers. Storage must mark a series stale or queries must constrain freshness.

### Histogram

A histogram records a distribution as count, sum where meaningful, and bucket populations:

~~~text
request duration observations
  -> <= 5 ms
  -> <= 10 ms
  -> <= 25 ms
  -> ...
  -> +Inf
~~~

Classic cumulative buckets multiply series by the number of boundaries. Exponential/native histogram representations can adjust resolution across magnitude and merge compatible scales, trading precision for compactness.

Histograms support:

- threshold compliance directly from bucket counts;
- average from sum/count when the quantity supports summation;
- quantile estimation after aggregation;
- exemplars linking a bucket observation to a trace.

Quantiles estimated from buckets are approximate. Accuracy depends on boundaries and distribution inside the selected bucket. Choose boundaries around decision points and SLO thresholds, not a copied generic list.

### Why not percentile gauges?

If instance A reports p99 10 ms for 100 requests and instance B reports p99 1 s for one million requests, averaging their p99 values has no population meaning. Preserve mergeable distributions or query raw observations in a system designed for them.

Client-side summaries/quantile sketches may be appropriate when their error guarantees and merge semantics are understood. A non-mergeable quantile cannot be aggregated across replicas or regions.

## Aggregation and Temporality

### Delta and cumulative

For additive instruments:

- **cumulative temporality** reports change from a start time through the point;
- **delta temporality** reports change since the previous point.

Cumulative streams tolerate collector restarts because the producer retains state, but resets need detection and long-lived values can lose numeric precision in some representations. Delta streams are compact for rate-oriented pipelines but a lost interval is unrecoverable unless buffered, and collector state is needed when converting.

### Single-writer principle

One series identity should have non-overlapping writers. During rolling replacement:

~~~text
old process emits cumulative series resource.instance=A
new process emits cumulative series resource.instance=B
aggregation removes instance only at query/recording-rule layer
~~~

If both write the identical series identity, overlapping intervals make resets and deltas ambiguous. Resource incarnation is operational metadata even if dashboards later aggregate it away.

### Reset and gap state machine

~~~mermaid
stateDiagram-v2
    [*] --> Active
    Active --> Active: next contiguous point
    Active --> Reset: new start time or lower cumulative value
    Reset --> Active: establish new baseline
    Active --> Gap: missed interval or stale marker
    Gap --> Active: new baseline/point
    Active --> Ended: resource disappears
    Ended --> [*]
~~~

Queries must not bridge a rate across an unknown gap or treat a reset as a large negative rate. Alert and SLO evaluators define missing-data behavior separately.

### Spatial and temporal aggregation

Spatial aggregation removes dimensions:

$$
M_{\text{service}}(t)
=
\sum_{\text{instance}} M_{\text{instance}}(t)
$$

only for additive compatible streams. Temporal aggregation combines points into larger windows: sums for deltas, end-minus-start for cumulative counters, and merged bucket counts for histograms.

Downsampling is irreversible. Preserve sums/counts/buckets rather than precomputed averages or quantiles so later queries remain composable.

## Attribute and Cardinality Budgets

Each unique attribute set creates a series. If a metric has independent value counts $c_1,\ldots,c_n$, the theoretical series count is:

$$
S_{\max} = \prod_{i=1}^{n} c_i.
$$

Values are rarely independent, but the product reveals explosive dimensions. A route label with hundreds of templates can be acceptable; adding raw user ID, request ID, URL, exception message, and container ID is not.

### Attribute classes

| Class | Examples | Policy |
|---|---|---|
| Stable aggregation | service, operation, status class, region | allowed and reviewed |
| Bounded deployment | version, zone, workload class | allowed with explicit domain |
| High-cardinality evidence | user, order, request, trace, raw URL | logs/traces, never ordinary metric labels |
| Untrusted value | header, query, exception text | normalize to bounded enum or drop |
| Tenant | tenant tier/cell for fleet metrics; exact tenant only in intentionally partitioned metering | access and cardinality budget |

Use route templates, error classes, and protocol codes. Unknown values map to a bounded *other/unknown* class with a separate count; silently dropping them hides schema drift.

### Budget enforcement

At instrumentation review:

- declare allowed keys and value domains;
- estimate active series per resource and fleet;
- include histogram bucket multiplication;
- set per-tenant/metric new-series rate limits;
- reject or relabel unbounded attributes at collection;
- measure series churn, not only active count; and
- version schema changes.

A hard backend series limit without producer feedback produces blind loss. Export dropped-series count, offending metric/key class under bounded labels, and active policy revision.

## Collection Models

### Scrape/pull

A collector discovers targets and periodically fetches their current cumulative metric state.

**Strengths**

- collector controls cadence, timeout, and concurrency;
- target health is observable from scrape success;
- cumulative state can be retried on next scrape;
- simple service-discovery integration.

**Risks**

- unreachable short-lived jobs disappear before scrape;
- target cardinality and scrape fan-out;
- synchronized scrape load;
- endpoint exposure and authentication;
- duplicate HA scrapers need replica labeling/dedup;
- a successful endpoint response can still omit a metric family.

Jitter scrape scheduling and bound response size/parse time. A scrape timeout is missing data, not a zero sample.

### Push/OTLP

SDKs or collectors push batches to an ingest service.

**Strengths**

- works across network boundaries and for ephemeral work;
- delta or cumulative streams;
- shared collector pipeline and explicit partial success;
- producer-side batching.

**Risks**

- producer queues/retries can affect applications;
- ingest must authenticate tenant/resource;
- overlapping writers and temporality conversion;
- loss can be invisible without acknowledgements/drop counters.

Prefer local collectors so applications do not hold thousands of backend connections or synchronize retry storms.

### Remote write

A scraper/local metrics system writes samples and metadata to remote durable storage. It decouples collection from global retention/query but creates a write-ahead queue and replay boundary.

Define:

- shard/concurrency and backoff;
- queue durability and maximum age;
- ordering and out-of-order window;
- tenant/auth mapping;
- HA replica deduplication;
- partial rejection handling;
- reshard behavior;
- backfill policy; and
- what happens when the remote endpoint is unavailable longer than local retention.

### Choosing a model

| Workload | Bias |
|---|---|
| Long-lived reachable service | scrape is simple and independently observable |
| Short-lived job | push to a local collector or durable job metric gateway with lifecycle semantics |
| Mobile/client telemetry | authenticated push with privacy aggregation; never expose scrape |
| Multi-cluster global query | regional scrape/ingest plus remote write/federation |
| High-value metering | durable event/accounting pipeline may be required; best-effort metrics alone are insufficient |

## Storage and Query Architecture

### Write path

~~~mermaid
flowchart LR
    COL[Scrapers and collectors] --> AUTH[Authenticate and validate]
    AUTH --> WAL[Write-ahead log]
    WAL --> HEAD[Mutable head series]
    HEAD --> BLOCK[Immutable time blocks]
    BLOCK --> OBJ[(Regional durable storage)]
    OBJ --> INDEX[Series/postings index]
    INDEX --> QUERY[Query engine]
    QUERY --> RULE[Recording and alert rules]
~~~

The exact implementation varies, but the state transitions matter:

1. validate schema, tenant, timestamp, and limits;
2. append durably under defined acknowledgement;
3. update in-memory/head representation;
4. compact into immutable blocks/chunks;
5. replicate or upload;
6. enforce retention/tombstones; and
7. expose query consistency.

### Query cost

A query first selects series, then scans samples/buckets:

$$
C_{\text{query}}
\propto
S_{\text{matched}}
\sum_{s \in \text{matched}} P_s
$$

where $P_s$ is points read for series $s$. A broad regex across a long range can dominate even if the output is one line.

Controls:

- bounded time range and returned series;
- query concurrency and per-tenant fairness;
- indexed exact matchers before regex;
- recording rules for repeatedly used expensive aggregates;
- downsampled long-retention tiers;
- query cost estimation/cancellation;
- cache keys including tenant and data revision semantics; and
- separate interactive from batch/reporting pools.

Recording rules trade write/storage cost and staleness for predictable query cost. They need versioned expressions, evaluation timestamps, gap behavior, and backfill policy.

## Capacity and Cost Model

Assume:

- $S$ active series;
- $f$ samples per series per second;
- $\bar{b}$ compressed stored bytes per sample including amortized index;
- $r$ replication factor;
- $T$ retention seconds;
- $h$ average histogram series multiplier;
- $q$ queries per second; and
- $\bar{P}$ samples scanned per query.

Ingest sample rate:

$$
\lambda_{\text{samples}} = S f.
$$

For classic histograms, if $S$ excludes their bucket expansion:

$$
\lambda_{\text{effective}} \approx S f h.
$$

Retained storage:

$$
V \approx S f h \bar{b} T r.
$$

Approximate query scan rate:

$$
Q_{\text{scan}} \approx q \bar{P}.
$$

State assumptions: compression changes with label repetition and scrape interval; indexes add series/churn cost; object-store replication and cache are extra; recording rules add series; exemplars add trace references; HA collection can double ingest before dedup.

### Churn matters

Short-lived resources create new series even if active $S$ stays flat. Index and compaction work follows series creations and tombstones:

$$
\lambda_{\text{series-create}}
=
\sum_{\text{resource starts}}
\text{metric streams per resource}.
$$

Aggregate away container/process identity at a regional recording layer for long retention while preserving a short forensic tier.

## Security, Privacy, and Multi-Region Operations

Metrics reveal:

- service and tenant names;
- traffic/customer volume;
- rollout and capacity;
- failure rates;
- infrastructure topology; and
- sometimes business or personal attributes.

Authenticate collection endpoints, authorize per-tenant ingestion/query, encrypt transport/storage, and apply retention/export rules. Do not let a producer select another tenant through a label. Resource identity should come from authenticated workload metadata or a trusted collector.

### Regional design

Keep collection and alert-critical rules regional so a WAN failure does not erase visibility into the affected region. Global views can:

- federate queries;
- asynchronously replicate blocks;
- consume regionally aggregated recording rules; or
- route tenants to a metric home.

Global aggregation must avoid double counting HA replicas and replicated regions. Tag source region and replica internally, deduplicate before removing those dimensions, and publish freshness/watermark.

During partition, regional SLO/alerts continue from local data. A global total is explicitly partial or stale; never present it as complete without a coverage indicator.

## Failure Traces

### One raw path label exhausts the metrics system

~~~text
deployment adds request.path as a label
-> every ID creates a new series
-> collector and store accept rapidly growing cardinality
-> head memory, index, and remote-write queue expand
-> rule evaluations slow and alerts become stale
-> backend begins dropping the exact service’s metrics
~~~

**Detection:** new-series rate, active/churn by metric, label value cardinality estimates, ingest rejection, head memory, rule evaluation lag.

**Response:** collector relabel/drop policy, rollback instrumentation, preserve bounded aggregate, isolate tenant, compact/tombstone under controlled load, and record the blind interval.

### Counter reset appears as recovery

~~~text
fleet restarts during incident
-> cumulative counters reset
-> naive dashboard subtracts new from old or clamps negative to zero
-> error rate appears to fall while requests still fail
~~~

**Controls:** resource incarnation/start time, reset-aware rate, concurrent success/total counters, restart overlay, and synthetic verification.

### Remote-write recovery overwhelms storage

~~~text
remote store unavailable -> regional WAL queues hours of samples
-> store recovers -> every shard replays at maximum speed
-> live samples compete with backlog
-> query/rule freshness remains bad despite endpoint recovery
~~~

**Controls:** replay rate limit, live/backfill priority, per-tenant fairness, queue age SLO, capacity for bounded catch-up, and explicit data watermark.

## Operating the Metrics System

Track:

- discovered/scraped targets, scrape duration/size/error, and missing families;
- accepted/rejected samples and reason;
- active series, new-series rate, label cardinality, and histogram expansion;
- WAL bytes/age, remote-write queue, retry, drop, and out-of-order points;
- compaction, block upload, replication, retention, and tombstone backlog;
- rule evaluation duration, missed intervals, and active revision;
- query matched series, scanned points/bytes, queue, cancel, and cache;
- region/replica coverage and global watermark; and
- storage/compute/network cost per retained useful series or query.

An external heartbeat or black-box probe detects total monitoring disappearance. Internal self-metrics alone cannot page if their own entire path is gone.

## Verification Strategy

| Test layer | What to prove |
|---|---|
| Instrument contract | Name, unit, type, monotonicity, attributes, and aggregation match semantics |
| Counter model | Reset, gap, duplicate, overlap, and process replacement produce correct rates |
| Gauge tests | Staleness and fleet aggregation match the declared operator |
| Histogram tests | Boundary, inclusivity, merge, scale conversion, quantile error, and SLO threshold |
| Cardinality tests | Worst-case attribute domains and malicious values stay within budget |
| Collection tests | Missing target, partial response, timeout, push rejection, and duplicate HA scrape |
| Remote-write fault | Throttle, outage, disk full, replay, out-of-order, and partial rejection |
| Query tests | Cost limits, cancellation, tenant isolation, recording-rule equivalence, and gap behavior |
| Region tests | Partition, partial global view, replica dedup, and regional rule continuity |
| Load tests | Churn burst, histogram expansion, backlog replay, broad query, and rule peak |

Use a synthetic canary that emits known counter, gauge, and histogram patterns with exemplars. Verify collection, aggregation, query, and alert-rule visibility end to end.

## Decision Framework

1. Is the observation additive change, sampled state, or distribution?
2. What unit, monotonicity, start/reset, and aggregation semantics apply?
3. Which attribute values are bounded and useful for aggregate decisions?
4. What active-series and churn budget does the full product imply?
5. Which histogram representation and resolution answer the required thresholds?
6. Is cumulative or delta temporality safer across the chosen collector topology?
7. Should the workload be scraped, push to a local collector, or use another durable accounting path?
8. What acknowledgement, queue, and loss semantics are required?
9. What retention/downsampling preserves future valid aggregation?
10. Which queries/rules need precomputation and cost isolation?
11. How do regional rules continue while global views expose partial coverage?
12. What telemetry proves the metrics system itself is fresh and complete?

## Key Takeaways

1. Metric correctness starts with instrument, unit, aggregation, temporality, and reset semantics.
2. Counters measure change, gauges sample state, and histograms preserve mergeable distributions.
3. Never average per-instance percentiles.
4. Single-writer identity and start time make resets and deltas interpretable.
5. Missing data is unknown, not zero.
6. Attribute products and series churn dominate storage and query cost.
7. Scrape, push, and remote write have different state and failure boundaries.
8. Retain composable sums, counts, and buckets before irreversible downsampling.
9. Keep alert-critical collection regional and expose global coverage/freshness.
10. Test resets, gaps, cardinality, backlog replay, query cost, and tenant isolation.

---

## References

- [OpenTelemetry Metrics Data Model](https://opentelemetry.io/docs/specs/otel/metrics/data-model/) — streams, temporality, resets, gaps, overlap, histograms, and exemplars
- [OpenTelemetry Metrics SDK Specification](https://opentelemetry.io/docs/specs/otel/metrics/sdk/) — instruments, views, aggregation, and cardinality limits
- [Prometheus Data Model](https://prometheus.io/docs/concepts/data_model/) — time-series identity, labels, samples, and staleness
- [OpenMetrics Specification](https://github.com/prometheus/OpenMetrics/blob/main/specification/OpenMetrics.md) — metric exposition types and wire semantics
- [Prometheus Remote Write Specification](https://prometheus.io/docs/specs/prw/remote_write_spec/) — sample transport and compatibility
- [Prometheus Storage](https://prometheus.io/docs/prometheus/latest/storage/) — WAL, local time-series blocks, retention, and operational behavior
- [Monarch: Google’s Planet-Scale In-Memory Time Series Database](https://www.vldb.org/pvldb/vol13/p3181-adams.pdf) — distributed metric ingestion, indexing, query, and regional operation
- [Distributed Tracing and Telemetry Pipelines](./01-distributed-tracing.md) — shared collection/control plane and trace exemplars
- [Alert Evaluation and Notification](./04-alerting.md) — rule state, routing, and missing-data behavior
- [SLOs and Error-Budget Control](./05-slos-error-budgets.md) — SLI aggregation and burn-rate interpretation
