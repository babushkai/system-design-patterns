# Distributed Tracing and Telemetry Pipelines

## TL;DR

Distributed tracing reconstructs the causal path of one operation across process, protocol, queue, and region boundaries. Its correctness depends on context propagation, stable span semantics, explicit retry and asynchronous links, and a sampling policy that preserves the incidents and populations you intend to analyze.

Tracing is also the natural place to introduce the shared telemetry architecture used by this section: instrumentation emits typed signals into local or regional collectors; collectors validate, enrich, redact, sample/aggregate, buffer, and export; storage/query systems index each signal under different cost and consistency models; a versioned control plane distributes schemas, sampling, routing, and retention policy. Metrics, logs, alerts, SLOs, cost, and incidents cross-link this architecture instead of redefining it.

Head sampling decides cheaply before the trace is known. Tail sampling can retain errors and high latency but must buffer incomplete traces and make decisions under timeout, memory, and multi-region constraints. A hybrid usually combines a deterministic baseline with bounded tail rules. Trace context and baggage are untrusted at external boundaries: they are correlation hints, not authentication, and baggage needs an allowlist, size budget, and privacy policy.

---

## Shared Telemetry Data and Control Planes

The signals have different semantics:

- **traces** preserve causal structure for selected operations;
- **metrics** aggregate numeric observations across populations;
- **logs** preserve discrete event evidence;
- **profiles** sample resource consumption over code locations;
- **alerts** are evaluated policy state derived from signals; and
- **SLOs and cost models** interpret those signals against service and business contracts.

They can share transport, resource identity, schema governance, and collection infrastructure without being forced into one storage model.

~~~mermaid
flowchart LR
    APP[Applications and infrastructure] --> SDK[Instrumentation and local buffers]
    SDK --> AGENT[Node/workload collector]
    AGENT --> REG[Regional ingest gateway]
    REG --> PROC[Validate, enrich, redact, route]
    PROC --> TRACE[(Trace store)]
    PROC --> METRIC[(Metrics store)]
    PROC --> LOG[(Log/event store)]
    PROC --> ARCHIVE[(Durable archive)]
    TRACE --> QUERY[Query and correlation]
    METRIC --> QUERY
    LOG --> QUERY
    QUERY --> ALERT[Alert evaluation]
    QUERY --> SLO[SLO and cost computation]

    CONTROL[Schema, sampling, routing, retention, and access policy] --> SDK
    CONTROL --> AGENT
    CONTROL --> REG
    CONTROL --> QUERY
~~~

### Shared contracts

Every telemetry envelope should identify:

~~~text
tenant and trust domain
resource identity and deployment revision
instrumentation scope and schema version
signal type and event/interval timestamps
collection time and collector path
data classification and retention class
trace/correlation identity where applicable
~~~

The control plane publishes immutable, versioned policy. Collectors report desired, received, validated, and active revisions. Invalid policy leaves a valid prior snapshot active. A telemetry outage must not block the application request path unless the application’s compliance contract explicitly requires durable audit acceptance.

### Shared invariants

1. Telemetry is never trusted more than its authenticated producer and collection path.
2. Tenant and classification are part of every queue, storage, cache, query, and export key.
3. Resource and schema identity survive transformations.
4. Backpressure is bounded and has an explicit drop/spill policy per signal class.
5. Redaction happens before data crosses a boundary that is not allowed to see the raw value.
6. Collection reports rejected and dropped data; “export succeeded” is not inferred from absence of errors.
7. Control-plane failure does not silently disable privacy, residency, or tenant policy.
8. Observability infrastructure has independent health signals and failure domains from the services it observes.

The later chapters own signal-specific collection and storage details. This chapter now focuses on traces.

## Trace Workload and Contract

A trace is a graph of spans representing causal work for one logical operation. It is not necessarily a tree: asynchronous fan-out, batching, queues, retries, and workflows require links between contexts.

### Span model

A span contains:

| Field | Meaning |
|---|---|
| Trace and span IDs | Correlation and node identity |
| Parent or links | Causal relationship |
| Kind | Server, client, producer, consumer, or internal boundary |
| Name | Stable low-cardinality operation |
| Start and end | Local interval under stated clock semantics |
| Status | Protocol/instrumentation outcome, not arbitrary log severity |
| Attributes | Bounded typed dimensions |
| Events | Time-stamped occurrences inside the interval |
| Resource/scope | Producing workload and instrumentation library/version |

Span names describe operations such as *HTTP GET /orders/{id}* or *queue consume*, never a raw URL, user ID, SQL statement, or exception message. High-cardinality instance evidence belongs in controlled attributes or linked logs, not operation names.

### Trace invariants

1. A valid trace ID is propagated unchanged across a logical trace boundary.
2. Each recorded span ID is unique within the trace.
3. A child starts from the active context that caused it, not whichever context is globally convenient.
4. Retries are separate attempt spans under one logical operation span.
5. Queue/batch causality uses producer/consumer relationships or links, not a fabricated synchronous parent.
6. Status and errors reflect the span’s own operation; a handled downstream error need not mark every ancestor identically.
7. End timestamps are emitted exactly once or recovered as incomplete.
8. Attribute keys and units follow a versioned semantic convention.
9. Propagated context never grants identity or authorization.
10. Sampling decisions are observable and analyzable as part of the trace contract.

## Context Propagation

### In-process context

Execution context must follow the actual concurrency primitive:

- synchronous call stack;
- async task/future;
- thread or coroutine handoff;
- callback registration;
- stream message;
- scheduled job; or
- explicit workflow/activity invocation.

Thread-local storage fails when logical work moves threads; one global mutable context leaks trace IDs between requests. Libraries should accept or capture an immutable execution context according to the language runtime.

### Network propagation

At an inbound boundary:

1. parse the standard trace context strictly and within header limits;
2. decide whether the boundary is trusted to continue the trace;
3. reject invalid context or start a new trace according to policy;
4. create a server/consumer span;
5. make its context active only for the request lifetime; and
6. inject the resulting context into permitted outbound carriers.

W3C Trace Context standardizes *traceparent* and *tracestate*. The sampled flag is a recommendation, not authority. A public client can set it on every request, so a service may cap or replace the sampling decision.

At trust boundaries, consider starting a new internal trace and linking to the external context. This prevents an attacker from choosing internal trace IDs, forcing expensive sampling, or correlating activity across tenants.

### Asynchronous messaging

For one message:

~~~mermaid
sequenceDiagram
    participant P as Producer
    participant B as Broker
    participant C as Consumer
    P->>P: producer span
    P->>B: inject message trace context
    B-->>C: deliver or redeliver
    C->>C: consumer/process span linked to message context
    C->>C: attempt and domain-operation spans
~~~

The envelope should carry trace context under a defined schema. A retry/redelivery creates a new delivery attempt span linked to the same message-producing context. Do not overwrite a durable domain event’s identity with a transient trace ID.

For a batch containing many messages, one consumer span can link to bounded representative input contexts or the batch operation can emit per-item spans under sampling. Making one item the parent falsely claims it caused the others; linking every item can exceed span limits.

For workflows that outlive trace retention, store stable workflow/run/activity IDs in durable history and link each bounded activity trace. See [Workflow Observability and Replay](../18-workflow-job-systems/09-workflow-observability-replay.md).

### Fan-out and retries

One logical client operation may create multiple attempts:

~~~text
client operation span
  -> attempt 1 span: timeout
  -> attempt 2 span: success
~~~

This distinguishes user-visible request count from transport attempt count. It also makes retry amplification diagnosable without treating parallel hedges as sequential children.

## Sampling Algorithms

Recording every span is often unaffordable. Sampling has two separate decisions:

- **recording:** whether instrumentation creates/retains detailed span data locally;
- **export:** whether the completed data leaves the process or collector.

Dropping early saves CPU and bytes. Deferring export preserves decision flexibility but still pays recording and buffer cost.

### Deterministic head sampling

Use a uniform trace ID and a stable threshold so every service makes a consistent probability decision:

$$
\text{keep} =
\left[
H(\text{trace-id}) < p \cdot 2^k
\right]
$$

where $p$ is the target probability and $H$ yields a $k$-bit uniform value. Deterministic sampling avoids fragmented traces caused by independent coins at every service.

Parent-based policy generally respects an upstream decision inside a trusted system, while applying local limits at untrusted ingress. Probability may vary by operation/tenant class, but the decision rule and effective probability need to travel with or be inferable from the trace so analysts understand bias.

**Strengths:** immediate, low memory, predictable volume, complete retained traces when consistently propagated.

**Weaknesses:** cannot know final latency/status, rare failures can be missed, low-volume routes may receive no samples.

### Tail sampling

A tail sampler groups spans by trace ID, waits for completion or decision timeout, evaluates rules, and retains or drops the trace:

~~~text
on span:
  validate tenant and trace identity
  append to bounded per-trace buffer
  update summary: error, latency, route, attributes, completeness
  if decisive rule fires -> decide early when semantics permit
  else wait until root completion or decision deadline

on deadline:
  decide using available summary
  mark trace incomplete if expected work may still arrive
  export or discard and tombstone decision for late spans
~~~

Candidate rules include errors, latency relative to route SLO, explicit debug authorization, statistically rare operation class, and a deterministic baseline. Rule order and quotas matter: “keep every error” can overload the pipeline during an outage precisely when it is most needed.

**Strengths:** preserves anomalous traces and can enforce per-class quotas.

**Weaknesses:** buffer memory, trace affinity, late spans, partial traces, regional fan-in, decision latency, and outage amplification.

### Hybrid policy

A robust policy often has:

1. deterministic head baseline for population analysis;
2. bounded tail quotas for error/latency classes;
3. minimum reservoir for low-volume operations;
4. per-tenant fairness;
5. emergency rate cap;
6. authorized time-bounded debug sampling; and
7. recorded effective sampling reason/probability.

Biased tail samples cannot be naively used to estimate global error rate or latency distribution. Metrics own population aggregates; traces explain examples. If probability sampling supports estimation, retain inclusion probability and apply statistically valid weighting only for compatible queries.

## Collector Pipeline

### Topology

| Layer | Responsibility | Failure concern |
|---|---|---|
| SDK/local exporter | span lifecycle, local batch, context | application overhead and blocking |
| Node/workload collector | local receive, resource enrichment, first redaction | host/pod blast radius |
| Regional gateway | authentication, tenant routing, tail affinity, durable buffer | regional surge and hot tenant |
| Backend ingest | validation, indexing, retention | storage/query contention |

Not every deployment needs every layer. A local collector reduces application connections and centralizes enrichment. A regional gateway isolates backends and supports tail sampling. Each additional queue adds latency, duplication, and another state boundary.

### Pipeline state

~~~mermaid
stateDiagram-v2
    [*] --> Received
    Received --> Validated
    Received --> Rejected: schema/auth/limit failure
    Validated --> Processed: enrich/redact/sample
    Processed --> Buffered
    Buffered --> Exported
    Buffered --> Dropped: expiry/overflow/policy
    Exported --> Acknowledged
    Exported --> Buffered: retryable failure
~~~

Transport acknowledgement needs defined semantics. If the collector acknowledges after memory receipt, a crash loses accepted spans. If it waits for durable storage, latency and backpressure rise. State the durability class and keep compliance audit logs out of a best-effort trace path.

OTLP can report partial success. Exporters must count rejected spans and not treat a successful transport status as full acceptance.

### Backpressure and retry

Order of preference:

1. apply configured sampling before queue saturation;
2. batch within latency/size bounds;
3. spill to a bounded local or regional write-ahead queue if durability justifies it;
4. retry transient export errors with jitter and a retry budget;
5. shed by explicit signal/tenant/priority policy; and
6. preserve drop counters and reason.

Never let an unbounded telemetry queue exhaust application memory. When the backend is slow, retrying from every SDK can form a synchronized storm; local collectors absorb and coordinate that pressure.

At-least-once export can duplicate spans after an ambiguous acknowledgement. Backends deduplicate by tenant, trace ID, span ID, and producer identity/revision within a defined window. Deduplication must not merge two malicious or broken producers that reused IDs across tenants.

## Baggage, Security, and Privacy

Baggage propagates application-defined key/value context. It is sent to downstream services and can cross organization boundaries. It is not automatically recorded in spans and must never carry:

- credentials, session tokens, or authorization decisions;
- raw personal or health data;
- secrets or encryption keys;
- unrestricted user input;
- high-cardinality values copied to every span; or
- internal topology not intended for the next hop.

Use an allowlist with:

~~~text
key and semantic owner
per-value and total encoded size
allowed ingress and egress trust domains
maximum lifetime/hops where enforceable
whether it may become a span/log/metric attribute
classification and retention
redaction or hashing rule
~~~

At public ingress, drop unknown baggage and validate trace context length/format. At egress to a third party, construct a new allowed carrier. Trace IDs are pseudonymous correlation identifiers but can still enable cross-system tracking; apply access control and retention.

### Attribute cardinality

Bound:

- attributes and events per span;
- key/value length;
- links per span;
- exception/stack size;
- SQL/URL/message capture;
- resource dimensions;
- baggage bytes; and
- dynamic instrumentation.

Truncate with explicit markers rather than silent ambiguity. Prefer route template over raw path and database operation/table over raw statement. Link to controlled logs when detailed evidence is necessary.

## Capacity and Cost Model

Assume:

- $\lambda$ root operations per second;
- $\bar{s}$ recorded spans per trace;
- $p_h$ head-recording probability;
- $p_t$ fraction of recorded traces retained by tail policy;
- $\bar{b}$ encoded bytes per span after batching/compression;
- $T_d$ tail decision horizon;
- $r$ replication factor; and
- $T_r$ retention duration.

Recorded span rate:

$$
\lambda_{\text{span,recorded}} = \lambda \bar{s} p_h.
$$

Exported byte rate:

$$
B_{\text{export}} \approx
\lambda \bar{s} p_h p_t \bar{b}.
$$

Tail-buffer memory before overhead:

$$
M_{\text{tail}} \approx
\lambda \bar{s} p_h \bar{b} T_d.
$$

Retained raw storage before indexes/compaction:

$$
S_{\text{retained}} \approx
B_{\text{export}} T_r r.
$$

State assumptions explicitly: retries and fan-out change $\bar{s}$; incidents change $p_t$ and span size; replication and indexes multiply storage; compression ratio depends on attribute repetition; late spans extend effective $T_d$.

### Size exceptional load

- outage causes error-tail rule to match most traces;
- one tenant sends attacker-chosen sampled headers;
- rollout doubles spans through duplicate instrumentation;
- collector recovery flushes disk queues;
- region partition forces tail decisions on incomplete traces;
- a high-fan-out request creates thousands of spans; and
- dynamic debug sampling overlaps peak traffic.

Apply per-tenant and per-rule quotas so anomaly retention cannot consume the deterministic baseline. Budget telemetry CPU, network, and storage as a percentage of service resources and measure actual overhead.

## Multi-Region Operations

Prefer regional ingestion and buffering. Cross-region synchronous export makes observability share fate with WAN failure and can violate residency.

A regional trace can be queried globally by:

- federated query over regional stores;
- asynchronous replication of redacted retained traces;
- global index pointing to regional data; or
- explicit trace-home routing.

Cross-region traces challenge tail sampling: all spans must meet at one decision point or regional samplers make partial decisions. Options:

- route trace IDs deterministically to a home region;
- keep per-region fragments and link them in query;
- make a head decision globally and skip tail completeness;
- tail-sample only within a region and mark partial; or
- replicate span summaries to a decision service while retaining payload locally.

Choose based on WAN cost, residency, decision delay, and acceptable completeness. Never block application traffic to preserve a globally complete trace.

Clock synchronization improves timelines but trace parent/link causality is more reliable than wall-clock ordering. Record collection time and clock-skew diagnostics; do not “fix” timestamps in a way that hides source evidence.

## Failure Trace

### Tail sampler collapses during an outage

~~~text
database latency rises
-> error/latency tail rules match most traces
-> retained volume exceeds normal capacity
-> tail buffers and exporter queues fill
-> collectors retry the overloaded backend
-> traces from the incident are dropped while healthy baseline traffic crowds queues
~~~

**Detection**

- retained fraction by sampling reason;
- tail buffer bytes/traces and oldest age;
- decision timeout and incomplete-trace rate;
- exporter queue and retry age;
- dropped spans by tenant/reason; and
- backend ingest saturation.

**Response**

1. Preserve a bounded deterministic baseline.
2. Enforce per-rule and per-tenant tail quotas.
3. Lower expensive attribute/event capture before dropping whole trace classes.
4. Spill only within a bounded durable queue.
5. Stop multiplicative exporter retries.
6. Record the policy revision and loss interval for incident evidence.

### Broken async propagation produces plausible but false traces

~~~text
worker reuses ambient context from previous message
-> new message spans attach to another tenant’s trace
-> trace UI shows a causal path that never occurred
-> baggage and identifiers leak across requests
~~~

**Controls:** explicit message context extraction, context reset in finally/defer paths, tenant-bound propagation tests, invalid-context counters, and cross-tenant trace validation.

## Operating the Trace System

The trace pipeline needs its own service levels:

- ingestion acceptance and rejection by tenant;
- end-to-end availability delay from span end to query;
- trace completeness under known instrumentation;
- collector active configuration convergence;
- sampling probability/reason distribution;
- queue age, retry, spill, drop, and partial-success rate;
- backend index/query latency and failed queries;
- schema/version drift; and
- privacy/redaction policy violations.

Use an independent external or minimal-path signal to detect total telemetry failure. A pipeline cannot reliably alert on its own complete disappearance unless something outside it expects a heartbeat.

## Verification Strategy

| Test layer | What to prove |
|---|---|
| Propagator conformance | Valid/invalid W3C fields, case, duplicates, limits, and unknown versions |
| Async context tests | Threads, coroutines, queues, batches, retries, cancellation, and context cleanup |
| Graph invariants | Unique span IDs, valid parent/link, retry attempts, and bounded fan-out |
| Sampling property tests | Stable deterministic decisions, configured probability, parent behavior, and quota fairness |
| Tail model tests | Late spans, timeout, incomplete traces, hot trace, region split, and tombstone behavior |
| Collector fault tests | Partial success, duplicate export, crash, disk full, backend throttle, and corrupt policy |
| Privacy tests | Baggage/attribute allowlist, egress stripping, redaction before export, and tenant isolation |
| Load tests | Normal peak, incident match-all, reconnect/flush storm, debug sampling, and giant traces |
| Compatibility tests | SDK, semantic schema, OTLP, collector, and backend versions |
| Query validation | Known synthetic trace remains searchable and correlated end to end |

Inject a synthetic trace across each supported protocol and failure domain. Verify not only that a trace appears, but that parent/link semantics, resource identity, sampling reason, attributes, and redaction are correct.

## Decision Framework

1. Which user or operator questions require causal traces rather than population metrics or event logs?
2. Where are trust boundaries that should start a new trace or strip baggage?
3. Which synchronous, asynchronous, batch, retry, and long-workflow relationships must be modeled?
4. What baseline sampling supports unbiased population examples?
5. Which anomaly rules deserve tail retention, and what are their quotas?
6. What decision horizon and incomplete-trace behavior are acceptable?
7. Which collector layers are required for isolation, enrichment, buffering, and residency?
8. What durability is promised at each acknowledgement?
9. What attribute, event, baggage, and trace-size budgets apply?
10. Can the system survive an incident-driven match-all sampling surge?
11. How are regional fragments queried without making application traffic depend on global tracing?
12. Can operators observe loss, schema drift, policy revision, and trace completeness?

## Key Takeaways

1. A trace is a causal graph, not necessarily a synchronous tree or total timeline.
2. Propagation must follow actual concurrency and messaging boundaries.
3. External trace context and baggage are untrusted correlation data, never authorization.
4. Head sampling is cheap and predictable; tail sampling is selective but stateful and failure-prone.
5. Hybrid sampling preserves a deterministic baseline while bounding anomaly retention.
6. Metrics, not biased trace samples, own population-level rates and percentiles.
7. Collector acknowledgements, queues, retries, partial success, and drops need explicit semantics.
8. Tail-buffer and outage-match capacity—not average trace traffic—drive design.
9. Regional collection protects latency and residency; global trace completeness is a tradeoff.
10. Verify causal correctness, privacy, loss behavior, and queryability end to end.

---

## References

- [W3C Trace Context](https://www.w3.org/TR/trace-context/) — interoperable trace propagation, mutation, privacy, and security
- [W3C Baggage](https://www.w3.org/TR/baggage/) — distributed application context format and limits
- [OpenTelemetry Trace Specification](https://opentelemetry.io/docs/specs/otel/trace/) — span model, SDK processing, sampling, and export
- [OpenTelemetry Context Specification](https://opentelemetry.io/docs/specs/otel/context/) — immutable execution-scoped context
- [OTLP Specification](https://opentelemetry.io/docs/specs/otlp/) — signal transport, retry, throttling, and partial success
- [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/) — stable cross-signal attribute and operation semantics
- [Dapper: A Large-Scale Distributed Systems Tracing Infrastructure](https://research.google/pubs/dapper-a-large-scale-distributed-systems-tracing-infrastructure/) — production tracing architecture and sampling
- [Workflow Observability and Replay](../18-workflow-job-systems/09-workflow-observability-replay.md) — correlation for workflows that outlive a trace
- [Logging Architecture](./03-logging.md) — discrete evidence, ordering, retention, and trace correlation
- [Metrics Systems](./02-metrics-monitoring.md) — population aggregation, histograms, and storage
