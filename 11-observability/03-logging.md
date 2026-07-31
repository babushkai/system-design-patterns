# Production Logging Architecture

## TL;DR

A production log is a typed event record, not a formatted sentence. Its contract defines event name, schema, severity, event and observation time, resource identity, correlation, classification, retention, and expected volume. Logs are valuable because they preserve discrete evidence that metrics aggregate away and traces may sample out.

Distributed logs do not have one trustworthy total order. Preserve source sequence, event time, observed/ingest time, resource incarnation, and causal identifiers. Use monotonic clocks for local durations and trace/message/workflow links for causality. Do not sort by timestamp and pretend clock skew disappeared.

Collection must survive file rotation, container death, duplicate delivery, partial records, backpressure, and backend outage without blocking the application indefinitely. Index only fields needed for interactive search; tier the full immutable event body into cheaper retention. Redaction and tenant routing happen before data crosses an unauthorized boundary.

[Distributed Tracing and Telemetry Pipelines](./01-distributed-tracing.md) defines the shared telemetry plane. This chapter covers event schema, collection state, ordering, indexing, retention, privacy, and correlation; [Metrics](./02-metrics-monitoring.md) covers aggregates and [Incident Management](./07-incident-management.md) covers evidence handling.

---

## Log Workload and Contract

An event envelope:

~~~json
{
  "event_name": "payment.authorization.completed",
  "schema_version": "2.1",
  "event_time": "2026-07-18T10:15:22.381Z",
  "observed_time": "2026-07-18T10:15:22.417Z",
  "severity": "INFO",
  "tenant": "tenant-a",
  "resource": {
    "service": "payments",
    "instance": "instance-7",
    "deployment": "sha256:..."
  },
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "request_id": "req_8f2",
  "attributes": {
    "operation": "authorize",
    "outcome": "declined",
    "reason_code": "insufficient_funds"
  },
  "body": "Authorization completed with a domain decline"
}
~~~

The example intentionally excludes raw card data, customer email, credential, request body, and free-form exception as indexed fields.

### Event invariants

1. Event name and schema version identify stable semantics.
2. Resource and tenant identity come from authenticated runtime/collector context where possible.
3. Event time and observation time remain distinct.
4. A source sequence/incarnation disambiguates restart and local order when required.
5. Severity describes operational significance under a documented policy; it is not the event type.
6. Trace, request, message, workflow, and domain IDs are correlation fields, not replacements for event identity.
7. Sensitive fields carry classification and are redacted/tokenized before unauthorized export.
8. Collection can duplicate; consumers and indexes tolerate at-least-once records.
9. Truncation, parse failure, and dropped records are explicit events/metrics.
10. Retention and legal hold operate by classified field/event family, tenant, and jurisdiction.

## Event Schema Design

### Stable event name, structured attributes, bounded body

Use the event name for machine queries and the body for human context. Avoid encoding variables into the name:

~~~text
good:
  event_name = database.query.failed
  db.system = postgresql
  error.type = timeout

bad:
  event_name = Query_to_orders_for_user_842_failed_after_3784ms
~~~

A schema registry or reviewed convention defines:

- required and optional fields;
- types, units, and bounded enums;
- classification and indexing permission;
- producer and semantic owner;
- correlation fields;
- maximum lengths/collection sizes;
- compatibility rules; and
- retention class.

### Event families

| Family | Purpose | Durability/privacy |
|---|---|---|
| Application diagnostic | Explain code/dependency behavior | Best effort, bounded retention, sampled/rate-limited |
| Domain event evidence | Record a meaningful state transition | May need durable business store; log is not source of truth |
| Security event | Authentication, authorization, tamper, admin activity | Restricted access, integrity, defined retention |
| Audit record | Evidence for a control or regulated action | Separate durable pipeline/immutability may be required |
| Access record | Request boundary and outcome | High volume, strict redaction, useful for correlation |
| Platform/system | Scheduler, kernel, proxy, database events | Resource enrichment and independent trust |

Do not claim compliance-grade audit durability for ordinary asynchronous application logs. If an effect must not commit without audit evidence, design a transactional audit/event path with an explicit failure contract.

### Severity

A useful severity scale distinguishes:

- fine-grained diagnostic detail, normally disabled or sampled;
- normal lifecycle/domain event;
- degraded but handled condition requiring no immediate human;
- operation failure that affects this request/job;
- process/fleet integrity failure requiring escalation.

Severity does not decide paging; alert policy owns that. One expected domain rejection may be INFO, while a sustained rate becomes an SLO signal. Repeatedly logging the same retry failure at every layer inflates noise; log one logical outcome plus attempt evidence where needed.

### Schema evolution

Add optional fields only when old consumers ignore them safely. Adding a new enum requires an unknown branch. Renaming fields uses an explicit schema transform or dual-read window. Never reuse an event name for different semantics.

Collectors can transform known schema versions, but must retain original provenance and report failures. A malformed new event should not be silently parsed into the previous schema.

## Time, Order, and Causality

### Four useful notions of time

- **event time:** when the producer says the event occurred;
- **observed time:** when an instrumentation/collector observed it;
- **ingest time:** when storage accepted it;
- **monotonic elapsed time:** duration measured within one process/clock domain.

Event time is best for user timelines but vulnerable to clock error and delayed buffers. Ingest time is reliable for storage operations but reorders delayed events. Preserve both.

### No global total order

Given events A and B from different machines, timestamps alone may not establish order. Prefer:

- trace parent/link causality;
- message/event sequence within a partition or aggregate;
- workflow history position;
- database transaction/commit sequence;
- producer incarnation plus local sequence; and
- explicit “unknown/concurrent” when no relation exists.

Clock synchronization reduces confusion but does not create consensus. Record clock-offset health and avoid rewriting source timestamps destructively.

### Source sequence

For a producer that needs gap detection:

~~~text
(resource identity, incarnation, sequence)
~~~

Sequence resets only with a new incarnation. The collector can detect duplicates, gaps, and reordering. A global auto-increment in the logging backend adds ingest order, not occurrence order.

## Collection and Delivery State

### File/stdout collection

A local agent tails files or container streams. Its checkpoint includes:

~~~text
tenant/resource identity
filesystem/device and file identity
path and rotation generation
byte offset
parser/schema version
last acknowledged downstream batch
~~~

Path alone is insufficient: rotation can rename the old file and create a new one at the same path. Copy-truncate can shrink a file under the same identity. The collector needs explicit policies for rename, truncate, deletion, and files that disappear before acknowledgement.

Multiline reconstruction is risky. Prefer one framed structured record per line or a protocol with length framing. If legacy stack traces require multiline parsing, bound record bytes and wait time; emit parse/truncation evidence.

### Direct network export

An application appender can emit typed records directly to a local collector. It avoids file parsing but must not make every log call a synchronous remote dependency.

Use:

- bounded in-process queue;
- batch by byte and time;
- nonblocking or bounded-block policy per event class;
- local collector connection;
- explicit partial rejection/drop count;
- flush only at controlled lifecycle boundaries; and
- separate durable audit channel where required.

### Delivery state machine

~~~mermaid
stateDiagram-v2
    [*] --> Emitted
    Emitted --> Buffered
    Buffered --> Parsed
    Buffered --> Dropped: local overflow/policy
    Parsed --> Redacted
    Parsed --> Quarantined: invalid schema
    Redacted --> Routed
    Routed --> Persisted
    Routed --> Retrying: transient backend failure
    Retrying --> Persisted
    Retrying --> Dropped: expiry/storage exhaustion
    Persisted --> Indexed
    Persisted --> Archived
~~~

Each boundary states acknowledgement durability. At-least-once retry can duplicate records. Assign an event/record ID when meaningful or deduplicate using authenticated producer identity, incarnation, sequence, and content hash within a bounded window.

## Buffering and Backpressure

Logging must not convert an observability outage into an application outage unless a specific audit contract requires it.

### Priority classes

| Class | Example | Overflow behavior |
|---|---|---|
| Required audit/security | privileged admin effect | durable admission or reject the governed effect |
| Operational error evidence | request failure summary | bounded disk/WAL, then explicit loss emergency |
| Normal lifecycle/access | successful request | sample/aggregate/drop under pressure |
| Debug detail | variable/stack diagnostics | first to shed |

Do not use severity alone as a durability class. A high-severity attacker-controlled message can exhaust storage; a low-severity audit event may be mandatory.

### Queue design

Bound:

- producer queue records and bytes;
- local disk use and age;
- collector tenant queues;
- retry attempts and elapsed horizon;
- batch record/byte count;
- maximum event and expanded body;
- quarantine storage; and
- flush concurrency.

Reserve capacity for collector self-health and loss records. If disk fills, prevent log storage from consuming filesystem space required by the application/database. Separate quotas or filesystems where the risk justifies it.

Backpressure options:

1. drop debug/normal records by declared priority;
2. sample repetitive event families;
3. aggregate counts into metrics;
4. spill bounded records to disk;
5. reject new diagnostic configuration;
6. block only for contractually durable audit; and
7. expose exact loss interval and reason.

## Indexing, Storage, and Query

### Tiered representation

~~~mermaid
flowchart LR
    INGEST[Validated redacted records] --> HOT[(Hot indexed store)]
    INGEST --> FULL[(Compressed full-event blocks)]
    HOT --> SEARCH[Interactive search]
    FULL --> SCAN[Batch/forensic scan]
    FULL --> ARCHIVE[(Archive/hold)]
    HOT --> EXPIRE[Short index retention]
    FULL --> EXPIRE2[Class-based body retention]
~~~

Index fields used for selective queries:

- tenant and resource/service;
- event name and severity class;
- bounded outcome/error type;
- deployment/region;
- trace/request/message/workflow correlation;
- event-time range; and
- schema version.

Do not index every arbitrary JSON key. Dynamic mapping can create an index field per user key, exhaust metadata, and produce incompatible types. Unknown attributes stay in a structured blob or a controlled schema-on-read tier.

### Partitioning

Partition by authenticated tenant, time, and optionally service/event family. Time partitioning supports retention; tenant partitioning supports isolation; excessive tiny partitions hurt metadata/query performance.

Late records need an accepted lateness window. After a hot partition closes, route them to a late-data segment or controlled rewrite rather than silently losing them.

### Query contract

Interactive search has:

- maximum time range;
- matched partitions/bytes estimate;
- result and aggregation limits;
- timeout/cancellation;
- tenant/access filters applied before query;
- field-level redaction;
- query audit;
- asynchronous export for large results; and
- stable pagination/continuation if results exceed one response.

A leading-wildcard full-text search across months is batch work, not an unlimited interactive query.

## Retention, Redaction, and Evidence

### Classification before storage

Classify fields as public, internal, confidential, personal, secret, regulated, or audit-controlled under the organization’s taxonomy. The policy decides:

- whether collection is allowed;
- transformation/tokenization;
- searchable versus body-only storage;
- allowed regions/exporters;
- access roles;
- retention/deletion;
- legal hold; and
- incident disclosure handling.

### Redaction layers

1. **At source:** avoid creating the sensitive value.
2. **Local collector:** remove/tokenize before leaving workload/host.
3. **Regional gateway:** enforce tenant policy and detect violations.
4. **Storage/query:** field-level access and defense-in-depth masking.
5. **Export:** audience-specific redaction.

Regex replacement over a rendered message is not sufficient. Use structured field classification, allowlisted headers, parameterized database metadata, and secret scanning as a fallback.

Hashing a low-entropy identifier is reversible by enumeration. Use keyed pseudonymization/tokenization where correlation is required, rotate keys under a documented linkage/retention policy, and restrict reidentification.

### Deletion and legal hold

Immutable block storage complicates selective deletion. Options:

- tenant-separated encryption and cryptographic erasure;
- deletion/tombstone indexes applied to queries and later compaction;
- small partition boundaries;
- field token vault deletion; or
- avoid collecting the field.

Legal hold freezes specified data under authorized scope while ordinary retention continues elsewhere. Both deletion and hold actions are audited and tenant-isolated.

## Correlation Across Signals

Use shared:

- resource identity and deployment revision;
- trace and span ID;
- request ID at public boundaries;
- message/event and workflow/run ID;
- operation/domain resource ID under access policy;
- region/zone/cell; and
- telemetry schema version.

Trace IDs in logs allow a sampled trace to find related events, but unsampled requests still need useful logs/metrics. Metrics exemplars can link a histogram observation to a representative trace without turning trace ID into a metric label.

Correlation IDs are high-cardinality indexes with retention/access cost. Do not emit user-controlled correlation values without validation or use them as authorization.

## Capacity and Cost Model

Assume:

- $\lambda_e$ emitted events per second;
- $\bar{b}_e$ average encoded bytes after source formatting;
- $p$ retained fraction after sampling/filtering;
- $c$ compression ratio expressed as compressed/raw;
- $i$ hot-index multiplier relative to compressed body;
- $r$ replication factor;
- $T_h$ hot retention seconds; and
- $T_a$ archive/full-body retention seconds.

Ingest byte rate:

$$
B_{\text{ingest}} = \lambda_e \bar{b}_e.
$$

Retained compressed body:

$$
S_{\text{body}} \approx
\lambda_e \bar{b}_e p c T_a r.
$$

Hot indexed storage:

$$
S_{\text{hot}} \approx
\lambda_e \bar{b}_e p c (1+i) T_h r.
$$

These omit block metadata, write amplification, caches, object versioning, and query temporary space. Compression worsens with high-cardinality/free-form values. Index multiplier depends on indexed fields and tokenization.

### Backlog sizing

For backend outage $T_o$ and retained incoming rate $B_r$:

$$
Q_{\text{required}} \ge B_r T_o
$$

plus headroom for burst and replay metadata. Recovery must handle live ingest plus catch-up:

$$
B_{\text{backend capacity}}
>
B_r + B_{\text{replay}}.
$$

Unlimited outage tolerance means unlimited disk and is not a design. Declare queue horizon and loss policy by event class.

### Volume drivers

- access log per request and retry attempt;
- stack traces repeated at multiple layers;
- raw request/response bodies;
- debug mode across a large fleet;
- unbounded arrays/exception chains;
- multiline/parser expansion;
- audit duplication;
- high-cardinality indexes; and
- incident backlog replay.

Measure bytes per event family and useful query frequency. Remove or aggregate events that cost heavily but never answer an operational, security, product, or compliance question.

## Multi-Region Operations and Security

Collect regionally for latency, outage independence, and residency. Store a region-local immutable copy or bounded queue before cross-region replication where evidence requirements justify it.

Global search can federate:

- query plan sent to authorized regional stores;
- redacted/index metadata replicated globally;
- asynchronous full-event replication for allowed classes; or
- tenant home-region routing.

Results include coverage and watermark per region. A global search during partition must say which regions are missing.

Protect ingestion from spoofing and injection:

- authenticate workloads/collectors;
- derive tenant/resource identity out of band;
- escape control characters in text renderers;
- bound event size/nesting;
- reject schema bombs and dynamic fields;
- encrypt transport/storage;
- authorize query/export at field level;
- audit searches and bulk downloads; and
- isolate query/ingest capacity by tenant.

Logs often contain the most sensitive operational data. Incident access is time-bounded and reviewed, not a blanket bypass of tenant/privacy policy.

## Failure Traces

### Logging disk fills the application filesystem

~~~text
backend unavailable -> local agent retries
-> application keeps writing verbose access/debug logs
-> shared disk queue grows without quota
-> database/application cannot write state
-> service fails because observability storage consumed its disk
~~~

**Controls:** separate quota/filesystem, priority shedding, bounded queue age/bytes, debug kill switch, disk reservation, and explicit loss telemetry.

### File rotation loses and duplicates evidence

~~~text
collector checkpoints path + offset
-> file rotates and a new file reuses the path
-> collector resumes old offset in new file, skipping records
-> retry also rereads renamed old file, duplicating others
~~~

**Controls:** file identity plus generation, rotation-aware checkpoint, truncate detection, downstream idempotent record identity, and rotation fault tests.

### Sensitive token enters every replica and archive

~~~text
new error handler logs full request headers
-> bearer credential reaches local file, regional collector, hot index, archive, and support export
-> deleting one indexed copy leaves replicas and queued batches
~~~

**Controls:** source allowlist, structured redaction before export, secret detection/quarantine, credential revocation, deletion map across every copy, access audit, and post-incident schema test.

### Clock skew produces a false root cause

~~~text
service B clock is ahead
-> its failure log sorts before service A’s causal request
-> responder blames B as initiator and rolls back wrong component
~~~

**Controls:** trace/message causality, event and observed times, clock-offset signal, source sequence, and timeline annotations.

## Operating the Logging System

Track:

- emitted/accepted/parsed/redacted/quarantined/persisted/indexed/dropped records and bytes;
- collection checkpoint age, file discovery, rotation/truncate, and parser failure;
- producer/local/regional queue bytes and oldest age;
- backend ingest acknowledgement, retry, duplicate, and late event;
- index fields/cardinality, segment/block count, compaction, and retention lag;
- query scanned bytes, partitions, queue, cancel, and bulk export;
- redaction/secret violations and policy revision;
- region coverage/watermark and replication lag; and
- storage, indexing, query, and egress cost per useful event family.

Maintain a minimal independent sink for logging-pipeline health or a heartbeat expected outside the pipeline.

## Verification Strategy

| Test layer | What to prove |
|---|---|
| Schema contract | Types, units, required fields, unknown fields/enums, bounds, and version migration |
| Context tests | Request/trace/message/workflow correlation survives async execution without leakage |
| Time/order tests | Clock skew, delayed delivery, restart, sequence gap, and concurrent events stay explicit |
| File collection | Rename, copy-truncate, delete, container restart, partial line, multiline timeout, and checkpoint crash |
| Delivery faults | Duplicate, partial rejection, backend throttle, outage, disk full, and replay |
| Backpressure | Priority shedding preserves required classes and never grows unbounded |
| Privacy/security | Header/body allowlists, tokenization, secret scan, query ACL, tenant substitution, and export redaction |
| Retention | Expiry, selective deletion, legal hold, replica/archive coverage, and cryptographic erasure |
| Query | Index selection, time/byte limits, cancellation, field redaction, and regional partial results |
| Load | Peak access logs, debug enablement, giant errors, high-cardinality fields, and catch-up replay |

Keep golden encoded events for every schema and verify the exact redacted representation at each boundary. Test that forbidden values never appear, not only that a redaction function was called.

## Decision Framework

1. Which discrete event cannot be answered reliably by metrics or traces?
2. Is this diagnostic, security, audit, access, domain, or platform evidence?
3. What stable event name/schema and bounded attributes express it?
4. Which time, sequence, and causal fields are available?
5. What acknowledgement/durability does the event class require?
6. Can collection duplicate, and how is identity/dedup handled?
7. What happens when every queue and disk boundary is full?
8. Which fields may be indexed, retained, exported, or correlated?
9. Where must redaction/tokenization occur before a trust boundary?
10. What hot, full-body, archive, deletion, and hold periods apply?
11. How will regional coverage and delayed events appear in queries?
12. Does measured query value justify the event and index cost?

## Key Takeaways

1. Logs are versioned typed events; message strings are a presentation field.
2. Event, observed, and ingest time answer different questions.
3. Causal IDs and source sequences are safer than assuming a global timestamp order.
4. File identity and checkpoint semantics determine whether rotation loses or duplicates logs.
5. At-least-once collection requires duplicate-tolerant storage/query.
6. Buffering is bounded and prioritized; ordinary logging must not exhaust application resources.
7. Index a controlled subset and retain compressed full events separately.
8. Redact before unauthorized export and design deletion across replicas, queues, and archives.
9. Regional collection protects outage visibility and residency; global queries expose coverage.
10. Verify exact encoded/redacted events under rotation, outage, replay, and privacy attacks.

---

## References

- [OpenTelemetry Logs Data Model](https://opentelemetry.io/docs/specs/otel/logs/data-model/): event/observed timestamps, severity, body, attributes, trace context, and resources
- [OpenTelemetry Logging Specification](https://opentelemetry.io/docs/specs/otel/logs/): collection, correlation, appenders, and legacy sources
- [OTLP Specification](https://opentelemetry.io/docs/specs/otlp/): transport acknowledgement, retry, throttling, and partial success
- [RFC 5424: The Syslog Protocol](https://www.rfc-editor.org/rfc/rfc5424): structured event transport, timestamps, and origin
- [RFC 3339: Date and Time on the Internet](https://www.rfc-editor.org/rfc/rfc3339): interoperable timestamps
- [OWASP Logging Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html): security event design, injection, sensitive data, verification, and protection
- [NIST SP 800-92: Guide to Computer Security Log Management](https://csrc.nist.gov/pubs/sp/800/92/final): log infrastructure, retention, protection, and operations
- [Distributed Tracing and Telemetry Pipelines](./01-distributed-tracing.md): shared collectors, context, and trace correlation
- [Metrics Systems and Monitoring](./02-metrics-monitoring.md): aggregate event rates and pipeline health
- [Incident Command and Learning](./07-incident-management.md): evidence timelines, access, preservation, and postmortems
