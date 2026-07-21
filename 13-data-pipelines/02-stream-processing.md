# Stream Execution: Time, State, Recovery, and Backpressure

## TL;DR

A streaming system repeatedly updates state while input remains open. Correctness is the contract among **event time, progress estimation, allowed lateness, emitted revisions, durable state, replayable input, and sink effects**—not a choice between “real time” and accuracy.

An event may be processed more than once after failure even when operator state is exactly-once. A watermark may trigger an on-time result even though a later event can still arrive. A checkpoint can recover the dataflow while an external API has already performed an irreversible action. State these boundaries explicitly; framework labels cannot state them for you.

---

## 1. Specify the Streaming Contract

Before selecting Kafka, Flink, Beam, or another runtime, write down:

| Contract field | Required decision |
|---|---|
| **Event identity** | Stable event ID, source coordinate, key, and producer identity used for ordering or deduplication. |
| **Source replay** | Retention, offset semantics, truncation behavior, and whether replay reproduces the original bytes and schema. |
| **Time domain** | Event, ingestion, or processing time; timezone and timestamp quality. |
| **Progress** | Watermark generation, idle-source treatment, alignment, and expected out-of-orderness. |
| **Emission** | Append, update, or retraction; early/on-time/late triggers and accumulation mode. |
| **State** | Keying, state schema, timers, TTL, maximum cardinality, and legal deletion. |
| **Recovery** | Checkpoint interval, recovery point, recovery time, and compatible source/sink protocol. |
| **Sink effect** | Transactional commit, idempotent versioned upsert, deduplication, or explicitly at-least-once effect. |
| **Overload** | Buffer limit, backpressure behavior, load shedding, and what may be delayed or dropped. |

### Invariants

Required invariants:

1. Each event is assigned to a deterministic key and time domain.
2. Operator state and source positions restore to one consistent recovery cut.
3. The lateness policy defines when state is retained, updated, side-output, or discarded.
4. A sink can reject an older replay after a newer version is visible.
5. State is bounded by a business rule, not only by available disk.
6. Input retention covers the full recovery and replay envelope.
7. Backpressure slows or rejects work in a declared way rather than growing memory without bound.

---

## 2. Continuous Execution versus Replay

Batch and stream execution compose in three ways:

- **Continuous execution** minimizes observation-to-result latency but carries live state and checkpoint operations.
- **Bounded replay** reads a closed offset range into a new result generation.
- **Dual serving paths** may combine a slow base with a fast delta, but they must share a precise merge boundary and semantics.

A stream-only, Kappa-style rebuild works when the retained log is the authoritative input and a replacement deployment can catch up before retention or the recovery deadline. It fails when mutable lookups cannot be reproduced, historical schemas cannot be decoded, state rebuild exceeds the deadline, or the sink cannot host a shadow generation.

A batch correction path is justified when a global algorithm needs bounded input, the event log does not retain enough history, or a periodic source snapshot is the only reproducible truth. It does **not** justify separately implementing every business rule. Reuse one logical specification where possible and test equality at the same input boundary.

---

## 3. Data Plane and Control Plane

~~~mermaid
flowchart LR
    subgraph DP["Data plane"]
        S[("Replayable source<br/>partitioned log")]
        O1["Parse / validate"]
        O2["Keyed operators<br/>state and timers"]
        O3["Window / join"]
        K[("Sink<br/>transaction or versioned upsert")]
        S --> O1 --> O2 --> O3 --> K
    end

    subgraph CP["Control plane"]
        JM["Coordinator<br/>DAG, assignments, barriers"]
        CS[("Checkpoint storage<br/>state + source positions")]
        MD["Deployment metadata<br/>operator IDs, schemas, policy"]
        OBS["Lag, watermark, state,<br/>checkpoint, sink metrics"]
        JM --> CS
        MD --> JM
    end

    JM -.coordinates.-> S
    JM -.coordinates.-> O2
    JM -.coordinates.-> K
    DP -.telemetry.-> OBS
~~~

The data plane moves records and updates keyed state. The control plane assigns partitions, initiates checkpoints, persists deployment metadata, restores tasks, and coordinates sink commits. A healthy source does not imply a healthy checkpoint coordinator; a READY job does not imply the sink has committed current results.

Source parallelism is commonly bounded by source partitions. Downstream operators may use a different parallelism after redistribution. Adding consumers beyond the source partition count cannot increase source read concurrency, while adding downstream key groups may still help a costly transformation.

---

## 4. Time Semantics: Which Question Are You Answering?

An event carries several possible clocks:

- **Event time:** when the business event occurred according to its producer.
- **Ingestion time:** when a platform first accepted it.
- **Processing time:** when an operator observed it.

Processing time is not “wrong.” It answers a different question. A processing-time window measures what the processor saw during an interval; an event-time window measures what producers say occurred during that interval. Retries, queueing, offline clients, clock skew, and network batching make those populations differ.

Choose processing time for operational questions such as current arrival rate when replay reproducibility is unnecessary. Choose event time for user sessions, billing periods, and historical analytics, then define timestamp trust:

- What if the producer clock is in the future?
- Which timestamp wins when events traverse multiple systems?
- Can a correction change event time?
- Is timezone conversion part of ingestion or query?
- Does a replay preserve the original timestamp?

### Windows

Common logical groupings are:

- **Tumbling:** fixed, non-overlapping intervals.
- **Sliding:** fixed intervals emitted at a smaller step, so records contribute to multiple windows.
- **Session:** activity separated by an inactivity gap; late events may merge previously distinct sessions.
- **Global:** one unbounded logical window, useful only with triggers and bounded state logic.

Window assignment alone does not say when results appear or whether they can change.

---

## 5. Watermarks, Triggers, and Revisions

A watermark is a runtime estimate of event-time progress: the system expects that most future events have timestamps later than the watermark. It is **not proof** that no earlier event will arrive.

For a multi-input or parallel operator, the effective watermark is generally constrained by the minimum watermark among non-idle inputs. One silent partition can therefore hold every downstream event-time timer. Idle-input detection prevents a genuinely idle partition from blocking progress, but a wrong idleness timeout can advance results while a merely slow partition is still active.

Document the watermark generator:

- bounded out-of-orderness, source-provided progress, or custom observation;
- per-partition generation and downstream combination;
- future-timestamp validation;
- idle partition detection;
- alignment for sources that advance at very different rates.

### Triggers and accumulation

The trigger answers **when to emit**:

- early processing-time panes for responsiveness;
- an on-time pane when the watermark passes the window end;
- late panes when accepted late input arrives.

The accumulation mode answers **what each emission means**:

- **discarding:** each pane contains only new contributions;
- **accumulating:** each pane contains the full result so far;
- **retracting/changelog:** a later record withdraws or updates a previous result.

The sink and consumer must understand this mode. Appending accumulating panes as if they were independent facts double-counts. A dashboard may accept revisions; a one-time payment cannot simply retract a transfer.

### Finalization and state cleanup

Allowed lateness is both a product policy and a state-retention policy. When the runtime removes window state, a still-later event can only be dropped, sent to a correction path, or recomputed from another store. “Final” therefore means final under a declared policy, not proof that reality cannot change.

---

## 6. Stateful Operators

State may include:

- keyed values, maps, lists, and aggregates;
- event-time and processing-time timers;
- window contents and trigger metadata;
- join buffers waiting for a counterpart;
- deduplication IDs or source versions;
- model or rule versions associated with prior events.

Partition keyed state by a stable key so all updates for the key reach one logical state owner. Runtimes often subdivide the key space into fixed key groups so rescaling transfers groups rather than rehashing individual records.

### Bound state deliberately

For each state family, estimate:

> state bytes ≈ active keys × bytes per key + buffered events × bytes per event + indexes and backend overhead

Then explain what bounds active keys and buffered events:

- TTL since last activity;
- window end plus allowed lateness;
- maximum join interval;
- source-version horizon for deduplication;
- explicit business closure event.

TTL changes semantics. If deduplication state expires before a replayed event returns, the event is no longer deduplicated. If join state expires early, a valid late counterpart is lost. Treat TTL changes as data migrations, not only performance tuning.

Hot keys remain indivisible under ordinary key partitioning. Mitigate them with hierarchical aggregation, key splitting with a correct merge phase, isolated processing, or a redesigned business key. Rescaling cannot divide one key’s state without algorithm support.

---

## 7. Checkpoints, Barriers, and Restore

A checkpoint records a consistent cut of:

- operator and timer state;
- source positions from which replay resumes;
- in-flight channel data when the algorithm includes it;
- pending sink transactions or committables.

With **aligned checkpoints**, barriers flow through the graph. An operator waits for a barrier on each input and temporarily blocks inputs that arrived first, preventing post-barrier records from entering the snapshot. Under backpressure or skew, alignment can become the dominant checkpoint delay.

With **unaligned checkpoints**, in-flight buffers may be captured so barriers can overtake queued data. This reduces barrier delay under backpressure but increases checkpoint I/O and recovery work. It does not remove the underlying throughput bottleneck.

Incremental checkpoints upload changed state rather than every state file, but compaction and shared-file lifetime affect actual bytes. Measure full and incremental size, upload duration, barrier travel, alignment, and restore—not just checkpoint interval.

### Checkpoints are not savepoints

Operational checkpoints are runtime-managed recovery artifacts and may be deleted according to retention. Savepoints or equivalent deployment snapshots are deliberately retained migration artifacts. Whether state can restore across code changes depends on stable operator identity and compatible serializers, not on the artifact’s name.

### Recovery behavior

Recovery restores state and replays input from the checkpointed positions. Records after the recovery cut may physically traverse operators again. Correctness requires the resulting committed state and effects to be equivalent to one logical application.

---

## 8. Define the Scope of Exactly Once

“Exactly once” can refer to different boundaries:

1. **Operator state:** after recovery, every input contribution is reflected once in managed state.
2. **Source plus state:** source positions and operator state share one recovery cut.
3. **Sink visibility:** results become visible once through a transactional or idempotent protocol.
4. **End-to-end business effect:** the user observes one charge, message, inventory decrement, or table update.

A framework checkpoint typically guarantees the first two when the source is replayable and durable. It does not mean each record physically flows only once. The external sink must collaborate.

### Transactional sink

A checkpoint-aware sink can write into a transaction for checkpoint N, prepare it, and commit only after the coordinator completes N. Recovery aborts or reconciles pending transactions. Readers must use an isolation mode that hides uncommitted data; otherwise prepared writes can leak.

### Versioned idempotent sink

For a key-value or table sink, apply only if the incoming source version is newer:

> update key K to value V and version S only when stored version < S

The comparison and update must be atomic at the sink. Event ID deduplication is similar but requires durable IDs retained at least as long as replay is possible.

### Irreversible external effect

Consider an HTTP payment call:

1. Operator calls the provider with no idempotency key.
2. Provider commits the charge.
3. Worker fails before the next checkpoint completes.
4. Recovery restores the earlier source position.
5. The event is replayed and the provider charges again.

Operator state can still be exactly-once; the business effect is not. Use a provider-supported idempotency key, or atomically write an outbox/intent to a transactional sink and let a separately deduplicated dispatcher perform the effect. If neither is possible, document at-least-once effects and build reconciliation.

---

## 9. Backpressure and Overload

Backpressure begins when a downstream task cannot obtain output buffer capacity. The slowdown propagates upstream until sources reduce consumption. This is a safety mechanism: it replaces unbounded in-memory growth with rising lag. It does not create capacity.

Diagnose from sink to source:

1. Identify the first operator that is busy but not itself backpressured.
2. Inspect sink throttling, remote latency, transaction duration, or commit serialization.
3. Inspect hot keys, timer storms, state backend I/O, serialization, and garbage collection.
4. Compare per-subtask rates; averages hide one saturated partition.
5. Check checkpoint alignment separately from steady-state record processing.

Responses include:

- remove sink or network throttling;
- scale a parallelizable operator and repartition deliberately;
- batch or asynchronously pipeline I/O with bounded concurrency;
- reduce record size or pre-aggregate;
- isolate hot keys;
- shed optional traffic or degrade enrichment under an explicit product policy;
- apply producer admission control when end-to-end lag exceeds the recovery envelope.

Unaligned checkpoints may keep checkpoints completing during backpressure, but the source backlog continues growing if sustainable processing remains below arrival.

---

## 10. Capacity, Replay, and Recovery Math

Use measured rates at the actual key and payload distribution.

Let:

- <code>r_in</code>: peak sustained arrival records/s.
- <code>s_event</code>: encoded bytes/record after broker overhead.
- <code>r_proc</code>: sustainable processing rate with checkpointing and sink commits enabled.
- <code>L</code>: replay backlog records.
- <code>S</code>: durable operator-state bytes.
- <code>delta_S</code>: state bytes changed per checkpoint interval.
- <code>b_cp</code>: effective checkpoint-storage bandwidth.
- <code>I</code>: checkpoint interval.

### Stability and catch-up

Steady state requires <code>r_proc > r_in</code> with margin for failures and maintenance. Catch-up time after an outage is:

> catch-up time = L / (r_proc − r_in)

If <code>r_proc ≤ r_in</code>, the job cannot catch up by waiting. It needs more capacity, less work, or controlled input.

Required source retention is at least:

> detection + repair/provisioning + restore + catch-up + validation margin

Broker bytes over the retention window are approximately arrival byte rate × retained seconds × replication factor, plus indexes and protocol overhead.

### Checkpoint and restore envelope

A lower bound on full checkpoint upload is <code>S / b_cp</code>; an incremental lower bound uses changed and newly referenced bytes, not always only <code>delta_S</code>. Restore includes metadata enumeration, state download, local reconstruction, and replay.

If checkpoint duration approaches the interval, checkpoints overlap or leave little processing time, depending on runtime policy. If recovery starts from a checkpoint <code>I</code> seconds old, normal-case replay begins around <code>r_in × I</code> records, plus any uncommitted sink work and outage backlog.

Capacity-test state growth, checkpoint storage request rate, sink transaction rate, and the slowest subtask. Record count alone misses payload, state, and timer costs.

---

## 11. Operations and Migrations

### Safe deployment

1. Assign stable operator identifiers and inventory state serializers.
2. Take a migration artifact at a known source position.
3. Validate compatibility on a copy, including timers and pending sink transactions.
4. For incompatible changes, start a new deployment from retained input into a shadow sink generation.
5. Catch up, compare old and new results at the same source coordinate, then switch readers.
6. Retain rollback state and the old sink generation until the new path meets its observation window.

Changing key selection redistributes state and is usually a rebuild. Shortening TTL discards information. Changing watermark or trigger policy can change which revisions were emitted even when final aggregates match. Treat each as a semantic migration.

### Failure and upgrade runbooks

- expired source offsets: choose restore from another snapshot, bounded rebuild, or declared data loss;
- checkpoint corruption: restore the last verified artifact and account for added replay;
- poison event: quarantine with source coordinate and schema, but do not silently advance a transactional sink past an unhandled record;
- sink outage: cap pending transactions and backlog; protect source retention;
- schema change: preserve old decoders through the maximum replay horizon.

---

## 12. Security and Data Lifecycle

Streaming state and checkpoints often contain raw personal or secret data even if the sink contains aggregates.

- encrypt source logs, network transport, state backend, checkpoints, and savepoints;
- restrict checkpoint paths because they may bypass row-level sink controls;
- use least-privilege source and sink credentials per job;
- redact event payloads and keys from logs, traces, dead letters, and metrics labels;
- carry tenant and classification boundaries through repartitioning;
- include retained log, state, checkpoints, and materialized views in deletion policy;
- audit manual offset changes, savepoint restores, replays, and sink cutovers.

Immediate erasure can conflict with replay and time-travel guarantees. Resolve that conflict in the product and legal contract before relying on retention as a recovery strategy.

---

## 13. Observability

### Time and freshness

- source head time minus last durably processed source time;
- event-time lag = current wall clock minus latest safely reflected event time;
- watermark value and rate of advance by input and operator;
- on-time, late-accepted, late-corrected, side-output, and dropped event counts.

Event-time lag must subtract the reflected event time from the current time; reversing the operands hides incidents behind negative values.

### State and recovery

- state bytes and key count by operator/subtask;
- timer count and firing rate;
- checkpoint start delay, alignment, snapshot, upload, total duration, size, and failure reason;
- restore duration and replay volume;
- pending sink transactions and commit latency.

### Throughput and pressure

- input/output records and bytes per subtask;
- busy, idle, and backpressured time;
- partition lag distribution, not only total lag;
- hot-key indicators, buffer occupancy, garbage collection, remote-call latency, retries.

Alert on trends that threaten the contract: retention headroom, state growth beyond forecast, watermark stall, repeated checkpoint failure, or catch-up rate approaching arrival rate.

---

## 14. Verification and Failure Injection

Build a deterministic event corpus with out-of-order records, duplicates, future timestamps, deletes/corrections, idle partitions, hot keys, and session merges. Then:

- compare a continuous run with a bounded replay over the same source range;
- kill a worker before and after local state update;
- fail the coordinator while a checkpoint is completing;
- fail the sink before prepare, after prepare, and after commit acknowledgement;
- replay records across the checkpoint boundary and verify state and sink invariants;
- pause one source partition to test watermark and idleness behavior;
- overload the sink and prove memory remains bounded while lag rises;
- restore after rescaling and verify every key’s state and timers;
- restore old checkpoints with the new binary in a compatibility environment;
- exhaust source retention in a drill and exercise the rebuild decision;
- test TTL boundaries with duplicates and join counterparts arriving immediately before and after expiry.

Verify emitted revisions, not only final values. Two pipelines can converge to the same total while one sends a duplicate alert or irreversible action.

---

## 15. Decision Framework

| Design question | Stronger fit |
|---|---|
| Results may wait for a closed input boundary | Bounded batch |
| Users need updates before input can close | Continuous event-time processing |
| Source log and all dependencies are replayable long enough | Stream-only rebuild is feasible |
| Historical algorithm needs global bounded context | Batch correction or periodic rebuild |
| Sink supports atomic versioned upserts or checkpoint transactions | End-to-end exactly-once visibility may be achievable |
| Sink performs non-idempotent external effects | Outbox/idempotency layer or explicit at-least-once contract |
| Late input must revise prior results | Changelog-capable sink and consumers |
| State cannot be bounded or restored within the objective | Redesign aggregation, retain external state, or use bounded execution |

Select a runtime after these answers. Benchmark the complete path with checkpointing, serialization, state backend, and real sink enabled; an operator-only throughput number is not a system capacity number.

---

## Primary References

- Tyler Akidau et al., [The Dataflow Model: A Practical Approach to Balancing Correctness, Latency, and Cost](https://www.vldb.org/pvldb/vol8/p1792-Akidau.pdf).
- Apache Beam, [Programming Guide: Windowing, Watermarks, and Triggers](https://beam.apache.org/documentation/programming-guide/#windowing).
- Paris Carbone et al., [Lightweight Asynchronous Snapshots for Distributed Dataflows](https://arxiv.org/abs/1506.08603).
- Apache Flink, [Checkpointing](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/datastream/fault-tolerance/checkpointing/).
- Apache Flink, [CheckpointingMode API](https://nightlies.apache.org/flink/flink-docs-release-2.2/api/java/org/apache/flink/core/execution/CheckpointingMode.html).
- Apache Flink, [Generating Watermarks](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/datastream/event-time/generating_watermarks/).
- Apache Flink, [Monitoring Back Pressure](https://nightlies.apache.org/flink/flink-docs-stable/docs/ops/monitoring/back_pressure/).
- Apache Kafka, [Design: Transactional Messaging](https://kafka.apache.org/documentation/#semantics).

---

**Next:** [Change Data Capture](04-change-data-capture.md) explains how a database log becomes a replayable stream without losing the snapshot-to-tail boundary or source transaction semantics.
