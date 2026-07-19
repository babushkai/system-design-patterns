# Workflow Observability and Replay

## What This Chapter Owns

Workflow operations must answer questions that request telemetry cannot:

- What authoritative facts exist for this one run?
- What state can be derived from those facts, and how fresh is the view?
- Can this history be reconstructed with a particular code version?
- Does “replay” mean read-only reconstruction, re-executing unfinished work, or creating a new lineage?
- What is the workflow waiting for, and is that wait expected?
- Which operator changed it, under what authorization, and with what effect?

This chapter owns history-derived observability, visibility projections, replay/debug/snapshot/reset/fork semantics, stuck-state diagnosis, lineage, history evolution, privacy, and repair commands. It does not redesign telemetry plumbing. Use [Distributed Tracing](../11-observability/01-distributed-tracing.md), [Metrics and Monitoring](../11-observability/02-metrics-monitoring.md), [Production Logging](../11-observability/03-logging.md), and [Alerting](../11-observability/04-alerting.md) for generic collection and delivery.

The central rule is:

> Identify the authoritative record for each runtime, then make every view declare how far it has derived that record. Do not promote a convenient search index, log stream, or mutable metadata row into an execution history it is not.

---

## Not Every Workflow Product Has the Same History

“Workflow history” is not a portable storage contract.

| Runtime shape | Typical authoritative execution state | What replay can mean |
|---|---|---|
| Deterministic durable engine | Ordered event history sufficient to rebuild workflow state | Re-execute deterministic workflow code against recorded events without reissuing recorded effects |
| State-machine service with retained history | Service-owned execution state plus a bounded execution-event history | Inspect transitions, redrive or start another execution according to product semantics; not necessarily code replay |
| DAG orchestrator | Metadata database rows, task-attempt records, scheduler state, and external logs/artifacts | Clear/retry/rerun tasks or backfill a new run; usually not deterministic reconstruction from one immutable log |
| Queue and workers | Job row, attempts, queue state, checkpoints, and effect receipts | Retry/resume from durable boundaries, not reconstruct an arbitrary historical process |

Temporal’s architecture explicitly treats a workflow execution history as sufficient to recover mutable execution state. AWS Step Functions Standard records execution details and exposes execution history for a bounded retention period, while Express does not record execution history in Step Functions and relies on configured CloudWatch Logs for its console details. Airflow, Dagster, and Prefect expose rich run/task metadata and logs, but that does not make them interchangeable with a deterministic event-history engine.

Write a **history profile** for every workflow type:

1. Which store is authoritative for current and terminal state?
2. Is there an ordered, complete event record? What omissions are allowed?
3. How long is it retained, and can it be exported before expiry?
4. Does reconstruction require a specific definition or worker build?
5. Which fields are mutable metadata, derived visibility, telemetry, or external artifacts?
6. Which repair operations change the same run, and which create a successor?

Without that profile, an operator may mistake missing logs for missing execution, or stale search results for authoritative state.

---

## Canonical Record and Derived Views

For an event-history engine, a canonical event envelope can contain:

~~~text
namespace_id, workflow_id, run_id
event_sequence, event_id, event_type, event_schema_version
command_id, activity_id, attempt_id
causation_event_id, correlation_id, parent_run_id
definition_version, worker_build_id
recorded_at, observed_external_time
actor_type, actor_id, authorization_context
payload_ref, payload_digest, data_classification
integrity_metadata
~~~

event_sequence orders facts inside one run. A wall-clock timestamp supports diagnosis but should not replace the sequence. causation_event_id answers “which fact caused this command?” correlation_id connects external interactions without pretending they share a transaction. Payload indirection keeps large or sensitive data outside broadly indexed structural history.

An event-sourced engine should enforce:

1. **A run has one append order.** Event IDs are unique and sequence numbers do not silently fork except through an explicit, represented branch model.
2. **Execution state is derived from a history prefix.** A state transition that is absent from the authoritative prefix is not made true by a log line or index row.
3. **Projection application is idempotent and monotonic per run.** Duplicate events do not duplicate counters or regress status.
4. **Every derived view exposes a watermark.** “Running as of event 382, projected at 10:14:03” is honest; an unqualified “Running” may be stale.
5. **Snapshots bind to one exact prefix.** Run ID, last event sequence, prefix digest, state schema, and compatible definition/build accompany the snapshot.
6. **Read-only replay issues no external effects.** Recorded activity results, signals, timers, and side-effect markers are inputs, not invitations to call dependencies again.
7. **Reset and fork preserve lineage.** They do not rewrite the evidence of the original run.
8. **Operator commands are authorized, idempotent, version-checked, and audited.**
9. **Redaction preserves structural meaning and provenance.** It may remove payload availability, but it must not silently invent a different execution.

For a metadata-driven orchestrator, translate these invariants rather than inventing an append log. The metadata database may be authoritative, task logs may be partial, and the audit record may need a transactional outbox. A task status update and an external log line can disagree; document which wins.

---

## The History-to-Visibility Data Path

Keep the write path for execution authority narrow:

~~~text
worker or scheduler command
          |
          v
authoritative execution transition
          |
          +--> ordered history or metadata transaction
          |
          +--> projection feed / transactional outbox
                    |
                    +--> current-state visibility store
                    +--> search index
                    +--> aggregate metrics
                    +--> audit export
                    +--> notification rules
~~~

The visibility store answers common queries: open runs by tenant, workflows waiting on a signal, failed attempts by definition version, or runs whose deadline passed. It is deliberately denormalized and may lag. Store source sequence/watermark with each row and expose projection lag globally.

The UI should read the authoritative run when an operator opens a critical execution, or clearly label a projection-only view. Never use a search-index result alone to decide that a run is absent, terminal, or safe to retry. If the projector is broken:

- execution continues on the authoritative path;
- search and dashboards show a degraded/freshness warning;
- projector consumers resume from a durable offset;
- the index is rebuilt from retained source facts;
- no “repair” writes flow backward from the index into execution authority.

Projection state is disposable only if the complete source and deterministic projection code remain available. If the source retention is shorter than rebuild time, the index has quietly become a second authority; either extend retention, archive the source, or admit and design for that role.

---

## Six Operations Commonly Called Replay

The word “replay” is unsafe in an operator interface unless it names the operation.

### 1. State reconstruction

A deterministic engine re-executes workflow code against recorded history to rebuild in-memory state. Completed activity results, timer firings, signals, version markers, and recorded nondeterministic values come from history. Reconstruction must not call an API, send a message, charge a card, or read the current wall clock.

It reconstructs state represented by the engine at event boundaries. It is **not** arbitrary time-travel debugging: unrecorded heap values, external database contents at that time, unsampled network traffic, and deleted payloads are unavailable. Preserve the raw stored history even when a new build is replay-incompatible; incompatibility is evidence about the code, not permission to discard the record.

### 2. Offline debug replay

Export an immutable production history and run it in an isolated tool against the original and candidate workflow builds. Block network and production credentials. Capture the first divergent command, event sequence, code version, and payload decoder. This is a compatibility test and a bug reproducer, not a live state change.

### 3. Redrive or resume

Redrive schedules incomplete or failed logical work again. It may append a command to the same execution, mutate orchestrator metadata through a supported API, or create a new attempt. External effects can happen. The operation therefore requires the effect guarantees in [Retry, Idempotency, and Compensation](./06-retry-idempotency-compensation.md) and the attempt authority in [Leases, Heartbeats, and Recovery](./08-leases-heartbeats-recovery.md).

### 4. Reset

A reset chooses a supported prior boundary and creates a new execution branch or successor whose initial state derives from that boundary. Later events in the original remain evidence; they are not erased. The reset plan must state what happens to activities and effects after the boundary:

- reuse a recorded result;
- schedule the logical effect again under the same idempotency identity;
- schedule a deliberately new effect under a new identity;
- require reconciliation before proceeding.

Using the old effect ID indiscriminately can suppress work that should happen in the reset. Generating all-new IDs indiscriminately can duplicate money movement or notifications.

### 5. Fork

A fork creates an isolated what-if execution with a new run identity, parent pointer, source event sequence, data classification, and environment. Default-deny every production effect. A fork is useful for debugging, migration rehearsal, and simulation; it must never masquerade as continuation of the production run.

### 6. Snapshot plus suffix

A snapshot is a derived cache of execution state at a precise history prefix. Recovery validates the binding and replays only the suffix. This is related to the broader problem of recording consistent distributed state studied by Chandy and Lamport, but a workflow-engine snapshot is not automatically a Chandy–Lamport global snapshot: it covers only the state and in-flight facts the engine’s protocol defines.

If the snapshot digest, schema, run ID, definition compatibility, or prefix is wrong, discard it and rebuild from the retained history. Never “make it fit” by skipping unknown suffix events.

---

## State-Aware Stuck Detection

Age alone does not identify a stuck workflow. A three-day approval wait may be correct; a runnable task that has not been dispatched for three minutes may be an incident. Detection needs the expected state machine plus liveness, progress, and deadline evidence.

| Observed state | Required evidence | Possible diagnosis | Safe first action |
|---|---|---|---|
| Waiting on timer | fire time, timer registration, timer-queue watermark | Healthy before due; overdue timer or lost wake-up after due | Reconcile timer index against authority |
| Waiting on signal | signal contract, optional business deadline, correlation identity | Healthy indefinite wait, missing callback, or mismatched correlation | Inspect source and dedup record; do not fabricate signal |
| Runnable, not scheduled | runnable transition sequence, queue/outbox offset, scheduler watermark | Lost enqueue, projector lag, admission block | Reconcile runnable fact to dispatch path |
| Activity scheduled, no attempt | task-queue publication, worker poll/capacity, tenant admission | Queue outage, no compatible worker, starvation | Restore capacity or route; preserve original schedule fact |
| Attempt live, no durable progress | accepted heartbeat, state-specific progress marker and deadline | Deadlock, dependency hang, or legitimate non-incremental phase | Apply phase-specific timeout/cancel policy |
| Attempt lease expired | grant epoch, last accepted renewal, reclaim backlog | Worker loss or authority overload | Run idempotent reclaim; protect renewals from storm |
| Compensation pending/failed | original effect receipt, compensation identity, retry policy | Unresolved business obligation | Escalate with effect evidence |
| Projection says running, authority terminal | source sequence versus projection watermark | Projection lag or poison event | Repair/rebuild projection only |

Define a stuck predicate per state:

~~~text
state is expected to make progress
AND authority has not observed durable progress since state-specific threshold
AND no documented wait condition explains the silence
AND relevant deadline or recovery objective has been breached
~~~

Population percentiles help set expectations but are not correctness thresholds. A global “older than p99 equals stuck” alert mixes valid long waits with failures. Attach business deadline, state, tenant impact, and last authoritative event to every stuck alert.

---

## Lineage and Causality

Stable lineage turns repair into an explainable graph:

~~~text
schedule or API request
        |
        v
workflow run ----continue-as-new----> successor run
     |  \
     |   +----reset at event 214----> reset branch
     |
     +----child workflow------------> child run
     |
     +----activity------------------> attempts
     |
     +----fork at event 187---------> isolated debug run
~~~

Record typed edges: parent/child, continue-as-new, reset-from, fork-from, retried-by, compensates, caused-by-signal, consumes-artifact, and produces-artifact. Include source event sequence and namespace. Enforce acyclic lineage where the relationship demands it, while allowing correlation links that are not parentage.

Workflow ID, run ID, activity ID, attempt ID, command ID, logical effect ID, trace ID, and business key solve different identity problems. Do not overload one field. In particular, a retry shares a logical activity/effect identity but receives a new attempt identity; a reset or fork receives a new run identity but retains explicit ancestry.

The [W3C PROV model](https://www.w3.org/TR/prov-o/) provides portable vocabulary around entities, activities, agents, and derivation. Use it as an interchange model where cross-system provenance matters, not as a requirement to store RDF on the execution write path.

---

## Audit, Integrity, Privacy, and Redaction

Execution history is evidence only if access and mutation are controlled. For every operator or automated repair action, capture:

- authenticated actor and actor type;
- authorization decision and policy version;
- command, normalized parameters, and idempotency key;
- expected source version or event sequence;
- reason, incident/change ticket, and approval when required;
- before/after run identity and lineage;
- observed outcome and any external references.

Protect integrity with append-only permissions, restricted service identities, backups, and independently monitored exports. Hash chaining, signed checkpoints, Merkle structures, or WORM storage can make tampering detectable, but only if roots/keys are protected separately and verification is exercised. Do not label an ordinary mutable table “tamper-proof.”

Separate structural events from payloads:

~~~text
structural event: type, sequence, causation, actor class, payload digest/ref
payload object: encrypted content, tenant key, classification, retention policy
visibility fields: allowlisted, minimized, separately indexed
~~~

This permits payload deletion, field-level redaction, or cryptographic erasure while retaining that an event occurred. A redaction record should identify scope, authority, time, and prior digest. It must not silently renumber events or change causation.

Privacy creates a real trade-off: deleting a payload may make replay, debugging, or legal proof incomplete. Classify fields at design time:

- **control-critical:** required for future execution or replay; retain or transform compatibly;
- **forensic:** useful for investigation but not execution; retain under a justified policy;
- **visibility:** allowlisted search fields with bounded retention;
- **secret/regulated:** tokenize, encrypt with scoped keys, minimize, or keep out of history.

Enforce tenant isolation and purpose-based access in APIs and UIs. Audit payload reads as well as repair writes. Apply retention holds explicitly; a hidden backup that indefinitely restores “deleted” payloads defeats the policy.

---

## Capacity, Cardinality, and Retention

Let:

- $R$ = mean authoritative events per second across retained runs;
- $B$ = mean encoded bytes per event including payload references but excluding index overhead;
- $T$ = retention in seconds;
- $r$ = physical replication/storage multiplier;
- $C$ = projector processing capacity in events per second;
- $L$ = projection backlog in events;
- $S$ = snapshot interval in events.

Raw retained history is approximately:

$$
H = R B T r
$$

Add separate budgets for indices, object payloads, backups, integrity metadata, and compaction overhead. Model burst rate and event fan-out per workflow type; one mean hides a runaway workflow that emits millions of tiny events.

If arrivals remain $R$ and $C > R$, a projection backlog drains in:

$$
T_{projection\_drain} = \frac{L}{C-R}
$$

If $C \le R$, scaling consumers without fixing a poison event or hot partition does not restore freshness. Expose both event-count lag and time lag.

With a valid snapshot every $S$ events, and assuming restart points are uniformly distributed between snapshots, the average suffix to replay is about:

$$
E[events_{suffix}] \approx \frac{S}{2}
$$

Snapshot creation, validation, storage, and invalidation cost may dominate when runs are short. Measure replay CPU and latency by history shape, not only event count.

Keep unbounded identities out of metric labels. workflow_id, run_id, business key, exception text, and payload-derived values belong in an indexed store or exemplars, not time-series dimensions. Metrics use bounded workflow type, state, error class, queue, region, and tenant tier; links lead from an aggregate signal to filtered runs.

Retention must cover the longest of operational recovery, replay compatibility, audit, dispute, and regulatory windows—or explicitly archive the evidence needed for each. Verify restore time: a cheap archive that takes days to hydrate may not satisfy the repair objective.

---

## History and Projection Evolution

Treat recorded events as immutable protocol messages:

1. Assign each event type an explicit schema version.
2. Preserve raw bytes or a lossless canonical representation.
3. Decode with deterministic, side-effect-free readers/upcasters.
4. Fail closed on an unknown event needed for state; do not skip it and continue with plausible-looking corruption.
5. Regression-test old histories against every supported workflow build and decoder.
6. Retain the original build or a reproducible artifact manifest for the promised replay window.
7. Version deterministic workflow branches with recorded markers or route histories to compatible builds.
8. Bind snapshots to decoder and definition compatibility; invalidate them when either changes.
9. Rebuild visibility under a versioned projection, compare it with the old projection, then cut over with watermarks.

An upcaster that reads the current timezone database, feature flag, network service, or mutable lookup table is nondeterministic. Its output may change on the next replay. Make transformations pure and pin any reference data.

When new code is incompatible, quarantine execution advancement, preserve history, and route to a compatible worker or a reviewed migration/reset plan. Never delete the “bad” history to make the dashboard green.

---

## Repair Commands as a Control Plane

Offer typed commands rather than direct database access:

| Command | Authority affected | Required guard |
|---|---|---|
| Signal/update | Live execution | expected run identity, signal ID, payload schema |
| Cancel/terminate | Desired state or terminal state | current version, reason, effect/compensation policy |
| Retry/redrive | Attempt or failed logical step | retry policy, stable effect identities, expected state |
| Reset | New branch/successor from prior boundary | source event, compatibility and effect plan |
| Fork | Isolated new run | namespace/environment isolation, effects disabled |
| Reconcile timer/queue/effect | Derived obligation versus external fact | evidence source, idempotency, conditional transition |
| Rebuild projection/reindex | Derived store only | source range, projection version, checksum/watermark |

Each command supports dry-run planning, authenticated authorization, idempotency, expected history/version checks, reason, and outcome recording. In an event-history engine, represent the accepted command and result in history when the runtime’s protocol supports it. In a metadata orchestrator, commit the state change and an audit/outbox record atomically. Do not claim an append-only history where none exists.

Projection repair never changes execution truth. Actual authoritative-history repair should be exceptional: quarantine the run, preserve original bytes, record provenance and approvals, create a derived repaired copy or explicitly supported branch, and test reconstruction before resuming.

---

## Specialized Failure Traces

### A stale visibility row triggers duplicate repair

The search index says RUNNING at event 82; the authoritative run completed at event 89. An operator retries from the index and duplicates work. The UI must show the watermark and re-read authority before issuing a version-checked command.

### A snapshot belongs to another prefix

A cache-key collision loads state from event 500 while the run’s history is at event 470. Replaying the “suffix” skips facts that never occurred in this run. Bind snapshot to namespace, run, exact prefix sequence/digest, schema, and build.

### Debug replay calls production

The replay tool imports an activity implementation with network credentials and sends an email. Reconstruction must substitute recorded results and run in a network-denied, credential-free sandbox; effecting redrive is a separate command.

### Reset chooses the wrong effect identity

A payment completed after the reset boundary. A new ID charges again; blindly reusing the old ID may also suppress a deliberately new order. Require an effect-by-effect reset plan and reconcile ambiguous outcomes before starting the branch.

### An upcaster changes with current configuration

History decoded yesterday but diverges after a feature flag changes. Pin pure conversion rules by event version and test the stored bytes; configuration-dependent transformation is not history evolution.

### A direct row edit erases causality

An operator changes FAILED to RUNNING in the metadata database. No task is enqueued, no actor is recorded, and later reconciliation cannot explain the state. Supported repair must update authority, dispatch obligation, and audit atomically.

### Redaction removes control data

A deletion job removes a signal field needed to select the deterministic branch. Future reconstruction fails. Classify control-critical payload before collection and use compatible tokenization or a policy that terminates/migrates the execution before deletion.

### Run IDs explode metric storage

Publishing one time series per run creates millions of labels, slows queries, and raises cost while still retaining incomplete history. Keep bounded aggregates in metrics and resolve individual runs through visibility/history.

### Express logs are mistaken for Standard history

CloudWatch logging is disabled or filtered for an Express state machine, and the console cannot reconstruct full details. The team assumes the execution never ran. The history profile must record that Express detail depends on configured logs and retention, unlike Standard execution history.

---

## Verification Strategy

Test the data and control planes independently:

1. Reconstruct from full history and from every supported snapshot plus suffix; compare canonical state and pending commands.
2. Crash projectors before/after checkpointing offsets, duplicate and reorder deliveries where the feed permits, and rebuild an empty index.
3. Compare old and candidate workflow builds against a corpus of production histories; retain the first divergent event and command.
4. Run offline replay and fork in a network-denied environment and prove no production credential or effect path exists.
5. Race duplicate repair commands and stale expected versions; exactly one authorized transition should win.
6. Inject projection lag and verify UIs, alerts, and APIs expose watermarks and re-read authority before mutation.
7. Exercise every stuck predicate with healthy long waits, overdue timers, lost dispatch, live/no-progress attempts, and delayed projections.
8. Reset on each effect boundary and verify reuse/new/reconcile policy per logical effect.
9. Evolve event decoders, definition versions, and snapshots across the full supported compatibility matrix.
10. Corrupt, truncate, reorder, or substitute history/payload objects; integrity checks should fail closed and preserve evidence.
11. Test cross-tenant reads, payload redaction, key destruction, retention holds, archive restore, and audit of both reads and writes.
12. Rebuild visibility at production volume and prove capacity exceeds arrival while meeting the freshness objective.

Verification should produce artifacts: replay-compatibility reports, projection checksums and watermarks, lineage graphs for reset/fork tests, effect-free sandbox evidence, and restore timing.

---

## Design Decisions to Record

Before production, document:

- the history profile and authoritative store for each workflow runtime;
- visibility freshness objective, watermark semantics, and rebuild source;
- supported meanings of replay and the permissions for each;
- snapshot interval, binding, compatibility, and fallback;
- state-specific stuck predicates and owners;
- lineage identities and effect-ID policy for retry, reset, and fork;
- event, definition, and projection evolution windows;
- payload classes, access controls, retention, deletion, and audit-integrity mechanism;
- capacity assumptions for history, payloads, projection, archive, and restore;
- the typed repair catalog and the exceptional history-repair procedure.

A workflow platform is operationally mature when an operator can explain one run from authoritative evidence, distinguish stale views from truth, reproduce compatible state without effects, and change execution only through a guarded, attributable command.

---

## Primary Sources

1. [Temporal History Service architecture](https://github.com/temporalio/temporal/blob/main/docs/architecture/history-service.md) — official description of event history, mutable state, tasks, and reset branching.
2. [Temporal Python SDK: Workflow Replay](https://github.com/temporalio/sdk-python#workflow-replay) — official replay API and determinism guidance.
3. [AWS Step Functions execution details](https://docs.aws.amazon.com/step-functions/latest/dg/concepts-view-execution-details.html) and [GetExecutionHistory API](https://docs.aws.amazon.com/step-functions/latest/apireference/API_GetExecutionHistory.html) — official Standard/Express history and retention distinctions.
4. [Distributed Snapshots: Determining Global States of Distributed Systems](https://doi.org/10.1145/214451.214456) — Chandy and Lamport, ACM TOCS 1985.
5. [PROV-O: The PROV Ontology](https://www.w3.org/TR/prov-o/) — W3C Recommendation for entities, activities, agents, and provenance relationships.

---

## Related Patterns

- [Durable Execution and Workflow Engines](./04-durable-execution-workflow-engines.md)
- [DAG Orchestration](./05-dag-orchestration.md)
- [Retry, Idempotency, and Compensation](./06-retry-idempotency-compensation.md)
- [Leases, Heartbeats, and Recovery](./08-leases-heartbeats-recovery.md)
- [Background Jobs and Worker Pools](./02-background-jobs-worker-pools.md)
- [Distributed Tracing and Telemetry Pipelines](../11-observability/01-distributed-tracing.md)
- [Metrics and Monitoring](../11-observability/02-metrics-monitoring.md)
- [Production Logging Architecture](../11-observability/03-logging.md)
- [Alert Evaluation and Notification](../11-observability/04-alerting.md)
- [Incident Command and Learning](../11-observability/07-incident-management.md)
- [Disaster Recovery](../15-deployment/05-disaster-recovery.md)
