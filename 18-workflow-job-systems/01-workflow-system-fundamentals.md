# Workflow System Fundamentals

A workflow system turns a request to perform work into durable, inspectable execution state that can outlive every process involved. Its first promise is not “the function ran.” It is that accepted work has an identity, an authority, a recoverable next action, and a declared outcome when workers, networks, control planes, or dependencies fail.

Workflow fundamentals define workload and durability contracts, execution identity, common control/execution-plane boundaries, durable acceptance, mechanism selection, and system invariants. [Background Jobs and Worker Pools](./02-background-jobs-worker-pools.md) covers queue claims and execution; [Distributed Scheduling](./03-distributed-cron-scheduling.md) covers occurrence materialization; [Durable Execution](./04-durable-execution-workflow-engines.md) covers history replay; [DAG Orchestration](./05-dag-orchestration.md) covers graph scheduling; [Effect Commit Protocols](./06-retry-idempotency-compensation.md) covers external effects; [Leases and Recovery](./08-leases-heartbeats-recovery.md) covers attempt authority; [Workflow Observability](./09-workflow-observability-replay.md) covers operator views and forensic replay.

## Primary Evidence and Scope

| Primary evidence | What it establishes | Boundary |
|---|---|---|
| Amazon SQS documentation | A durable queue can retain accepted messages, hide received messages temporarily, and redeliver when processing is not acknowledged | Broker semantics, not a multi-step workflow state model |
| Kubernetes Job controller documentation | A controller can reconcile desired completion against replaceable worker Pods | Kubernetes-specific execution controller; external effects remain application responsibilities |
| Apache Airflow architecture documentation | DAG definitions, scheduler, metadata database, executor, and workers form distinct orchestration roles | Batch/DAG model, not code-replay semantics |
| Temporal event-history documentation | Per-execution append-only events and replay reconstruct durable workflow progress | Temporal-specific implementation and limits |
| Azure Durable Task programming model | Orchestrators, activities, durable timers, external events, event sourcing, and replay separate decisions from effects | Durable Task family; other engines may interpret a state machine instead of replaying user code |

## Workload and guarantee contract

“Run this later” leaves the important decisions undefined. Record:

| Field | Required answer |
|---|---|
| **Logical work identity** | Which user/business command is this, and how are duplicate start requests recognized? |
| **Acceptance boundary** | Which durable commit must complete before the caller receives “accepted”? |
| **Completion contract** | What observable result, deadline, and terminal state constitute success? |
| **Recovery grain** | Restart the whole job, one graph node, one explicit state transition, or replay to the next undecided command? |
| **Control-flow shape** | One task, static graph, dynamic code, recurring occurrence, human/external signals, or a combination? |
| **Failure semantics** | May work be lost, repeated, delayed, compensated, quarantined, or require operator judgment? |
| **Effect boundary** | Which database/API/message effects lie outside the workflow state transaction? |
| **Time semantics** | Immediate, delayed, recurring, deadline-bound, or waiting for an external event? |
| **Scale shape** | Starts/s, transitions/s, concurrent active/sleeping executions, fan-out, timers, history bytes, and hottest tenant? |
| **Evolution horizon** | How long can old payloads, definitions, histories, and workers remain live? |
| **Security boundary** | Who can start, signal, cancel, inspect, reset, or terminate which execution and payload? |
| **Repair authority** | Which automated reconciler or operator action advances every nonterminal state? |

The contract names at least three outcomes separately:

- **accepted:** durable responsibility transferred to the workflow system;
- **started:** some worker began an attempt;
- **completed:** the declared result became durable and externally observable.

Returning HTTP `202 Accepted` after putting an item in process memory satisfies none of the durable promises. Conversely, a queue message can be durably accepted while the business entity that motivated it was rolled back. The acceptance transaction must match the product promise.

## Taxonomy: mechanisms are not one ladder

Scheduling, dependency shape, and recovery granularity are orthogonal. A distributed cron service decides **when** to create work; that work may be a queue job, DAG run, or durable workflow. A DAG engine is not simply a weaker durable-execution engine: it makes graph nodes and data intervals explicit, while a code-replay engine makes dynamic per-entity control flow and long waits natural.

| Mechanism | Durable authority | Recovery grain | Strong fit | Main tax |
|---|---|---|---|---|
| Best-effort in-process task | None beyond process memory | None | Disposable cache warm, advisory telemetry | Work disappears on restart |
| Queue-backed background job | Broker message or job row | Whole job/explicit checkpoint | Independent bounded task, retry from start | Duplicate attempts and limited mid-job progress |
| Database state machine | Entity/transition rows | Explicit application state | Few stable states near one database transaction | Polling, transition code, and home-grown tooling |
| Distributed schedule/timer | Schedule and occurrence records | Trigger occurrence | Recurring or delayed creation of other work | Civil time, misfire, overlap, and catch-up policy |
| DAG orchestrator | Graph-run and node state | Node or partition | Batch/data pipeline with explicit dependencies and artifacts | Scheduler metadata, backfill, and graph evolution |
| Durable workflow engine | Event history or interpreted state-machine state | Next undecided command/state | Long-lived dynamic process, signals, durable waits, step-level recovery | History cost, determinism/versioning or DSL constraints |

Select with several axes, not a framework popularity list:

| Axis | Simpler mechanism is sufficient when | Stronger orchestration is justified when |
|---|---|---|
| **Restart cost** | Whole task can cheaply and safely restart | Hours of progress or completed steps must survive |
| **Wait duration** | Worker remains active for a bounded short task | Process waits minutes to months without holding compute |
| **Flow shape** | One handler or small explicit state table | Dynamic branches, signals, child processes, or large DAG |
| **Effect risk** | Repeat is naturally convergent | Ambiguous effects require durable per-step evidence and repair |
| **Audit** | Terminal result and logs suffice | Every decision, input, timer, and operator action must be reconstructed |
| **Throughput** | Per-item orchestration overhead is acceptable | Extremely fine-grained work should be batched or remain in a stream/data engine |
| **Evolution** | Work completes within one deployment window | Instances outlive many code, schema, or policy versions |

Do not choose durable execution merely because a side effect is dangerous. An engine cannot make an arbitrary remote effect atomic with its history. Dangerous effects need the protocol in [Effect Commit Protocols](./06-retry-idempotency-compensation.md) regardless of orchestration surface.

## Common reference architecture

~~~mermaid
flowchart LR
    C[Caller / trigger]
    G[Start and signal API]
    S[(Authoritative execution state)]
    N[Durable wakeup / task transport]
    T[Timer and dependency evaluator]
    W[Replaceable workers]
    E[External effect systems]
    R[Reconciler / repair controller]
    V[Visibility projection]
    P[Definition, policy, version control plane]

    C --> G
    G --> S
    S --> N
    T --> S
    N --> W
    W --> S
    W --> E
    R --> S
    S --> V
    P --> G
    P --> T
    P --> W
~~~

The **execution data plane** accepts starts/signals, advances one execution, dispatches runnable work, records results, and materializes timers. The **control plane** manages definitions, compatible worker versions, quotas, pause/drain policy, routing, retention, and operator authorization. The **visibility plane** is a query-optimized projection; it may lag and must not become the authority for transition decisions.

Workers are replaceable compute. The authoritative state store answers which logical work exists and which transitions are committed. The wakeup transport lowers dispatch latency; in a hybrid design it need not be authoritative if a reconciler can rediscover ready state. The timer service makes future work eligible. The reconciler closes gaps created by lost notifications, crashes, and partially completed control actions.

**Reference-design availability boundary:** loss of the definition/deployment control plane stops new versions and risky administrative changes, but compatible workers continue advancing already accepted executions from last-known-good policy. Security revocation, quota changes, and cancellation may require a shorter stale-policy interval than ordinary execution.

## Durable identity and state

A common execution envelope contains:

```text
namespace / tenant
logical operation key and start-request identity
execution ID, run ID, parent/root IDs
definition type, immutable definition version, handler/build constraints
input schema, payload reference, digest, and encryption metadata
status, status reason, and state revision
created/accepted/eligible/deadline/terminal timestamps
current step or frontier summary
outstanding task, timer, child, and external-signal identities
cancellation/termination intent and actor
result reference or terminal error classification
retention, legal-hold, and audit metadata
```

The full representation differs by mechanism: a job row may have one claim; a DAG run has node instances; a replay engine has an ordered history and derived mutable state. The envelope provides stable identity across APIs, logs, audit, and business reconciliation.

**System-level invariants:**

1. A successful acceptance response names an execution whose durable state can be found after every allowed process/host failure.
2. Retrying a start with the same logical request identity does not create an unrelated execution or silently change parameters.
3. State advances by compare-and-set, transaction, or serialized history revision; a stale worker/control action cannot overwrite a newer transition.
4. Claiming or scheduling work is not evidence that its external effect committed.
5. Every nonterminal state has a durable next trigger, a bounded wait/deadline, or an explicit operator-owned exception.
6. Terminal state is immutable except through a new, audited reset/reopen/run identity whose semantics are visible.
7. Cancellation is durable intent and cooperative transition policy; it is never presented as proof that already committed effects were undone.
8. Inputs, code/definition, payload codec, and keys remain interpretable for at least the execution and forensic retention horizon.
9. A delayed or duplicate notification cannot create a second logical occurrence without a distinct occurrence identity.
10. Visibility/search lag cannot authorize execution, cancellation, or repair against stale state.

## Durable acceptance protocols

### Idempotent start API

The caller sends a stable request identity plus an immutable parameter digest. The system atomically creates the execution when the key is unused, returns the existing execution when key and digest match, and rejects key reuse with different parameters. A timeout after the commit is resolved by querying/retrying that same identity, not by inventing a new key.

Keep logical operation ID separate from physical run/attempt ID. The same payroll occurrence may require a repaired run without becoming a second payroll obligation.

### Business transaction and asynchronous work

When a database change creates the obligation, commit an execution intent or [outbox](../05-messaging/07-outbox-pattern.md) record in the same transaction. A relay/wakeup may be delivered repeatedly; the execution identity makes dispatch idempotent. If the broker message alone is authoritative, do not acknowledge acceptance until the broker's documented durability confirmation succeeds, and preserve the producer request ID across an ambiguous send.

### Reconciliation closes notification gaps

“State says ready” and “worker was notified” cannot generally be one transaction across a database and broker. Make one authoritative and reconcile the other:

1. commit the execution/transition and a durable dispatch intent;
2. publish a wakeup carrying execution ID and revision;
3. mark/advance dispatch evidence when the transport confirms;
4. periodically scan undispatched or ready-but-unclaimed state;
5. republish safely using the same identity.

The scan is a correctness mechanism, not merely an incident script. Its lag contributes directly to the execution-start SLO.

## Failure and effect boundaries

Durable orchestration can serialize its own state transitions. It cannot atomically include arbitrary payment, email, object store, or partner API state. The unavoidable ambiguity is:

```text
external effect committed
        -> worker or reply fails
        -> workflow completion not recorded
        -> another attempt may run
```

The workload contract records that boundary. [Effect Commit Protocols](./06-retry-idempotency-compensation.md) covers stable effect identity, dedup retention, reconciliation, reservation, and compensation.

Likewise, a lease decides when another worker may try; it cannot prove the first worker stopped. Detailed renewal/fencing belongs to [Leases and Recovery](./08-leases-heartbeats-recovery.md).

## Capacity model across planes

Plan execution transitions independently from worker CPU. Let start rate be $\lambda_{\mathrm{start}}$, mean durable transitions per completed execution be $\mathbb{E}[N_{\mathrm{transitions/run}}]$, and incoming signal/timer/recovery rates be separate:

$$
\lambda_{\mathrm{transition}}
= \lambda_{\mathrm{start}}\,\mathbb{E}[N_{\mathrm{transitions/run}}]
  + \lambda_{\mathrm{signal}} + \lambda_{\mathrm{timer}} + \lambda_{\mathrm{recovery}}
$$

If each transition appends $b$ encoded bytes, raw history/state-log ingest is approximately $\lambda_{\mathrm{transition}}b$ before indexes, replicas, backups, and visibility projections. Open executions also retain timers, pending tasks, mutable-state summaries, and payloads even when worker compute is zero.

**Illustrative:** 200 starts/s at 18 durable transitions per run, plus 800 signals/s and 600 timer/recovery transitions/s, produces about 5,000 transitions/s. At 1.5 KiB encoded per transition, raw ingest is about 7.3 MiB/s before replication and indexes. A design sized only for 200 business operations/s misses the actual persistence workload by an order of magnitude.

Worker demand remains class-specific:

$$
L_{\mathrm{workers}} = \sum_j \lambda_j\,\mathbb{E}[S_j]
$$

Include retry attempts, fan-out, poll/dispatch traffic, payload fetch, visibility indexing, timer bursts, replay CPU, and reconciliation scans. A million sleeping workflows may consume negligible worker CPU and still dominate timer indexes, storage, key retention, and control-plane cardinality.

If backlog $B$ remains while new work arrives at $\lambda$ and safe completion capacity is $\mu$, optimistic drain time is:

$$
T_{\mathrm{drain}} \ge \frac{B}{\mu-\lambda}, \qquad \mu>\lambda
$$

Mechanism selection does not remove this conservation law; see [Capacity Planning](../01-foundations/10-capacity-planning.md).

## Security and tenant isolation

Starting, signaling, canceling, resetting, and reading an execution are distinct permissions. Authorize namespace, workflow type, execution identity, and action; do not accept a caller-supplied tenant only because it appears in the payload. Worker task queues and handler registrations are capabilities: a worker that can poll sensitive tasks can receive inputs and report results.

Execution inputs, results, errors, and histories can contain secrets and personal/business data for far longer than request logs. Encrypt in transit and at rest, minimize payloads, store large/sensitive bodies behind access-controlled references, and retain the codec/key version required to read history. Search/visibility attributes should contain only data safe for broader indexing.

Bound starts, signals, fan-out, timers, payload bytes, retained history, and operator queries per tenant. A malicious workflow definition or input can create infinite children, timer storms, expensive replays, or downstream calls even when ordinary request rate limits look healthy.

Definitions and worker artifacts are production code. Sign/provenance-check them, restrict registration and routing changes, stage activation, and audit emergency reset/terminate/re-drive operations. Administrative repair must preserve tenant authorization and business invariants.

## Evolution and compatibility

An execution may outlive many deployments. Treat these as independently versioned contracts:

- start/signal/cancel API and request identity;
- input, result, and error payload schemas/codecs;
- workflow definition or state-machine version;
- worker handler/activity versions and capabilities;
- durable state/history schema;
- visibility projection schema;
- external effect parameter contract.

Pin or record the versions needed to interpret each execution. Deploy readers/upcasters before writers emit a new payload. Route old work only to compatible workers, and prove compatible capacity exists before removing an old build. A visibility reindex can run asynchronously; changing authoritative transition meaning requires a staged state migration or a new run/version.

Rollback restores compatible execution code and routing; it does not delete newly appended state. If a new definition emitted a command or value the old version cannot interpret, rollback is a forward compatibility operation.

## Specialized failure traces

### Acceptance response precedes durable state

1. API validates work and returns `202`.
2. Process crashes before enqueue/state commit.
3. Caller retains an execution ID that no authority knows.

Commit first, return the committed identity second. Reconcile the business entity against accepted executions to detect older gaps.

### State commits but wakeup is lost

1. `READY` becomes durable.
2. Broker publication times out or the relay crashes.
3. No worker sees the execution; queue dashboards look empty.

A durable dispatch intent and ready-state reconciler republish by execution ID/revision. Queue depth alone is not end-to-end work debt.

### Control-plane outage stops the execution plane

Workers synchronously fetch mutable definition/config on every transition. The deployment service fails, so already accepted work cannot progress despite healthy execution state. Distribute immutable, versioned artifacts and last-known-good policy; reserve synchronous control calls for decisions that truly require fresh authority.

### Mechanism recovery grain is too coarse

A 10-hour job completes nine hours of work, the worker restarts, and the broker correctly redelivers from the beginning. The queue did not lose work; the chosen recovery grain lost progress. Split/checkpoint the job or use DAG/durable execution when step-level recovery justifies it.

### Payload outlives its decoder

A delayed job or sleeping workflow wakes after the only worker supporting input schema 3 was removed. Retries cannot repair missing code. Inventory live versions, retain compatible workers/upcasters, and gate retirement on zero reachable executions.

### Cancellation is mistaken for undo

An operator marks the execution canceled after a payment activity committed. The worker stops, but no refund occurs. Cancellation prevents or requests future work; compensation is a separate forward effect with its own state and failure modes.

### Visibility lag drives destructive repair

Search says an execution is still running, but authoritative state already completed. An operator resets it from a stale projection and repeats later steps. Repair reads and conditionally updates the authoritative execution revision; visibility is navigation, not authority.

## Observability and verification

At the system boundary, measure:

- starts offered/accepted/deduplicated/rejected/ambiguous and acceptance latency;
- accepted-to-ready, ready-to-start, transition, completion, and terminal latency by definition/version/tenant;
- nonterminal count and age by reason: runnable, running, waiting timer/signal/dependency, retry, paused, or repair-needed;
- transition rate/bytes, state-store conflicts/lag, dispatch intents, lost-wakeup reconciliation lag;
- workers/pollers and compatible capacity by task/definition/build version;
- active/sleeping executions, timers due/fired late, child/fan-out cardinality, history/state bytes;
- visibility projection lag and authoritative-versus-projection mismatch;
- cancellation/reset/termination/repair actions and unresolved business reconciliation.

[Workflow Observability](./09-workflow-observability-replay.md) owns per-execution timelines, lineage, stuck detection, and forensic/operator UX.

Verification should include:

1. state-machine/model tests proving invariants and terminal monotonicity;
2. duplicate and ambiguous start/signal/cancel requests with parameter mismatch;
3. crash injection before/after every state, wakeup, task, result, and terminal commit;
4. state committed with notification lost, duplicated, reordered, or delayed;
5. control-plane/visibility outage while existing executions progress;
6. old/new payload, definition, worker, and state-schema compatibility;
7. restore with pending tasks/timers and reconciliation from authoritative state;
8. tenant authorization/quotas on start, signal, inspect, and repair;
9. production-shaped transition, timer, fan-out, backlog, and recovery load;
10. business-ledger reconciliation proving accepted obligations reach a declared terminal outcome.

## Decision framework

1. What exact durable commit makes the product allowed to say “accepted”?
2. What are the logical operation, execution, run, occurrence, task, and effect identities?
3. What recovery grain is required, and can the whole unit safely restart?
4. Is the flow one task, an explicit state table, a static graph, or dynamic signal/time-driven code?
5. Is scheduling an orthogonal trigger rather than the execution mechanism itself?
6. Which effects lie outside the workflow transaction, and which chapter-owned protocol closes each ambiguity?
7. Does every nonterminal state have an automated trigger, deadline, reconciler, or named operator owner?
8. What transition/timer/history/visibility workload exists in addition to worker compute?
9. Which payload, definition, worker, codec, and key versions must survive the longest execution?
10. Can the execution plane continue safely through control-plane and visibility loss?
11. Who may start, signal, inspect, cancel, reset, terminate, or repair each tenant's executions?
12. Does the selected mechanism buy necessary recovery semantics without imposing an unnecessary replay/graph/operations tax?

## Primary references

- [Amazon SQS, *What is Amazon Simple Queue Service?*](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/welcome.html)
- [Kubernetes, *Jobs*](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [Apache Airflow, *Architecture Overview*](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/overview.html)
- [Temporal, *Workflow Execution overview*](https://docs.temporal.io/workflow-execution)
- [Temporal, *Events and Event History*](https://docs.temporal.io/workflow-execution/event)
- [Microsoft, *Durable Task programming model*](https://learn.microsoft.com/en-us/azure/azure-functions/durable/programming-model-overview)
- [Microsoft, *Durable orchestrations*](https://learn.microsoft.com/en-us/azure/durable-task/common/durable-task-orchestrations)
- [Pat Helland, *Life Beyond Distributed Transactions: an Apostate's Opinion* (CIDR 2007)](https://www.cidrdb.org/cidr2007/papers/cidr07p15.pdf)
