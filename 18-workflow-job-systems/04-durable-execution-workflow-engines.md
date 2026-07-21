# Durable Execution and Workflow Engines

A durable workflow engine persists orchestration decisions so a process can wait, crash, move to another worker, and continue without treating volatile stack memory as progress. In a code-replay engine, workers reconstruct local workflow state by rerunning deterministic orchestration code against an ordered event history. In an interpreted engine, the service advances a persisted declarative state machine. Both models separate durable orchestration decisions from external activity effects.

Durable execution covers event history, deterministic replay matching, orchestration/activity separation, timers and signals, history rollover, replay performance, worker/definition versioning, and engine evolution. [Workflow System Fundamentals](./01-workflow-system-fundamentals.md) covers mechanism selection; [Effect Commit Protocols](./06-retry-idempotency-compensation.md) covers activity effects; [Leases and Recovery](./08-leases-heartbeats-recovery.md) covers heartbeat and fencing; [Workflow Observability](./09-workflow-observability-replay.md) covers operator timelines; [DAG Orchestration](./05-dag-orchestration.md) covers static graphs and data intervals.

## Primary Evidence and Scope

| Primary evidence | What it establishes | Boundary |
|---|---|---|
| Temporal workflow/event-history documentation | Commands are checked against event history during replay; history is append-only durable execution state | Temporal semantics and current implementation limits |
| Temporal History Service architecture | History/mutable-state transitions and internal transfer/timer tasks use transactional state plus an outbox-like queue before Matching dispatch | Open-source Temporal implementation, not a universal service topology |
| Microsoft Durable Task documentation | Orchestrator state is rebuilt through event sourcing/replay; orchestrators must be deterministic; activities may run at least once | Durable Task family and storage-provider qualifications |
| Cadence topology and versioning documentation | Frontend, History, Matching, persistence, external workers, and explicit code version markers form a durable code-replay system | Cadence terminology/APIs |
| AWS Step Functions workflow-type documentation | Standard and Express workflow types have different execution semantics and interpret persisted state-machine definitions | Do not project code-replay constraints or one delivery claim across all Step Functions modes |

## Durable-execution contract

Define:

| Field | Required answer |
|---|---|
| **Workflow identity** | Namespace, stable workflow/business ID, physical run ID, parent/root, and duplicate-start policy? |
| **Definition model** | Replayed code, interpreted state-machine version, or application-owned explicit state table? |
| **History authority** | Which ordered events/transition record are canonical, and which mutable/visibility views are derived? |
| **Command boundary** | Which operations are durable orchestration commands versus external activities/effects? |
| **Message contract** | Signals/events/updates, identity, ordering, schema, sender authorization, and duplicate handling? |
| **Timer contract** | Fire-at semantics, lateness SLO, cancellation race, and simultaneous-timer burst? |
| **Activity contract** | Task queue/capability, timeouts, progress checkpoint, cancellation, result size, and logical-effect identity? |
| **Evolution** | Which histories run on which build/definition, and how are incompatible changes introduced/retired? |
| **History lifecycle** | Event/byte limit, rollover rule, closed retention, archival, encryption key, and restore? |
| **Repair** | Who may reset, terminate, cancel, pause, skip/patch, or start a compensating/new run? |

Durable execution does not mean physical exactly-once execution. Workflow code may replay many times. Activity/task semantics differ by engine and configuration, and arbitrary remote effects remain outside the history transaction.

## Two engine families

### Code replay

Temporal, Cadence, and Azure Durable orchestrators expose ordinary-looking code. The engine records commands and externally supplied outcomes; on activation the SDK reruns orchestration code and feeds recorded results back until it reaches the history frontier. Correctness requires the current code and SDK to emit a command sequence compatible with that history.

The workflow code is a deterministic state-machine generator. Local variables are reconstructed results, not independently durable facts. The history (not a worker cache or stack snapshot) is authority.

### Interpreted state machine

AWS Step Functions and other declarative engines persist a definition/state and have the service interpret the next transition. User orchestration code is not replayed, so language-level deterministic restrictions do not apply in the same form. Definition versioning, input/output transforms, task semantics, quotas, and state-transition compatibility still matter.

Do not flatten product guarantees. AWS currently documents different semantics for Step Functions Standard, asynchronous Express, and synchronous Express workflows. Even when a service describes a state transition/task execution as exactly once, an explicitly configured retry or an external API's commit/response ambiguity can repeat effects. Record the exact workflow type and boundary.

## Reference architecture

~~~mermaid
flowchart LR
    C[Client / signal sender]
    F[Frontend API and authorization]
    H[History shard / transition authority]
    P[(History + mutable-state persistence)]
    I[Durable internal transfer/timer tasks]
    M[Matching / task queues]
    WW[Workflow workers]
    AW[Activity workers]
    E[External systems]
    V[(Visibility/search projection)]

    C --> F
    F --> H
    H --> P
    H --> I
    I --> M
    M --> WW
    M --> AW
    WW --> H
    AW --> H
    AW --> E
    H --> V
~~~

The **History/transition authority** serializes events for one run and maintains a derived summary of pending activities, timers, children, messages, and status. **Internal transfer tasks** durably request workflow/activity dispatch; **timer tasks** wake due executions. **Matching** connects task queues to compatible long-polling workers. Workflow workers make deterministic decisions; activity workers interact with the outside world. **Visibility** indexes searchable metadata asynchronously and is never transition authority.

**Documented, Temporal implementation:** a History state transition transactionally updates mutable state and internal history tasks; queue processors eventually transfer workflow/activity tasks to Matching. This is an outbox boundary: Matching can receive a duplicate transfer, while History state determines whether a completion is current.

The engine cluster does not normally execute user code. Application workers poll outbound through authenticated task queues, which improves network isolation but makes task-queue routing and worker availability part of progress.

## Durable state and invariants

An engine commonly persists:

```text
namespace, workflow ID, run ID, root/parent/continued-run IDs
workflow type and definition/build/version-routing metadata
start input schema/reference/digest and identity
ordered history branch, next event ID, and history checksum/size
mutable-state revision and execution status
pending workflow task and last accepted task frontier
pending activities, attempts/timeouts, and result references
pending timers and due times
children and external-message/signal/update state
cancellation/termination/pause intent
search/visibility attributes and projection revision
retention, archival, encryption codec/key version, and audit principal
```

An event envelope contains monotonically ordered event ID within a run, type, server-accepted time, typed attributes, causation/command identity, payload codec/schema, and audit attribution where supported. Do not let a user-supplied timestamp establish history order.

**Reference-design invariants:**

1. Events for one workflow run have one serialized order and are never mutated in place.
2. A history transition atomically advances authoritative mutable state and durably records every internal task required to make the transition progress.
3. Only a workflow-task completion compatible with the current task token/history frontier can append its command batch; duplicate or stale completions cannot fork the run.
4. Replay consumes the same recorded outcomes and emits the same compatible command sequence before the frontier.
5. Workflow code performs no unrecorded external I/O or nondeterministic decision whose value affects commands.
6. A timer wait persists state and releases application-worker compute; firing creates a durable history transition/task rather than relying on a sleeping process.
7. Activity completion is correlated to the exact scheduled activity/attempt authority; recording completion is separate from the external effect.
8. Visibility, caches, and snapshots can lag or disappear without changing execution truth.
9. Rollover/continue-as-new links runs and transfers explicit logical state without rewriting old history.
10. A worker/definition version is not retired while any reachable history still requires it, unless an explicitly compatible path has been proven.
11. Restore preserves history order, run identity, timer/task reconciliation, and version/key availability.

## Workflow-task replay protocol

~~~mermaid
sequenceDiagram
    participant S as History service
    participant Q as Matching/task queue
    participant W as Workflow SDK worker

    S->>Q: enqueue workflow task(run, frontier, task token)
    W->>Q: long-poll compatible task queue
    Q-->>W: task + history/events
    W->>W: replay deterministic workflow from start/checkpoint
    W->>W: match emitted commands to recorded events
    Note over W: first unmatched command is the progress frontier
    W->>S: complete task(token, command batch, query/update results)
    S->>S: validate current token/frontier
    S->>S: append events + mutable state + internal tasks atomically
    S-->>W: accepted or stale/conflict
~~~

Detailed flow:

1. The service commits an input event (start, activity result, timer fire, signal, cancellation, or child event) and makes a workflow task available.
2. A compatible worker receives a task token and history suffix/full history according to SDK/cache state.
3. The SDK invokes workflow code in a deterministic scheduler. Each durable API call emits a command or awaits a recorded outcome.
4. Before the frontier, commands are matched against history and recorded results are returned; no already-recorded activity or timer is newly scheduled.
5. At the first undecided command, the workflow runs until it blocks or completes and returns a batch.
6. The service validates task authority/current execution state, appends the implied events, updates mutable state, and creates required transfer/timer/visibility work.
7. If the completion is stale or ambiguous, the worker reloads/replays; it never invents a second history branch locally.

The system records the commands and externally observed outcomes needed to reconstruct orchestration, not necessarily every physical retry event. For example, Temporal's current documentation says activity retries can remain represented by the scheduled event until a terminal activity event is written. Treat provider event density as measured capacity data.

## Deterministic workflow code

Determinism means: given the same workflow code version, start input, and ordered history, replay must make a compatible sequence of durable commands. It does not require every CPU instruction or log line to be identical.

### Allowed sources of decisions

- immutable workflow input and prior activity/child results from history;
- signals/messages in their history-accepted order;
- SDK workflow time, timers, deterministic random/UUID APIs, and version markers;
- deterministic pure computation and collections with stable ordering;
- configuration captured in input/history/version marker rather than read live;
- SDK-provided deterministic concurrency primitives.

### Unsafe inside replayed workflow code

- direct network, database, filesystem, process, or environment reads;
- wall clock, ordinary randomness/UUID, locale/time-zone defaults, or host identity;
- unordered map/set iteration when order changes commands;
- native threads, blocking synchronization, or nondeterministic race winners;
- mutable global/singleton state;
- dependency/library/runtime upgrades that change ordering, serialization, numeric, or exception behavior;
- logging/metrics with side effects unless replay-aware.

Push external work into activities. Use a product's recorded side-effect API only for a small nondeterministic value it explicitly supports; it is not a substitute for a retryable activity or effect protocol. Temporal's current docs, for example, warn that a Side Effect function that can fail may execute more than once.

Nondeterminism is often latent: a warm sticky worker may retain reconstructed state and avoid full replay until deploy, eviction, failover, or cache miss. Replay tests are release gates, not optional recovery tests.

## Activities and the external-effect boundary

A workflow command schedules an activity task with logical activity identity, input, task queue/capability, and timeout policy. An activity worker executes unrestricted code and reports result/failure. The history service records the accepted outcome and wakes the workflow.

The unsafe gap remains:

```text
activity commits remote effect
        -> completion reply is lost or worker dies
        -> history lacks completion
        -> engine may schedule another attempt
```

Temporal and Durable Task document at-least-once activity behavior under this failure. Other workflow types can document different task semantics, especially without configured retry, but no engine can retroactively make an arbitrary external system share the history transaction. [Effect Commit Protocols](./06-retry-idempotency-compensation.md) owns stable effect keys, provider retention/parameter matching, local transactions, reconciliation, and compensation.

Timeouts answer different questions. Temporal-style names are useful vocabulary:

| Timeout | Question answered |
|---|---|
| **Schedule-to-start** | How long may a task wait for a compatible worker? |
| **Start-to-close** | How long may one physical activity attempt run? |
| **Schedule-to-close** | How long may all waits/attempts consume for this logical activity? |
| **Heartbeat** | How long may a heartbeat-enabled attempt go without liveness/progress evidence? |

An activity heartbeat can carry a bounded checkpoint so a retry resumes application progress, but its durability/frequency is product-specific. It does not prove a remote effect did not commit, and it does not fence a stale worker by itself.

Large activity inputs/results make every history fetch, replication, archival, and replay expensive. Store large immutable artifacts externally and put authenticated reference, version, size, and checksum in history. The artifact lifetime and encryption key must cover workflow retention/replay.

## Durable timers, messages, and cancellation

A durable timer is state such as `(run, timer_id, fire_at, status)`, indexed by due time. Starting it appends/commits the timer decision; the workflow task completes and the application worker can forget the execution. At/after `fire_at`, the timer processor records a fire transition and schedules another workflow task.

No application thread remains allocated during the wait, but the service still consumes history bytes, timer-index entries, replication, storage, and future dispatch capacity. “Fire at 09:00” means eligible according to the engine's time semantics; queueing/failover can make delivery later. Define lateness SLO and cancellation/fire race behavior.

Signals or external events are durable only after the authoritative service accepts/records them. Client retry needs a message identity or convergent handler because a lost response can make acceptance ambiguous. Concurrent timer, signal, cancellation, and activity events obtain some serialization order in history; workflow logic must make every allowed order safe rather than infer sender wall-clock order.

History prevents the classic lost wakeup: an event accepted before workflow code reaches `await signal` remains available when replay later evaluates the wait. The workflow still needs a durable condition/state, bounded message cardinality, schema evolution, sender authorization, and timeout/escalation policy.

Cancellation is a recorded request. Workflow/activity code must observe it at defined yield/heartbeat points and decide cleanup/compensation. Termination is an administrative stop that may bypass cleanup. Neither undoes external effects.

## History growth, caches, snapshots, and rollover

History is an audit/recovery log and a capacity liability. Every activity, timer, child, message, retry outcome, and workflow task can add transitions/events. Products impose event/byte/message limits that change by version; store current configured limits as policy and alert well before them.

Code-replay SDKs use sticky worker caches, incremental history, and derived mutable state to avoid full replay on every activation. These are performance optimizations. Cache loss must fall back to authoritative history with identical decisions.

An engine may persist a snapshot/checkpoint of reconstructed state. Bind it to workflow/run, exact history event ID/hash, SDK/definition version, and payload codec. On mismatch or corruption, discard it and replay. A snapshot that cannot be verified is a second authority and a recovery risk.

**Continue-as-new/history rollover** closes one run and starts a linked run with explicit compacted state and a fresh history. Temporal preserves Workflow ID and creates a new Run ID. Rollover must define:

- which state is carried as new input and how it is validated;
- how pending messages/signals and active handlers are drained or forwarded;
- whether activities/children remain attached to the old run;
- definition/build version selection for the new run;
- lineage, business identity, cancellation, and result semantics across the chain.

Use rollover before the emergency threshold, driven by history length/bytes, replay latency, signal count, and a business-safe boundary. It is not garbage collection for an arbitrary mid-effect state.

## Capacity and performance model

Size at least four workloads:

1. **history/state transitions:** starts, workflow tasks, activities, timers, messages, children, cancels, resets;
2. **matching/polling:** task enqueue, sync/async match, long polls, compatible worker routing;
3. **workflow replay:** activations multiplied by history/replay work and cache-miss rate;
4. **activity execution:** external worker resources and downstream capacity.

Let workflow start rate be $\lambda_w$, mean history events per completed run be $\mathbb{E}[N_{\mathrm{events/run}}]$, mean encoded event size be $b$, and other independent event rate be $\lambda_{\mathrm{other}}$:

$$
\begin{aligned}
\lambda_{\mathrm{events}}
  &= \lambda_w\,\mathbb{E}[N_{\mathrm{events/run}}] + \lambda_{\mathrm{other}}, \\
B_{\mathrm{history/second}}
  &= \lambda_{\mathrm{events}} b
\end{aligned}
$$

**Illustrative:** 500 workflow starts/s with 80 events/run produce 40,000 events/s before signals or recovery. At 1.2 KiB/event, raw ingest is about 46.9 MiB/s. Seven days of that closed-history volume is roughly 27 TiB before replication, indexes, open histories, backups, and compression. Measure real encoded distributions; large payload tails dominate.

Workflow-task rate is driven by activations, not starts alone:

$$
\lambda_{\mathrm{workflow\ tasks}} \approx
\lambda_{\mathrm{starts}} + \lambda_{\mathrm{activity\ completions}}
 + \lambda_{\mathrm{timer\ fires}} + \lambda_{\mathrm{messages}}
 + \lambda_{\mathrm{child\ events}} + \lambda_{\mathrm{recovery}}
$$

Replay CPU is roughly the sum of history/command processing across cache-miss activations. One old workflow with 40,000 events activated every second can consume more replay CPU than thousands of sleeping workflows. Track instructions/time per replay and move to rollover before latency approaches the workflow-task timeout.

Timer capacity follows the due-time distribution, not only sleeping count. If 20 million timers are spread uniformly across ten minutes, average fire rate is about 33,333/s; if they share one deadline, the instantaneous burst is much larger. Shard/bucket timers, jitter product semantics where allowed, and reserve matching/worker/downstream capacity for the wakeup wave.

Activities follow the worker/backlog model in [Background Jobs](./02-background-jobs-worker-pools.md). More workflow workers do not fix history persistence, matching, or downstream saturation.

## Safe workflow evolution

Long-lived histories make code deployment a data-compatibility migration. Classify changes by whether they alter durable command sequence for existing history.

Potentially replay-safe changes include comments, replay-aware logs, and pure refactors proven to preserve command order/parameters. Breaking changes include adding/removing/reordering an activity/timer/child before an existing command, changing branch results for recorded input, changing message-handler scheduling, changing serialization, or reading a new mutable configuration.

Use one or more explicit mechanisms:

### History version/patch marker

Workflow code asks the SDK for a version at a stable code point. The chosen version is recorded; old histories take the old command path and new histories take the new path. Cadence documents `GetVersion`; Temporal supports patch/version mechanisms. Retain the old branch until no history can replay through it.

### Compatible worker/build routing

Tag worker builds and route pinned histories/task queues only to compatible code. Temporal's current Worker Versioning model includes pinned and auto-upgrade behavior with product-specific constraints. Capacity planning includes every live compatibility cohort; “old workers exist” is insufficient if they cannot keep up.

### Instance/definition version

Associate each instance with an immutable orchestration/definition version and keep versioned code or declarative state machines side by side. Azure Durable Task documents built-in orchestration versioning. New starts use the new version; old instances drain or migrate at an explicit safe point.

### New run/workflow type

At a business-safe boundary, continue/start a new version with validated explicit state and lineage. This avoids replaying old history through new semantics but requires a protocol for messages, children, effects, and rollback across the handoff.

Release gates:

1. statically lint known nondeterministic APIs where SDK tooling exists;
2. replay a corpus of real production histories against the candidate build;
3. canary new starts and forced cache eviction/replay;
4. route a bounded compatibility cohort;
5. observe nondeterminism, replay latency, task age, and stuck versions;
6. remove old branches/builds only after authoritative live-history inventory reaches zero plus retention/repair policy.

Rollback routes tasks back to compatible code. It does not remove events written by the new build; if old code cannot replay them, recovery requires the compatible branch/build or a new forward repair.

## Specialized failure traces

### Sticky cache hides nondeterminism

1. A new workflow build reads an unordered map but stays on a warm worker.
2. Local reconstructed state advances without full replay for days.
3. Deployment evicts the cache; replay chooses a different command order and the instance stops.

Force cold replay in CI/canary against real histories. Healthy happy-path execution is not replay compatibility evidence.

### Duplicate workflow task tries to fork history

1. A workflow-task completion is committed but its response is lost.
2. The worker retries or another copy returns a different batch.
3. Without task-token/frontier validation, both append successors to one state.

History authority accepts only a completion valid for the current task/frontier and makes duplicate completion idempotent or stale. Workers reload rather than merge histories.

### Activity effect succeeds; history completion is absent

The payment API commits, but the activity worker dies before History records completion. The engine eventually tries again. This is normal at-least-once activity behavior in code-replay systems, not a history-corruption bug. Stable effect identity/reconciliation closes the gap.

### Timer wakeup wave overloads the system

Millions of subscriptions share midnight expiry. Timer processors enqueue workflow tasks, workflow workers schedule activities, and activities overload the database simultaneously. Capacity-test the entire timer-to-effect chain; shard, rate-limit, stage, or jitter only where business semantics permit.

### History approaches the hard limit during an incident

A chatty signal loop and retries grow history; replay slows, creating workflow-task timeouts and more transitions. The instance can no longer reach the code that would roll it over. Trigger rollover on an early soft threshold and have a product-supported repair path; do not wait for the hard limit.

### Old version has no compatible poller

Routing correctly pins a history to build 12, but deployment scaled build 12 to zero. The execution remains “running” while task age grows. Monitor compatible poller/capacity per version and block retirement until no reachable histories remain.

### Rollover drops application-level pending work

Workflow compacts only its main state and starts a new run while its application-level inbox/handler work is not incorporated. A message accepted near the boundary is neither reflected nor deliberately forwarded. Define a quiesce/drain/transfer protocol and fault-test messages at every rollover boundary.

### Visibility projection drives an invalid reset

Search lags and says a workflow is stuck before an activity; authoritative history already passed it. An operator resets from the stale view and reopens downstream effects. Repair tools resolve and condition on current history/run revision before mutating state.

### Encryption key expires before history

Events persist for audit/replay, but the codec key or artifact referenced by an activity result is deleted first. Failover can load bytes but cannot reconstruct state. Couple payload/artifact/key retention to every open, archived, resettable, and legally held history.

## Security and abuse boundaries

Authorize start, signal/update, query, cancel, pause, reset, terminate, and history read separately by namespace/workflow/tenant. Treat task-queue polling and worker registration as capabilities: a malicious worker can read activity inputs, execute effects, and submit results unless the service authenticates and authorizes it.

History is a high-value long-retention record. Minimize payloads, use client-side payload codecs/encryption where required, rotate keys without losing replay, and keep sensitive values out of broadly indexed visibility attributes. Redact error/stack data and external artifact URLs. Deletion/retention policy must account for backups and archives.

Validate and quota signals, updates, child fan-out, timers, history bytes/events, payload sizes, query cost, and reset frequency. An authorized tenant can otherwise create a replay/history/timer denial of service. Workflow code and SDK dependencies are supply-chain authority over long-lived business processes; sign artifacts, restrict build routing, and audit version rules.

Reset/terminate/skip/patch actions can repeat or bypass business effects. Require reason, approval where needed, dry-run/current-revision evidence, immutable audit, and post-action reconciliation. Do not give observability users implicit mutation privileges.

## Observability and verification

Observe per namespace, workflow type/version/build, task queue, and tenant:

- start/message/cancel/update acceptance, deduplication, and authorization;
- history events/bytes, transition rate, mutable-state conflicts, persistence and replication latency;
- workflow-task schedule-to-start, execution/replay time, cache hit/miss, replayed events, stale completions, and nondeterminism;
- matching backlog/oldest task, pollers and compatible capacity by build;
- activities scheduled/started/completed with timeout/heartbeat/result-size evidence;
- timers outstanding/due/fire lateness and simultaneous-deadline bursts;
- open/closed/continued chains, rollover thresholds, run lineage, and stuck version cohorts;
- visibility lag versus authoritative history;
- reset/pause/cancel/terminate/version-rule changes and unresolved effect reconciliation.

[Workflow Observability](./09-workflow-observability-replay.md) owns the operator timeline, search model, trace lineage, and forensic replay interface.

Verification includes:

1. deterministic unit tests using SDK virtual time/scheduler;
2. replay of golden and sampled production histories on every candidate build;
3. history-prefix/property fuzzing across every await, message, timer, cancel, and exception order;
4. crash before/after workflow-task dispatch, command commit, activity dispatch/completion, timer fire, and visibility update;
5. duplicate/stale workflow-task completion and history-service failover;
6. sticky-cache eviction, snapshot corruption/mismatch, and full-history fallback;
7. activity effect committed with lost completion and stale attempt return;
8. timer storms, message floods, large histories/results, rollover, and restore;
9. mixed-version workers, missing compatible pollers, patch markers, rollback, and old-branch retirement;
10. payload/codec/key/artifact retention plus authorization and malicious fan-out tests.

## Decision framework

1. Does the process require durable waits/dynamic control flow and finer recovery than a bounded job or DAG node?
2. Is the engine replaying user code or interpreting a persisted definition, and which constraints follow?
3. Which ordered history/state is authoritative, and which caches/mutable/visibility views are derived?
4. How does one workflow-task token/frontier prevent duplicate workers from forking history?
5. Which values enter decisions only through deterministic SDK/history APIs?
6. Which operations are activities, and what effect protocol covers their commit gap?
7. What timer/message ordering, lateness, deduplication, and cancellation races are part of the contract?
8. What event/byte/replay/timer limits trigger rollover well before failure?
9. Which histories route to which compatible builds/definitions, and how is capacity proven for every cohort?
10. Can candidate code replay real histories cold before deployment and after rollback?
11. Do history payload, artifacts, codecs, and encryption keys survive every replay/reset/retention horizon?
12. Can operators repair from authoritative state without silently repeating or bypassing effects?

## Primary references

- [Temporal, *Workflow Execution overview*](https://docs.temporal.io/workflow-execution)
- [Temporal, *Events and Event History*](https://docs.temporal.io/workflow-execution/event)
- [Temporal, *Continue-As-New*](https://docs.temporal.io/workflow-execution/continue-as-new)
- [Temporal source, *History Service architecture*](https://github.com/temporalio/temporal/blob/main/docs/architecture/history-service.md)
- [Temporal, *Activity Execution*](https://docs.temporal.io/activity-execution)
- [Temporal, *Worker Versioning*](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning)
- [Cadence, *Deployment topology*](https://cadenceworkflow.io/docs/concepts/topology)
- [Cadence, *Workflow versioning*](https://cadenceworkflow.io/docs/go-client/workflow-versioning)
- [Microsoft, *Durable orchestrations*](https://learn.microsoft.com/en-us/azure/durable-task/common/durable-task-orchestrations)
- [Microsoft, *Durable orchestrator code constraints*](https://learn.microsoft.com/en-us/azure/durable-task/common/durable-task-code-constraints)
- [Microsoft, *Durable orchestration versioning*](https://learn.microsoft.com/en-us/azure/durable-task/common/durable-orchestration-versioning)
- [AWS Step Functions, *Choosing workflow type*](https://docs.aws.amazon.com/step-functions/latest/dg/choosing-workflow-type.html)
- [Amazon States Language specification](https://states-language.net/spec.html)
