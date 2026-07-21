# Background Jobs and Worker Pools

A background-job system accepts a bounded unit of work, makes it eligible through a durable queue or job table, gives a replaceable worker temporary authority to attempt it, and records a terminal or retryable outcome. Its safety depends on the exact claim token, state transition, and shutdown protocol—not on the existence of “a queue.”

Background-job execution covers job records and transitions, receive/claim/complete protocols, broker acknowledgement interaction, database-backed claims, concurrency, restart boundaries, and graceful shutdown. [Message Queue Architecture](../05-messaging/01-message-queues.md) covers broker internals; [Poison-Message Quarantine](../05-messaging/08-dead-letter-queues.md) covers quarantine and redrive; [Priority, Fairness, and Backpressure](./07-priority-fairness-backpressure.md) covers allocation and admission; [Effect Commit Protocols](./06-retry-idempotency-compensation.md) covers external effects; [Leases and Recovery](./08-leases-heartbeats-recovery.md) covers lease timing, progress, and fencing.

## Primary Evidence and Scope

| Primary evidence | What it establishes | Boundary |
|---|---|---|
| Amazon SQS visibility and delete documentation | Receive returns a receipt handle and temporarily hides a message; visibility can change; delete uses the latest handle; standard queues may still redeliver | SQS semantics, not RabbitMQ or database-row claims |
| RabbitMQ consumer acknowledgement documentation | Manual acknowledgements are channel/delivery-tag scoped; unacknowledged deliveries requeue on channel/connection loss; prefetch bounds the unacknowledged window | AMQP/RabbitMQ semantics; not a renewable per-message visibility lease |
| PostgreSQL 18 `SELECT` documentation | `FOR UPDATE SKIP LOCKED` can avoid row-lock contention for multiple consumers of a queue-like table | Inconsistent view by design; the application supplies lease, generation, and recovery state |
| Kubernetes Pod termination documentation | Termination is a bounded sequence with a grace period; forced termination remains possible | Process lifecycle, not proof that an external effect or broker ack completed |
| Sidekiq and Celery task guidance | Jobs should carry simple/versionable inputs, be idempotent, and use late acknowledgement/retry behavior deliberately | Framework-specific defaults must still be verified by version/configuration |

## Job execution contract

Define before enqueue:

| Field | Required answer |
|---|---|
| **Logical identity** | Which obligation does this job fulfill, and what start key deduplicates producer retries? |
| **Job type/version** | Which handler and payload schema can interpret it throughout retention? |
| **Input semantics** | Immutable snapshot/version, or “reconcile entity to latest desired state”? |
| **Eligibility** | Immediate or `available_at`; dependency and cancellation conditions? |
| **Completion** | Which durable result/state makes the job done? |
| **Attempt authority** | Receipt handle/channel delivery tag or owner+claim generation+deadline? |
| **Execution bound** | Expected/max runtime, checkpoint policy, deadline, memory/CPU/GPU, and fan-out? |
| **Failure result** | Retry wait, permanent failure, quarantine, cancellation, or operator review? |
| **Effect protocol** | Which stable logical-effect identity protects every external mutation? |
| **Concurrency budget** | Per process, job class, tenant, and downstream dependency? |
| **Shutdown** | Can an active job finish inside grace, checkpoint/relinquish, or tolerate forced retry? |

Use a background job when one bounded handler can restart safely from the beginning or from an explicit application checkpoint. If correctness requires a durable graph frontier, many external signals, or weeks-long orchestration, use a [DAG](./05-dag-orchestration.md) or [durable workflow](./04-durable-execution-workflow-engines.md).

## Durable job and attempt state

Separate the logical job from physical attempts.

```text
job:
  namespace / tenant
  job ID and producer request key
  job type and immutable handler version constraints
  payload schema, reference, digest, and codec/key version
  state and state revision
  available_at, business deadline, created/terminal timestamps
  execution/checkpoint contract and latest checkpoint version/reference
  cancellation intent and actor
  terminal result reference or error classification

attempt:
  job ID, attempt ID, and ordinal
  worker process incarnation
  claim generation and claim/visibility deadline when applicable
  broker receipt handle or delivery tag kept at the required scope
  accepted/start/heartbeat/finish timestamps
  progress/checkpoint revision
  outcome, error class, and completion state revision
```

Do not use a hostname or PID alone as worker identity; both are reusable. A process incarnation is unique per start. Do not use the broker delivery count as the sole business-attempt identity: redelivery counters have provider-specific semantics and can reset during redrive or migration.

## State machine and invariants

~~~mermaid
stateDiagram-v2
    [*] --> Accepted
    Accepted --> Ready: eligible
    Ready --> Claimed: atomic receive/claim
    Claimed --> Running: handler admitted
    Running --> Succeeded: terminal commit
    Running --> RetryWait: retryable outcome
    RetryWait --> Ready: next eligibility
    Claimed --> Ready: claim abandoned/expired
    Running --> Ready: claim expired; new generation
    Accepted --> Canceled: cancel before start
    Ready --> Canceled
    Running --> Canceled: cooperative stop committed
    Running --> Failed: permanent outcome
    RetryWait --> Failed: policy exhausted
~~~

“Expired” creates eligibility for another attempt; it does not prove the first process stopped. The old attempt may still execute and must not overwrite a successor's terminal state.

**Reference-design invariants:**

1. Accepted jobs are durable before producer success is returned.
2. One state revision has at most one current claim generation, while physical duplicate execution remains possible under broker or network failure.
3. Claim, renew, relinquish, completion, and cancellation identify the exact attempt authority; an old receipt/owner/generation cannot complete a newer attempt.
4. A job becomes terminal only after its declared result is durable; broker acknowledgement/deletion follows that commit.
5. Redelivery of a terminal job is recognized and acknowledged without repeating the logical work.
6. A job waiting locally for a worker slot does not consume an unbounded hidden/claimed broker window.
7. Cancellation prevents or interrupts future work according to policy but never claims to undo an already committed effect.
8. Every nonterminal job is discoverable by broker delivery, an eligibility scan, claim expiry, or reconciliation.
9. Handler/payload/key versions remain available until no retained or delayed job can require them.
10. Forced worker death may cause retry, not silent terminal success.

## Worker execution path

~~~mermaid
sequenceDiagram
    participant Q as Queue / eligible-job index
    participant W as Worker incarnation
    participant J as Authoritative job store
    participant E as Effect system

    W->>W: reserve local class/downstream capacity
    W->>Q: long-poll or atomically claim
    Q-->>W: job + attempt authority token
    W->>J: CAS claim/start with generation
    W->>W: validate handler, payload, deadline, cancellation
    W->>E: execute with stable logical-effect identity
    W->>J: conditional terminal/retry transition
    J-->>W: committed state revision
    W->>Q: ack/delete exact delivery token
~~~

Reserve or bound local execution capacity before pulling a large batch. Otherwise the client hides/prefetches jobs that are not running, ages their visibility/ack deadline, and steals them from workers that have capacity.

The terminal-state commit and external effect may not share a transaction. The diagram intentionally leaves that gap visible; [Effect Commit Protocols](./06-retry-idempotency-compensation.md) owns how the logical effect converges.

## Three claim protocols that must not be conflated

### SQS-style visibility receipt

**Documented, Amazon SQS:** receiving a message leaves it in the queue but makes it temporarily invisible and returns a receipt handle. A worker deletes with the most recent receipt handle after processing; `ChangeMessageVisibility` adjusts the current delivery's remaining visibility. Standard queues are at-least-once and can deliver a duplicate even during a visibility interval or after one replicated copy missed a delete.

Worker rules:

1. long-poll only when an execution slot and downstream budget are available;
2. bind the latest receipt handle to that physical attempt;
3. choose initial visibility to cover dispatch-to-start plus a measured execution interval;
4. renew before expiry with margin and jitter when work remains authorized;
5. stop initiating new effects when renewal/authority is lost, while assuming in-flight calls may still complete;
6. durably record the job outcome, then delete using the current receipt handle;
7. on duplicate delivery, read terminal/effect state and converge.

SQS currently documents a maximum visibility interval of 12 hours from the receive request. Jobs exceeding a broker's maximum claim window need segmentation/checkpointing or another mechanism; repeated extension does not create infinite ownership.

### RabbitMQ-style manual acknowledgement

RabbitMQ manual acknowledgement is not a renewable SQS visibility timeout. A delivery tag is scoped to its channel. The broker maintains an unacknowledged delivery window bounded by prefetch; if the channel or connection closes before ack, unacknowledged deliveries are requeued. Acknowledging on another channel or acknowledging the same/unknown tag is a channel error.

Worker rules:

- use manual acknowledgement for work that must survive consumer failure;
- keep prefetch bounded relative to active handler capacity and payload memory;
- ack on the original channel only after durable completion;
- reject/nack according to an explicit retry/quarantine policy;
- on consumer cancellation, finish or intentionally abandon deliveries already in flight before closing the channel.

RabbitMQ consumer concurrency, channel count, prefetch, and downstream concurrency are distinct controls. Prefetch bounds delivered-but-unacknowledged messages; it does not prove that many handlers or effects are active.

### Database-backed atomic claim

A database can make the job row authoritative and use a short transaction to claim eligible rows. PostgreSQL example:

```sql
WITH candidates AS MATERIALIZED (
    SELECT id
      FROM jobs
     WHERE state = 'READY'
       AND available_at <= clock_timestamp()
     ORDER BY available_at, id
     FOR UPDATE SKIP LOCKED
     LIMIT :batch_size
)
UPDATE jobs AS j
   SET state = 'CLAIMED',
       state_revision = j.state_revision + 1,
       claim_generation = j.claim_generation + 1,
       claim_owner = :worker_incarnation,
       claim_expires_at = clock_timestamp() + :claim_interval
  FROM candidates AS c
 WHERE j.id = c.id
RETURNING j.*;
```

Commit this transaction before external execution. Holding the row lock during network/CPU work turns worker runtime into database lock time and makes recovery depend on a long transaction.

Completion is conditional:

```sql
UPDATE jobs
   SET state = 'SUCCEEDED',
       state_revision = state_revision + 1,
       terminal_at = clock_timestamp(),
       result_ref = :result_ref,
       claim_owner = NULL,
       claim_expires_at = NULL
 WHERE id = :job_id
   AND state IN ('CLAIMED', 'RUNNING')
   AND claim_owner = :worker_incarnation
   AND claim_generation = :claim_generation;
```

Zero updated rows means authority changed; it is not success. A reclaimer makes expired rows eligible using conditional state/revision rules, and the next claim receives a greater generation. External resources still need fencing/idempotency; a row update cannot recall a stale worker's already-sent API request.

PostgreSQL explicitly says `SKIP LOCKED` gives an inconsistent view and is suitable for queue-like consumers, not general queries. Deterministic ordering helps age fairness, but locked/hot rows may be skipped repeatedly; fleet fairness remains the policy in [Priority, Fairness, and Backpressure](./07-priority-fairness-backpressure.md).

### Hybrid job truth plus broker wakeup

For complex/critical jobs, keep lifecycle state in a database and use the broker only to wake workers. Messages carry job ID and expected revision, not the sole copy of job truth. A missing wakeup is repaired by an eligibility scan; duplicate wakeups lose the claim CAS or observe terminal state. This adds two systems and reconciliation but makes inspection, version gating, cancellation, and repair explicit.

## Job inputs and checkpoints

Prefer a small, immutable envelope over serialized application objects. If it references mutable data, declare one of two semantics:

- **snapshot:** include entity/object version, content digest, or `as_of` token; fail/reconcile if it changed;
- **convergent latest:** the job means “move entity X toward current desired state,” and repeated old jobs safely converge.

A bare row ID is not automatically safe; it can transform an enqueue-time command into an execution-time read of unrelated newer state.

Checkpoints are application state, not broker acknowledgements. A useful checkpoint contains input/version, completed range or cursor, output manifest/checksum, and monotonic checkpoint generation. Persist it before releasing the claim. On retry, validate it against the same logical job and immutable input; never trust a partially uploaded artifact merely because its key exists.

Split work when one attempt cannot fit the maximum claim/shutdown window or when restart cost exceeds the orchestration cost. Do not chain jobs with “job A completes, then best-effort enqueue B”: atomically commit A's transition plus a dispatch intent, or use a workflow engine, so crash does not lose or duplicate the continuation.

## Concurrency and capacity

Concurrency is a vector:

```text
broker prefetch / hidden deliveries
local handler slots by job class
CPU, memory, GPU, file, and subprocess slots
database/client connection pools
per-tenant limits
per-downstream concurrency and rate budgets
fan-out inside each job
```

Pool size alone bounds none of the later items if one handler starts several calls or tasks. Admit a job only when all required bulkheads can be acquired in a fixed/nonblocking protocol; avoid holding one scarce resource while waiting indefinitely for another.

For admitted job rate $\lambda$ and mean slot occupancy $W$, Little's Law gives mean active handlers:

$$
L = \lambda W
$$

For homogeneous workers $i$ with $c_i$ safe slots and mean service time $S$, an optimistic service-rate bound is:

$$
\mu \le \frac{\sum_i c_i}{\mathbb{E}[S]}, \qquad \lambda < \mu \text{ for a stable backlog}
$$

The inequality is optimistic because downstream quotas, heavy tails, retries, tenant skew, and worker loss reduce useful capacity. Plan to the bottleneck dependency and the longest acceptable queue age, not CPU utilization alone.

To drain backlog $B$ within target time $T$ while arrival rate continues:

$$
\mu_{\mathrm{safe}} \ge \lambda + \frac{B}{T}
$$

**Illustrative:** 240 jobs/s with 0.8-second mean slot occupancy require 192 active slots on average. Twelve instances with 24 safe slots provide 288; after one instance fails, 264 remain, or about 73% mean utilization before variability. If an outage creates 900,000 queued jobs and safe completion is 360/s while 240/s continues arriving, optimistic drain time is $900{,}000/120 = 7{,}500$ seconds, or 125 minutes. More workers help only if dependencies sustain 360/s.

Let prefetch/hidden count be $P$ and active local slots be $C$. If $P$ greatly exceeds $C$, up to $P-C$ jobs wait inside clients rather than the visible queue. Their queue-age metric may look improved while end-to-end latency and visibility expiry worsen. Count prefetched-but-not-started jobs and include their wait in claim timing.

[Auto-Scaling](../06-scaling/08-auto-scaling.md) owns controller stability. The worker-specific input is required safe slots by class and downstream; oldest eligible age and predicted drain time are usually stronger SLO signals than raw depth or CPU.

## Graceful shutdown and deployment

Shutdown is a protocol, not `SIGTERM` plus hope:

1. mark the worker draining and stop new receive/claim calls;
2. cancel outstanding long polls/subscriptions without accepting more work;
3. release or make visible any prefetched job that has not started, according to broker protocol;
4. keep authority/renewals for active attempts while they finish inside the grace budget;
5. stop starting new irreversible effects after the remaining grace time falls below their safe bound;
6. durably checkpoint or terminally transition completed work;
7. acknowledge/delete only after that transition;
8. intentionally relinquish unfinished work, then close broker/database resources;
9. allow forced kill to produce redelivery/reconciliation, never fabricated success.

For RabbitMQ, cancel the consumer to stop future deliveries, drain/ack current in-flight deliveries, then close the channel; closing first requeues unacknowledged work. For SQS, stop `ReceiveMessage`, finish/delete active receipts or deliberately set visibility to zero when safe. A long-running external call may complete after process termination or claim loss, so shutdown still depends on effect identity/fencing.

Deployment capacity includes old and new pools concurrently, plus drain. Route only payload/handler versions a worker supports. Canary one job class/tenant, compare duration/error/effect evidence, then expand. Rollback retains the new payload decoder and any state transitions already emitted; an old binary that cannot read queued jobs is not a rollback.

## Specialized failure traces

### Prefetch expires before execution starts

1. A process has 20 handler slots but prefetches 1,000 messages with 60-second visibility.
2. Jobs wait locally for several minutes.
3. Visibility expires and other workers receive them while the first process later starts its copies.

Bound receive to available slots, measure dispatch-to-start time, and size visibility for queue-client wait plus execution—not execution alone.

### Database claim holds a lock through external I/O

A worker selects `FOR UPDATE`, calls a partner for 40 seconds before commit, and blocks claim/recovery transactions while accumulating old snapshots/locks. Commit a persisted claim in a short transaction, then execute outside it using generation-conditional completion.

### Stale completion overwrites a successor

Attempt generation 7 pauses; the claim expires and generation 8 succeeds. Generation 7 resumes and unconditionally writes `SUCCEEDED` with its older result. Completion must compare owner and generation, and protected effects must reject stale authority where possible.

### Completion commits but broker ack is lost

The job store records success, then the connection closes before delete/ack. Redelivery is correct. The next worker reads terminal state, verifies the same logical job, and acknowledges without redoing work. If there is no terminal/effect ledger, the effect protocol must handle the duplicate.

### Ack precedes durable completion

The worker acknowledges, then crashes before result/state commit. The broker has no work and the job store says running. Acknowledge only after durable completion; reconcile running claims that never terminally transition.

### Scale-out overloads a dependency

Queue age triggers a tenfold worker increase, but each job makes four database queries and the database is already at its safe concurrency. Throughput falls while attempts time out. Scale against per-dependency budgets; shed/defer work when the bottleneck cannot expand.

### Shutdown loses locally buffered work

The worker reports ready until termination, receives a large batch, then closes after acknowledging or auto-acking it without executing. Stop claims first, use manual/late ack, account for prefetched items, and rehearse forced kill at every drain phase.

### Handler version disappears

A 14-day delayed job references payload schema 4, but the last schema-4 worker was removed after two days. Generic retries cannot create a decoder. Gate worker retirement on retained/delayed job inventory and keep an upcaster or compatible drain pool.

## Security and abuse boundaries

Authorize producers by tenant and allowed job type; workers revalidate tenant/object authority at execution time because permissions and data may change while queued. Never dispatch arbitrary module/class/function names from untrusted payloads. Use an allowlisted handler registry and safe serialization—language-native object deserialization can execute code.

Payload references need integrity hashes, access controls, expiry compatible with delay/retention, and tenant binding. Do not place long-lived credentials or sensitive records directly in queue bodies, error strings, metrics, or quarantine evidence. Receipt handles, broker credentials, and database claim tokens are attempt authority and should not be logged broadly.

Limit payload bytes, enqueue rate, outstanding jobs, runtime, memory, fan-out, checkpoint bytes, and downstream calls per tenant/type. Sandbox untrusted transforms and restrict network/filesystem access. Separate permissions for enqueue, inspect payload, cancel, force retry, change handler routing, and bulk repair; audit every override.

## Observability and verification

Measure by tenant, job type, handler version, queue/partition, and downstream:

- offered/accepted/deduplicated/rejected enqueue and enqueue-to-eligible latency;
- ready count and oldest ready age, including prefetched-but-not-started work;
- receive/claim latency, claim conflicts, current generation, expiry, renewal margin, and reclaims;
- active handlers, slot utilization, local waiting, memory, and fan-out;
- start-to-finish and checkpoint progress distributions, not only mean runtime;
- terminal outcomes, stale-completion rejection, duplicate deliveries/attempts, and ambiguous effects;
- broker unacknowledged/invisible counts, poll efficiency, ack/delete errors, and receipt/channel errors;
- worker incarnations/pollers by supported version and graceful/forced shutdown residuals;
- predicted completion and backlog drain time under current safe downstream capacity.

Quarantine/redrive metrics belong to [Poison-Message Quarantine](../05-messaging/08-dead-letter-queues.md); retry/effect evidence belongs to [Effect Commit Protocols](./06-retry-idempotency-compensation.md).

Verification includes:

1. model/property tests for allowed state transitions and terminal monotonicity;
2. simultaneous claims, `SKIP LOCKED`, claim expiry, generation rollover, and stale completion;
3. duplicate delivery during valid visibility and after terminal commit;
4. crash before/after claim, effect, checkpoint, terminal commit, and broker ack;
5. SQS current/stale receipt handles and visibility-renewal loss;
6. RabbitMQ channel loss, consumer cancellation, bounded prefetch, requeue, and ack-on-wrong-channel rejection;
7. shutdown with unstarted prefetch, short jobs, jobs longer than grace, hung calls, and forced kill;
8. handler/payload rolling compatibility and delayed old jobs;
9. open-loop load with production duration/fan-out skew, downstream saturation, backlog, and recovery;
10. authorization, malicious payload, oversized fan-out, and operator repair controls.

## Decision framework

1. Can one bounded handler safely restart, or is finer durable progress required?
2. Is the broker message, database row, or hybrid ledger authoritative for job lifecycle?
3. What exact receipt/channel/owner-generation token authorizes this attempt?
4. Does completion conditionally record the exact attempt before broker acknowledgement?
5. How are missing and duplicate wakeups reconciled?
6. What input version/snapshot semantics survive queue delay and data mutation?
7. Are broker prefetch, local handlers, fan-out, and every downstream concurrency budget independently bounded?
8. What safe service rate and failure capacity meet oldest-age and backlog-drain objectives?
9. Can a job checkpoint/relinquish within the deployment grace period?
10. Which old handlers, codecs, payloads, keys, and checkpoint schemas remain reachable?
11. Do authorization and tenant quotas apply at both enqueue and execution?
12. Have ambiguous effect, stale worker, forced shutdown, and duplicate delivery been fault-tested?

## Primary references

- [Amazon SQS, *Visibility timeout*](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-visibility-timeout.html)
- [Amazon SQS API, *DeleteMessage*](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/APIReference/API_DeleteMessage.html)
- [Amazon SQS, *Short and long polling*](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-short-and-long-polling.html)
- [RabbitMQ, *Consumer acknowledgements and publisher confirms*](https://www.rabbitmq.com/docs/confirms)
- [RabbitMQ, *Consumers*](https://www.rabbitmq.com/docs/consumers)
- [RabbitMQ, *Consumer prefetch*](https://www.rabbitmq.com/docs/consumer-prefetch)
- [PostgreSQL 18, *SELECT: locking clause and SKIP LOCKED*](https://www.postgresql.org/docs/current/sql-select.html)
- [PostgreSQL 18, *UPDATE*](https://www.postgresql.org/docs/current/sql-update.html)
- [Kubernetes, *Pod lifecycle: termination of Pods*](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination)
- [Sidekiq, *Best Practices*](https://github.com/sidekiq/sidekiq/wiki/Best-Practices)
- [Celery, *Tasks*](https://docs.celeryq.dev/en/stable/userguide/tasks.html)
