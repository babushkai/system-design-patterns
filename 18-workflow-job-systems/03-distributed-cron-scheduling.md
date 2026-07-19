# Distributed Scheduling and Timer Services

## TL;DR

A distributed scheduler turns time-based intent into durable, claimable work. The difficult parts are not parsing cron expressions. They are materializing each logical occurrence exactly once in scheduler state, surviving missed ticks and failover, preventing stale owners from committing, distributing millions of timers without hot partitions, admitting only work the fleet can finish, and preserving schedule semantics across time zones and version changes.

Separate two responsibilities:

- the **schedule/timer control plane** decides *which logical occurrence is due*;
- the **execution plane** claims, runs, and records attempts at least once.

Give every occurrence a stable identity, persist it before dispatch, use compare-and-swap claims with fencing epochs, make effects safe under redelivery, and expose deterministic repair operations. A lease may ensure that one scheduler normally owns a shard; it does not make worker effects exactly once.

---

## 1. Workload and Contract

Scheduling workloads fall into three families:

1. **recurring schedules:** “run at 02:00 Europe/Amsterdam each business day”;
2. **one-shot timers:** “wake workflow 30 days after approval”;
3. **deferred jobs:** “make this job eligible no earlier than 2026-08-01T10:00Z.”

The service contract should state:

- time expression and time-zone semantics;
- earliest eligible time and optional deadline;
- misfire policy after downtime;
- overlap/concurrency policy;
- delivery and effect semantics;
- cancellation and update behavior;
- schedule/occurrence retention;
- payload schema and code revision;
- tenancy, priority, quota, and resource class.

### 1.1 Core invariants

1. **Unique occurrence:** one logical schedule time maps to one occurrence identity.
2. **Durable materialization:** a due occurrence is stored before dispatch can be lost.
3. **At-least-once execution:** an unacknowledged occurrence becomes claimable again.
4. **Fenced ownership:** a stale scheduler or worker cannot overwrite a newer owner.
5. **Monotonic terminal state:** success is not reverted by a late timeout or cancellation.
6. **Explicit misfire:** downtime behavior is a declared policy, not an accidental burst.
7. **Bounded admission:** accepted work has a capacity or rejection/defer policy.
8. **Tenant isolation:** one tenant's hot schedule cannot monopolize scan or worker capacity.
9. **Versioned semantics:** changing time expression, payload, or handler code does not reinterpret past occurrences.
10. **Repairability:** missed, duplicated, stuck, or corrupt state can be inspected and repaired through audited operations.

“Exactly once at 09:00” is usually two requirements incorrectly combined. The scheduler can uniquely materialize occurrence `O`; delivery and external effects still require the protocols in [Effect Commit Protocols](./06-retry-idempotency-compensation.md).

---

## 2. Data Model and State Machines

```text
schedule:
  tenant_id
  schedule_id
  schedule_revision
  expression
  time_zone
  calendar_policy
  misfire_policy
  overlap_policy
  payload_template
  handler_revision
  priority_class
  shard_key
  next_fire_at
  state
  created_at
  updated_at

occurrence:
  tenant_id
  occurrence_id
  schedule_id
  schedule_revision
  logical_fire_at
  eligible_at
  deadline_at
  payload_ref
  state
  claim_epoch
  claimed_by
  lease_expires_at
  attempt_count
  result_ref
  created_at
  terminal_at
```

A recurring occurrence ID can be derived from:

```text
occurrence_id = hash(
  tenant_id,
  schedule_id,
  schedule_revision,
  canonical_logical_fire_at
)
```

The unique constraint on that tuple makes repeated materialization harmless.

### 2.1 Schedule state

```text
DRAFT -> ACTIVE -> PAUSED -> ACTIVE
ACTIVE -> DRAINING -> RETIRED
ACTIVE/PAUSED -> INVALID
```

`PAUSED` prevents new occurrences but retains state. `DRAINING` prevents new materialization while existing occurrences finish. `RETIRED` is terminal after retention requirements are met.

### 2.2 Occurrence state

```text
SCHEDULED -> READY -> CLAIMED -> RUNNING -> SUCCEEDED
                       |          |       -> FAILED_FINAL
                       |          -> READY       # lease expiry/retry
                       -> CANCELLED

READY/RUNNING -> DEAD_LETTERED
```

Keep logical occurrence and execution attempts separate. One occurrence can have multiple attempts:

```text
attempt:
  occurrence_id
  attempt_number
  claim_epoch
  worker_id
  started_at
  heartbeat_at
  finished_at
  outcome
  error_class
```

This avoids losing forensic history when a new attempt overwrites a `RUNNING` row.

---

## 3. Materializing Recurring Schedules

A scheduler shard repeatedly:

1. reads active schedules whose `next_fire_at` is within a lookahead horizon;
2. computes due logical fire times under the stored schedule revision;
3. inserts occurrences using the unique occurrence identity;
4. advances `next_fire_at` with compare-and-swap;
5. commits occurrence insertion and cursor advancement atomically;
6. publishes a wake-up hint or lets executors scan durable `READY` state.

### 3.1 Transactional materialization

```text
BEGIN
  SELECT schedule WHERE id = ? FOR UPDATE
  due = compute_occurrences(next_fire_at, now, misfire_policy)
  INSERT due occurrences ON CONFLICT DO NOTHING
  UPDATE schedule
    SET next_fire_at = compute_next(...)
    WHERE schedule_revision = expected_revision
COMMIT
```

If the process crashes before commit, neither cursor nor occurrences advance. If it crashes after commit but before notifying workers, durable occurrences remain discoverable. The notification is an optimization, not the source of truth.

### 3.2 Lookahead horizon

Materializing slightly ahead of time hides storage and dispatch latency. Too short a horizon risks late work during control-plane pauses; too long creates unnecessary state and makes schedule edits/cancellations harder.

Let:

- maximum expected control-plane outage be 40 seconds;
- p99 materialization plus publication latency be 3 seconds;
- clock and operational margin be 7 seconds.

Then a minimum lookahead is:

```text
40 s + 3 s + 7 s = 50 s
```

Choose based on the service's failure envelope, not a universal constant. One-shot timers months away should not all become active queue entries months early; keep them in a timer index until within the execution horizon.

---

## 4. Timer Indexes

A full table scan for every tick is not a scheduler.

### 4.1 Ordered time buckets

Partition timers by coarse time bucket and shard:

```text
partition = UTC_date_or_hour(eligible_at)
shard = hash(tenant_id, timer_id) mod N
sort key = eligible_at, timer_id
```

Workers query only current and recovery buckets. Include a tie-breaker so pagination is stable. Avoid a partition key that is only the minute/hour; every timer due at a round boundary would land on one hot partition.

### 4.2 Hierarchical timing wheel

A timing wheel groups deadlines into circular buckets at multiple resolutions:

```text
level 0: seconds within a minute
level 1: minutes within an hour
level 2: hours within a day
level 3: days within a larger horizon
```

As time advances, higher-level buckets cascade into lower levels. Insert and expiry can be close to constant time, making wheels effective for large in-memory timer sets. Durable systems checkpoint wheel state or rebuild it from an authoritative timer store after restart.

Resolution creates bounded lateness. A one-second wheel cannot promise microsecond dispatch. Cascading also creates bursts at bucket boundaries; shard buckets and cap work per tick.

### 4.3 Calendar queues and ordered heaps

A priority heap offers simple exact ordering but costs logarithmic insertion/removal and becomes a single-memory bottleneck. Calendar queues bucket by expected inter-arrival spacing and can approach constant-time operations when distributions are stable, but need resizing when density changes.

Choose the index by timer count, mutation rate, required resolution, persistence model, and distribution. Many durable schedulers use a database/index for long horizon and an in-memory heap or wheel for the near horizon.

---

## 5. Time Semantics

### 5.1 Use wall time for intent, monotonic time for elapsed durations

Recurring calendar intent depends on civil time and a named time zone. Lease duration and timeout measurement depend on elapsed time and should use a monotonic clock within a process.

Do not compare wall clocks from two nodes to decide which owner is current. Ownership comes from a consensus-backed transaction/lease and monotonically increasing epoch.

### 5.2 Time zones and daylight-saving transitions

Store the IANA time-zone identifier and schedule revision, not only a UTC offset. Offsets change.

For a local time that does not exist during a spring-forward transition, policy might be:

- skip;
- run at the next valid instant;
- run at an explicitly mapped UTC time.

For an ambiguous local time repeated during fall-back:

- run once at the first instance;
- run once at the second;
- run twice with distinct logical instants.

There is no universally correct answer. Persist the selected calendar policy and include the canonical UTC logical time in occurrence identity.

### 5.3 Leap seconds and clock corrections

Most application schedulers rely on platform time behavior rather than modeling leap seconds directly. They must still tolerate time moving forward in steps or being smeared. Derive each next occurrence from the prior logical schedule time, not by repeatedly adding a measured duration to “now,” or calendar schedules drift.

---

## 6. Claims, Leases, and Fencing

Executors claim ready work atomically:

```text
UPDATE occurrence
SET state = 'CLAIMED',
    claimed_by = worker,
    claim_epoch = claim_epoch + 1,
    lease_expires_at = authority_now + lease_duration
WHERE occurrence_id = ?
  AND state = 'READY'
RETURNING claim_epoch
```

On expiry, a recovery transaction changes `CLAIMED/RUNNING` back to `READY` and increments or preserves a fencing epoch according to the storage design.

### 6.1 Why leader election is insufficient

A scheduler leader can pause beyond its lease, recover, and continue dispatching. The new leader also dispatches. Consensus elects a current leader; it cannot erase messages already emitted by the stale one.

Downstream state accepts only the current `claim_epoch`, or effects use stable operation identity. See [Leases, Heartbeats, and Recovery](./08-leases-heartbeats-recovery.md) for the ownership protocol.

### 6.2 Heartbeat and lease sizing

Let:

- heartbeat interval be `H`;
- tolerated missed heartbeats be `K`;
- p99.9 stop-the-world/network pause be `P`;
- storage transaction margin be `M`.

A starting lease bound is:

```text
lease_duration >= K * H + P + M
```

Shorter leases recover faster but produce false takeovers. Longer leases delay recovery. For long work, checkpoint progress and renew; do not set a six-hour lease merely because a job may run six hours.

---

## 7. Misfires, Catch-Up, and Overlap

After downtime, a schedule may have thousands of missed occurrences. Declare a policy:

| Policy | Behavior | Suitable for |
|---|---|---|
| Skip | advance to next future occurrence | sampling, cache refresh |
| Fire once now | collapse missed times into one run | reconciliation, “ensure current” tasks |
| Catch up all | materialize each missed occurrence | ledgers, interval-complete pipelines |
| Catch up bounded | materialize last N or last duration | operational maintenance |
| Fail for review | pause and surface ambiguity | high-risk external effects |

Catch-up identity must preserve the original logical fire time so retries and downstream partitions remain stable.

### 7.1 Overlap policies

- **allow:** multiple occurrences run concurrently;
- **forbid:** later occurrence waits or skips while one is active;
- **replace:** request cancellation of prior work, then start new;
- **serialize:** queue all occurrences in logical order;
- **coalesce:** combine several due occurrences into a larger interval.

`replace` is only safe if cancellation is meaningful and effect commit handles races. “Forbid overlap” needs an authority record; querying whether a worker appears active is racy.

### 7.2 Backfill amplification

If 100,000 schedules fire every minute and the control plane is down for 30 minutes, “catch up all” creates:

```text
100,000 * 30 = 3,000,000 occurrences
```

Releasing them instantly produces a recovery storm. Materialize durably, then admit through tenant and resource-class budgets.

---

## 8. Sharding and Hot Partitions

Separate schedule ownership from occurrence execution.

Control-plane shards can own ranges of `hash(tenant_id, schedule_id)`. Each shard uses a lease/epoch stored in a consensus-backed system, periodically checkpoints scan cursors, and is recoverable by another owner.

Execution queues may shard by tenant, resource class, priority, or occurrence ID. The best control-plane distribution is not always the best worker distribution.

### 8.1 Resharding protocol

1. publish routing map revision `R+1`;
2. stop old shard materialization at a recorded cursor;
3. transfer or rebuild near-horizon timer state;
4. acquire new shard epoch;
5. rescan an overlap window;
6. rely on unique occurrence inserts to absorb duplicates;
7. retire old ownership only after observed convergence.

An overlap scan is safer than a gap. Uniqueness turns overlap into extra reads; a gap loses occurrences.

### 8.2 Hot tenants and synchronized schedules

Round times such as midnight and the top of the hour create natural spikes. Mitigations:

- allow a declared jitter/window for tasks that do not require exact wall time;
- shard one large tenant's schedules without losing tenant quotas;
- pre-materialize within a lookahead;
- reserve worker capacity by deadline/priority;
- spread maintenance schedules administratively;
- isolate high-risk or high-volume tenants.

Never silently jitter a financial or legal deadline. Timing flexibility is part of the schedule contract.

---

## 9. Admission, Fairness, and Capacity

Scheduling decides eligibility; admission decides whether execution can begin.

Suppose:

- steady arrival is 12,000 occurrences per second;
- mean service time is 2.5 seconds;
- target utilization is 70 percent;
- workers run 40 concurrent slots each.

Required concurrent slots at target utilization:

```text
12,000 * 2.5 / 0.70 = 42,858 slots
```

Required workers:

```text
ceil(42,858 / 40) = 1,072 workers
```

This mean-based estimate is insufficient when service times are heavy-tailed or resource classes differ. Size CPU, memory, accelerator, network, and downstream concurrency independently; simulate bursts and retries.

### 9.1 Drain-time model

For backlog `B`, arrival rate `lambda`, and service rate `mu` where `mu > lambda`:

```text
drain_time = B / (mu - lambda)
```

If `mu <= lambda`, backlog never drains. Scaling on queue depth alone can lag; include oldest eligible age, arrival/service rates, and startup delay.

### 9.2 Scheduling policy

The execution scheduler should support:

- bounded priority classes with aging;
- weighted or deficit fairness across tenants;
- per-tenant and per-resource concurrency quotas;
- deadline-aware admission where deadlines are trustworthy;
- retry work charged to the originating tenant and effect budget;
- separate capacity for control-plane repair.

The canonical fairness algorithms and overload boundary belong to [Priority, Fairness, and Backpressure](./07-priority-fairness-backpressure.md).

---

## 10. Schedule Evolution and Payload Versioning

Editing a schedule creates a new revision. Define what happens to already materialized occurrences:

- keep them bound to old revision;
- cancel future unclaimed old-revision occurrences;
- regenerate under new revision with an audited mapping;
- let an operator choose at edit time.

Do not mutate occurrence meaning in place. Store handler/code revision and payload schema version so old work remains decodable.

Large payloads should live in immutable object storage or a durable record referenced by digest. Embedding megabytes in queue records increases replication, scan, and retry cost. The reference must remain valid through the maximum execution and repair horizon.

Worker rollouts need compatibility across queued payload versions. Use upcasters or versioned handlers, and verify replay of the oldest retained payload before retiring code.

---

## 11. Multi-Region Design

Common models:

### Home region per schedule

One region materializes and normally executes each schedule. Failover transfers shard epoch and rescans an overlap. Simple uniqueness; higher latency for remote effects.

### Global materialization, regional execution

A globally consistent ledger creates occurrences; regional queues execute according to data locality. Strong control plane, potentially expensive/latent writes.

### Disjoint regional ownership

Schedules belong permanently to regions. Simple and scalable, but global tenants need explicit ownership and failover mapping.

Avoid active-active materialization from eventually consistent copies without a uniqueness authority. Two regions can both decide the same occurrence is due.

During region failover:

1. fence old shard ownership;
2. load durable schedule cursors;
3. rebuild the near-horizon index;
4. rescan an overlap window;
5. insert occurrences idempotently;
6. release work through recovery admission budgets.

Recovery point for schedules includes the ledger, timer state, payload references, occurrence states, and effect outcomes—not merely cron definitions.

---

## 12. Security and Multi-Tenant Isolation

Scheduling is authority to execute future code. Protect:

- schedule creation, update, pause, and manual fire with scoped authorization;
- handler allowlists and versioned payload schemas;
- tenant-bound queue and storage keys;
- per-tenant quotas for schedule count, fire rate, payload size, and concurrency;
- signed internal dispatch envelopes;
- workload identity for workers;
- secrets resolved at execution time, not copied into durable payloads;
- audit history for mutations and manual repair;
- retention and deletion propagation to occurrences, payloads, logs, and dead letters.

Cron expressions, time zones, and payload templates are untrusted input. Bound expansion: reject expressions that exceed permitted fire rate, cyclic calendar rules, oversized target lists, or payloads that cause uncontrolled fan-out.

A tenant must not infer another tenant's schedule timing from shared identifiers or retrieve its results through predictable occurrence IDs. Authorize before lookup and include tenant scope in identities.

---

## 13. Failure Traces

### 13.1 Cursor advances before occurrence commit

1. Scheduler computes the 10:00 occurrence.
2. It updates `next_fire_at` to 11:00.
3. Process crashes before inserting the occurrence.
4. 10:00 is never discovered again.

**Prevention:** insert occurrence and advance cursor in one transaction.

### 13.2 Notification lost after commit

1. Occurrence commits.
2. Queue notification fails.
3. No worker receives the hint.

**Prevention:** executors scan durable ready state or an outbox republishes; notification is not authority.

### 13.3 Failover duplicates a tick

1. Old leader pauses after dispatch.
2. Lease transfers and new leader rescans overlap.
3. Both materialize the same logical time.

**Prevention:** unique occurrence identity absorbs duplicate materialization; effect identity absorbs duplicate dispatch.

### 13.4 DST produces a double payroll

1. “01:30 local” occurs twice at fall-back.
2. Scheduler keys by formatted local string without offset/fold.
3. It either collides unpredictably or runs twice unintentionally.

**Prevention:** explicit ambiguous-time policy and canonical UTC logical instant.

### 13.5 Misfire storm overloads dependency

1. Scheduler is unavailable for an hour.
2. Every missed occurrence becomes immediately ready.
3. Worker autoscaling floods a database with catch-up traffic.
4. Recovery causes a second outage.

**Prevention:** durable catch-up plus admission budgets, fairness, and dependency-specific concurrency caps.

### 13.6 Stale worker commits after lease transfer

1. Worker A pauses beyond lease.
2. Worker B claims with epoch 18 and succeeds.
3. Worker A resumes with epoch 17 and writes a result.

**Prevention:** downstream fence on epoch or stable idempotent effect key; monotonic terminal state.

### 13.7 Schedule edit rewrites history

1. Payload template changes in place.
2. A queued old occurrence dereferences the mutable template.
3. It runs with new parameters but old occurrence identity.

**Prevention:** bind occurrence to immutable schedule revision and payload digest.

---

## 14. Observability and Repair

Control-plane signals:

- schedules scanned and materialized by shard/revision;
- scan cursor lag and lookahead coverage;
- materialization transaction conflicts;
- duplicate occurrence insert rate;
- timer-bucket size and cascade latency;
- shard lease age, failover, and stale-owner rejection;
- invalid schedule and expansion-limit rejection.

Execution-plane signals:

- ready/claimed/running/terminal counts;
- oldest eligible age and deadline misses;
- arrival, service, retry, and completion rates;
- lease expiry and reclaim rate;
- attempts per occurrence;
- concurrency and resource saturation by class/tenant;
- misfire/catch-up volume and drain-time estimate;
- dead-letter and repair backlog age.

Repair APIs should support:

- recompute and preview occurrences for a time range;
- rescan a shard overlap window;
- materialize a missing occurrence idempotently;
- cancel/requeue with expected phase revision;
- move a corrupt item to quarantine;
- rebuild the near-horizon timer index;
- compare authoritative occurrence state with queue/index projections.

Every repair emits an audit event and uses the same transition rules as normal execution.

---

## 15. Verification

1. **Calendar vectors:** DST gaps/folds, leap days, month ends, time-zone rule updates.
2. **State-machine properties:** terminal monotonicity, unique occurrence, legal phase transitions.
3. **Crash-point tests:** before/after occurrence insert, cursor advance, notification, claim, outcome, ack.
4. **Clock fault tests:** skew, jumps, smear, monotonic/wall divergence.
5. **Failover tests:** paused old leader, overlapping scan, routing-map update, regional evacuation.
6. **Load tests:** synchronized boundaries, hot tenants, millions of long timers, heavy-tailed job duration.
7. **Misfire tests:** each policy after controlled downtime, with admission limits.
8. **Schema tests:** oldest retained payload on newest worker and mixed-version fleets.
9. **Tenant tests:** quotas, cross-tenant lookup, fair-share under one tenant flood.
10. **Model-based simulation:** random schedules, crashes, leases, retries, edits, and repairs checked against invariants.

Use a virtual clock in deterministic simulation. Wall-clock integration tests alone are slow and miss rare interleavings.

---

## 16. Decision Framework

Use operating-system cron when one host, best-effort execution, and manual recovery are acceptable. Use a database-backed scheduler when volume is moderate, transactional materialization is valuable, and the database can sustain indexed due-time scans. Use a broker's delayed delivery when delays fit its retention/resolution contract and recurring-calendar semantics are minimal. Use a durable workflow engine when timers are part of a long-running state machine with replay, cancellation, and effect recovery.

Before designing a scheduler, answer:

1. What is the unique identity of a logical occurrence?
2. Which time semantics apply: UTC instant, elapsed delay, or named-zone calendar?
3. What happens to missed and overlapping occurrences?
4. How many timers exist, how often do they mutate, and how synchronized are deadlines?
5. Where is occurrence materialization committed?
6. How are stale schedulers and workers fenced?
7. Can accepted backlog finish before its deadlines?
8. How are tenant fairness and downstream limits enforced?
9. Which schedule and payload revisions must old work retain?
10. How is state reconstructed after regional loss?
11. What repair operations exist for a missing, duplicated, or stuck occurrence?

The smallest scheduler is usually the best one, but “small” still requires durable identity and explicit failure semantics. Adding replicas without those properties only duplicates the uncertainty.

---

## Primary References

- [Varghese and Lauck: Hashed and Hierarchical Timing Wheels](https://dl.acm.org/doi/10.1145/41457.37504)
- [Brown: Calendar Queues](https://dl.acm.org/doi/10.1145/63039.63045)
- [RFC 5545: Internet Calendaring and Scheduling Core Object Specification](https://www.rfc-editor.org/rfc/rfc5545)
- [Kubernetes: CronJob](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)
- [Temporal Documentation: Timers](https://docs.temporal.io/develop/java/timers)
- [Google Research: Large-scale Cluster Management at Google with Borg](https://research.google/pubs/large-scale-cluster-management-at-google-with-borg/)

---

## Related Chapters

- [Background Jobs and Worker Pools](./02-background-jobs-worker-pools.md)
- [Effect Commit Protocols for Workflows](./06-retry-idempotency-compensation.md)
- [Priority, Fairness, and Backpressure](./07-priority-fairness-backpressure.md)
- [Leases, Heartbeats, and Recovery](./08-leases-heartbeats-recovery.md)
- [Durable Execution and Workflow Engines](./04-durable-execution-workflow-engines.md)
