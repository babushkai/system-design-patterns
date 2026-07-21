# Leases, Heartbeats, and Recovery

## Job-Attempt Authority Protocol

A lease grants one attempt temporary authority; an epoch names that grant; recovery advances authority after expiry or revocation. Stale-attempt fencing—not the timeout itself—prevents a replaced attempt from committing state or effects.

The job-attempt protocol covers claim epochs, renewal, separate liveness and progress signals, deadlines, reclaim, checkpoint resume, fencing, shutdown, and reconciliation. [Distributed Locks](../01-foundations/09-distributed-locks.md) covers generic lock acquisition; [Failure Modes](../01-foundations/06-failure-modes.md) covers failure detection; [Retry, Idempotency, and Compensation](./06-retry-idempotency-compensation.md) covers end-to-end effect protocols.

Do not assume every ownership mechanism has these semantics. A database lease, an SQS visibility timeout, a RabbitMQ delivery acknowledgement, an etcd lease, and a ZooKeeper session/version are different contracts. A visibility timeout makes a message eligible for redelivery; it does not fence the first consumer. A RabbitMQ acknowledgement is scoped to its delivery and channel. An etcd lease TTL and an MVCC revision are distinct values. A ZooKeeper version or sequence number is useful only within the comparison protocol built around it. Translate the invariant, not the product name.

---

## Start With the Recovery Contract

Choose the contract before choosing a heartbeat interval:

| Question | Example answer |
|---|---|
| When may another attempt start? | After grantor time passes the stored lease expiry, or after explicit revocation |
| What overlap is acceptable? | Computation may overlap; committed job state and protected effects may not |
| What is the recovery objective? | Reassign within 45 seconds of loss of renewal |
| What is the work deadline? | Finish by 02:00 UTC, independent of worker liveness |
| Can work resume? | Only from a checkpoint compatible with the job input and current code |
| What if an external effect is ambiguous? | Stop automatic completion and reconcile by logical effect ID |

“At most one worker is running” is usually neither observable nor enforceable: a paused or partitioned process can keep executing. A useful guarantee is narrower and stronger:

> At any authority version, only the current grant epoch may commit job state; after supersession, the old epoch cannot commit. Every effect boundary either rejects stale epochs or deduplicates and reconciles a stable logical effect.

That guarantee permits redundant CPU work while protecting durable outcomes.

---

## Durable State and Invariants

Keep logical job state separate from physical attempt state. A compact schema is:

~~~text
Job
  job_id, tenant_id, state
  input_digest, definition_version
  current_epoch, current_grant_id, current_attempt_id, owner_session
  lease_until_authority_time
  desired_state, absolute_deadline
  retry_budget_used
  checkpoint_ref, checkpoint_schema, checkpoint_input_digest
  completion_ref, version

Attempt
  attempt_id, job_id, epoch, grant_id, worker_session
  state, claimed_at_authority_time
  last_liveness_seq, last_liveness_at
  last_progress_seq, progress_value, last_progress_at
  checkpoint_ref, outcome, finished_at

EffectAuthority
  protected_resource_id, job_id
  accepted_epoch, accepted_grant_id, version

EffectReceipt
  job_id, logical_effect_id, epoch, grant_id
  request_digest, outcome, provider_reference
~~~

Use an unguessable grant_id as well as a monotonic epoch. The epoch orders grants; the grant ID prevents an implementation bug or restored database from confusing two grants that happen to carry the same number. worker_session distinguishes a restarted process from its predecessor on the same host.

The state machine should enforce these invariants:

1. **Every authority transition advances the epoch.** An epoch never returns to a previous value for the same job.
2. **Attempt mutation is conditional.** Renewal, progress, checkpoint publication, completion, and relinquish require the exact tuple (job_id, epoch, grant_id, attempt_id, worker_session).
3. **A renewal cannot resurrect a superseded grant.** It may extend only the current non-terminal grant.
4. **Job terminal states are monotonic.** Only a separately authorized, audited repair operation may create a successor run or change the interpretation of a bad terminal result.
5. **Liveness, progress, and deadline are independent facts.** None is inferred from another.
6. **A checkpoint pointer becomes visible only after its artifact is durable**, and it is bound to the input digest, definition version, schema, and producer.
7. **A stale epoch cannot commit protected effects.** Where the recipient cannot enforce an epoch, the logical effect has a stable idempotency key and an explicit reconciliation path.
8. **Reclaim is idempotent.** Concurrent controllers consume retry budget and change job state at most once.

These invariants belong in storage constraints and conditional updates, not only in worker code.

---

## Claim and Renewal Protocol

### Claim

The authority service claims a ready job in one transaction:

1. Read a candidate whose state is READY or RETRYABLE.
2. Check admission, cancellation, deadline, and retry policy.
3. Increment current_epoch; generate a fresh grant, attempt, and worker-session binding.
4. Set lease_until_authority_time to authority_now plus lease_duration.
5. Insert the attempt and change the job to RUNNING.
6. Return the exact grant tuple, lease duration, job specification, and compatible checkpoint candidate.

The database or consensus service that serializes the grant supplies **authoritative time**. A worker-provided timestamp must not extend a lease. A worker can use its local monotonic clock to schedule an early renewal, but it cannot prove from its own wall clock that the grantor still considers the lease valid.

~~~sql
UPDATE jobs
   SET state = 'RUNNING',
       current_epoch = current_epoch + 1,
       current_grant_id = :grant_id,
       current_attempt_id = :attempt_id,
       owner_session = :worker_session,
       lease_until = authority_now() + :lease_duration,
       version = version + 1
 WHERE job_id = :job_id
   AND version = :observed_version
   AND state IN ('READY', 'RETRYABLE');
~~~

This sketch assumes authority_now() is evaluated by the serialized authority. If a separate scheduler and store disagree about time, define which one is authoritative and make the transition atomic with that decision.

### Renew

A renewal carries the exact grant tuple and a monotonically increasing liveness sequence. The authority extends the lease only if the job is still RUNNING, every identity field matches, the liveness sequence is newer, and policy has not requested cancellation or revocation.

~~~sql
UPDATE jobs
   SET lease_until = authority_now() + :lease_duration,
       version = version + 1
 WHERE job_id = :job_id
   AND state = 'RUNNING'
   AND current_epoch = :epoch
   AND current_grant_id = :grant_id
   AND current_attempt_id = :attempt_id
   AND owner_session = :worker_session;
~~~

The corresponding attempt-row update records the accepted liveness sequence. Duplicate renewal requests are harmless; an old sequence does not move diagnostic state backward.

A successful write followed by a lost response is deliberately awkward. The grant may have been renewed, but the worker cannot know. It should retry within a bounded uncertainty window using the same grant tuple. If it cannot obtain an authoritative success before its conservative self-deadline, it **self-demotes**: stop starting effects, stop publishing progress, attempt only safe cancellation, and let recovery decide. Continuing because “the renewal probably worked” turns an availability ambiguity into a correctness violation.

Lease duration is a policy, not a proof. Size it from measured scheduling pauses and authority-store tail latency. A practical budget is:

$$
D_{lease} > h + L_{schedule,p99.9} + L_{renew,p99.9} + U_{clock} + M
$$

where $h$ is the renewal interval, $U_{clock}$ covers the worker’s conservative scheduling uncertainty, and $M$ is a safety margin. This is an engineering bound under stated observations, not a universal theorem. Alert when the observed tail consumes the margin.

---

## Three Signals, Three Decisions

One “heartbeat timestamp” cannot answer all operational questions.

| Signal | Meaning | Typical field | Decision it supports |
|---|---|---|---|
| Liveness | The attempt can still reach the authority loop | liveness sequence and accepted time | Keep or expire the grant |
| Progress | Durable application work advanced | records committed, cursor, phase, bytes | Diagnose a wedge or estimate completion |
| Deadline | The contract’s time budget remains | absolute job or step deadline | Cancel, fail, escalate, or compensate |

A dedicated heartbeat thread can remain healthy while application threads deadlock. That attempt is live but not progressing. A worker can commit output just before a network partition and then fail to report liveness; the durable output progressed even though the lease will expire. A worker may be live and progressing after a customer deadline; renewing it is not permission to violate the deadline.

Progress must describe a durable boundary. Reporting “row 900 processed” before row 900 and its checkpoint are committed produces a false resume point. Prefer a monotonic domain cursor tied to a committed artifact or transaction. For work whose progress cannot be monotonic, report a phase plus evidence, not a made-up percentage.

Progress-stall policy is state-specific. Ten minutes without progress may be a fault for a CPU transform, normal for a workflow waiting on a callback, or ambiguous for one blocking third-party request. [Workflow Observability and Replay](./09-workflow-observability-replay.md) owns population views and stuck-state diagnosis.

---

## Reclaim, Reassign, and Resume

A recovery controller queries an index on (state, lease_until) using grantor-authoritative time. For each expired candidate it executes a compare-and-swap against the **observed grant**, not merely lease_until less than now:

~~~text
RUNNING(epoch=e, grant=g, attempt=a)
  -- CAS exact e/g/a and expired at authority time -->
EXPIRED(epoch=e+1, no current owner)
~~~

Advancing the epoch during revocation immediately makes the old grant stale in the job store. A later claim may advance it again. Epochs need to be ordered, not contiguous. The transaction marks the old attempt EXPIRED, records a reason, clears ownership, and consumes retry budget once. Multiple reclaimers racing on the same row leave one winner.

Reassignment is a scheduling decision, not part of expiry itself. Apply retry backoff, tenant admission, priority, and dependency-health checks before returning the job to READY; otherwise a storage outage can create a reclaim storm that overloads the same storage needed for renewal. [Priority, Fairness, and Backpressure](./07-priority-fairness-backpressure.md) owns that admission policy.

### Checkpoint publication

For an object-store checkpoint:

1. Write an immutable artifact under a content-addressed or unique key.
2. Verify its checksum and durability policy.
3. Conditionally publish its pointer on the current attempt row.
4. Optionally advance the job’s resume pointer in the same job-store transaction.

The manifest should include at least:

~~~json
{
  "schema_version": 4,
  "definition_version": "invoice-export/7",
  "input_digest": "sha256:...",
  "logical_cursor": {"partition": 12, "last_committed_id": 88421},
  "committed_outputs": ["sha256:..."],
  "producer_build": "git:9b27...",
  "checksum": "sha256:..."
}
~~~

The successor validates tenant, job ID, input digest, schema, definition compatibility, artifact checksum, and authorization before loading. If compatibility is absent, either run a tested deterministic migration or restart from a known-safe boundary. Never deserialize a checkpoint merely because it is the newest object. Treat it as untrusted input; apply size limits and safe parsers.

### Completion

Completion is another conditional transition on the exact grant tuple. The attempt first ensures every required effect has a durable receipt or an explicitly reconciled outcome, then tries RUNNING to SUCCEEDED. Zero rows updated means the attempt is stale; it must not “fix” the row with an unconditional write.

---

## Fencing Effects Correctly

Job-store CAS protects job metadata. It does not protect a file server, payment API, email provider, or database reached by the worker.

At a cooperative effect gateway, store an accepted (epoch, grant_id) pair for the logical protected resource, separately from per-operation receipts. Each operation carries (job_id, logical_effect_id, epoch, grant_id, request_digest). In one transaction the gateway:

1. rejects epoch lower than accepted_epoch;
2. if epoch is higher than accepted_epoch, validates the grant and advances the accepted epoch/grant pair;
3. permits **equal** epoch and accepted_epoch only when grant_id also matches, so one authorized attempt can make multiple operations;
4. deduplicates logical_effect_id and rejects a digest mismatch;
5. records the outcome before acknowledging.

The common predicate “stored fence is lower than token” is insufficient as the write predicate for ordinary work: after the first write at epoch 42, it rejects the same authorized attempt’s second write at epoch 42. The accepted authority epoch and individual effect receipts are separate state.

There are two revocation strengths:

- **Successor-activated fencing:** an old epoch is rejected after a higher epoch first reaches the effect boundary. This is the usual high-water-mark contract.
- **Eager fencing:** the authority advances the boundary’s minimum accepted epoch as part of revoke or handoff, before enabling the successor. This closes the interval in which a revoked attempt could act before its successor’s first effect, but it requires a transactional gateway or a carefully reconciled cross-system protocol.

A monotonically increasing number attached only to requests is not a fence. The recipient must compare it atomically with the effect. Third-party APIs commonly cannot do this. There, use a stable logical effect ID independent of attempt number, an outbox or effect broker when possible, provider-side idempotency if its retention window is sufficient, and reconciliation by provider reference. Do not claim exactly-once behavior from a lease.

---

## Graceful Shutdown Without Premature Release

On deploy or scale-in, a worker enters DRAINING:

1. Stop accepting new claims.
2. Continue renewing active grants while work remains authorized.
3. Ask application work to stop at an effect-safe or checkpoint-safe boundary.
4. Wait for all local effect threads and buffered writes to finish or become durably handed off.
5. Publish a compatible checkpoint.
6. Conditionally relinquish the exact grant, advancing authority or returning the job to retry policy.
7. Revoke per-attempt credentials and exit.

Never release the grant while a background thread can still issue effects. If the drain deadline arrives before a safe boundary, stop renewing and exit; let expiry and reconciliation handle the uncertain attempt. Shutdown improves recovery latency, but correctness must still survive SIGKILL, host loss, and network isolation at every step.

---

## Reconciliation Is Part of Recovery

Recovery cannot infer an external result from worker death. Consider: the provider accepted a charge, the response was lost, the worker died, and the lease expired. Retrying blindly may duplicate the charge; marking success blindly may hide a rejection.

A reconciler consumes logical effect IDs with unknown outcomes, queries the authoritative provider when possible, compares request digests and provider references, and writes a durable receipt. It then conditionally advances the job or routes the case to manual review. Reconciliation commands carry actor, reason, ticket, idempotency key, and expected job version. The audit record must distinguish observed provider fact from operator assertion.

Also reconcile impossible internal combinations: a terminal attempt with a running job, a checkpoint pointer whose artifact is absent, a current grant with no attempt, or retry budget counted twice. Prefer repair by normal conditional transitions. Direct row editing destroys the evidence needed to understand the incident.

---

## Capacity and Recovery Budget

State assumptions before applying formulas.

Let:

- $A$ = concurrently leased attempts;
- $h$ = heartbeat interval in seconds;
- $b$ = mean renewals combined per authority write, with $b=1$ when unbatched;
- $q$ = checkpoints per active attempt per second;
- $S$ = mean encoded checkpoint bytes;
- $E$ = expired-attempt backlog;
- $\lambda$ = newly expiring attempts per second;
- $\mu$ = reclaim-controller service rate per second.

Steady renewal write rate is approximately:

$$
W_{renew} = \frac{A}{h b}
$$

Batching reduces write count but enlarges failure domains and may increase tail latency. Checkpoint ingress is:

$$
B_{checkpoint} = A q S
$$

excluding replication, object metadata, and abandoned artifacts. If $\mu > \lambda$, an existing expiry backlog drains in roughly:

$$
T_{drain} = \frac{E}{\mu - \lambda}
$$

If $\mu \le \lambda$, it never drains in steady state. Provision the authority store for renewal peaks plus claims, completions, progress, and reclaim—then test it under the correlated pause or outage that causes many leases to expire together. Randomly jitter renewals so a deploy does not synchronize the fleet.

---

## Specialized Failure Traces

### Heartbeat thread hides a deadlock

The renewal loop runs in a dedicated thread while all work threads wait on a lock. Leases remain fresh forever. Detection requires state-specific progress evidence and a progress deadline; reclaiming solely on liveness would never fire.

### A progress cursor outruns durability

An attempt reports offset 50,000, then its output buffer is lost. The successor resumes at 50,001 and silently skips data. Publish progress only with the committed output/checkpoint boundary, or make the sink able to prove which offsets are durable.

### Lost renewal response

The authority commits a renewal, but the response disappears. The worker continues past its conservative uncertainty window and issues an effect while recovery has already revoked it. Correct behavior is retrying the same renewal, then self-demoting without authoritative success; the effect boundary still fences the race.

### A zombie completes over its successor

Attempt 8 pauses, attempt 9 completes, then attempt 8 unconditionally marks the job failed. Exact epoch/grant CAS rejects attempt 8. Terminal monotonicity prevents a cleanup path from overwriting success.

### The fence never reaches the resource

The authority grants epoch 19, but the database writer accepts any authenticated call and never compares epochs. Epoch 18 can still write. Fix the effect boundary or admit that the resource is protected only by logical idempotency and reconciliation.

### A new build loads an old checkpoint

The serialized cursor changed meaning between definition versions. The new worker loads it successfully but omits a partition. A compatibility matrix, input digest, schema version, and replay/migration test reject or transform the checkpoint before work starts.

### Drain releases before work stops

The worker relinquishes, a successor starts, and an old asynchronous upload finishes. The drain protocol must join or durably transfer effect work before relinquish; fencing/idempotency remains the crash-safe fallback.

### Reclaim causes a renewal collapse

A store latency spike expires thousands of leases. Unbounded reclaim immediately creates claims and checkpoints, further loading the store and expiring healthy work. Rate-limit reclaim separately, reserve authority capacity for renewals, and shed nonessential progress writes.

---

## Security Boundaries

- Bind each grant to a tenant and worker identity; authorize every renewal, checkpoint, completion, and effect against that binding.
- Use short-lived, least-privilege attempt credentials. Revocation narrows exposure, but fencing—not credential propagation speed—must protect correctness.
- Make epoch allocation authority-controlled. A worker must not choose a larger token to impersonate a successor.
- Encrypt checkpoint and effect payloads, validate manifests and content types, and prevent cross-tenant object references.
- Separate repair permission from ordinary worker permission. Require strong authentication, expected-version checks, reason codes, and immutable audit entries.
- Avoid placing secrets or raw regulated payloads in heartbeat/progress fields; those fields are broadly indexed and retained.

---

## Verification Strategy

Model the job/attempt state machine and check the invariants under reordered renewals, duplicate claims, concurrent reclaimers, and delayed completion. Then exercise real boundaries:

1. Kill the process before and after claim commit, checkpoint upload, pointer publication, effect acknowledgement, and completion commit.
2. Pause application threads while renewal continues, then pause the entire process beyond the lease.
3. Drop renewal responses after the authority commits.
4. Skew wall clocks and verify only grantor time decides expiry.
5. Run old and new attempts concurrently against the effect gateway; include multiple valid writes from the same epoch.
6. Exhaust the authority store and object store independently; verify renewal capacity is reserved and reclaim is rate-limited.
7. Replay checkpoint compatibility across every supported definition/schema pair; corrupt, truncate, oversize, and cross-tenant the artifact.
8. Terminate workers throughout drain and prove no released grant retains an active effect path.
9. Reconcile provider outcomes for success, rejection, timeout-after-accept, duplicate ID, and mismatched digest.

Record recovery time from last accepted renewal to a successor’s authorized start, not merely scanner latency. Track stale-CAS rejections, effect-fence rejections, uncertain effects, progress-stall transitions, reclaim backlog, and checkpoint-validation failures. Generic metric, log, trace, and alert pipeline design belongs to the [observability section](../11-observability/02-metrics-monitoring.md).

---

## Design Decisions to Record

Before production, write down:

- the grantor and its authoritative clock;
- lease duration, renewal cadence, uncertainty window, and measured tail assumptions;
- exact identity tuple checked by every mutation;
- progress semantics for each job type;
- deadline and cancellation precedence;
- effect resources that enforce epochs, those that only deduplicate, and their reconciliation owners;
- checkpoint compatibility and migration policy;
- reclaim rate limits and renewal-capacity reservation;
- drain deadline and safe boundary;
- audited commands allowed to repair or create successor work.

If any protected effect has neither an enforceable epoch nor a stable deduplication/reconciliation protocol, recovery is knowingly at-least-once at that boundary.

---

## Related Patterns

- [Background Jobs and Worker Pools](./02-background-jobs-worker-pools.md)
- [Retry, Idempotency, and Compensation](./06-retry-idempotency-compensation.md)
- [Priority, Fairness, and Backpressure](./07-priority-fairness-backpressure.md)
- [Workflow Observability and Replay](./09-workflow-observability-replay.md)
- [Distributed Locks](../01-foundations/09-distributed-locks.md)
- [Failure Modes](../01-foundations/06-failure-modes.md)
- [Distributed Time](../01-foundations/05-distributed-time.md)
- [Leader Election](../02-distributed-databases/09-leader-election.md)
- [Delivery Guarantees](../05-messaging/04-delivery-guarantees.md)

---

## Primary Sources

1. [Leases: An Efficient Fault-Tolerant Mechanism for Distributed File Cache Consistency](https://doi.org/10.1145/74850.74870) — Gray and Cheriton, SOSP 1989.
2. [The Chubby Lock Service for Loosely-Coupled Distributed Systems](https://www.usenix.org/legacy/events/osdi06/tech/full_papers/burrows/burrows.pdf) — Burrows, OSDI 2006; sequencers illustrate effect-side stale-owner rejection.
3. [ZooKeeper: Wait-free Coordination for Internet-scale Systems](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Hunt.pdf) — Hunt et al., USENIX ATC 2010; sessions, versions, and ordered coordination state.
4. [etcd API guarantees](https://etcd.io/docs/latest/learning/api_guarantees/) and [etcd lease API](https://etcd.io/docs/latest/learning/api/#lease-api) — official guarantees distinguish revisions, transactions, and lease TTL.
5. [Temporal Java SDK ActivityExecutionContext](https://javadoc.io/static/io.temporal/temporal-sdk/1.32.0/io/temporal/activity/ActivityExecutionContext.html) — official SDK contract for activity heartbeat details and retry resume data.
