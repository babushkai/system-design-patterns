# Poison-Message Quarantine and Redrive

A poison message is a delivery that cannot make progress under the current code, data, policy, or dependency state. Quarantine removes it from the hot delivery path while preserving enough evidence and identity to diagnose, repair, and replay it safely. A dead-letter queue is therefore an incident and repair subsystem—not a place to hide exhausted generic retries.

Scope: failure classification, quarantine state, evidence, bulk isolation, repair, and controlled redrive. Retry timing/idempotency/compensation belong to [Workflow Effect Protocols](../18-workflow-job-systems/06-retry-idempotency-compensation.md). Queue claims are in [Message Queue Architecture](01-message-queues.md), and end-to-end duplicate effects are in [Delivery Guarantees](04-delivery-guarantees.md).

## Workload and contract

Quarantine operations should be explicit:

```text
quarantine(delivery, classification, evidence, policy_version)
inspect(quarantine_id)
assign(quarantine_id, owner)
repair(quarantine_id, repair_artifact)
redrive(selection, destination, transform_version, rate_budget, dry_run)
resolve(quarantine_id, outcome)
expire(quarantine_id, retention_policy)
```

Define:

- which failures are retryable, permanent, ambiguous, or policy-blocked;
- who classifies them and with which policy version;
- whether quarantine advances the main subscription position;
- evidence captured and sensitive-data handling;
- retention/legal hold and payload availability;
- repair and transformation authority;
- redrive identity, ordering, destination, rate, and duplicate-effect safeguards;
- ownership/SLO for oldest untriaged and unresolved items;
- behavior during systemic failure when millions of messages fail together.

Moving a record to quarantine is a state transition with durable evidence. If the broker cannot atomically record quarantine and advance source progress, the operation may duplicate; stable identities make it reconcilable.

## State and invariants

A quarantine record includes:

```text
quarantine_id
original_event_id and source topic/partition/position
original schema/type/tenant and payload digest/reference
consumer/service/release/policy versions
first/last failure time and attempt summary
normalized failure class/code and redacted diagnostic reference
claim/order/entity versions at failure
status, owner, retention/hold, repair/redrive lineage
```

Enforce:

**Original identity is preserved.** Redrive does not silently create a new logical business intent. If a new event is deliberately created, it records causal lineage and why dedup semantics differ.

**Evidence is immutable.** Triage notes and repair actions append to history; original payload digest, failure class, and source position are not overwritten.

**Quarantine is isolated from hot delivery.** A poison item cannot immediately cycle back and consume workers without an explicit redrive decision.

**Source progress is unambiguous.** The main subscription records that the source item was quarantined, not successfully applied. Reconciliation can distinguish the two.

**Repair is reproducible.** Transform/code/config/schema artifact versions are recorded; “edited JSON manually” is not a production repair protocol.

**Redrive is bounded and reversible.** Selection, destination, rate, idempotency horizon, stop conditions, and outcomes are known before execution.

**Retention preserves dependencies.** A quarantine envelope is not useful if its referenced payload, schema, encryption key, or source context expires first.

## Failure classification

Classify before deciding to retry or quarantine:

| Class | Examples | Normal action |
|---|---|---|
| transient dependency | timeout, rate limit, leader failover | bounded retry/backoff; do not quarantine every item |
| deterministic payload/schema | malformed field, unknown required version | quarantine with schema evidence |
| deterministic domain conflict | impossible transition, stale prerequisite | quarantine or resolve through domain repair |
| authorization/policy | revoked tenant, prohibited content, residency violation | fail closed; restricted quarantine/review |
| consumer defect | release crashes on valid record | halt/circuit-break release or isolate cohort |
| ambiguous outcome | external request timed out after possible commit | reconcile status before retry/redrive |
| resource/pathological | decompression bomb, huge expansion, hot key | isolate and enforce budgets |

Attempt count alone is a weak classifier. Ten identical schema errors are poison; ten timeouts during a known regional outage may be normal transient failure. Conversely, one invalid signature should fail permanently without repeated work.

Classification is a versioned decision function over typed error code, retryability declaration, dependency health, attempt history, message/schema version, and policy. Do not parse arbitrary exception strings as the primary contract.

Consumers return structured failure outcomes. The queue/quarantine service applies central safety limits: maximum delivery age, attempts, cumulative execution, lease extensions, and unknown-error budget. Application teams can narrow but not bypass platform bounds.

## Data plane and control plane

The **data plane** receives structured failures, persists quarantine records/payload references, advances or records source disposition, serves bounded inspection, and executes rate-limited redrive. It supports high-volume batch quarantine without calling a human-review service per message.

The **control plane** owns classification policy, retention, access, ownership/routing, repair transform registry, redrive approvals, destinations, quotas, legal holds, and dashboards. Emergency policy changes are versioned, audited, canaried, and rollbackable.

Separate quarantine storage from the source queue so poison evidence survives source retention/purge and cannot block its partitions. Keep a source-position index for reconciliation and an entity/type/error index for triage. Avoid arbitrary high-cardinality indexes on raw payload fields.

Human triage UI reads redacted metadata by default; privileged payload access is just-in-time and audited. Bulk APIs operate on stable query snapshots/manifests so a changing filter does not cause an unrepeatable redrive set.

## Quarantine transition

For a broker with native dead-lettering, configure a dead-letter destination plus maximum delivery/classification policy. Still verify what is preserved: original topic, position, headers, delivery attempts, and payload. Broker-native redrive may assign a new transport position and may not retain source identity unless included in the envelope.

For an application-managed consumer, a safe local-database path is:

1. begin the same transaction used for inbox/progress;
2. insert quarantine disposition keyed by `(consumer, event_id)` with payload reference/evidence;
3. update entity/projection status if the product exposes blocked work;
4. advance the contiguous source checkpoint only under policy that permits skipping into quarantine;
5. insert an outbox record for the quarantine store if it is external;
6. commit, then acknowledge source.

If quarantine storage is remote, first persist a local durable disposition/outbox; calling remote quarantine then checkpointing source is a dual write. A crash can either lose evidence or duplicate records.

Strict ordered streams need special handling. Quarantining sequence 18 and applying 19 may violate the domain even though queue throughput recovers. Policies include:

- block only that entity/key while unrelated keys continue;
- quarantine the blocked suffix together;
- rebuild/repair 18 before proceeding;
- explicitly skip 18 only if domain semantics allow it and record the gap.

Moving poison out of the queue does not remove ordering invariants.

## Systemic failures and bulk isolation

A bad deployment or schema rollout can make most messages fail. Individually retrying and quarantining them floods storage, alerts, and operator interfaces. Detect correlated failure by normalized fingerprint, release, schema, partition, and time window.

When a failure cohort crosses a learned/declared threshold:

1. open a circuit for the affected consumer release/message class;
2. pause or slow source delivery for that cohort;
3. store one incident/fingerprint record plus compact references to affected source ranges where safe;
4. preserve source retention/checkpoints;
5. roll back or repair the consumer;
6. replay from source under a controlled generation rather than materializing millions of duplicate payloads.

Range quarantine is safe only when the source remains retained and immutable. If source retention can expire, copy payloads or extend retention. The incident manifest lists exact partitions/positions and checksums.

Protect healthy tenants/types through bulkheads. A global circuit should be the last resort; isolate by consumer, release, schema, tenant, and partition without letting unbounded label cardinality exhaust the controller.

## Triage and repair workflow

Quarantine states can be:

```text
NEW -> TRIAGED -> REPAIR_READY -> REDRIVING -> RESOLVED
  \-> POLICY_HOLD
  \-> DISCARDED (with authority and evidence)
  \-> REDRIVE_FAILED -> TRIAGED
```

Triage groups records by stable error fingerprint and finds first-seen release/schema plus representative samples. The owner determines whether the fix is consumer code, schema adapter, source correction, dependency reconciliation, policy exception, or intentional discard.

Prefer fixing code/config and replaying the original immutable event. If payload transformation is necessary, register a pure versioned transform that takes original envelope and returns repaired envelope plus validation report. Keep original bytes/digest and transformation lineage. Validate against schema, domain invariants, tenant policy, and idempotency key behavior.

Source-data correction often should produce a new domain event rather than mutate the failed event. Link the corrective event to the quarantined identity and resolve only after downstream reconciliation proves the intended state.

Ambiguous external effects enter reconciliation, not automatic redrive. Query the downstream status using business/idempotency reference. Mark applied if it committed, retry with the same key if absent, or escalate if unknown cannot be resolved.

## Controlled redrive

Redrive is a production deployment. Its manifest contains:

- immutable selection snapshot/count/bytes/error classes;
- source and destination;
- original/repaired schema and transform artifact digest;
- identity/idempotency policy;
- ordering/entity grouping;
- rate/concurrency/downstream budget;
- canary subset and comparison;
- stop thresholds for failure, latency, or duplicates;
- operator/approver and audit ticket;
- rollback/abort behavior.

Dry run decodes and validates without effects, estimating destination partitions, schema outcomes, and downstream work. Canary a deterministic small cohort, verify effects/reconciliation, then ramp. Redrive traffic has a separate priority/budget from live traffic so old poison cannot starve new work.

Preserve original `event_id` when the logical intent is unchanged; inbox duplicates then safely detect already-applied effects. If the consumer’s inbox has expired, reconcile or restore identity evidence before redrive. Generating new IDs to “get around dedup” is dangerous and requires explicit new-intent semantics.

For ordered keys, replay missing items before suffixes and respect current source versions. An old state-replacement event may now be stale and should resolve as superseded rather than overwrite current state. Delta/financial events may still require exact application.

## Retention and privacy

Retention derives from investigation time, source replay, legal/audit, privacy deletion, and repair value. Apply separate horizons by error/data class. Monitor records and bytes approaching expiry without resolution.

Payload references, schemas, encryption keys, and diagnostic traces must live at least as long as quarantine—or the record declares that payload was intentionally erased and can no longer be redriven. Copying a payload into quarantine creates another governed data store. Propagate deletion, residency, legal hold, and tenant-key destruction.

Store stack traces and raw exceptions separately with redaction; they may contain secrets or record content. Quarantine UI/search should not expose payload values as metric labels or broad full-text indexes.

Discard is an explicit terminal outcome with reason, authority, affected business identity, and reconciliation. TTL expiry is not silent discard. Before expiry, notify owner and apply policy: archive, extend under approval, repair, or record irreversible loss.

## Capacity and cost model

Illustrative platform:

- live delivery 100,000 messages/s at 1.2 KiB;
- normal quarantine rate 0.015%;
- quarantine envelope/evidence adds 900 bytes;
- payload retained inline for 30 days with two replicas;
- a bad release can fail 35% of traffic for 12 minutes before circuit/rollback;
- controlled redrive target 8,000/s.

Normal quarantine is 15/s. At 2.1 KiB per record, 30 days yields about 38.9 million records and 76 GiB logical; two replicas are 152 GiB before indexes/backups.

The 12-minute incident produces `100,000 * 0.35 * 720 = 25.2 million` failures. Copying each 2.1 KiB record adds about 49 GiB logical in minutes and can overload metadata/indexes. Cohort/range manifests plus retained source are essential for systemic failures.

At 8,000/s, redriving 25.2 million takes 52.5 minutes before retries. If each effect consumes 3 ms of downstream CPU, redrive adds 24 CPU-seconds/s. Live traffic must retain priority and downstream utilization headroom.

Operator capacity also matters. Fifteen unrelated poison messages/s is 1.3 million/day; manual per-message review is impossible. Group by fingerprint/entity and automate known-safe repair while measuring false grouping.

## Concrete failure trace: unsafe bulk redrive

A schema bug quarantines two million payment events. After deploying a decoder fix, an operator selects “all payments” in a UI whose query is not snapshotted and redrives at unlimited speed with new event IDs. The selection grows while running, old events whose payments already committed bypass inbox dedup, and the payment provider is saturated by duplicates.

Containment stops redrive and payment calls, then reconciles by payment intent. Repair records original IDs, restores downstream idempotency/status, and replays only proven absent effects. Prevention requires immutable selection manifests, original identity preservation, dry-run/canary, approvals, rate/downstream budgets, automatic stop thresholds, and a privileged API that cannot silently generate new logical IDs.

## Operations and observability

Track by source, consumer, release, schema, tenant, fingerprint, and state:

- quarantine rate/count/bytes and ratio to live traffic;
- oldest new/untriaged/unresolved age and owner/SLO;
- failure-class/fingerprint cardinality and first/last seen;
- systemic circuit state and retained source ranges;
- payload/reference availability, schema/key expiry, and retention risk;
- repair transform outcomes and validation failures;
- dry-run/canary/redrive rate, lag, duplicates, downstream saturation, and stop events;
- resolved/discarded/expired outcomes and source/effect reconciliation;
- unauthorized payload access and control-plane changes.

Alert on rate change and age, not merely “DLQ non-empty.” A nonempty quarantine may be normal; an unowned week-old financial event or sudden correlated spike is urgent.

Runbooks cover schema poison, bad release cohort, external ambiguity, ordered-key blockage, missing payload/key, quarantine-store outage, privacy deletion, and runaway redrive.

## Security and isolation

Separate permissions to quarantine automatically, inspect metadata, reveal payload, upload repair transforms, approve redrive, change destination/rate, discard, and alter retention. Use least privilege and immutable audit.

Authenticate original source identity and bind event ID to tenant/payload digest. Quarantined payloads are hostile: validate size, compression, schema, content type, and sandbox any transform/parser. Never render unescaped payload content in an operator UI.

Redrive can invoke high-value effects and is equivalent to production write access. Require short-lived authorization, change record, dual control for high-risk domains, and per-tenant/destination quotas. Cross-tenant selection or cache leakage is a critical incident.

## Verification strategy

- model-test quarantine state transitions, source disposition, and idempotent duplicate quarantine;
- fault-inject local/remote quarantine dual-write boundaries;
- generate each structured failure class and verify policy/version/evidence;
- simulate a bad release affecting millions and ensure cohort circuit/range manifests protect storage;
- replay ordered streams with a quarantined gap and validate blocking/repair policy;
- test dry-run, immutable selection, canary, stop thresholds, rate priority, and cancellation;
- expire inbox/payload/schema/key independently and verify redrive is blocked or reconciled;
- attempt unauthorized inspection, transformation, discard, and cross-tenant redrive.

## Decision framework

Quarantine when a message cannot safely progress now but retaining it has repair/audit value. Drop only when loss is explicitly acceptable and observed. Pause/circuit-break a cohort instead of individually quarantining every event during a systemic consumer/dependency failure.

Before enabling a dead-letter path:

1. Which typed failures lead to retry, quarantine, reconciliation, pause, or drop?
2. Can source disposition and quarantine evidence be made atomic/recoverable?
3. What ordering scope remains blocked after one item is quarantined?
4. Which evidence/payload/schema/key is needed to repair, and how long is it retained?
5. Can systemic failure be represented as ranges/cohorts instead of millions of copies?
6. How does redrive preserve identity and protect live/downstream capacity?
7. Who owns every item and which terminal outcomes are audited?

## References

- [Amazon SQS: Dead-Letter Queues](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-dead-letter-queues.html)
- [RabbitMQ: Dead Letter Exchanges](https://www.rabbitmq.com/docs/dlx)
- [Google Cloud Pub/Sub: Dead-Letter Topics](https://cloud.google.com/pubsub/docs/dead-letter-topics)
- [Apache Kafka Connect: Error Handling and Dead Letter Queues](https://kafka.apache.org/documentation/#connectconfigs_errors.deadletterqueue.topic.name)
- [CloudEvents Specification](https://github.com/cloudevents/spec)
- [NIST SP 800-61 Rev. 3: Incident Response Recommendations and Considerations](https://csrc.nist.gov/pubs/sp/800/61/r3/final)
