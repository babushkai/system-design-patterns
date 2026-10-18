# Poison-Message Quarantine and Redrive

## TL;DR

A poison message is a delivery that cannot make progress under the current code, data, policy, or dependency state. Quarantine removes it from the hot delivery path while preserving enough evidence and identity to diagnose, repair, and replay it safely. A dead-letter queue is therefore an incident and repair subsystem, not a place to hide exhausted generic retries.

A quarantine subsystem must define failure classification, quarantine state, retained evidence, bulk isolation, repair, and controlled redrive. Retry timing/idempotency/compensation belong to [Workflow Effect Protocols](../18-workflow-job-systems/06-retry-idempotency-compensation.md). Queue claims are in [Message Queue Architecture](01-message-queues.md), and end-to-end duplicate effects are in [Delivery Guarantees](04-delivery-guarantees.md).

## Workload and Contract

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

## State and Invariants

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

## Failure Classification

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

## Data Plane and Control Plane

The **data plane** receives structured failures, persists quarantine records/payload references, advances or records source disposition, serves bounded inspection, and executes rate-limited redrive. It supports high-volume batch quarantine without calling a human-review service per message.

The **control plane** owns classification policy, retention, access, ownership/routing, repair transform registry, redrive approvals, destinations, quotas, legal holds, and dashboards. Emergency policy changes are versioned, audited, canaried, and rollbackable.

Separate quarantine storage from the source queue so poison evidence survives source retention/purge and cannot block its partitions. Keep a source-position index for reconciliation and an entity/type/error index for triage. Avoid arbitrary high-cardinality indexes on raw payload fields.

Human triage UI reads redacted metadata by default; privileged payload access is just-in-time and audited. Bulk APIs operate on stable query snapshots/manifests so a changing filter does not cause an unrepeatable redrive set.

## Quarantine Transition

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

## Systemic Failures and Bulk Isolation

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

## Triage and Repair Workflow

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

## Controlled Redrive

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

## Retention and Privacy

Retention derives from investigation time, source replay, legal/audit, privacy deletion, and repair value. Apply separate horizons by error/data class. Monitor records and bytes approaching expiry without resolution.

Payload references, schemas, encryption keys, and diagnostic traces must live at least as long as quarantine, or the record declares that payload was intentionally erased and can no longer be redriven. Copying a payload into quarantine creates another governed data store. Propagate deletion, residency, legal hold, and tenant-key destruction.

Store stack traces and raw exceptions separately with redaction; they may contain secrets or record content. Quarantine UI/search should not expose payload values as metric labels or broad full-text indexes.

Discard is an explicit terminal outcome with reason, authority, affected business identity, and reconciliation. TTL expiry is not silent discard. Before expiry, notify owner and apply policy: archive, extend under approval, repair, or record irreversible loss.

## Product Knob Map

A quarantine is configured in two places and only one of them is the dead-letter queue: the delivery path decides when a record is classified as poison, while the quarantine's own retention decides how long anyone has to do something about it. Teams audit the first and inherit the second from a template. Read both ends off the running platform before trusting this table, because the defaults were chosen for queues that were never meant to hold evidence.

| Product and scope | Knob | Invariant it enforces | Default and how it misleads |
| --- | --- | --- | --- |
| Amazon SQS source queue with a redrive policy | `maxReceiveCount`, `VisibilityTimeout` | a record leaves the hot path after a bounded number of deliveries | default ten, and it counts retrievals rather than failures, so a handler whose visibility timeout expires is charged a delivery it never completed and a "retry three times" policy becomes one retry under latency. Set visibility above p99 handler duration before trusting the number, and read `ApproximateReceiveCount` off a real message instead of inferring it. |
| Amazon SQS dead-letter queue | `MessageRetentionPeriod` on the quarantine itself | evidence outlives the investigation window | a dead-letter queue is an ordinary queue and inherits the four-day default, so Friday's poison record is deleted Tuesday with no error emitted anywhere. Fourteen days is the maximum, so any retention objective longer than two weeks needs exported evidence in durable storage. Derive the value from oldest-untriaged age plus holiday cover, never from the source queue's value. |
| Amazon SQS redrive | `MaxNumberOfMessagesPerSecond`, `CancelMessageMoveTask` | replay runs at a declared rate and stops when told | left blank, the service chooses the rate from backlog size and varies it while the task runs; 500 messages/s is the ceiling you can declare. Only one movement task per queue is active at a time, so a second operator's task fails instead of queueing. Canceling does not revert messages already moved, so the destination holds an unmarked partial set and consumer-side idempotency still has to hold. |
| RabbitMQ quorum queue | `x-delivery-limit` with `x-dead-letter-exchange` | poison is removed after a bounded number of acknowledgment attempts | delivery limits exist only on quorum queues; a classic queue dead-letters on reject, TTL expiry, and max-length eviction but never counts attempts, so an infinite requeue loop looks like a healthy queue. Changing queue type silently changes the quarantine contract. |
| RabbitMQ dead-letter exchange | `default-requeue-rejected`, `x-first-death-reason` headers | the failure cause survives the hop into quarantine | requeue-on-reject defaults on, so a nack without explicit requeue=false returns the record to the same consumer forever. TTL expiry and max-length eviction land in the same queue as failures, so "dead-lettered" is not a synonym for "failed" and triage must read the death header. |
| Google Cloud Pub/Sub subscription | `deadLetterPolicy.maxDeliveryAttempts` | exhausted deliveries stop consuming subscriber budget | between five and one hundred, default five, counted as one plus NACKs plus ack-deadline overruns, and honored on a best-effort basis: the tracked count can reset to zero, most often on pull subscriptions with inactive subscribers, so poison is delivered well past the maximum. Attempts are counted only when the Pub/Sub service account holds publisher and subscriber roles on the dead-letter topic, granted after that topic exists; without them nothing forwards and no error surfaces. |
| Google Cloud Pub/Sub dead-letter topic | `messageRetentionDuration` plus a subscription attached to it | quarantined payloads survive long enough to be triaged | topic retention is unset by default, which hands the decision to subscriptions; the subscription you attach defaults to seven days measured from publish time, bounded between ten minutes and thirty-one days. A message published to a topic with no subscription is dropped, so an unattached dead-letter topic is a silent delete. Forwarding wraps the payload in a new message with new identity, so replay means republishing and the original `messageId` is already gone. |
| AWS Lambda async invocations and event-source mappings | `MaximumRetryAttempts`, `MaximumRecordAgeInSeconds`, `OnFailure` destination | a failed record survives retry exhaustion somewhere queryable | for stream sources both limits default to -1, infinite, so the default policy retries a poison batch forever and blocks its shard instead of dead-lettering it, and destinations fire only for records those limits discard. The quiet trap is reserved concurrency set to zero: the poller idles, source backlog grows, and the function reports zero failures while nothing reaches the destination. |
| Azure Service Bus queue or subscription | `MaxDeliveryCount`, `DefaultMessageTimeToLive` | poison leaves the delivery path after bounded redelivery | default ten, incremented by an abandoned message or an expired lock rather than by a processing failure, and the behavior cannot be disabled, only raised. Expiry-driven and failure-driven quarantining land in the same place with different reason codes, so triage has to read the reason. The quarantine is a sub-queue of the entity, so recreating that entity in infrastructure code discards it while active-depth metrics stay flat. |
| Kafka Streams / Kafka Connect | `default.deserialization.exception.handler`; `errors.tolerance` plus dead-letter topic settings | one unreadable record cannot halt a topology or connector | the default fail handler stops the task, which is usually right. Log-and-continue and `errors.tolerance=all` advance the committed position after writing the dead-letter copy, so replay needs an explicit offset reset or a re-import from the dead-letter topic, and that topic is the only surviving copy of payload and headers once source retention moves on. Context headers are copied only when explicitly enabled. |

```bash
# read-only: prove what both ends of the path will actually do, before the incident
aws sqs get-queue-attributes --queue-url "$SOURCE_URL" \
  --attribute-names RedrivePolicy MessageRetentionPeriod VisibilityTimeout
aws sqs get-queue-attributes --queue-url "$DLQ_URL" \
  --attribute-names MessageRetentionPeriod ApproximateNumberOfMessages \
  --query 'Attributes.{retention:MessageRetentionPeriod,depth:ApproximateNumberOfMessages}'
# deliveries already spent by what sits in quarantine; zero visibility hides nothing
aws sqs receive-message --queue-url "$DLQ_URL" --max-number-of-messages 10 \
  --visibility-timeout 0 --message-attribute-names All \
  --query 'Messages[].{id:MessageId,deaths:MessageAttributes."aws:deadLetterQueue.RedriveCount".StringValue}'
# redrive is rate-limited and cancellable; the last ten tasks are the audit trail
aws sqs list-message-move-tasks --source-arn "$DLQ_ARN" --max-results 10 \
  --query 'MessageMoveTasks[].{status:Status,rate:MaxNumberOfMessagesPerSecond,moved:NumMovedMessages}'

rabbitmqctl list_queues name type arguments policy
rabbitmqctl list_policies -p orders

gcloud pubsub subscriptions describe orders-worker \
  --format 'value(topic,deadLetterPolicy,ackDeadlineSeconds,messageRetentionDuration)'
gcloud pubsub topics describe orders-dlq --format 'value(name,messageRetentionDuration)'
# attempts are counted only if the service account holds both roles on the dead-letter topic
gcloud pubsub topics get-iam-policy orders-dlq \
  --query 'bindings[?role!=null].{role:role,members:members}'

aws lambda get-function --function-name billing-provision \
  --query 'Configuration.{reserved:ReservedConcurrentExecutions,state:State}'
aws lambda get-function-event-invoke-config --function-name billing-provision \
  --query '{retries:MaximumRetryAttempts,age:MaximumRecordAgeInSeconds,onFailure:DestinationConfig.OnFailure.Destination}'
aws lambda list-event-source-mappings --function-name billing-provision \
  --query 'EventSourceMappings[].{state:State,retries:MaximumRetryAttempts,age:MaximumRecordAgeInSeconds,failureDestination:DestinationConfig.OnFailure.Destination}'

curl -s "$CONNECT_BASE/connectors/orders-sink/config" \
  | jq '{tolerance:."errors.tolerance",dlq:."errors.deadletterqueue.topic.name",retry:."errors.retry.timeout"}'
kafka-get-offsets.sh --bootstrap-server "$BROKERS" --topic orders-dlq

az servicebus queue show --name orders --namespace-name "$NS" \
  --query '{maxDelivery:maxDeliveryCount,ttl:defaultMessageTimeToLive,deadLetters:countDetails.deadLetterMessageCount}'
```

## Capacity and Cost Model

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

## Concrete Failure Trace: Unsafe Bulk Redrive

A schema bug quarantines two million payment events. After deploying a decoder fix, an operator selects “all payments” in a UI whose query is not snapshotted and redrives at unlimited speed with new event IDs. The selection grows while running, old events whose payments already committed bypass inbox dedup, and the payment provider is saturated by duplicates.

Containment stops redrive and payment calls, then reconciles by payment intent. Repair records original IDs, restores downstream idempotency/status, and replays only proven absent effects. Prevention requires immutable selection manifests, original identity preservation, dry-run/canary, approvals, rate/downstream budgets, automatic stop thresholds, and a privileged API that cannot silently generate new logical IDs.

## Operations and Observability

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

## Incident Playbook

Quarantine incidents are about evidence more often than throughput: the records are already out of the hot path, and the only question is whether what survives can explain them and be replayed safely. Establish what is provably recoverable before touching a redrive tool, because the irreversible actions in this subsystem are usually taken by an operator trying to make a dashboard go green.

### The dead-letter queue is non-empty and nobody owns it

Read age and fingerprint cardinality before depth: ten thousand records sharing one error class seen in the last hour is a release, while forty records across twenty classes with the oldest nine days old is an ownership failure. Route by error class and tenant to a named owner, and if no route matches then the missing route is the defect rather than the messages being defective. Where the platform exposes no owner field, attach ownership in the quarantine record so it survives redeliveries. Verify by watching oldest-unowned-age return to zero and confirming every active error class resolves to a team. Do not purge to silence the page, and do not alert on non-empty depth as the SLO: a flat nonzero queue holding one unowned week-old financial event is worse than a large spike someone is actively working.

### Records bounce between the main queue and the quarantine

The signature is a redrive that completes and then refills the dead-letter queue at nearly the same count within minutes, with redelivery counters climbing monotonically across passes. That means the poison condition is still present (unrolled release, missing schema adapter, degraded dependency) or the redrive re-entered with fresh identities so dedup cannot see the earlier attempts. Stop redriving, fix the consumer or isolate the cohort, then prove it on a deterministic canary of ten records and watch attempt counters stay flat. Do not raise `maxReceiveCount` or delivery limits to give the record "a few more chances", and do not strip message attributes so it looks new: both convert a bounded quarantine into an unbounded loop that consumes worker capacity indefinitely.

### A bad release is dead-lettering most of traffic

Treat this as one incident with millions of rows rather than millions of incidents, and confirm correlation first: same release digest, same normalized fingerprint, first-seen timestamp aligned to the deploy time. Open the circuit or pause delivery for that consumer cohort, roll the release back, and preserve source retention and checkpoints so replay reads immutable retained history instead of copies; record a range manifest of partitions, position bounds, and checksums rather than materializing payloads. After rollback, replay one bounded generation under an explicit rate budget and confirm quarantine ratio returns to baseline. Do not let the platform copy twenty-five million records into quarantine just because the redrive policy says to: you will spend the incident on storage and index pressure, and the copies are worse evidence than the retained source, which still carries original position and ordering.

### A redrive is saturating downstream or duplicating effects

Judge a running move by destination saturation and duplicate-detection counters, not by how fast the quarantine is draining; a fast drain into a saturated consumer is the failure mode, not progress. Stop conditions in the manifest exist to be triggered, so cancel early rather than negotiating. Before resuming at a smaller budget, check whether quarantined records have outlived the consumer inbox retention that made replay idempotent: past that horizon a redrive is a new effect rather than a safe retry and needs reconciliation against downstream state first. Verify by sampling entity state against the authoritative source instead of counting move successes. Do not finish a move because it is already half done; a partial redrive is recoverable, while a complete redrive of identity-expired records produces duplicate effects that no later run can undo.

### A record can never be repaired

The evidence is gone rather than the will: payload reference expired, encryption key destroyed, schema registry version pruned, or the upstream system that could reissue it decommissioned. Ask the domain whether the business intent still has to happen; if it does, reconstruct from the authoritative source as a new event with explicit causal lineage instead of pretending to replay the original. If reconstruction is impossible, record irreversible loss with named authority, affected business identity, and reconciliation of dependent totals while the inputs are still legible. Verify by confirming the loss is visible wherever the successful equivalent would have been counted, so downstream sums reconcile rather than merely differing. Do not let TTL force the decision into a silent DISCARDED: expiry without authority and a reconciliation note is how a handful of lost events becomes an unexplainable month-end difference.

## Security and Isolation

Separate permissions to quarantine automatically, inspect metadata, reveal payload, upload repair transforms, approve redrive, change destination/rate, discard, and alter retention. Use least privilege and immutable audit.

Authenticate original source identity and bind event ID to tenant/payload digest. Quarantined payloads are hostile: validate size, compression, schema, content type, and sandbox any transform/parser. Never render unescaped payload content in an operator UI.

Redrive can invoke high-value effects and is equivalent to production write access. Require short-lived authorization, change record, dual control for high-risk domains, and per-tenant/destination quotas. Cross-tenant selection or cache leakage is a critical incident.

## Verification Strategy

- model-test quarantine state transitions, source disposition, and idempotent duplicate quarantine;
- fault-inject local/remote quarantine dual-write boundaries;
- generate each structured failure class and verify policy/version/evidence;
- simulate a bad release affecting millions and ensure cohort circuit/range manifests protect storage;
- replay ordered streams with a quarantined gap and validate blocking/repair policy;
- test dry-run, immutable selection, canary, stop thresholds, rate priority, and cancellation;
- expire inbox/payload/schema/key independently and verify redrive is blocked or reconciled;
- attempt unauthorized inspection, transformation, discard, and cross-tenant redrive.

## Decision Framework

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
- [Amazon SQS: StartMessageMoveTask](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/APIReference/API_StartMessageMoveTask.html)
- [RabbitMQ: Quorum Queue Dead Lettering and Delivery Limits](https://www.rabbitmq.com/docs/quorum-queues#dead-lettering)
- [Apache Kafka Connect: Dead Letter Queue Error Collector](https://kafka.apache.org/documentation/#connect_ref_errorcollector_deadletterqueue)
- [Google Cloud Pub/Sub: Handling Failures](https://cloud.google.com/pubsub/docs/handling-failures): best-effort attempt counting and the IAM roles that dead-lettering silently depends on
- [AWS Lambda: Asynchronous Invocation Destinations](https://docs.aws.amazon.com/lambda/latest/dg/invocation-async.html)
- [Azure Service Bus: Dead-Letter and Other In-Queue Messages](https://learn.microsoft.com/en-us/azure/service-bus-messaging/service-bus-dead-letter-queues)

---

## Related Chapters

- [Message Queue Architecture](01-message-queues.md): the queue these poison messages came from, and the retry policy that produced them
- [Effect Commit Protocols for Workflows](../18-workflow-job-systems/06-retry-idempotency-compensation.md): redrive is a retry with human judgment attached, so it needs the same idempotency rules
- [Background Jobs and Worker Pools](../18-workflow-job-systems/02-background-jobs-worker-pools.md): the worker that quarantines a message, and the lease that expires while it does
- [Incident Command and Learning](../11-observability/07-incident-management.md): a growing quarantine is an incident with a queue name attached to it
