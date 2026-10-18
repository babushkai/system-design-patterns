# Delivery Guarantees and Effect Boundaries

## TL;DR

Delivery guarantees describe uncertainty between producer, broker, consumer, and effect store. A broker can prove that a record was durably appended, fetched, or transactionally written to another broker partition. It cannot, by itself, prove that a payment, email, filesystem write, or unrelated database transaction happened exactly once. Correct design names each atomicity boundary and closes the gaps end to end.

Delivery guarantees are end-to-end contracts over broker semantics, acknowledgement ambiguity, producer retry deduplication, consumer effect commit, and the limits of “exactly once.” [Outbox and Inbox](07-outbox-pattern.md) owns atomic database-to-broker publication and transactional consumer deduplication. [Ordering](03-message-ordering.md) owns sequences. Workflow-specific retries and compensation belong to [Effect Commit Protocols](../18-workflow-job-systems/06-retry-idempotency-compensation.md).

## Workload and Guarantee Contract

Draw the complete path:

```text
source transaction
  -> producer/relay
  -> broker leader and replicas
  -> subscription delivery/checkpoint
  -> consumer transaction
  -> external effects
```

For every arrow state:

- what constitutes success;
- where durable evidence lives;
- what a timeout means;
- whether retries can duplicate;
- how long deduplication evidence is retained;
- which failure domains the acknowledgement survives;
- who reconciles ambiguous outcomes.

Use precise vocabulary:

**At-most-once observation** means a logical delivery is attempted no more than once in the defined scope; loss is possible. **At-least-once observation** means retries continue so a retained message is eventually attempted, but duplicates are possible. **Effectively-once effect** means duplicate attempts converge to one committed logical effect under an idempotency/transaction protocol. **Exactly-once processing** is valid only inside a named transactional domain with defined inputs, outputs, epochs, and retention.

Never shorten these to “exactly once” in an architecture contract.

## State and Invariants

Relevant state includes producer identity/epoch/sequence, broker record identity and position, replica commit watermark, delivery/consumer generation, consumer checkpoint, idempotency/inbox record, effect transaction, and reconciliation status.

Enforce:

**Broker acknowledgement matches replication policy.** A producer success means the record is durably present on the declared failure domains, not merely accepted into one process buffer.

**Logical identity survives retry.** Every retry of the same intent carries the same event/idempotency identity; a new intent gets a new identity.

**Producer epochs fence old sessions.** A restarted or failed-over producer cannot continue an earlier sequence namespace and overwrite/deduplicate unrelated records.

**Effect and processed marker are atomic where possible.** Recording “done” before the effect can lose work; recording it afterward can duplicate the effect.

**Checkpoint follows durable effect.** A consumer does not advance past work whose required effect could still be lost.

**Deduplication retention covers the retry/replay contract.** Expiring identity evidence earlier silently weakens the guarantee.

**Ambiguity is recoverable.** A timeout has a status lookup or reconciliation path; blind retries are not the only option.

## Failure Windows

The central ambiguity is the **lost acknowledgement**. Suppose the broker commits event E and sends success, but the response is lost. The producer cannot distinguish committed from uncommitted. Retrying with a new ID creates two logical events; abandoning the attempt can lose it. Retrying with the same identity lets the broker return the existing result.

On the consumer side, suppose the handler writes the effect and crashes before acknowledgement. Redelivery is necessary because the broker cannot know the effect committed. If the effect and inbox marker share one database transaction, the second attempt observes the marker and acknowledges without repeating the mutation. If the effect is an external API, that API must accept the same idempotency key or the workflow must reconcile/compensate.

Other windows include:

- broker leader acknowledges before required replicas and then fails;
- producer retry crosses a producer-session/epoch reset;
- consumer checkpoints a batch before its database commit;
- broker transaction commits outputs but an external side effect fails;
- dedup state expires before a delayed replay;
- failover restores broker data but not the external dedup store;
- operator replay changes event IDs and defeats identity.

Build the protocol around these windows rather than happy-path SDK configuration.

## Producer-Side Semantics

### Fire-and-forget

The producer sends and does not await durable acknowledgement. It minimizes latency and resource use but permits silent loss during client, network, or broker failure. Restrict it to observations whose loss is acceptable and measured, such as sampled diagnostics, not business intent.

### Confirmed at-least-once publication

The producer waits for a broker acknowledgement matching the replication policy. On timeout it retries the same logical event with bounded backoff/deadline. Without broker deduplication, the log may contain duplicates; consumers still need effect safety.

### Idempotent producer session

The broker tracks `(producer_id, epoch, sequence)` per partition. A retry of the same sequence returns the existing append; an older epoch is fenced; a gap or conflicting payload is rejected. This prevents duplicates caused by retries within the protocol’s state/retention scope. It does not deduplicate the same business operation emitted under a new producer identity after an application restart unless the stable event ID is also enforced.

Producer identity state must be durable enough for failover. If the broker forgets it during restore while the producer assumes it remains valid, the guarantee changes. Include snapshot/backup compatibility and dedup watermark in recovery tests.

The strongest database-to-broker producer design is not “retry after commit”; it is a [transactional outbox](07-outbox-pattern.md) or authoritative event log whose relay retries with stable identity.

## Broker Durability and Replication

A leader append passes stages: accepted, written locally, replicated, committed under the quorum/in-sync policy, and retained in a published segment. Producer acknowledgements choose one of these stages. The contract must state whether process crash, node loss, zone loss, and simultaneous failures are covered.

Replication settings are not independent toggles. A quorum acknowledgement is meaningful only if replica membership, failure-domain placement, minimum healthy replicas, and unclean leader-election policy preserve the claimed durability. Allowing an out-of-date replica to become leader restores availability by accepting data loss.

Fsync policy determines power-loss durability; replication does not protect against correlated software deletion or poison writes. Backups/archives and delayed replicas address different failures. Validate checksums and generation manifests during recovery.

Retention bounds redelivery. Once the only retained copy is deleted, at-least-once cannot be honored for a consumer that has not checkpointed it. Tie topic retention to maximum permitted lag/replay, or make lagging subscriptions expire/fail explicitly.

## Consumer Acknowledgement Strategies

**Acknowledge before effect** gives at-most-once processing: a crash loses work. **Acknowledge after effect** gives at-least-once attempt: a crash after effect can repeat it. **Acknowledge atomically with effect** is possible only when broker progress and effect participate in one transactional domain or when a durable inbox/effect transaction lets redelivery converge.

For batch consumption, process and commit a contiguous prefix. If item 4 fails after items 1–3 commit, do not advance a single checkpoint through item 10 unless individual durable identities allow safe replay. Store per-item inbox state or split the batch.

Acknowledgement tokens contain consumer-group generation/claim generation. A consumer revoked during rebalance cannot move the checkpoint after a new member owns the partition. This prevents progress regression or skipping; it does not eliminate duplicate execution during an unclean handoff.

Cancellation and deadlines matter. If the broker times out a consumer while its database transaction continues, it may redeliver concurrently. The effect store should serialize/idempotently reject the duplicate, and the consumer should stop work when authority/deadline is lost where safe.

## Effectively-Once Database Effects

For an effect in one transactional database, use a uniqueness-constrained inbox:

```text
BEGIN
  INSERT inbox(consumer, event_id, received_at)
    -- unique(consumer, event_id); duplicate means already committed
  UPDATE domain_state ...
  UPDATE projection_checkpoint ...
COMMIT
ack broker
```

On duplicate, the unique conflict proves that this consumer’s prior database transaction committed, so it can acknowledge. A separate `SELECT has_seen` followed by effect is racy. A marker in a separate store recreates a dual write.

Idempotency must reflect the logical effect. `event_id` may be appropriate for one event; `payment_intent_id` may be the correct identity across several delivery records. A uniqueness constraint that is too narrow permits business duplicates; one too broad suppresses legitimate repetitions.

For monotonically versioned state, `UPDATE ... WHERE current_version < incoming_version` can converge naturally, but it discards intermediate deltas. Use it only when the event carries replacement state or the domain allows skipping.

## External Effects

An external API should accept a caller-generated idempotency key, durably bind it to request semantics, and return the original result for retries. It must reject reuse of the key with different parameters. The retention window must cover caller retry, delayed broker replay, and disaster recovery.

If the downstream system lacks idempotency, options are:

- place an idempotent adapter in front of it with authoritative status/reconciliation;
- allocate a natural unique business reference the downstream enforces;
- make the effect detectable and reconcile before retry;
- accept at-most-once and possible loss;
- accept duplicates and run repair/compensation;
- redesign the boundary.

Recording “request sent” locally is not proof that the remote effect happened. Recording “done” only from a response is not proof it did not happen when the response is lost. Model `UNKNOWN` as a durable state and resolve it through status lookup or reconciliation.

## Transactional Consume-Transform-Produce

Some brokers allow a consumer to read records, write output records, and commit input positions in one broker transaction. The coordinator tracks producer epoch and transaction markers; read-committed consumers hide aborted output. This can provide exactly-once processing **within that broker transaction domain**.

The guarantee stops at non-participating state. A handler that updates PostgreSQL or charges a card inside the same function has an unclosed boundary even if broker input/output are transactional. Use an outbox/inbox, downstream idempotency, or a workflow effect protocol.

Long broker transactions pin unstable data and can time out during processing. Keep them bounded in records, bytes, and duration. External calls inside the transaction couple broker health to remote latency and should be avoided.

## Deduplication Scope and Storage

Dedup state is indexed by effect scope and stable identity. It stores status, request digest, result reference, first/last observation, and expiry. Partition it with the effect when atomicity is required.

Capacity is rate times retention. Illustrative consumer:

- 40,000 delivered events/s;
- 72-hour maximum replay/retry horizon;
- 80 encoded bytes per inbox record including index overhead;
- two database replicas.

The live window contains `40,000 * 259,200 = 10.368 billion` identities. One logical copy is about 772 GiB; two are about 1.51 TiB before storage-engine amplification, backups, and reserve. A global dedup table is therefore often impractical. Prefer transactional per-domain inboxes, natural unique keys, version gating, compact partitions dropped by time, and replay horizons justified by recovery policy.

Bloom filters can avoid some negative lookups but cannot be the correctness authority because false positives would suppress legitimate work. They may front an exact store only if positives are verified.

Expiry is a semantic event. After 72 hours, the same ID can repeat unless the effect itself has a durable natural uniqueness constraint. Document this in replay tooling and disaster-recovery plans.

## Product Knob Map

Guarantee settings are spread across producer, broker, coordinator, and the downstream endpoint, and no single product surface shows the composed contract. The rows below name the knob whose default was picked for throughput or compatibility, together with the guarantee it silently decides. Treat every default as unverified until you read it from your own deployment, because managed offerings differ by region and API version.

| Product and scope | Knob | Invariant it enforces | Default and how it misleads |
| --- | --- | --- | --- |
| Apache Kafka producer and topic | `acks=all` with `min.insync.replicas` | an acknowledged append survives the loss of any in-sync replica except the last | `min.insync.replicas` defaults to 1, so `acks=all` degrades to "written to the leader" and one broker loss discards acknowledged records. Set both together and assert them at deploy time. |
| Apache Kafka transactions | `transactional.id`, `transaction.timeout.ms`, `transaction.state.log.replication.factor` | only one live session per transactional identity can commit, and coordinator loss preserves that fact | the internal transaction state topic is functional at a replication factor of 1 with in-sync minimum 1, so an environment sized on defaults can lose coordinator state and fence nothing. Raising the client timeout past the broker ceiling also lengthens how long read-committed consumers wait for a stable watermark. |
| Apache Kafka consumer group | `enable.auto.commit`, commit ordering | progress advances only over effects that are durable | background time-based commit runs independently of your handler, so a rebalance loses the in-flight window; committing after a batch that partially failed skips the failed members. Commit an explicit contiguous prefix after the effect commits. |
| Amazon SQS | `VisibilityTimeout`, `RedrivePolicy`, FIFO `ContentBasedDeduplication` | retry is bounded and same-intent retries collapse | FIFO deduplication covers a short identity window, so it removes an immediate retry storm and not a replay issued days later; the redrive counter bounds delivery attempts rather than outcomes, and standard queues suppress only best-effort duplicates. |
| RabbitMQ classic mirrored versus quorum queues | publisher confirms plus queue type | a confirm means the record survived the failure domains you claim | a confirm on a classic mirrored queue is issued by the primary alone, so failover can drop confirmed publishes; quorum queues confirm after a quorum write. The acknowledgement semantics are identical, which is why the migration looks risk-free in tests. |
| RabbitMQ AMQP channel transactions | `select_transaction` scope | local consumption and publication commit together | the transaction covers only broker operations on that channel, never your database or external call, and it serialises throughput. Reaching for it usually means the boundary belongs in an outbox. |
| Amazon SNS and Lambda asynchronous invocation | endpoint retry policy, function event-invoke destinations | a failed delivery is retained somewhere queryable | retries stop at the configured ceiling and, without an on-failure destination, the payload is discarded rather than quarantined; the retry counter is per invocation, so fan-out multiplies it across protocols. |
| Google Cloud Pub/Sub exactly-once delivery | `enableExactlyOnceDelivery` with ordering keys and storage policy | at most one outstanding delivery per ordering key in a region | the guarantee is scoped per ordering key and per region rather than globally; a global or topic-mapped topic can hold one outstanding delivery in each region, so "exactly once" needs its scope named out loud. It also raises the minimum acknowledge deadline and lowers throughput ceilings. |

```bash
# read-only: assemble the end-to-end contract from live settings
kafka-configs.sh --bootstrap-server "$BROKERS" --describe --all \
  --entity-type topics --entity-name orders \
  | grep -E 'min.insync.replicas|unclean|retention.ms'
kafka-topics.sh --bootstrap-server "$BROKERS" --describe \
  --topic __transaction_state | tr ',' '\n' | grep -E 'ReplicationFactor|Isr'
rg -n 'acks|enable\.idempotence|transactional\.id|delivery\.timeout\.ms|enable\.auto\.commit' config/
aws sqs get-queue-attributes --queue-url "$QUEUE_URL" \
  --attribute-names VisibilityTimeout RedrivePolicy FifoQueue ContentBasedDeduplication
aws lambda get-function-event-invoke-config --function-name charges \
  --query '{retries:MaximumRetryAttempts,maxAge:MaximumEventAgeInSeconds,destinations:DestinationConfig}'
gcloud pubsub subscriptions describe billing-worker \
  --format 'config(enableMessageOrdering,exactlyOnceDeliveryConfig,retryPolicy,deadLetterPolicy)'
psql -c "SELECT min(received_at) AS oldest_identity,
                now() - max(retire_before) AS horizon_slack FROM inbox"
```

## Capacity and Latency Model

Illustrative broker path:

- 60,000 publications/s at 1 KiB;
- three replicas;
- producer batches average 200 records;
- quorum commit adds a measured 4 ms p50 and 18 ms p99;
- 0.15% of attempts time out and retry;
- four independent subscriptions.

Replicated ingress is about 176 MiB/s before protocol/index overhead. Retry traffic adds only 90 attempts/s on average, but correlated broker slowdown can push timeout/retry rates far higher; use retry budgets and absolute deadlines.

Fan-out delivery is `60,000 * 4 = 240,000` records/s. If consumer effect transactions sustain a measured 2,500/s per database shard at safe utilization, each full-rate subscription needs at least 24 effect shards before skew and failure reserve. Broker throughput is not end-to-end capacity.

Batching amortizes acknowledgements but increases ambiguity/replay suffix and latency. Model batch fill time at low traffic, maximum transaction bytes, and recovery reprocessing, not only peak throughput.

## Concrete Failure Trace: Disaster Restore Forgets Dedup State

A broker and consumer database are restored from backups after regional loss. The broker archive includes 48 hours of events, but the consumer inbox backup is 36 hours old. Replaying the broker suffix repeats 12 hours of already executed effects. The team assumed “at-least-once plus inbox equals exactly once,” but restored the two evidence streams to inconsistent recovery points.

Containment pauses consumers and identifies effects with natural business references. Repair reconciles the 12-hour window before controlled replay. Prevention aligns recovery-point contracts, includes inbox/effect state in recovery manifests, retains downstream idempotency keys beyond maximum replay, and exercises cross-system restore, not independent component restore.

## Operations and Observability

Track each boundary:

- producer attempts, stable IDs, acknowledgements, timeouts, retry age, epoch fences;
- leader append/replicate/commit latency and healthy failure domains;
- unclean election, truncation, checksum/corruption, and retention exposure;
- delivered/acknowledged/redelivered rates and consumer generation rejects;
- inbox duplicates, unique conflicts, effect commit latency, and checkpoint lag;
- external idempotency result reuse, key conflicts, unknown outcomes, and reconciliation age;
- broker transaction open duration/bytes, aborts, and read-committed lag;
- dedup storage bytes, oldest identity, expiry, and replay horizon mismatch.

Runbooks cover producer timeout ambiguity, broker data loss, replay beyond dedup horizon, downstream idempotency outage, stuck broker transaction, and inconsistent disaster restore.

## Incident Playbook

Guarantee incidents are rarely caused by a broker silently dropping data; they are caused by two components holding different ideas of what an acknowledgement meant. Each entry below names the evidence that closes that ambiguity first, because the tempting action in these moments usually destroys the evidence needed for repair.

### Effects repeated after failover or restore

Compare recovery points before you compare code: the broker archive watermark, the inbox's oldest retained identity, and the replay start position. Repetition requires that replay began below the dedup floor, which happens when components are restored independently even though every individual restore reported success. Contain by pausing consumers, enumerating affected effects from natural business references, reconciling the overlap window, then advancing the checkpoint deliberately through it. Do not truncate the inbox to make redelivery converge: that turns a bounded duplicate window into unbounded uncertainty about which effects ever ran, and removes the only evidence reconciliation had.

### A publish timed out and you do not know whether it landed

Treat the timeout as a durable `UNKNOWN` state rather than a failure to retry, then resolve it by identity: look the stable event or intent id up in the broker, in the relay position, or at the downstream status endpoint. Retry with the same identity only when the lookup is unavailable or inconclusive, and bound that loop with an absolute deadline so one ambiguous record cannot hold a partition hostage. Escalate to reconciliation when the lookup itself is down. Do not mint a fresh identity to "get it through", because two logical intents are exactly what the business contract forbids, and the second copy surfaces in a finance review rather than in your dashboards.

### A transactional pipeline stalls after a coordinator change

Read transactional state first: open transaction age against `transaction.timeout.ms`, producer fencing errors, and whether the coordinator's own state topic has replication and in-sync minimums worth trusting. A stuck transaction holds the stable watermark back, so consumers look starved rather than slow; a fenced zombie writer appears as client churn to anyone reading only consumer logs. Contain by aborting the specific transactional identity rather than raising the timeout globally, since a longer timeout silently adds latency for every read-committed consumer. Do not restart consumers repeatedly hoping for a clean election; record the coordinator epoch and member id first, then verify healing by watching the stable watermark advance while open-transaction bytes fall.

### Redelivery climbs in lockstep with throughput

A duplicate ratio that tracks volume points at lease arithmetic rather than a poison message: compare the acknowledgement deadline or visibility timeout against handler p99 including queueing behind a busy dependency, and check whether batch commits advance past individual failures. Contain by shrinking the unit of work per poll and extending the lease for the affected class only, since global changes reach unrelated consumers that were healthy. Verify by watching redelivery rate fall while completion rate holds, not by watching lag alone. Do not disable retries or restart the group from the newest position as relief: that discards retained history and reappears as missing work instead of duplicate work.

### Downstream idempotency expires before your replay horizon does

Put both horizons in one view: broker retention plus maximum acceptable lag on one side, and the partner API's key window or your inbox partition's expiry on the other. The mismatch is silent until the incident you most need it for, because a restored region replays further back than a live system ever chooses to. Prefer a natural unique constraint in the effect store, which outlives any TTL; otherwise route the affected window to quarantine and reconcile against the partner's own statement of record. Do not treat a longer dedup TTL as the whole fix while the replay horizon still exceeds the partner's retention: that moves the mismatch instead of closing it, and the next restore repeats it at greater scale.

## Security and Abuse Resistance

Authenticate producer identity/epoch issuance and consumer groups. An attacker who can choose another tenant’s event or idempotency IDs can suppress legitimate work or retrieve prior results. Namespace keys by authenticated caller and effect scope; bind them to a request digest.

Protect acknowledgement/checkpoint operations with generation-scoped tokens. Limit retry rate, batch size, transaction duration, and replay privileges. Broker logs, inboxes, and reconciliation records contain business identifiers and result references; apply least privilege, encryption, retention, and audit.

Never deserialize untrusted payloads before schema/size checks. Duplicate storms are an availability attack even when effects are idempotent: the duplicate path still consumes broker, network, lookup, and transaction resources.

## Verification Strategy

- enumerate every crash point between append, acknowledgement, effect, inbox, checkpoint, and response;
- property-test stable identity reuse and reject same key/different request;
- partition leaders/producers to verify epoch fencing and acknowledged durability;
- crash consumers before/after database commit and prove one logical effect;
- test external API timeout with status reconciliation and duplicate key reuse;
- expire dedup partitions and attempt delayed replay;
- restore broker, effect store, and inbox to skewed recovery points;
- load-test retry storms, duplicate hot keys, transaction aborts, and downstream saturation.

## Decision Framework

Select guarantees per effect, not per broker brand:

1. What loss and duplication harm does the business tolerate?
2. What stable identity names one logical intent/effect?
3. Which steps can share one transaction?
4. Where is the first external boundary, and does it support idempotency/status?
5. What do producer/consumer timeouts mean and how are they reconciled?
6. How long must identity evidence survive replay and disaster recovery?
7. Can component backups restore a consistent end-to-end guarantee?

Use at-most-once only when loss is cheaper than duplication. Use at-least-once with idempotent/transactional effects for most durable work. Claim exactly-once processing only inside the precise domain you can prove.

## References

- [Jerome H. Saltzer, David P. Reed, and David D. Clark: End-to-End Arguments in System Design](https://web.mit.edu/Saltzer/www/publications/endtoend/endtoend.pdf)
- [Apache Kafka: Message Delivery Semantics](https://kafka.apache.org/documentation/#semantics)
- [Apache Kafka: Transactions](https://kafka.apache.org/documentation/#transactions)
- [RabbitMQ: Consumer Acknowledgements and Publisher Confirms](https://www.rabbitmq.com/docs/confirms)
- [Amazon SQS: At-Least-Once Delivery](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/standard-queues-at-least-once-delivery.html)
- [Martin Kleppmann et al.: Online Event Processing: Achieving Consistency Where Distributed Transactions Have Failed](https://doi.org/10.1145/3329672.3329679)
- [Google Cloud Pub/Sub: Exactly-Once Delivery](https://cloud.google.com/pubsub/docs/exactly-once-delivery)
- [AWS Lambda: Asynchronous Invocation and Failure Destinations](https://docs.aws.amazon.com/lambda/latest/dg/invocation-async.html)
- [Amazon SQS: Idempotency Tokens](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/using-idempotency-tokens.html)
---

## Related Chapters

- [WhatsApp](../08-case-studies/10-whatsapp.md): end-to-end encryption turns the acknowledgement contract into a device-state problem
- [Idempotency and Operation Identity](../01-foundations/08-idempotency.md): at-least-once is survivable only when the effect can be recognized a second time
- [Transactional Outbox, Inbox, and CDC Publication](07-outbox-pattern.md): making the write and the publish commit as one decision
- [Distributed Transactions](../02-distributed-databases/07-distributed-transactions.md): the alternative to reconciling two systems: one commit domain
