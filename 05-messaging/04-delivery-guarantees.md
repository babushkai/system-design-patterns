# Delivery Guarantees and Effect Boundaries

Delivery guarantees describe uncertainty between producer, broker, consumer, and effect store. A broker can prove that a record was durably appended, fetched, or transactionally written to another broker partition. It cannot, by itself, prove that a payment, email, filesystem write, or unrelated database transaction happened exactly once. Correct design names each atomicity boundary and closes the gaps end to end.

Scope: broker delivery semantics, acknowledgement ambiguity, producer retry deduplication, consumer effect commit, and the limits of “exactly once.” [Outbox and Inbox](07-outbox-pattern.md) owns atomic database-to-broker publication and transactional consumer deduplication. [Ordering](03-message-ordering.md) owns sequences. Workflow-specific retries and compensation belong to [Effect Commit Protocols](../18-workflow-job-systems/06-retry-idempotency-compensation.md).

## Workload and guarantee contract

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

## State and invariants

Relevant state includes producer identity/epoch/sequence, broker record identity and position, replica commit watermark, delivery/consumer generation, consumer checkpoint, idempotency/inbox record, effect transaction, and reconciliation status.

Enforce:

**Broker acknowledgement matches replication policy.** A producer success means the record is durably present on the declared failure domains, not merely accepted into one process buffer.

**Logical identity survives retry.** Every retry of the same intent carries the same event/idempotency identity; a new intent gets a new identity.

**Producer epochs fence old sessions.** A restarted or failed-over producer cannot continue an earlier sequence namespace and overwrite/deduplicate unrelated records.

**Effect and processed marker are atomic where possible.** Recording “done” before the effect can lose work; recording it afterward can duplicate the effect.

**Checkpoint follows durable effect.** A consumer does not advance past work whose required effect could still be lost.

**Deduplication retention covers the retry/replay contract.** Expiring identity evidence earlier silently weakens the guarantee.

**Ambiguity is recoverable.** A timeout has a status lookup or reconciliation path; blind retries are not the only option.

## Failure windows

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

## Producer-side semantics

### Fire-and-forget

The producer sends and does not await durable acknowledgement. It minimizes latency and resource use but permits silent loss during client, network, or broker failure. Restrict it to observations whose loss is acceptable and measured, such as sampled diagnostics—not business intent.

### Confirmed at-least-once publication

The producer waits for a broker acknowledgement matching the replication policy. On timeout it retries the same logical event with bounded backoff/deadline. Without broker deduplication, the log may contain duplicates; consumers still need effect safety.

### Idempotent producer session

The broker tracks `(producer_id, epoch, sequence)` per partition. A retry of the same sequence returns the existing append; an older epoch is fenced; a gap or conflicting payload is rejected. This prevents duplicates caused by retries within the protocol’s state/retention scope. It does not deduplicate the same business operation emitted under a new producer identity after an application restart unless the stable event ID is also enforced.

Producer identity state must be durable enough for failover. If the broker forgets it during restore while the producer assumes it remains valid, the guarantee changes. Include snapshot/backup compatibility and dedup watermark in recovery tests.

The strongest database-to-broker producer design is not “retry after commit”; it is a [transactional outbox](07-outbox-pattern.md) or authoritative event log whose relay retries with stable identity.

## Broker durability and replication

A leader append passes stages: accepted, written locally, replicated, committed under the quorum/in-sync policy, and retained in a published segment. Producer acknowledgements choose one of these stages. The contract must state whether process crash, node loss, zone loss, and simultaneous failures are covered.

Replication settings are not independent toggles. A quorum acknowledgement is meaningful only if replica membership, failure-domain placement, minimum healthy replicas, and unclean leader-election policy preserve the claimed durability. Allowing an out-of-date replica to become leader restores availability by accepting data loss.

Fsync policy determines power-loss durability; replication does not protect against correlated software deletion or poison writes. Backups/archives and delayed replicas address different failures. Validate checksums and generation manifests during recovery.

Retention bounds redelivery. Once the only retained copy is deleted, at-least-once cannot be honored for a consumer that has not checkpointed it. Tie topic retention to maximum permitted lag/replay, or make lagging subscriptions expire/fail explicitly.

## Consumer acknowledgement strategies

**Acknowledge before effect** gives at-most-once processing: a crash loses work. **Acknowledge after effect** gives at-least-once attempt: a crash after effect can repeat it. **Acknowledge atomically with effect** is possible only when broker progress and effect participate in one transactional domain or when a durable inbox/effect transaction lets redelivery converge.

For batch consumption, process and commit a contiguous prefix. If item 4 fails after items 1–3 commit, do not advance a single checkpoint through item 10 unless individual durable identities allow safe replay. Store per-item inbox state or split the batch.

Acknowledgement tokens contain consumer-group generation/claim generation. A consumer revoked during rebalance cannot move the checkpoint after a new member owns the partition. This prevents progress regression or skipping; it does not eliminate duplicate execution during an unclean handoff.

Cancellation and deadlines matter. If the broker times out a consumer while its database transaction continues, it may redeliver concurrently. The effect store should serialize/idempotently reject the duplicate, and the consumer should stop work when authority/deadline is lost where safe.

## Effectively-once database effects

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

## External effects

An external API should accept a caller-generated idempotency key, durably bind it to request semantics, and return the original result for retries. It must reject reuse of the key with different parameters. The retention window must cover caller retry, delayed broker replay, and disaster recovery.

If the downstream system lacks idempotency, options are:

- place an idempotent adapter in front of it with authoritative status/reconciliation;
- allocate a natural unique business reference the downstream enforces;
- make the effect detectable and reconcile before retry;
- accept at-most-once and possible loss;
- accept duplicates and run repair/compensation;
- redesign the boundary.

Recording “request sent” locally is not proof that the remote effect happened. Recording “done” only from a response is not proof it did not happen when the response is lost. Model `UNKNOWN` as a durable state and resolve it through status lookup or reconciliation.

## Transactional consume-transform-produce

Some brokers allow a consumer to read records, write output records, and commit input positions in one broker transaction. The coordinator tracks producer epoch and transaction markers; read-committed consumers hide aborted output. This can provide exactly-once processing **within that broker transaction domain**.

The guarantee stops at non-participating state. A handler that updates PostgreSQL or charges a card inside the same function has an unclosed boundary even if broker input/output are transactional. Use an outbox/inbox, downstream idempotency, or a workflow effect protocol.

Long broker transactions pin unstable data and can time out during processing. Keep them bounded in records, bytes, and duration. External calls inside the transaction couple broker health to remote latency and should be avoided.

## Deduplication scope and storage

Dedup state is indexed by effect scope and stable identity. It stores status, request digest, result reference, first/last observation, and expiry. Partition it with the effect when atomicity is required.

Capacity is rate times retention. Illustrative consumer:

- 40,000 delivered events/s;
- 72-hour maximum replay/retry horizon;
- 80 encoded bytes per inbox record including index overhead;
- two database replicas.

The live window contains `40,000 * 259,200 = 10.368 billion` identities. One logical copy is about 772 GiB; two are about 1.51 TiB before storage-engine amplification, backups, and reserve. A global dedup table is therefore often impractical. Prefer transactional per-domain inboxes, natural unique keys, version gating, compact partitions dropped by time, and replay horizons justified by recovery policy.

Bloom filters can avoid some negative lookups but cannot be the correctness authority because false positives would suppress legitimate work. They may front an exact store only if positives are verified.

Expiry is a semantic event. After 72 hours, the same ID can repeat unless the effect itself has a durable natural uniqueness constraint. Document this in replay tooling and disaster-recovery plans.

## Capacity and latency model

Illustrative broker path:

- 60,000 publications/s at 1 KiB;
- three replicas;
- producer batches average 200 records;
- quorum commit adds a measured 4 ms p50 and 18 ms p99;
- 0.15% of attempts time out and retry;
- four independent subscriptions.

Replicated ingress is about 176 MiB/s before protocol/index overhead. Retry traffic adds only 90 attempts/s on average, but correlated broker slowdown can push timeout/retry rates far higher; use retry budgets and absolute deadlines.

Fan-out delivery is `60,000 * 4 = 240,000` records/s. If consumer effect transactions sustain a measured 2,500/s per database shard at safe utilization, each full-rate subscription needs at least 24 effect shards before skew and failure reserve. Broker throughput is not end-to-end capacity.

Batching amortizes acknowledgements but increases ambiguity/replay suffix and latency. Model batch fill time at low traffic, maximum transaction bytes, and recovery reprocessing—not only peak throughput.

## Concrete failure trace: disaster restore forgets dedup state

A broker and consumer database are restored from backups after regional loss. The broker archive includes 48 hours of events, but the consumer inbox backup is 36 hours old. Replaying the broker suffix repeats 12 hours of already executed effects. The team assumed “at-least-once plus inbox equals exactly once,” but restored the two evidence streams to inconsistent recovery points.

Containment pauses consumers and identifies effects with natural business references. Repair reconciles the 12-hour window before controlled replay. Prevention aligns recovery-point contracts, includes inbox/effect state in recovery manifests, retains downstream idempotency keys beyond maximum replay, and exercises cross-system restore—not independent component restore.

## Operations and observability

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

## Security and abuse resistance

Authenticate producer identity/epoch issuance and consumer groups. An attacker who can choose another tenant’s event or idempotency IDs can suppress legitimate work or retrieve prior results. Namespace keys by authenticated caller and effect scope; bind them to a request digest.

Protect acknowledgement/checkpoint operations with generation-scoped tokens. Limit retry rate, batch size, transaction duration, and replay privileges. Broker logs, inboxes, and reconciliation records contain business identifiers and result references; apply least privilege, encryption, retention, and audit.

Never deserialize untrusted payloads before schema/size checks. Duplicate storms are an availability attack even when effects are idempotent—the duplicate path still consumes broker, network, lookup, and transaction resources.

## Verification strategy

- enumerate every crash point between append, acknowledgement, effect, inbox, checkpoint, and response;
- property-test stable identity reuse and reject same key/different request;
- partition leaders/producers to verify epoch fencing and acknowledged durability;
- crash consumers before/after database commit and prove one logical effect;
- test external API timeout with status reconciliation and duplicate key reuse;
- expire dedup partitions and attempt delayed replay;
- restore broker, effect store, and inbox to skewed recovery points;
- load-test retry storms, duplicate hot keys, transaction aborts, and downstream saturation.

## Decision framework

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
- [Martin Kleppmann et al.: Online Event Processing—Achieving Consistency Where Distributed Transactions Have Failed](https://doi.org/10.1145/3329672.3329679)
