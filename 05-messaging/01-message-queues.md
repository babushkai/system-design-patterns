# Message Queue Architecture

## TL;DR

A message queue is a durable ownership-transfer mechanism for asynchronous work. Producers append an intent; one worker at a time receives a temporary claim; completion becomes final only when the queue records an acknowledgment. The core design problem is managing that state under crashes, overload, rebalancing, and retention, not choosing an SDK call.

A queue contract must specify enqueue, claim and visibility, acknowledgment, partitioning, worker flow control, backlog, retention, and capacity. [Message Ordering](03-message-ordering.md) owns sequence guarantees, [Delivery Guarantees](04-delivery-guarantees.md) owns duplicate/loss boundaries, and [Poison-Message Quarantine](08-dead-letter-queues.md) owns failed-message repair.

## Workload and Contract

Define the API in state-machine terms:

```text
enqueue(queue, message_id, routing_key, payload_ref, not_before, expiry)
claim(queue, worker, max_items, lease_duration) -> deliveries
extend(delivery_token, lease_duration)
ack(delivery_token)
release(delivery_token, reason, not_before)
```

A delivery token binds queue, message, claim generation, worker, and lease. Acknowledging only by message ID lets an expired worker delete work now owned by another worker.

Specify separately:

- durability required before enqueue acknowledgment;
- maximum accepted payload and whether payloads are inline or referenced;
- time-to-visible for immediate and scheduled messages;
- claim/visibility semantics and maximum extension;
- backlog retention and message expiry;
- partitioning/routing and fairness policy;
- behavior when the queue is full or its control plane is unavailable;
- whether approximate depth and age metrics are sufficient.

The queue contract should not promise that a handler’s external effect happens once. [Delivery Guarantees](04-delivery-guarantees.md) covers acknowledgment and side-effect ambiguity end to end.

## State and Invariants

A durable queue tracks:

| State | Purpose |
|---|---|
| message record | immutable ID, routing key, payload reference, enqueue time, expiry |
| availability state | ready, scheduled, claimed, acknowledged, quarantined, expired |
| claim generation | fences late acknowledgment or extension |
| partition position | append/scan location or priority/time bucket |
| consumer membership | worker identity, capacity, liveness, assignment |
| retention checkpoints | lowest data still required by consumers and repair policy |

The design enforces these invariants:

**Accepted work is reconstructable.** Once the durability contract is acknowledged, a permitted failure cannot erase the message or all replicas of its payload.

**At most one current claim generation.** Multiple workers can observe redelivery across time, but only the latest unexpired generation can acknowledge or extend that claim.

**A claim is a lease, not ownership.** If the worker disappears, the queue can make the message eligible again without consulting the dead process.

**Acknowledged work is not normally claimable.** Retained bytes may remain for audit or compaction, but serving state excludes them.

**Retention does not outrun recovery.** The queue does not delete data still required by an active claim, configured replay window, replica recovery, or quarantine evidence.

**Admission is bounded.** Producers cannot consume unbounded disk simply because consumers are slow.

## Data Plane and Control Plane

The **data plane** appends messages, replicates records, indexes ready/scheduled state, issues claims, validates claim generations, records acknowledgments, and streams payloads. Its hot path should use a cached versioned queue assignment rather than a global metadata lookup per message.

The **control plane** creates queues, defines durability/retention/quotas, assigns partitions and replicas, manages consumer membership, publishes encryption/schema policy, and coordinates movement. Nodes continue on a pinned assignment during a short control-plane outage, but reject unsafe mutations when their partition epoch is stale.

Separate queue metadata from message payloads. Large payloads amplify replication, memory, and redelivery cost. A claim-check design places an immutable encrypted object in durable storage and queues a content-addressed reference plus integrity digest. Publication must not expose the reference before the object is durable, and retention must not delete the object while any message or replay can reference it.

## Claim and Acknowledgment Protocol

For a work-table design, claiming is one transaction:

1. select eligible rows in a bounded partition/time order while skipping rows locked by other claimers;
2. update each row from `ready` to `claimed` with worker ID, lease deadline, and incremented claim generation;
3. return payload references and signed delivery tokens;
4. commit before the worker begins effects.

A log-based queue may leave the record immutable and store delivery state in a separate subscription/consumer cursor plus in-flight map. The invariant is the same: current claim state must survive coordinator failure.

Visibility timeout is a recovery detector. Too short and slow healthy work is delivered concurrently; too long and a dead worker stalls recovery. Use task-specific initial leases plus bounded heartbeats/extensions. A worker extends only while making progress and stops extending before it loses local authority. The broker caps total lease lifetime so a wedged worker cannot hide work forever.

Acknowledgment includes the latest claim generation. If claim 8 expires and worker B receives claim 9, a late `ack(claim=8)` is rejected. This protects the queue record. It does not fence an external database or payment effect; that boundary needs idempotency or a downstream fencing key.

Batch claims reduce round trips but increase invisible work, memory, and recovery delay. A worker should claim according to available execution slots, not prefetch thousands while only ten can run. Track claimed-but-not-started age separately from execution time.

## Ready, Delayed, and Priority Structures

A single FIFO list is insufficient when messages can be delayed, expire, or carry priority. Common structures are:

- append log plus per-partition consumer positions for streaming workloads;
- ready queue plus time-ordered delayed structure;
- time buckets or timing wheel for large scheduled populations;
- bounded priority bands, each with fairness and aging;
- database work table with indexes on `(state, not_before, partition)`.

Moving scheduled messages into ready state must be idempotent. A crash between removal from a delay structure and insertion into ready cannot lose the message. Keep the authoritative record in one store and derive readiness, or transact the move.

Strict priority can starve normal work. Use weighted service, reserved capacity, or age promotion. Priority is an admission/scheduling policy, not a field that workers independently interpret. Backfills and retries need separate budgets so they cannot crowd out new work.

## Partitioning and Consumer Assignment

Partitions are the unit of storage placement and parallelism. Hash routing balances arbitrary keys; entity routing keeps related work together; tenant routing supports isolation but creates skew. [Message Ordering](03-message-ordering.md) covers the ordering consequences.

Partition count limits maximum useful parallelism and affects metadata, file handles, recovery, and fan-out. More partitions are not free. Use many logical/virtual buckets mapped to physical partitions when rebalancing is expected, but pin the mapping version into each routing decision so producers and consumers agree.

Consumer assignment is epoch-based. A coordinator publishes `(group, partition, owner, epoch)`. During rebalance, the old owner stops claiming, drains or releases in-flight messages according to policy, and checkpoints before the new owner starts. If the system cannot guarantee a clean handoff, it must tolerate overlapping delivery while fencing stale acknowledgments.

Hot partitions require better routing or isolated capacity, not merely more consumers: one partition may allow only limited concurrent claims or one ordered lane. Split a hot tenant/entity only if its semantic scope permits it. Otherwise apply per-key admission and expose that key as the bottleneck.

## Worker Flow Control and Overload

The queue is a buffer, not infinite capacity. Apply backpressure at three boundaries:

- producer admission based on stored bytes, oldest age, and tenant quota;
- broker-to-worker flow based on free execution slots and bytes;
- worker concurrency based on downstream saturation and deadline budget.

Queue depth alone is ambiguous. Ten million tiny messages may be cheap; ten thousand large or slow tasks may represent days of work. Track **work backlog** using estimated service cost and **age of oldest eligible message**. Service estimates can be learned by message class and updated from observations.

When consumers fall behind, retries and lease expirations can create positive feedback. Stop speculative prefetch, lengthen leases only for demonstrably progressing tasks, reduce producer acceptance for low-priority classes, and protect downstream services with concurrency budgets. Scaling workers into a saturated database makes the incident worse.

## Retention, Compaction, and Payload Lifecycle

Work queues can remove acknowledged records after a repair/audit window; log queues retain by time/size independently of acknowledgment. Define retention from replay, incident investigation, legal, and cost requirements. “Keep forever” is not an operational plan.

Deletion proceeds from a watermark: all required replicas have the segment; no live claim or replay window needs it; quarantine references are preserved; snapshots/checkpoints cover recovery; payload references have no remaining owners. Compaction by message key is not equivalent to queue acknowledgment and can erase intermediate work, so use it only for state-like streams whose contract permits replacement.

External payloads need reference accounting or expiry derived from the queue’s maximum retention plus safety margin. A periodic reconciliation compares live message references with objects and detects both missing payloads and orphaned cost.

## Product Knob Map

Queue products disagree about which side of a claim owns time, so the mapping below starts from the invariant and walks to the setting that enforces it. In each product exactly one knob carries the correctness promise, and its default was chosen for throughput or developer convenience rather than for your deadline. Read the defaults off the running deployment before trusting any table, including this one.

| Product and scope | Knob | Invariant it enforces | Default and how it misleads |
| --- | --- | --- | --- |
| Amazon SQS, per queue or per receive | `VisibilityTimeout` | a claim is exclusive for a bounded window | 30 s. It is a redelivery timer, not an execution deadline: a healthy worker still running at second 31 gets a duplicate peer rather than an extension. Renew from the worker's own heartbeat with `ChangeMessageVisibility` (12 h ceiling). |
| Amazon SQS, per queue | `RedrivePolicy.maxReceiveCount` | retry attempts are bounded and quarantine happens automatically | counts deliveries, not failures, so a crash loop burns the budget before anyone reads the payload; the DLQ then holds one message with twenty attempts. |
| RabbitMQ, per channel | `basic.qos` prefetch count | in-flight work per consumer is bounded | 0 means unbounded, which turns one slow consumer into the queue's cold storage and makes rebalancing take minutes. Set it to a concurrency budget, not to "a bit more". |
| RabbitMQ quorum queues | `x-delivery-limit`, `x-retry-interval` | poison work is quarantined without manual intervention | classic mirrored queues have no delivery limit at all, so a migration that keeps them silently loses the bounded-retry invariant. |
| Apache Kafka consumer group | `max.poll.records`, `max.poll.interval.ms` | a member that stops making progress is removed | 500 records and 300 s. A batch whose p99 exceeds the interval produces a rebalance loop, not an error at the failing record. |
| Apache Kafka consumer group | `enable.auto.commit`, `auto.offset.reset` | progress is recorded only after effects complete | background time-based commit advances the offset ahead of processing, so a rebalance loses exactly the in-flight window; `latest` skips history on a new group unless you reset it deliberately. |
| Redis Streams consumer group | `XAUTOCLAIM` / `XCLAIM` min-idle-time | work abandoned by a dead consumer is reassigned | nothing reassigns by itself: entries stay in the pending entries list forever, and `maxmemory-policy` eviction deletes stream entries rather than rejecting the write. |
| PostgreSQL as a queue | `SELECT … FOR UPDATE SKIP LOCKED` plus a `(status, visible_at)` partial index | claim contention stays bounded as the table grows | no delay, retry, or expiry semantics come free; without the partial index every claim scans and autovacuum falls behind the delete churn. |
| NATS JetStream workqueue | `ack_wait`, `max_deliver`, ack-pending window | unacked work is redelivered with a bound | redelivery is driven by wall clock since delivery, not by worker liveness, so an overloaded pool sees the same message again while it is still running. |

```bash
# what the queue actually promises right now (script this into the service runbook)
aws sqs get-queue-attributes --queue-url "$QUEUE_URL" \
  --attribute-names VisibilityTimeout MaximumMessageSize MessageRetentionPeriod \
    RedrivePolicy ReceiveWaitTimeSeconds
rabbitmqctl list_queues name type arguments state messages messages_unacknowledged consumers
redis-cli XINFO GROUPS orders:stream                       # pending vs largest-delivered id
kafka-consumer-groups.sh --bootstrap-server "$BROKERS" --describe --group billing-workers
psql -c "SELECT status, count(*) AS n, now() - min(visible_at) AS oldest_wait
         FROM jobs GROUP BY status ORDER BY oldest_wait DESC"
```

## Capacity and Cost Model

Consider an illustrative queue:

- peak 25,000 enqueues/s;
- average encoded record 1.6 KiB including metadata;
- three replicas;
- consumer service time averages 18 ms with a measured coefficient of variation;
- peak producer burst lasts 20 minutes at 25,000/s while consumers sustain 19,000/s;
- target worker utilization 60%.

Logical ingress is about `25,000 * 1.6 KiB = 39 MiB/s`; three replicas write roughly 117 MiB/s before log, index, checksum, and compaction amplification. One day of unreclaimed replicated ingress is about 9.9 TiB, so retention and disk reserve dominate broker sizing.

The burst adds `(25,000 - 19,000) * 1,200 = 7.2 million` messages. At the post-burst surplus drain rate, if consumers are raised to 23,000/s while ingress returns to 19,000/s, drain time is `7.2M / 4,000 = 1,800 seconds`, or 30 minutes. Capacity planning needs this recovery interval, not only steady state.

At 18 ms mean service time, 25,000/s consumes 450 concurrent worker-seconds. At 60% target utilization, plan about 750 execution slots before failure reserve. Validate with the actual service-time tail and downstream limits; Little’s Law relates average in-flight work to throughput and residence time but does not promise acceptable tails.

## Concrete Failure Trace: Lease Shorter Than Execution

A media task normally takes 40 seconds, but its visibility lease is 30 seconds. Worker A starts and remains healthy. At 30 seconds the queue exposes the message; worker B receives claim generation 12 while A still holds generation 11. Both upload the result. A’s late acknowledgment is correctly rejected, yet the duplicate external upload already occurred.

Containment reduces concurrency for that task class and uses a safe longer lease. Repair identifies duplicate outputs through the message/effect idempotency key. Prevention adds progress-based extension, a bound on claimed-but-not-started time, claim-generation propagation to the effect service, and a test that pauses a worker across lease expiry. The queue behaved according to its contract; the end-to-end effect protocol was incomplete.

## Operations and Observability

Observe by queue, partition, tenant, message class, and consumer group:

- enqueue/claim/ack/release rates and latency;
- ready, scheduled, claimed, quarantined, and expired counts and bytes;
- oldest ready age, service-time distribution, and predicted drain time;
- claim expirations, extensions, stale-token rejections, and redeliveries;
- producer admission/rejection and quota use;
- worker free slots, prefetch, claimed-not-started age, and downstream saturation;
- partition skew, reassignment epoch, replica lag, recovery ETA, and disk reserve;
- payload fetch failures, missing objects, and orphaned objects.

Runbooks cover growing backlog, hot partition, full disk, stuck scheduled mover, consumer rebalance loop, mass lease expiry, corrupt payload, and control-plane outage. Drills should use production-sized backlogs; empty-queue failover proves little.

## Incident Playbook

The order below is the triage order: each step names what to read, the decision it licenses, and the action that is safe while the backlog is still growing. All five assume the queue itself is serving traffic; a control-plane or partition outage is a different runbook.

### Backlog grows with no errors

Compare arrival rate with completion rate per message class, then check whether service time moved. Flat service time and flat throughput mean a capacity or placement limit: one hot partition, a prefetch ceiling, or a concurrency budget. Rising service time means a dependency, and adding workers converts a slow incident into an outage of that dependency. Decision rule: only add consumers when the saturated resource is the consumer itself; otherwise raise the downstream budget first or shed low-priority classes at admission. Expected drain time is backlog divided by surplus capacity, so state it in the incident channel as a number and re-derive it after each change.

### Duplicates appear although one worker owns the message

Read claim expirations and redeliveries next to the service-time distribution for that class. Expiry outrunning p99 execution is the usual cause, and the queue is behaving correctly. Contain by lengthening the lease for that class and lowering its concurrency; do not lower it globally, because unrelated classes inherit the change. Repair with the effect idempotency key rather than the message id when the effect lives in a third-party system, then prevent recurrence with heartbeat-driven extension and claim-generation fencing at the effect service.

### Consumers flap or rebalance in a loop

Check oldest in-flight age against the liveness deadline of the client library before touching any timeout. A member that exceeds its poll interval is being evicted for not finishing, so raising the session timeout only widens the window in which two members own the same partition. Shrink the unit of work per poll instead, and confirm recovery by watching assignment epochs stop incrementing while completion rate holds.

### Dead-letter queue grows during a deploy

Classify before redriving: sample twenty messages, group by error class, producer version, and tenant. One poison payload is quarantined individually; a systemic class means the consumer or producer is wrong and every redrive re-runs broken code against live effects. Fix forward or roll back, then replay in a controlled pass at a rate the downstream can absorb, with duplicate suppression verified on a small slice first. Purging to make the dashboard quiet destroys the only complete record of what failed.

### Queue reads empty while work is unfinished

Look for work that is stored but not visible: in-flight or claimed counts, delayed and scheduled lanes, redelivery backoff windows, subscription filters that match nothing, and consumers holding prefetched messages that died without releasing them. Only after those are ruled out should the incident be treated as message loss, because the recovery path (replaying producers from their own durable log) is expensive and itself creates duplicates.

## Security and Isolation

Authenticate producers and consumers separately and authorize queue, tenant, operation, and message type. A worker that can claim should not automatically be allowed to purge, replay, or inspect every payload. Delivery tokens are unguessable, scoped, short-lived capabilities.

Validate envelope size, headers, delay, expiry, routing-key length, and payload reference before expensive work. Encrypt transport and durable records; isolate encryption keys and storage partitions where tenant or residency policy requires it. Queue metrics and traces must not expose raw payloads or sensitive routing keys.

Control-plane changes (retention reduction, purge, replay, quota, consumer identity) need review, audit, and staged activation. Purge and bulk redrive are destructive operations with separate privileges and dry-run counts.

## Verification Strategy

- model-test random enqueue/claim/extend/expire/ack transitions against a simple state machine;
- kill brokers after every append/replication/publication boundary and verify acknowledged durability;
- partition coordinators and workers to verify assignment epochs and stale-token rejection;
- pause workers across lease expiry and validate duplicate-effect defenses;
- load-test measured message-size/service-time distributions, hot partitions, and downstream saturation;
- fill disks and stall replicas while checking admission and retention watermarks;
- reconcile inline/external payloads after expiry, quarantine, and replay;
- fuzz malformed envelopes, oversized headers, and unauthorized operations.

## Decision Framework

Use a work queue when one logical worker should perform each item and buffering isolates producers from consumers. Use pub-sub when multiple independent subscriptions need the same event. Use a durable workflow engine when work contains timers, multi-step state, compensation, or human waits; see [Durable Execution](../18-workflow-job-systems/04-durable-execution-workflow-engines.md).

Choose the queue design by answering:

1. What is durably accepted, and which failures may lose it?
2. What is the claim/lease state machine and stale-token fence?
3. How do payload size, service-time tail, and downstream limits shape flow control?
4. What partition key and consumer assignment contain skew?
5. How much burst backlog can be stored, and how long will it take to drain?
6. What retention/replay/quarantine requirements delay reclamation?
7. How are duplicate effects, poison data, tenant isolation, and destructive operations verified?

## References

- [Amazon SQS: Visibility Timeout](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-visibility-timeout.html)
- [RabbitMQ: Consumer Acknowledgements and Publisher Confirms](https://www.rabbitmq.com/docs/confirms)
- [Apache Kafka: Design](https://kafka.apache.org/documentation/#design)
- [PostgreSQL: SELECT, SKIP LOCKED](https://www.postgresql.org/docs/current/sql-select.html)
- [John D. C. Little: A Proof for the Queuing Formula L = λW](https://doi.org/10.1287/opre.9.3.383)
- [CloudEvents Specification](https://github.com/cloudevents/spec)
- [Apache Kafka: Consumer Configuration](https://kafka.apache.org/documentation/#consumerconfigs)
- [RabbitMQ: Quorum Queues](https://www.rabbitmq.com/docs/quorum-queues)
- [Redis: Streams, Declarative and Consumer Groups](https://redis.io/docs/latest/develop/data-types/streams/)
- [NATS JetStream: Consumer Configuration](https://docs.nats.io/using-nats/jetstream/manage_jetstream/consumers)
- [Amazon SQS: FIFO Queues and Duplicate Deduplication](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/standard-queues-fifo.html)
---

## Related Chapters

- [Kafka](../09-whitepapers/06-kafka.md): where the log-as-abstraction argument was first made at scale
- [Twitter/X Timelines](../08-case-studies/01-twitter.md): fan-out on write, and the viral publish that turns one event into a million writes
- [Delivery Guarantees and Effect Boundaries](04-delivery-guarantees.md): what a broker can promise about one message, and what it leaves to you
- [Backpressure](../06-scaling/07-backpressure.md): a queue is a buffer with an opinion: what happens when it fills
- [Background Jobs and Worker Pools](../18-workflow-job-systems/02-background-jobs-worker-pools.md): the consumer side, where a queue becomes a work system with retries and leases
