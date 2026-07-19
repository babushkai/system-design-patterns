# Message Queue Architecture

A message queue is a durable ownership-transfer mechanism for asynchronous work. Producers append an intent; one worker at a time receives a temporary claim; completion becomes final only when the queue records an acknowledgement. The core design problem is managing that state under crashes, overload, rebalancing, and retention—not choosing an SDK call.

This chapter owns queue mechanics: enqueue, claim/visibility, acknowledgement, partitioning, worker flow control, backlog, retention, and capacity. [Message Ordering](03-message-ordering.md) owns sequence guarantees, [Delivery Guarantees](04-delivery-guarantees.md) owns duplicate/loss boundaries, and [Poison-Message Quarantine](08-dead-letter-queues.md) owns failed-message repair.

## Workload and contract

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

- durability required before enqueue acknowledgement;
- maximum accepted payload and whether payloads are inline or referenced;
- time-to-visible for immediate and scheduled messages;
- claim/visibility semantics and maximum extension;
- backlog retention and message expiry;
- partitioning/routing and fairness policy;
- behavior when the queue is full or its control plane is unavailable;
- whether approximate depth and age metrics are sufficient.

The queue contract should not promise that a handler’s external effect happens once. Acknowledgement and side-effect ambiguity are end-to-end concerns covered in the delivery chapter.

## State and invariants

A durable queue tracks:

| State | Purpose |
|---|---|
| message record | immutable ID, routing key, payload reference, enqueue time, expiry |
| availability state | ready, scheduled, claimed, acknowledged, quarantined, expired |
| claim generation | fences late acknowledgement or extension |
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

## Data plane and control plane

The **data plane** appends messages, replicates records, indexes ready/scheduled state, issues claims, validates claim generations, records acknowledgements, and streams payloads. Its hot path should use a cached versioned queue assignment rather than a global metadata lookup per message.

The **control plane** creates queues, defines durability/retention/quotas, assigns partitions and replicas, manages consumer membership, publishes encryption/schema policy, and coordinates movement. Nodes continue on a pinned assignment during a short control-plane outage, but reject unsafe mutations when their partition epoch is stale.

Separate queue metadata from message payloads. Large payloads amplify replication, memory, and redelivery cost. A claim-check design places an immutable encrypted object in durable storage and queues a content-addressed reference plus integrity digest. Publication must not expose the reference before the object is durable, and retention must not delete the object while any message or replay can reference it.

## Claim and acknowledgement protocol

For a work-table design, claiming is one transaction:

1. select eligible rows in a bounded partition/time order while skipping rows locked by other claimers;
2. update each row from `ready` to `claimed` with worker ID, lease deadline, and incremented claim generation;
3. return payload references and signed delivery tokens;
4. commit before the worker begins effects.

A log-based queue may leave the record immutable and store delivery state in a separate subscription/consumer cursor plus in-flight map. The invariant is the same: current claim state must survive coordinator failure.

Visibility timeout is a recovery detector. Too short and slow healthy work is delivered concurrently; too long and a dead worker stalls recovery. Use task-specific initial leases plus bounded heartbeats/extensions. A worker extends only while making progress and stops extending before it loses local authority. The broker caps total lease lifetime so a wedged worker cannot hide work forever.

Acknowledgement includes the latest claim generation. If claim 8 expires and worker B receives claim 9, a late `ack(claim=8)` is rejected. This protects the queue record. It does not fence an external database or payment effect; that boundary needs idempotency or a downstream fencing key.

Batch claims reduce round trips but increase invisible work, memory, and recovery delay. A worker should claim according to available execution slots, not prefetch thousands while only ten can run. Track claimed-but-not-started age separately from execution time.

## Ready, delayed, and priority structures

A single FIFO list is insufficient when messages can be delayed, expire, or carry priority. Common structures are:

- append log plus per-partition consumer positions for streaming workloads;
- ready queue plus time-ordered delayed structure;
- time buckets or timing wheel for large scheduled populations;
- bounded priority bands, each with fairness and aging;
- database work table with indexes on `(state, not_before, partition)`.

Moving scheduled messages into ready state must be idempotent. A crash between removal from a delay structure and insertion into ready cannot lose the message. Keep the authoritative record in one store and derive readiness, or transact the move.

Strict priority can starve normal work. Use weighted service, reserved capacity, or age promotion. Priority is an admission/scheduling policy, not a field that workers independently interpret. Backfills and retries need separate budgets so they cannot crowd out new work.

## Partitioning and consumer assignment

Partitions are the unit of storage placement and parallelism. Hash routing balances arbitrary keys; entity routing keeps related work together; tenant routing supports isolation but creates skew. Ordering consequences belong to the ordering chapter.

Partition count limits maximum useful parallelism and affects metadata, file handles, recovery, and fan-out. More partitions are not free. Use many logical/virtual buckets mapped to physical partitions when rebalancing is expected, but pin the mapping version into each routing decision so producers and consumers agree.

Consumer assignment is epoch-based. A coordinator publishes `(group, partition, owner, epoch)`. During rebalance, the old owner stops claiming, drains or releases in-flight messages according to policy, and checkpoints before the new owner starts. If the system cannot guarantee a clean handoff, it must tolerate overlapping delivery while fencing stale acknowledgements.

Hot partitions require better routing or isolated capacity, not merely more consumers: one partition may allow only limited concurrent claims or one ordered lane. Split a hot tenant/entity only if its semantic scope permits it. Otherwise apply per-key admission and expose that key as the bottleneck.

## Worker flow control and overload

The queue is a buffer, not infinite capacity. Apply backpressure at three boundaries:

- producer admission based on stored bytes, oldest age, and tenant quota;
- broker-to-worker flow based on free execution slots and bytes;
- worker concurrency based on downstream saturation and deadline budget.

Queue depth alone is ambiguous. Ten million tiny messages may be cheap; ten thousand large or slow tasks may represent days of work. Track **work backlog** using estimated service cost and **age of oldest eligible message**. Service estimates can be learned by message class and updated from observations.

When consumers fall behind, retries and lease expirations can create positive feedback. Stop speculative prefetch, lengthen leases only for demonstrably progressing tasks, reduce producer acceptance for low-priority classes, and protect downstream services with concurrency budgets. Scaling workers into a saturated database makes the incident worse.

## Retention, compaction, and payload lifecycle

Work queues can remove acknowledged records after a repair/audit window; log queues retain by time/size independently of acknowledgement. Define retention from replay, incident investigation, legal, and cost requirements. “Keep forever” is not an operational plan.

Deletion proceeds from a watermark: all required replicas have the segment; no live claim or replay window needs it; quarantine references are preserved; snapshots/checkpoints cover recovery; payload references have no remaining owners. Compaction by message key is not equivalent to queue acknowledgement and can erase intermediate work, so use it only for state-like streams whose contract permits replacement.

External payloads need reference accounting or expiry derived from the queue’s maximum retention plus safety margin. A periodic reconciliation compares live message references with objects and detects both missing payloads and orphaned cost.

## Capacity and cost model

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

## Concrete failure trace: lease shorter than execution

A media task normally takes 40 seconds, but its visibility lease is 30 seconds. Worker A starts and remains healthy. At 30 seconds the queue exposes the message; worker B receives claim generation 12 while A still holds generation 11. Both upload the result. A’s late acknowledgement is correctly rejected, yet the duplicate external upload already occurred.

Containment reduces concurrency for that task class and uses a safe longer lease. Repair identifies duplicate outputs through the message/effect idempotency key. Prevention adds progress-based extension, a bound on claimed-but-not-started time, claim-generation propagation to the effect service, and a test that pauses a worker across lease expiry. The queue behaved according to its contract; the end-to-end effect protocol was incomplete.

## Operations and observability

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

## Security and isolation

Authenticate producers and consumers separately and authorize queue, tenant, operation, and message type. A worker that can claim should not automatically be allowed to purge, replay, or inspect every payload. Delivery tokens are unguessable, scoped, short-lived capabilities.

Validate envelope size, headers, delay, expiry, routing-key length, and payload reference before expensive work. Encrypt transport and durable records; isolate encryption keys and storage partitions where tenant or residency policy requires it. Queue metrics and traces must not expose raw payloads or sensitive routing keys.

Control-plane changes—retention reduction, purge, replay, quota, consumer identity—need review, audit, and staged activation. Purge and bulk redrive are destructive operations with separate privileges and dry-run counts.

## Verification strategy

- model-test random enqueue/claim/extend/expire/ack transitions against a simple state machine;
- kill brokers after every append/replication/publication boundary and verify acknowledged durability;
- partition coordinators and workers to verify assignment epochs and stale-token rejection;
- pause workers across lease expiry and validate duplicate-effect defenses;
- load-test measured message-size/service-time distributions, hot partitions, and downstream saturation;
- fill disks and stall replicas while checking admission and retention watermarks;
- reconcile inline/external payloads after expiry, quarantine, and replay;
- fuzz malformed envelopes, oversized headers, and unauthorized operations.

## Decision framework

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
