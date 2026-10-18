# Message Ordering

## TL;DR

Ordering is a scoped relation between events, not a property that a broker can provide “globally” without cost or qualification. A useful design names which events must be comparable, who assigns their sequence, what happens at gaps, and how epochs prevent an old writer from extending a sequence after failover.

Message ordering is defined by its scope, partition sequencing, causal metadata, gap handling, reorder buffers, epochs, rebalancing, and resharding. Queue claims are in [Message Queue Architecture](01-message-queues.md); duplicate/loss and acknowledgment ambiguity are in [Delivery Guarantees](04-delivery-guarantees.md).

## Workload and Order Contract

Start from the invariant, not “FIFO.” Examples:

- every state transition for one account applies in source commit order;
- all records in one database transaction become visible as one ordered batch;
- a reply is never processed before the request it references;
- configuration generation 42 supersedes 41 everywhere;
- no ordering is required across unrelated telemetry events.

Represent ordering metadata explicitly:

```text
OrderedEnvelope {
  stream_id
  writer_epoch
  sequence
  event_id
  source_version
  predecessor_ids[]?
  transaction_id?
  transaction_index?
}
```

Define:

- **scope**: entity, tenant, topic partition, transaction, or entire system;
- **relation**: total, per-key, FIFO-per-producer, or causal partial order;
- **sequencer**: authoritative database commit, partition leader, or dedicated service;
- **gap policy**: wait, retrieve missing data, skip with evidence, or rebuild;
- **consumer policy**: apply in order, buffer, version-gate, or treat operations as commutative;
- **reshard/cutover protocol** and how cursors translate;
- **deadline** after which liveness can override waiting, if the domain permits it.

“Ordered delivery” is incomplete unless these fields are known.

## State and Invariants

Sequencing state usually includes current writer epoch, next sequence per ordered stream, partition assignment generation, committed log end, transaction/batch boundaries, and each consumer’s last applied sequence plus reorder buffer.

Enforce:

**Uniqueness within an epoch.** For `(stream_id, writer_epoch)`, a sequence identifies at most one logical event.

**Epoch monotonicity.** Once epoch `e+1` is authoritative, no record or checkpoint from epoch `e` can advance the stream.

**Applied prefix.** A strict consumer’s durable state corresponds to a contiguous prefix through `last_applied`, plus explicitly stored future records. It never claims sequence 20 applied while silently missing 19.

**Ordering key matches invariant scope.** If two events must be ordered together, their routing cannot place them under independent sequencers.

**Sequence assignment follows source commitment.** A sequence attached before a transaction that later aborts creates a gap whose meaning must be explicit. A source commit position or transactional outbox sequence avoids guessing.

**Cutover has one pivot.** During repartitioning, every key has a declared last position in the old assignment and first epoch/position in the new one.

## Ordering Models

### Per-producer FIFO

One producer emits records in its local send order. This is weak: two producers updating the same entity have no shared order, and producer restart can reset local counters. Use it only when the producer is the sole authority for the scope and its epoch is durably fenced.

### Per-key or per-partition total order

All events for a key route to one partition leader, which appends a total sequence. This is the common scalable contract. Unrelated keys share a physical partition order, but applications normally depend only on each key’s subsequence. Throughput for one hot key is limited by its sequencer and consumer lane.

### Causal order

If event B was produced after observing A, A causally precedes B. Lamport clocks preserve a logical order consistent with causality but cannot distinguish concurrency. Vector clocks/version vectors can identify concurrent histories, at metadata cost proportional to participating writers unless compressed or scoped.

Many applications need only explicit dependencies: carry predecessor event IDs, source versions, or workflow step tokens. This is easier to audit than pretending wall-clock timestamps encode causality.

### Global total order

A global log assigns every event a comparable sequence. It simplifies deterministic replication and audit but centralizes sequencing/consensus, increases cross-region latency, and imposes order between unrelated events. Use it only when the invariant truly spans the entire scope. The consensus mechanics are covered in [Consensus Algorithms](../02-distributed-databases/08-consensus-algorithms.md).

## Sequence Assignment and Writer Epochs

A safe partition leader is elected under an epoch/term. Append requests contain producer identity, producer epoch where applicable, and a sequence. The leader accepts only the current authoritative epoch and persists the record before exposing its position according to the durability contract.

For domain aggregates, the authoritative database can assign `aggregate_version` with optimistic concurrency:

```text
append events for aggregate A
only if current_version = expected_version
publish versions expected_version+1 ... expected_version+n
```

The [event-store transaction](05-event-sourcing.md) prevents concurrent commands from claiming the same next version; downstream ordering should reuse that version rather than create another counter.

On producer retry, an idempotent append protocol can map `(producer_id, epoch, producer_sequence)` to the prior result. This prevents retry reordering inside that producer session. It does not order independent producers or guarantee an external effect; those are separate contracts.

Writer epochs require fencing at every acceptance point. If leader A loses authority but stays alive in a partition, leader B begins epoch 18. Consumers and replicas reject A’s epoch 17 records even if their numeric sequence is higher. Comparing sequence without epoch re-admits a stale writer.

## Partition Routing and Hot Keys

The routing function is versioned and deterministic. Hash `(tenant, entity_id)` to a virtual bucket, then map the bucket to a physical partition under an assignment generation. Avoid process-randomized hashes. Producers either receive the current map or send through a router that does.

Per-key order requires every producer to choose the same key. Missing, differently normalized, or semantically inconsistent keys are common ordering bugs. Validate key presence and canonical representation at ingress; emit sampled provenance that shows why a record chose its partition.

A hot key cannot be scaled by ordinary partition splitting while retaining one total order. Options are:

- keep one sequencer and parallelize downstream work that commutes;
- split the domain invariant into independent subkeys;
- batch/coalesce state updates;
- represent operations with commutative data types;
- admit less work for that key.

If none applies, the single-lane throughput is a real domain limit.

## Consumer Reorder Algorithm

A strict consumer stores `last_applied` and a bounded map keyed by `(epoch, sequence)`:

1. reject an older epoch; pause and reconcile on an unexpected newer epoch;
2. if `sequence <= last_applied`, treat it as an already-observed delivery under the delivery/effect policy;
3. if `sequence = last_applied + 1`, apply atomically with advancing `last_applied`;
4. repeatedly drain consecutive buffered records;
5. if `sequence > last_applied + 1`, persist it in the reorder buffer and record the missing range;
6. trigger gap repair when age/bytes exceed policy.

Persist future records before acknowledging them if the broker can otherwise discard the only copy available to this subscription. Buffer limits are in bytes and sequence span, not only item count. An attacker or corrupt producer can send a sequence far in the future and exhaust memory.

Applying state-setting events by source version can make out-of-order arrival naturally safe: `set profile to version 19` is ignored after version 20. Delta events such as `increment by 3` still need their sequence or a commutative operation identity. Choose event semantics that reduce ordering dependence where possible.

## Gap Detection and Repair

A gap may mean delayed delivery, filtered event, publisher abort, retention loss, corrupt segment, wrong routing key, or sequence allocation that permits holes. The sequence contract must say which gaps are valid.

Repair sources include broker replay, authoritative event store, source snapshot plus suffix, or producer reconciliation API. A safe response is:

1. stop applying later events for the affected strict scope;
2. classify whether the missing sequence should exist;
3. fetch/replay the missing range from an authoritative source;
4. verify event identity and epoch;
5. apply the repaired prefix and drain the buffer;
6. if the range is irrecoverable, rebuild state from a snapshot or escalate a domain-specific skip decision.

“Wait 30 seconds then continue” trades correctness for liveness and must be a named product policy. It is acceptable for best-effort telemetry, not silently for ledger transitions.

Gap age uses broker/source recorded time, not arbitrary client event time. Alert on the oldest blocked strict stream and total buffered bytes; aggregate out-of-order counts can hide one permanently wedged high-value entity.

## Atomic Batches and Transactions

Some domains require several records to appear together. Add `transaction_id`, ordered indices, count or end marker, and source commit position. Consumers buffer until the complete batch is durable, validate all members, then apply it atomically where their state store permits.

If a broker transaction spans partitions, consumers need a read-committed view and transaction markers replicated consistently. This only controls visibility inside the broker ecosystem. A database write or external API call still needs an end-to-end effect protocol.

Large transactions delay the stable/visible watermark and increase recovery buffers. Bound record count and bytes; prefer one domain event describing the committed fact over thousands of mechanically coupled messages when semantics permit.

## Rebalancing and Resharding

Consumer rebalance is a handoff of an ordered prefix. The old owner stops fetching, completes or persists in-flight records, commits a checkpoint under membership generation `g`, and revokes. The new owner starts from that checkpoint under `g+1`. If the old owner crashes, overlap/replay can occur; generation-fenced checkpoints prevent it from later moving progress.

Producer repartitioning is harder because moving a key between sequencers can interleave records. Use a cutover barrier:

1. publish assignment generation `m+1` with the key marked migrating;
2. stop/redirect new writes through a coordinator;
3. append a sealed `last_old_sequence` marker to the old partition;
4. wait until it is durably replicated and consumers can observe it;
5. initialize a new key epoch on the destination with a pointer to the barrier;
6. route new writes only to the destination;
7. retain translation metadata through all consumer replay windows.

Dual-producing without a barrier creates two incomparable total orders. Migration testing must include delayed producers holding the old map.

## Product Knob Map

Order is produced by client configuration more often than by broker configuration, which is why an ordering incident is usually a diff of effective producer settings rather than a server log review. The knobs below are the ones that decide whether a sequence exists at all, who may assign it, and what happens to a key's history when topology changes. Read them off the running clients and brokers before trusting this table or a design document.

| Product and scope | Knob | Invariant it enforces | Default and how it misleads |
| --- | --- | --- | --- |
| Apache Kafka producer, keyed records | `partitioner.adaptive.partitioning.enable` | records for one key are appended in the order the application produced them | enabled. Batching across partitions raises throughput and can reorder same-key records when retries are in flight; disable it wherever per-key order is the contract, accepting the throughput cost explicitly. |
| Apache Kafka producer | `enable.idempotence` with `acks` and `max.in.flight.requests.per.connection` | a retried batch cannot duplicate or reorder within a producer session | idempotence requires `acks=all`; an override to `acks=1` disables it without a startup error in some clients, so a latency tuning change silently removes the retry-ordering guarantee. Up to five in-flight requests stay ordered only while idempotent. |
| Apache Kafka consumer | `isolation.level` | consumers apply committed transaction prefixes only | `read_committed`. Switching to `read_uncommitted` for latency exposes aborted and not-yet-committed records, which appears downstream as phantom state rather than as a config error. |
| Apache Kafka topic and broker | `min.insync.replicas`, `unclean.leader.election.enable` | an accepted prefix is never truncated away by a leader change | `min.insync.replicas` defaults to 1, and unclean election is off in current releases but shipped on by default for years, so an inherited broker config can keep the lossy behavior: assert it per cluster rather than assuming. Enabling it for availability discards the tail that consumers already applied, so the sequence shrinks and every consumer must treat the gap as history loss. |
| Amazon SQS FIFO queue | `MessageGroupId`, `FIFOThroughputLimit` | one group is processed by at most one consumer at a time in receive order | throughput limit defaults to per-queue, which serializes unrelated groups behind the busiest key; sequence numbers are monotonic but explicitly not gap-free, so contiguity checks produce false alarms. |
| Google Cloud Pub/Sub subscription | `enableMessageOrdering`, exactly-once delivery | records sharing an ordering key are delivered in publish order to one attempt at a time | off by default, and it applies per storage partition rather than globally. A publisher that omits the ordering key on some messages gets unordered best-effort delivery for exactly those messages. Ordered delivery also raises the minimum acknowledge deadline. |
| Amazon Kinesis stream after a split | shard parent/child graph, `SequenceNumberForOrdering` | a partition key's history is read oldest shard first | the service preserves order across a split only if the consumer traverses parent to child itself. Nothing stops a parallel reader from starting the child shard while the parent still has records for that key. |
| RabbitMQ quorum queue / NATS JetStream consumer | single active consumer; `max_ack_pending` | one lane applies a scope without interleaved redelivery | single active consumer is off by default and does not create per-key order across publishers; an in-flight acknowledgment window above one means redelivered records can be applied after newer ones. |
| PostgreSQL logical decoding / Debezium | one replication slot per publication | the change stream carries one source log sequence number order | a single slot is one ordered lane, so throughput is bounded by that lane and its restart position; snapshot-phase rows and streaming-phase changes need explicit sequencing or the initial load lands after the changes it precedes. |

```bash
# read-only: prove which ordering contract the deployment is actually running
kafka-configs.sh --bootstrap-server "$BROKERS" --describe --all \
  --entity-type topics --entity-name orders \
  | grep -E 'min.insync.replicas|cleanup.policy|unclean'
rg -n 'enable\.idempotence|acks|partitioner\.adaptive|max\.in\.flight' config/producer.properties
aws sqs get-queue-attributes --queue-url "$QUEUE_URL" \
  --attribute-names FifoQueue ContentBasedDeduplication FIFOThroughputLimit \
    MessageGroupId DeduplicationScope
gcloud pubsub subscriptions describe orders-auditor \
  --format 'value(enableMessageOrdering,exactlyOnceDeliveryConfig.enableExactlyOnceDelivery)'
aws kinesis list-shards --stream-name orders \
  --query 'Shards[].{id:ShardId,parent:ParentShardId,range:[SequenceNumberRange.StartingSequenceNumber,SequenceNumberRange.EndingSequenceNumber]}'
psql -c "SELECT slot_name, restart_lsn, confirmed_flush_lsn,
                pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) AS retained
         FROM pg_replication_slots"
```

## Capacity and Cost Model

Illustrative ordered stream:

- 120,000 events/s across 96 partitions;
- average 700 bytes;
- one hot entity emits 4,000 events/s;
- consumer apply time averages 0.4 ms for ordinary keys and 1.1 ms for the hot entity;
- 0.02% of arrivals are temporarily out of order and buffer for a measured mean 3 seconds;
- three replicas.

Average partition load is 1,250 events/s, but the hot key alone is 4,000/s. Its serial apply demand is `4,000 * 1.1 ms = 4.4 CPU-seconds/s`; a single sequential worker cannot sustain it regardless of fleet size. The domain must coalesce, make operations commutative/parallel, split the key, or reject load.

Logical bandwidth is about 80 MiB/s and replicated append about 240 MiB/s before overhead. Reorder-buffer population by Little’s Law is approximately `120,000 * 0.0002 * 3 = 72` events on average, only about 50 KiB at 700 bytes. But a partition outage can buffer millions; size the hard bound from maximum gap-repair time and affected rate, not the normal average.

If an 8-minute partition gap affects 1,250/s, its suffix is 600,000 events or about 401 MiB before indexing/overhead. Decide whether to buffer, pause upstream, or rebuild rather than discovering the memory limit during failure.

## Concrete Failure Trace: Reshard Without a Barrier

Key `customer-9` maps to partition 3. A new assignment moves it to partition 11. Producer A refreshes immediately and emits update B to partition 11; producer C retains the old map and emits earlier update A to partition 3. Consumers read partitions independently and apply B then A, reverting state. Both partitions are internally ordered and healthy.

Containment pauses the key and reconstructs source commit order. Repair replays from the authoritative source version. Prevention uses the migration barrier/new key epoch, rejects old assignment generations after cutover, and makes consumers version-gate state. Monitoring compares routing generation and detects the same key active under two partitions/epochs.

## Operations and Observability

Track by ordered scope, partition, writer epoch, assignment, and consumer:

- append rate and sequence/epoch rejection;
- missing ranges, oldest gap age, buffered events/bytes, and repair outcome;
- last appended, replicated, delivered, and applied positions;
- hot-key share and serial service demand;
- producer routing-generation age and wrong-key/missing-key rejection;
- consumer rebalance duration, overlapping ownership, and stale checkpoint attempts;
- reshard barrier progress and keys active in multiple epochs;
- batch completion latency and open transaction bytes.

Runbooks cover stale producer epoch, permanent gap, hot key, consumer checkpoint regression, incomplete batch, and reshard rollback. Logs retain event ID, scope, epoch, sequence, partition, assignment generation, source version, and trace correlation without exposing payloads.

## Incident Playbook

Each entry below starts from the evidence that separates the plausible causes, because ordering symptoms are heavily aliased: a stale writer, a lost prefix, and a config regression all look like "old data came back". Order matters while a strict scope is blocked, since every minute of waiting widens the reorder buffer and the eventual repair.

### A gap will not close

Classify whether the missing sequence should exist before you fetch anything: publisher abort, an attribute filter that skipped it, retention that already reclaimed it, and a genuinely delayed delivery all present as the same blocked prefix. Compare the requested range against the log start offset or stream first position; if the missing record is below the retained floor, replay cannot help and only an authoritative source snapshot plus suffix can. Keep applying nothing later in that scope while you decide, then record the outcome as a named policy (repaired, rebuilt, or skipped with approval). Do not "wait thirty seconds and continue" on a ledger scope, and do not let a generic retry loop become the gap policy: it converts a bounded repair into silent eventual divergence that no test reproduces.

### State for one key moved backward

Read the epoch and routing generation attached to the two writes before touching application logic. Two healthy partitions each internally ordered is the signature of a cutover without a barrier: the key was writable under two assignment generations at once. Confirm by checking whether producers hold stale maps (routing-generation age) and whether consumers accepted records whose epoch is older than the current authoritative one. Contain by fencing the old generation at acceptance so late writes are rejected rather than applied, then repair from the authoritative source version. Do not "fix" it by rewriting sequence numbers or bumping a counter to outrun the stale write; that destroys the audit trail that tells you which consumer lane leaked, and the next reshard repeats the incident.

### Reorder buffer grows while throughput looks healthy

Separate one wedged scope from broad reordering by charting buffered bytes against the oldest gap age per scope. A single high-value entity pinned at a nonzero gap with everything else draining is a routing or sequencer problem for that key; uniform growth across scopes usually means a producer far ahead of a slow consumer lane, or a batch/transaction marker that never arrives. Check hot-key share next: if one key owns most of the serial demand, buffering more is not a fix because the lane cannot drain faster than its sequencer. Decide explicitly among coalescing state-setting events, splitting the invariant into subkeys, or admitting less work for that key. Do not raise the buffer ceiling as the response; it moves the failure from a stall to an out-of-memory kill and hides the scope that is blocked.

### Reordering appeared after a client or config change

Diff effective producer configuration against the previous deployment rather than reading the intended values: idempotence disabled by an `acks` override, adaptive partitioning enabled, retries raised above the in-flight ceiling, or a library upgrade that changed the default partitioner are all ordering regressions with healthy-looking brokers. Prove it by replaying one keyed session and comparing append order to production order, then roll the config back before adding consumer-side defenses. Verify recovery by watching duplicate-and-reorder counters return to baseline on new traffic only, not over the whole window. Do not add a downstream sort or timestamp-based reconciliation to absorb it; timestamps are not causality, and you will be carrying that reconciler after the config is fixed.

### A reshard left two readable positions for one key

Establish the pivot explicitly: the sealed last position in the old assignment and the first epoch/position in the new one, then confirm which consumers can still observe both. For shard-graph products, walk parent to child per partition key rather than per shard, and gate the child on the parent reaching its end iterator. Translate cursors using retained metadata for as long as any consumer may replay across the boundary; if translation is gone, rebuild affected state from a snapshot instead of stitching two orders. Do not dual-produce to both sides "until consumers catch up": that creates two incomparable total orders and guarantees every consumer picks a different one under failure.

## Security and Integrity

Only the authoritative sequencer may set epoch/sequence fields. Brokers reject publisher-supplied values outside its assigned producer session or domain stream. Sign or authenticate cross-region/source replication so an attacker cannot inject a higher epoch or enormous sequence to block a consumer.

Limit key length, buffer span, batch size, dependency count, and causal metadata. Tenant boundaries apply to routing, diagnostic gap APIs, replay, and logs. Sequence numbers can reveal activity volume; treat them as metadata subject to disclosure policy.

## Verification Strategy

- model-test sequence/epoch/reorder transitions against a simple ordered map;
- differential-test optimized buffers and batch application with exhaustive sorted replay;
- partition old/new leaders and prove stale epochs cannot append or checkpoint;
- delay, duplicate, omit, and reorder events while checking each declared gap policy;
- rebalance consumers at every checkpoint boundary;
- reshard keys with delayed producers holding every old routing generation;
- load-test real key skew and prove serial hot-key capacity assumptions;
- fuzz huge sequence jumps, dependency lists, and incomplete batches.

## Decision Framework

Choose the weakest relation that preserves the domain invariant:

- no order for independent commutative facts;
- version gating for state replacements;
- per-key total order for aggregate transitions;
- explicit predecessor/causal metadata for cross-stream dependencies;
- global order only for truly global invariants.

Then answer:

1. Which events must be comparable and why?
2. Who assigns sequence and epoch after source commit?
3. What makes stale writers and checkpoints powerless?
4. What is a valid gap, and which source repairs it?
5. Can event semantics commute or version-gate instead of buffering?
6. What is the hot-scope serial throughput ceiling?
7. How does rebalancing/resharding establish one cutover pivot?

## References

- [Leslie Lamport: Time, Clocks, and the Ordering of Events in a Distributed System](https://lamport.azurewebsites.net/pubs/time-clocks.pdf)
- [Colin J. Fidge: Timestamps in Message-Passing Systems That Preserve the Partial Ordering](https://doi.org/10.5555/647342.724880)
- [Apache Kafka: Message Delivery and Ordering](https://kafka.apache.org/documentation/#semantics)
- [Apache Kafka Protocol: Control Plane and Leader Epochs](https://kafka.apache.org/protocol.html)
- [CloudEvents Specification](https://github.com/cloudevents/spec)
- [Apache Kafka: Producer Configuration](https://kafka.apache.org/documentation/#producerconfigs)
- [Google Cloud Pub/Sub: Message Ordering](https://cloud.google.com/pubsub/docs/ordering)
- [Amazon Kinesis Data Streams: Completing a Resharding Action](https://docs.aws.amazon.com/streams/latest/dev/kinesis-using-sdk-java-after-resharding.html)

---

## Related Chapters

- [Kafka](../09-whitepapers/06-kafka.md): a partition is the ordering domain and the offset is the sequence number
- [Delivery Guarantees and Effect Boundaries](04-delivery-guarantees.md): ordering without deduplication still permits duplicates, which surprises most consumers
- [Collaborative Document Replication and CRDT Sync Engines](../07-real-time/07-crdts-collaborative-editing.md): convergence as the alternative when a total order is not available at any price
- [Idempotency and Operation Identity](../01-foundations/08-idempotency.md): the practical substitute for exactly-once, and the only one a broker can verify
