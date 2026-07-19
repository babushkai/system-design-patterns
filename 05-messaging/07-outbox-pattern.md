# Transactional Outbox, Inbox, and CDC Publication

The transactional outbox closes one specific dual-write gap: domain state and an intent to publish are committed in the same local transaction. A relay then delivers that durable intent to a broker at least once. The transactional inbox closes the corresponding consumer gap by committing message identity, local effect, and progress in one local transaction. Together they provide recoverable publication and effectively-once local effects without pretending two independent systems share one atomic commit.

This chapter is the canonical treatment of outbox/inbox mechanics and CDC-based relay. [Delivery Guarantees](04-delivery-guarantees.md) defines the end-to-end semantics; [Change Data Capture](../13-data-pipelines/04-change-data-capture.md) owns general database-log architecture, snapshots, and schema capture.

## Workload and contract

The producer transaction is:

```text
BEGIN
  mutate domain rows
  insert outbox event with stable identity and aggregate/source version
COMMIT
```

The asynchronous path is:

```text
committed outbox row
  -> poller or CDC reader
  -> broker append with stable event identity
  -> consumer delivery
  -> inbox + local effect + checkpoint transaction
  -> broker acknowledgement
```

Define:

- which local transaction makes the event eligible;
- event identity, aggregate/source version, partition key, schema, and destinations;
- relay durability, retry, lease, ordering, and deletion/retention policy;
- maximum commit-to-broker freshness and tolerated outage backlog;
- consumer idempotency/effect scope and inbox retention;
- CDC slot/log retention and bootstrap/recovery behavior;
- payload ownership, size, privacy, encryption, and claim-check lifecycle;
- reconciliation from domain state to outbox, broker, and consumer effects.

An outbox guarantees **no committed domain transition is missing its publication intent** if application code writes both correctly. It does not guarantee one broker record, one delivery, one consumer attempt, or one external effect.

## State and invariants

Producer-side state includes domain rows/event stream, outbox rows, relay claim/attempt state or CDC position, broker publication evidence, and cleanup watermark. Consumer-side state includes inbox identities, domain/projection effects, source/entity versions, and checkpoints.

Enforce:

**Local atomicity.** Domain mutation and outbox insert both commit or both roll back.

**Stable identity.** Every relay retry and CDC restart preserves `event_id`; consumers do not depend on a transport-assigned ID.

**Source provenance.** Outbox records identify aggregate/entity, source version or commit position, event type/schema, actor/tenant, and transaction/causation context.

**Claim fencing.** A polling relay marks success only while holding the current claim generation; a stale worker cannot delete/mark another worker’s row.

**Deletion follows durable evidence.** Rows are not reclaimed until the chosen relay can restart/reconcile without losing unpublished intent.

**Inbox and local effect commit together.** A duplicate inbox identity proves the corresponding local transaction committed.

**Retention horizons align.** Outbox history, CDC log/slot, broker replay, inbox identity, and external idempotency cover the declared recovery window.

## Outbox schema and write path

A minimal logical record contains:

```text
outbox_id / event_id            immutable globally unique identity
aggregate_type, aggregate_id    routing and provenance
aggregate_version               domain ordering/version gate
event_type, schema_id           contract
payload                         immutable bytes or claim-check reference
headers                         bounded typed metadata
created_at / commit_position    freshness and source order
destination_contract            logical route, not necessarily vendor topic
status/claim fields             only for polling relay
```

Enforce uniqueness on `event_id` and, where appropriate, `(aggregate_id, aggregate_version, event_type)`. The latter is domain-specific; one aggregate version may legitimately emit multiple event types.

The application does not publish after commit as a “best effort.” It writes the outbox in the same repository/unit-of-work call as the state mutation. Tests should fail if a state transition requiring publication lacks an outbox record. Database constraints or event-sourced append APIs can make this harder to omit.

Keep outbox rows append-only in semantic fields. Relay status is mutable operational state. Updating payload after commit makes replay nondeterministic and can let a compromised relay alter domain facts.

Payloads should be self-contained enough for consumers to act without synchronously calling the producer for mutable state. Reference-only events reintroduce temporal coupling: by fetch time the entity may have changed or disappeared. For large data, use an immutable content-addressed claim-check object with digest, schema, encryption metadata, and retention tied to the outbox/broker contract.

## Polling relay protocol

A scalable poller uses leased claims:

1. select a bounded batch of eligible rows ordered within the required scope, skipping rows locked/claimed by live workers;
2. atomically set `claim_owner`, increment `claim_generation`, and set `claim_until`;
3. publish each event using stable `event_id` and partition key;
4. wait for the broker durability acknowledgement;
5. mark published only if owner/generation still match;
6. release or let the lease expire after ambiguous failure;
7. clean up later under a separate watermark.

The publish/mark step cannot be atomic across database and broker. If publish succeeds and marking fails, the row is republished. This is why stable IDs and consumer inbox/effect idempotency are mandatory. Marking before broker acknowledgement creates loss.

Batching database claims and broker appends improves throughput, but preserve per-aggregate sequence. Publishing rows concurrently can reorder versions even if selected in order. Route each aggregate to one ordered lane/partition or gate completion/claim by preceding version. Cross-aggregate global order is normally not required.

Use absolute retry deadlines and backpressure from broker health. A relay outage grows the outbox; unbounded retries should not hold database transactions open. The relay claims, commits, then performs network I/O.

Polling indexes should cover eligibility without indexing large payloads. Partition rows by creation/commit range so cleanup drops sealed partitions rather than issuing huge deletes. Measure vacuum/compaction and primary-cache impact.

## CDC relay protocol

Log-based CDC reads committed changes from the database’s replication/change log. Because the outbox insert shares the domain transaction, the log exposes it only on commit and in database commit order. A connector transforms the row into the public envelope and publishes it.

CDC state includes source database identity/timeline, log position, transaction boundaries, schema history, connector generation, destination partition mapping, and last committed destination/checkpoint. Restart can replay from the last committed source position, so duplicates remain possible.

The database log is not infinite. A replication slot or equivalent retention mechanism can pin WAL/binlog while the connector is down and fill the primary disk. Cap and alert based on retained bytes/time and disk reserve. If the slot is lost or falls behind retained history, recovery needs an authoritative outbox/backfill or snapshot-plus-log protocol; “create a new slot at latest” silently loses publications.

Deleting outbox table rows does not necessarily remove their already-recorded log entries, but cleanup policy must account for connector recovery. If CDC position is lost and rows were deleted, there may be no source for replay. Keep a publication archive/reconciliation path or treat slot/checkpoint backup as tier-zero state.

CDC conveys database row schema; the public event contract is the outbox payload/schema, not arbitrary table-change events. Publishing raw domain-table CDC across team boundaries couples consumers to storage schema and can expose rolled-up implementation details. Use raw CDC for owned projections and explicit outbox events for integration intent.

## Polling versus CDC

Choose polling when volume is moderate, operational simplicity and database portability matter, and a well-indexed work table fits the primary. Choose CDC when publication volume/latency justifies operating log readers, source-log retention, schema history, connector recovery, and transactional ordering.

The outbox contract remains stable across both. Migration can run old and new relays in shadow using the same event IDs. Compare event identity/position coverage, then fence one publisher generation before activating the other. Running both without destination deduplication produces duplicates by design.

CDC is not inherently “exactly once.” Source read, destination append, and source checkpoint still have crash windows unless they share a proven transaction protocol; downstream effect idempotency remains required.

## Transactional inbox and local effects

For each consumer/effect scope, create an inbox with a unique key:

```text
BEGIN
  INSERT inbox(consumer_name, event_id, request_digest, source_position)
  -- if exact duplicate already committed: return prior outcome
  validate entity/source version and gap policy
  mutate local domain/projection rows
  update contiguous checkpoint or entity version
  insert local outbox records for further publication if needed
COMMIT
ack broker
```

This composes across services: each local transaction consumes an inbox event, changes local state, and emits its own outbox fact. It does not create one global transaction, but every boundary is durable and replayable.

The inbox key includes consumer/effect contract because two independent projections should each apply the same event. Bind the ID to a payload/request digest so an attacker or bug cannot reuse an ID with different content.

Inbox cleanup uses the maximum of broker replay, operator redrive, disaster recovery, and upstream outbox retention. Natural business uniqueness or entity version may allow compacting identities. Never delete inbox state merely because the broker’s normal retry window is short if operators can replay months later.

If an effect is in a remote system, the local inbox transaction can record an intent/outbox to call it; it cannot mark the remote effect complete. Pass a stable downstream idempotency key and reconcile ambiguous outcomes, or use a durable workflow.

## Ordering and transaction boundaries

Database commit position provides a total order within its source log, but broker partitioning may preserve only per-key order. Carry aggregate version and route consistently. Consumers should version-gate or repair gaps instead of depending on wall-clock timestamps.

One database transaction can emit multiple outbox rows. Include a transaction/batch ID and indices if consumers require atomic group visibility. Otherwise events to different partitions can arrive independently. Do not imply a cross-aggregate atomic view unless the destination protocol/consumer supports it.

Multiple source databases have no shared commit order. A relay timestamp does not create causality. Model cross-database processes as workflows with explicit correlation/dependencies, or consolidate the invariant into one transaction boundary.

## Cleanup and archival

Separate status update from physical deletion. A cleanup watermark advances only when:

- the relay has durable publication evidence/checkpoint beyond the rows;
- no active claim can still mark them;
- the recovery/reconciliation window is satisfied;
- required audit/legal/privacy policy is applied;
- external claim-check payload ownership is transferred or expired safely.

Time/range partitions make reclamation predictable. Archive only if there is a real replay/audit requirement; otherwise the broker or authoritative domain store may be the archive. Every extra archive becomes another sensitive copy and recovery dependency.

Periodic reconciliation samples or scans domain transitions requiring events and verifies corresponding outbox IDs, broker identities, consumer source versions, and effect invariants. Count equality is insufficient because duplicates can hide omissions.

## Schema and relay migration

Outbox event schemas evolve like public APIs. Producers write registered schema IDs. Relays preserve identity/type and do not apply mutable “latest” transforms without version provenance. Consumers deploy compatible readers before producers emit new required semantics.

Relay changes use generations:

1. pin source positions and old/new relay artifacts;
2. shadow publish to an isolated destination or compare envelopes before send;
3. reconcile IDs, routing keys, schemas, and order;
4. fence old relay generation;
5. activate new generation from a declared source position;
6. keep rollback checkpoint and artifacts;
7. monitor duplicates/gaps through the retention window.

Changing partition keys is an ordering migration requiring a barrier as described in the ordering chapter.

## Capacity and cost model

Illustrative producer database:

- 18,000 domain transactions/s;
- 35% create one outbox event;
- average outbox row 1.4 KiB including indexes;
- polling batch 500 rows;
- broker acknowledgement batch latency measured at 12 ms p50, 80 ms p99;
- broker outage budget 6 hours;
- three database replicas.

Outbox rate is 6,300/s and logical growth about 8.6 MiB/s. Six hours of outage accumulates 136 million rows and roughly 185 GiB logical before table/index/WAL amplification; three replicas plus WAL can multiply primary storage substantially. Preallocate reserve and test catch-up impact.

At 500 rows/batch, steady polling needs 12.6 batches/s. If one worker holds one batch until broker confirmation, p99 80 ms permits only 12.5 batches/s—already near steady rate. Parallel leased batches are needed, but per-key order and database/broker budgets cap concurrency.

To drain the six-hour backlog in three hours while live ingress continues, relay throughput must be `6,300 + 136M/10,800`, about 18,900 events/s. Broker and consumer capacity must support catch-up without overload.

Inbox storage at 6,300/s and 30-day replay is 16.3 billion identities. At a measured 72 bytes/identity, one logical copy is about 1.09 TiB. Partition/compact by time and effect scope, use natural uniqueness where valid, and align replay policy deliberately.

## Concrete failure trace: lost CDC slot after cleanup

A CDC relay is down for two days. The replication slot is accidentally dropped, and an aggressive job has already deleted published-looking outbox rows older than one day. A new slot starts at the current WAL position. Domain transactions from the missing day have neither replayable rows nor retained log entries, so their integration events disappear silently.

Containment stops the connector and downstream claims of completeness. Repair reconstructs required events from the authoritative domain/event store using stable deterministic IDs and publishes through a controlled backfill. Prevention backs up/checks connector source positions, ties cleanup watermark to recoverable publication evidence rather than row age, monitors retained WAL, and rehearses slot-loss recovery.

## Operations and observability

Track by source DB/shard, destination, relay generation, event type, consumer, and tenant:

- outbox insert rate/bytes and transactions missing required event intents;
- oldest unpublished/claimed row age, claim expiries, attempts, and backlog drain ETA;
- broker publish acknowledgement, timeout, duplicates, and routing/schema rejects;
- CDC source position, slot/log retained bytes, transaction lag, schema-history health;
- cleanup watermark, table/index/WAL bytes, partitions eligible/blocked for deletion;
- inbox duplicate/conflict rate, apply/checkpoint latency, and retention horizon;
- end-to-end source commit-to-effect lag and source/effect reconciliation mismatch;
- claim-check missing/orphaned objects and deletion progress.

Runbooks cover relay outage, broker ambiguity, database overload, stuck claims, CDC slot/log loss, schema rejection, duplicate storm, cleanup error, and replay beyond inbox retention.

## Security and privacy

Application roles may insert outbox rows only through owned domain transactions. Relay roles read/claim/publish but cannot mutate domain state or alter payloads. CDC connectors receive least-privilege replication access and isolate schema-history/checkpoint credentials.

Authenticate destination routes from trusted configuration, not payload-supplied topic names. Validate event type/schema, tenant, size, compression ratio, headers, and claim-check digest. Encrypt outbox/inbox, WAL archives, broker, and payload objects according to data classification.

Replay, seek, cleanup, connector reset, and bulk redrive are privileged audited operations. An event ID is not a secret, but allowing callers to choose another tenant’s ID can suppress effects through inbox uniqueness; namespace and bind identity to authenticated source plus digest.

## Verification strategy

- transaction tests prove every committed domain transition has its required outbox row and rollbacks publish none;
- kill pollers before/after broker acknowledgement and mark-published to prove duplicates but no loss;
- partition/fence poller claim generations and relay generations;
- restart CDC from checkpoints, lose a slot/log range, and execute the documented backfill;
- crash consumers before/after inbox/effect/checkpoint commit;
- replay beyond normal windows and verify retention contracts or explicit rejection;
- compare domain/outbox/broker/inbox/effect IDs and source versions under reconciliation;
- load-test six-hour outage accumulation plus bounded catch-up against primary/broker/downstream capacity.

## Decision framework

Use outbox when a local transaction must reliably cause asynchronous publication to another transactional domain. Use a simple local job table if the work remains inside the same database/service and a broker adds no value. Consider distributed transactions only when every participant supports the protocol and blocking/availability trade-offs fit; see [Distributed Transactions](../02-distributed-databases/07-distributed-transactions.md).

Choose relay and consumer design by answering:

1. What exact local transaction creates publication intent?
2. Which stable identity/version/order fields survive every retry?
3. Does polling or CDC fit volume and operational capability?
4. What outage backlog and catch-up rate can every tier sustain?
5. When can rows/logs/inbox identities/payloads be safely reclaimed?
6. How does each consumer commit inbox, effect, and progress atomically?
7. What source reconstructs missing publications after catastrophic relay-state loss?

## References

- [Chris Richardson: Transactional Outbox Pattern](https://microservices.io/patterns/data/transactional-outbox.html)
- [Debezium: Outbox Event Router](https://debezium.io/documentation/reference/stable/transformations/outbox-event-router.html)
- [PostgreSQL: Logical Decoding Concepts](https://www.postgresql.org/docs/current/logicaldecoding-explanation.html)
- [PostgreSQL: Replication Slots](https://www.postgresql.org/docs/current/warm-standby.html#STREAMING-REPLICATION-SLOTS)
- [AWS Prescriptive Guidance: Transactional Outbox Pattern](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html)
- [Martin Kleppmann et al.: Online Event Processing—Achieving Consistency Where Distributed Transactions Have Failed](https://doi.org/10.1145/3329672.3329679)
