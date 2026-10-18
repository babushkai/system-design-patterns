# Transactional Outbox, Inbox, and CDC Publication

## TL;DR

The transactional outbox closes one specific dual-write gap: domain state and an intent to publish are committed in the same local transaction. A relay then delivers that durable intent to a broker at least once. The transactional inbox closes the corresponding consumer gap by committing message identity, local effect, and progress in one local transaction. Together they provide recoverable publication and effectively-once local effects without pretending two independent systems share one atomic commit.

Outbox/inbox design includes transactional rows, polling or CDC relay, and consumer effect commit. [Delivery Guarantees](04-delivery-guarantees.md) defines end-to-end semantics; [Change Data Capture](../13-data-pipelines/04-change-data-capture.md) covers general log architecture, snapshots, and schema capture.

## Workload and Contract

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
  -> broker acknowledgment
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

## State and Invariants

Producer-side state includes domain rows/event stream, outbox rows, relay claim/attempt state or CDC position, broker publication evidence, and cleanup watermark. Consumer-side state includes inbox identities, domain/projection effects, source/entity versions, and checkpoints.

Enforce:

**Local atomicity.** Domain mutation and outbox insert both commit or both roll back.

**Stable identity.** Every relay retry and CDC restart preserves `event_id`; consumers do not depend on a transport-assigned ID.

**Source provenance.** Outbox records identify aggregate/entity, source version or commit position, event type/schema, actor/tenant, and transaction/causation context.

**Claim fencing.** A polling relay marks success only while holding the current claim generation; a stale worker cannot delete/mark another worker’s row.

**Deletion follows durable evidence.** Rows are not reclaimed until the chosen relay can restart/reconcile without losing unpublished intent.

**Inbox and local effect commit together.** A duplicate inbox identity proves the corresponding local transaction committed.

**Retention horizons align.** Outbox history, CDC log/slot, broker replay, inbox identity, and external idempotency cover the declared recovery window.

## Outbox Schema and Write Path

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

## Polling Relay Protocol

A scalable poller uses leased claims:

1. select a bounded batch of eligible rows ordered within the required scope, skipping rows locked/claimed by live workers;
2. atomically set `claim_owner`, increment `claim_generation`, and set `claim_until`;
3. publish each event using stable `event_id` and partition key;
4. wait for the broker durability acknowledgment;
5. mark published only if owner/generation still match;
6. release or let the lease expire after ambiguous failure;
7. clean up later under a separate watermark.

The publish/mark step cannot be atomic across database and broker. If publish succeeds and marking fails, the row is republished. This is why stable IDs and consumer inbox/effect idempotency are mandatory. Marking before broker acknowledgment creates loss.

Batching database claims and broker appends improves throughput, but preserve per-aggregate sequence. Publishing rows concurrently can reorder versions even if selected in order. Route each aggregate to one ordered lane/partition or gate completion/claim by preceding version. Cross-aggregate global order is normally not required.

Use absolute retry deadlines and backpressure from broker health. A relay outage grows the outbox; unbounded retries should not hold database transactions open. The relay claims, commits, then performs network I/O.

Polling indexes should cover eligibility without indexing large payloads. Partition rows by creation/commit range so cleanup drops sealed partitions rather than issuing huge deletes. Measure vacuum/compaction and primary-cache impact.

## CDC Relay Protocol

Log-based CDC reads committed changes from the database’s replication/change log. Because the outbox insert shares the domain transaction, the log exposes it only on commit and in database commit order. A connector transforms the row into the public envelope and publishes it.

CDC state includes source database identity/timeline, log position, transaction boundaries, schema history, connector generation, destination partition mapping, and last committed destination/checkpoint. Restart can replay from the last committed source position, so duplicates remain possible.

The database log is not infinite. A replication slot or equivalent retention mechanism can pin WAL/binlog while the connector is down and fill the primary disk. Cap and alert based on retained bytes/time and disk reserve. If the slot is lost or falls behind retained history, recovery needs an authoritative outbox/backfill or snapshot-plus-log protocol; “create a new slot at latest” silently loses publications.

Deleting outbox table rows does not necessarily remove their already-recorded log entries, but cleanup policy must account for connector recovery. If CDC position is lost and rows were deleted, there may be no source for replay. Keep a publication archive/reconciliation path or treat slot/checkpoint backup as tier-zero state.

CDC conveys database row schema; the public event contract is the outbox payload/schema, not arbitrary table-change events. Publishing raw domain-table CDC across team boundaries couples consumers to storage schema and can expose rolled-up implementation details. Use raw CDC for owned projections and explicit outbox events for integration intent.

## Polling versus CDC

Choose polling when volume is moderate, operational simplicity and database portability matter, and a well-indexed work table fits the primary. Choose CDC when publication volume/latency justifies operating log readers, source-log retention, schema history, connector recovery, and transactional ordering.

The outbox contract remains stable across both. Migration can run old and new relays in shadow using the same event IDs. Compare event identity/position coverage, then fence one publisher generation before activating the other. Running both without destination deduplication produces duplicates by design.

CDC is not inherently “exactly once.” Source read, destination append, and source checkpoint still have crash windows unless they share a proven transaction protocol; downstream effect idempotency remains required.

## Transactional Inbox and Local Effects

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

## Ordering and Transaction Boundaries

Database commit position provides a total order within its source log, but broker partitioning may preserve only per-key order. Carry aggregate version and route consistently. Consumers should version-gate or repair gaps instead of depending on wall-clock timestamps.

One database transaction can emit multiple outbox rows. Include a transaction/batch ID and indices if consumers require atomic group visibility. Otherwise events to different partitions can arrive independently. Do not imply a cross-aggregate atomic view unless the destination protocol/consumer supports it.

Multiple source databases have no shared commit order. A relay timestamp does not create causality. Model cross-database processes as workflows with explicit correlation/dependencies, or consolidate the invariant into one transaction boundary.

## Cleanup and Archival

Separate status update from physical deletion. A cleanup watermark advances only when:

- the relay has durable publication evidence/checkpoint beyond the rows;
- no active claim can still mark them;
- the recovery/reconciliation window is satisfied;
- required audit/legal/privacy policy is applied;
- external claim-check payload ownership is transferred or expired safely.

Time/range partitions make reclamation predictable. Archive only if there is a real replay/audit requirement; otherwise the broker or authoritative domain store may be the archive. Every extra archive becomes another sensitive copy and recovery dependency.

Periodic reconciliation samples or scans domain transitions requiring events and verifies corresponding outbox IDs, broker identities, consumer source versions, and effect invariants. Count equality is insufficient because duplicates can hide omissions.

## Schema and Relay Migration

Outbox event schemas evolve like public APIs. Producers write registered schema IDs. Relays preserve identity/type and do not apply mutable “latest” transforms without version provenance. Consumers deploy compatible readers before producers emit new required semantics.

Relay changes use generations:

1. pin source positions and old/new relay artifacts;
2. shadow publish to an isolated destination or compare envelopes before send;
3. reconcile IDs, routing keys, schemas, and order;
4. fence old relay generation;
5. activate new generation from a declared source position;
6. keep rollback checkpoint and artifacts;
7. monitor duplicates/gaps through the retention window.

Changing partition keys requires the barrier protocol in [Message Ordering](03-message-ordering.md).

## Product Knob Map

An outbox is small enough that its settings look incidental, and they are not: they decide whether an unpublished intent can still be published after a stall, and whether a stalled consumer can damage the primary. The knobs below are grouped by the failure each one bounds. Read them from the live instance before trusting a runbook written for another version.

| Product and scope | Knob | Invariant it enforces | Default and how it misleads |
| --- | --- | --- | --- |
| PostgreSQL logical replication slot | `max_slot_wal_keep_size`, `max_replication_slots` | a stalled reader cannot silently consume the primary's disk | unbounded by default, so a connector down for days pins WAL until the primary fills; set a ceiling and accept that exceeding it invalidates the slot, which is a correctness event requiring backfill rather than a warning to silence. |
| Debezium PostgreSQL connector | `slot.drop.on.stop`, `snapshot.mode`, publication auto-create behavior | restarting a relay resumes from its source position instead of rewriting history | dropping the slot on shutdown turns a routine redeploy into a fresh snapshot, and a table created after the publication was defined may not be published at all, so events for the new aggregate simply never appear while the connector reports running. |
| PostgreSQL polling relay | partial index on eligibility plus `SKIP LOCKED`, `idle_in_transaction_session_timeout` | claim cost stays bounded and no transaction outlives its network call | without a partial index every poll scans published rows; holding the transaction open across broker I/O pins the oldest xmin, so autovacuum cannot reclaim deleted rows and, sustained long enough, drives the cluster toward xid wraparound. Claim, commit, then do network I/O. |
| Apache Kafka producer used as the relay | `max.block.ms`, `buffer.memory`, `delivery.timeout.ms`, `linger.ms` | broker backpressure surfaces as relay latency, not lost or unbounded blocking work | a full buffer blocks the caller for up to `max.block.ms`, which is harmless only if that call is outside a database transaction; `delivery.timeout.ms` ends retries silently, so the relay must record the failure rather than assume eventual success. |
| Kafka Connect or similar relay runtime | `errors.tolerance`, `errors.limit`, dead-letter topic settings | a rejected record is retained with its source position | tolerance defaults to none, so one bad record parks the task and everything behind it; teams respond by enabling tolerance without a dead-letter topic, which converts a loud stall into records that vanish with only a log line as evidence. |
| DynamoDB Streams or similar time-bounded change feeds | stream retention horizon (a fixed 24 hours for DynamoDB Streams) | the relay can resume from any outage shorter than the recovery objective | the feed expires on a clock regardless of your consumer's health, so an outage past the horizon means a full authoritative rescan, not a resume. A relay designed as "just restart it" fails quietly exactly once per incident. |
| Cleanup job against the outbox table or partitions | delete batch size, partition detach-and-drop, vacuum/autovacuum settings, undo history limits | reclamation never competes with production traffic for the primary | a single large `DELETE` by age produces a WAL and lock burst, replica lag spike, and purge/vacuum backlog that outlasts the incident; time-range partitions detached concurrently are what make reclamation boring. Age is also the wrong predicate: it says nothing about publication evidence. |
| Inbox table on the consumer side | unique `(consumer_name, event_id)` with request digest, `ON CONFLICT` handling, time-partitioned retention | a duplicate proves a committed local transaction and never masks a different payload | `ON CONFLICT DO NOTHING` returns "no row inserted" for every conflict including same-id-different-content, so bind and compare a digest instead of trusting the absence of an error. Dropping old inbox partitions before the redrive and disaster-recovery horizon reopens duplicates that were already considered settled. |

```bash
# read-only: can this outbox still honor its recovery window?
psql "$SOURCE_DB" -c "SELECT slot_name, active, restart_lsn,
                             pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) AS retained
                      FROM pg_replication_slots"
psql "$SOURCE_DB" -c "SELECT max(age(datfrozenxid)) AS max_xid_age FROM pg_database"
psql "$SOURCE_DB" -c "SELECT status, count(*) AS rows, now() - min(created_at) AS oldest
                      FROM outbox GROUP BY status ORDER BY oldest DESC"
psql "$SOURCE_DB" -c "SELECT indexdef FROM pg_indexes WHERE tablename = 'outbox'"
curl -s "$KAFKA_CONNECT/connectors/orders-outbox/config" \
  | jq '{tolerance:."errors.tolerance", dlq:."errors.deadletterqueue.topic.name", limit:."errors.limit"}'
mysql -e "SHOW ENGINE INNODB STATUS\G" | grep -i 'History list length'   # purge backlog
```

## Capacity and Cost Model

Illustrative producer database:

- 18,000 domain transactions/s;
- 35% create one outbox event;
- average outbox row 1.4 KiB including indexes;
- polling batch 500 rows;
- broker acknowledgment batch latency measured at 12 ms p50, 80 ms p99;
- broker outage budget 6 hours;
- three database replicas.

Outbox rate is 6,300/s and logical growth about 8.6 MiB/s. Six hours of outage accumulates 136 million rows and roughly 185 GiB logical before table/index/WAL amplification; three replicas plus WAL can multiply primary storage substantially. Preallocate reserve and test catch-up impact.

At 500 rows/batch, steady polling needs 12.6 batches/s. If one worker holds one batch until broker confirmation, p99 80 ms permits only 12.5 batches/s, already near the steady rate. Parallel leased batches are needed, but per-key order and database/broker budgets cap concurrency.

To drain the six-hour backlog in three hours while live ingress continues, relay throughput must be `6,300 + 136M/10,800`, about 18,900 events/s. Broker and consumer capacity must support catch-up without overload.

Inbox storage at 6,300/s and 30-day replay is 16.3 billion identities. At a measured 72 bytes/identity, one logical copy is about 1.09 TiB. Partition/compact by time and effect scope, use natural uniqueness where valid, and align replay policy deliberately.

## Concrete Failure Trace: Lost CDC Slot After Cleanup

A CDC relay is down for two days. The replication slot is accidentally dropped, and an aggressive job has already deleted published-looking outbox rows older than one day. A new slot starts at the current WAL position. Domain transactions from the missing day have neither replayable rows nor retained log entries, so their integration events disappear silently.

Containment stops the connector and downstream claims of completeness. Repair reconstructs required events from the authoritative domain/event store using stable deterministic IDs and publishes through a controlled backfill. Prevention backs up/checks connector source positions, ties cleanup watermark to recoverable publication evidence rather than row age, monitors retained WAL, and rehearses slot-loss recovery.

## Operations and Observability

Track by source DB/shard, destination, relay generation, event type, consumer, and tenant:

- outbox insert rate/bytes and transactions missing required event intents;
- oldest unpublished/claimed row age, claim expiries, attempts, and backlog drain ETA;
- broker publish acknowledgment, timeout, duplicates, and routing/schema rejects;
- CDC source position, slot/log retained bytes, transaction lag, schema-history health;
- cleanup watermark, table/index/WAL bytes, partitions eligible/blocked for deletion;
- inbox duplicate/conflict rate, apply/checkpoint latency, and retention horizon;
- end-to-end source commit-to-effect lag and source/effect reconciliation mismatch;
- claim-check missing/orphaned objects and deletion progress.

Runbooks cover relay outage, broker ambiguity, database overload, stuck claims, CDC slot/log loss, schema rejection, duplicate storm, cleanup error, and replay beyond inbox retention.

## Incident Playbook

Outbox incidents split into two very different problems: unpublished intent that still exists somewhere recoverable, and intent whose only evidence has been reclaimed. Determine which one you have before taking any action, because the recovery for the first is a throughput decision and the second is a reconstruction project.

### Backlog grows while the database looks healthy

Check relay concurrency against the ordering constraint before adding workers: per-aggregate order means rows for one aggregate cannot be published out of version order, so a relay that appears idle may be deliberately serialized behind one hot aggregate. Read claim expiries and attempt counts next; expiring leases mean relays are losing work rather than failing to find it, which points at broker latency or lease duration shorter than the p99 publish path. Compute drain time as backlog divided by surplus relay capacity and post the number, then raise concurrency only if the saturated resource is the relay process itself. Do not delete or reset claims to "unstick" rows: leases must expire on their own, and forcing them lets two relays publish the same row while discarding the attempt history you need to explain duplicates downstream.

### A broker outage ended and duplicates appeared

This is the designed outcome of publish-then-mark, so verify the safety net rather than hunting a bug: confirm relay retries preserved `event_id`, that consumers rejected duplicates at the inbox rather than suppressing them with a filter, and that the duplicate rate decays as in-flight rows are marked. Then check the second-order effects, because a duplicate storm still consumes broker, lookup, and transaction capacity even when effects converge. Do not regenerate event identities to "clean" the window: consumers key their own state on those ids, and minting new ones converts an idempotent replay into a fresh set of effects, which is the one outcome this design exists to prevent.

### A CDC slot was lost or invalidated and a window is missing

Stop every downstream claim of completeness first, then bound the hole precisely: last committed source position plus timestamp, current position, and whether outbox rows for the gap still exist. If they do, republish through a controlled backfill with the original deterministic ids; if they do not, reconstruct from the authoritative domain or event store, because a new slot started at the current log position publishes the future and silently abandons the past. Announce the affected range to consumers so their reconciliation can act on it. Do not treat "connector is green again" as recovery, and do not let a re-snapshot stand in for the missing window without checking whether consumers can absorb historical rows: re-reading current table state emits one event per row today, which is neither the deleted set nor its ordering.

### Primary storage is filling although replication reports healthy

Look for an inactive or orphaned slot and retained WAL rather than data growth; a slot that is retained but not consumed keeps the log alive indefinitely and reports as active replication to anyone reading only connector status. Compare retained bytes against disk reserve and the configured ceiling, then make an explicit choice: raise the ceiling with storage headroom and an owner, or drop the slot and schedule the backfill it implies. Check transaction ID age at the same time, since slots that hold a xmin also block freeze progress. Do not drop the slot to relieve disk pressure without recording its restart position and confirming archive coverage; you are choosing data loss over downtime, and that trade belongs to an incident decision with a name attached rather than to a disk alarm.

### Cleanup removed rows that were never published

Establish what is provably unpublished: reconcile domain transitions requiring events against broker-recorded identities, since comparing row counts cannot see an omission hidden by a duplicate. Recover from the authoritative store using stable deterministic ids and publish through a bounded backfill with consumer-visible provenance. Then fix the predicate: the cleanup watermark must advance on durable publication evidence plus an expired claim window, never on row age alone. Prevention is the periodic reconciliation itself, run as a scheduled job whose failures page; do not rebuild trust by pausing cleanup permanently, because an unbounded outbox eventually becomes the primary's problem rather than the broker's.

## Security and Privacy

Application roles may insert outbox rows only through owned domain transactions. Relay roles read/claim/publish but cannot mutate domain state or alter payloads. CDC connectors receive least-privilege replication access and isolate schema-history/checkpoint credentials.

Authenticate destination routes from trusted configuration, not payload-supplied topic names. Validate event type/schema, tenant, size, compression ratio, headers, and claim-check digest. Encrypt outbox/inbox, WAL archives, broker, and payload objects according to data classification.

Replay, seek, cleanup, connector reset, and bulk redrive are privileged audited operations. An event ID is not a secret, but allowing callers to choose another tenant’s ID can suppress effects through inbox uniqueness; namespace and bind identity to authenticated source plus digest.

## Verification Strategy

- transaction tests prove every committed domain transition has its required outbox row and rollbacks publish none;
- kill pollers before/after broker acknowledgment and mark-published to prove duplicates but no loss;
- partition/fence poller claim generations and relay generations;
- restart CDC from checkpoints, lose a slot/log range, and execute the documented backfill;
- crash consumers before/after inbox/effect/checkpoint commit;
- replay beyond normal windows and verify retention contracts or explicit rejection;
- compare domain/outbox/broker/inbox/effect IDs and source versions under reconciliation;
- load-test six-hour outage accumulation plus bounded catch-up against primary/broker/downstream capacity.

## Decision Framework

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
- [Martin Kleppmann et al.: Online Event Processing: Achieving Consistency Where Distributed Transactions Have Failed](https://doi.org/10.1145/3329672.3329679)
- [PostgreSQL: Replication and WAL Retention Settings](https://www.postgresql.org/docs/current/runtime-config-replication.html)
- [Debezium: PostgreSQL Connector Configuration](https://debezium.io/documentation/reference/stable/connectors/postgresql.html)
- [Amazon DynamoDB: Change Data Capture with Streams](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Streams.html)

---

## Related Chapters

- [Delivery Guarantees and Effect Boundaries](04-delivery-guarantees.md): at-least-once is the honest promise here, and the inbox is what makes it survivable
- [Change Data Capture](../13-data-pipelines/04-change-data-capture.md): reading the log instead of polling the table, with a coupling cost of its own
- [Distributed Transactions](../02-distributed-databases/07-distributed-transactions.md): the alternative this pattern exists to avoid: one commit across two systems
- [Effect Commit Protocols for Workflows](../18-workflow-job-systems/06-retry-idempotency-compensation.md): when the relay dies halfway, recovery becomes a compensation problem
