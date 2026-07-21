# Change Data Capture: Snapshot, Tail, Apply, and Repair

## TL;DR

Change data capture (CDC) converts committed database changes into a downstream change stream. It is often safer than application dual writes because capture follows the database commit log, but it is not automatically complete, globally ordered, low overhead, or exactly once. Those properties depend on the source database, capture mode, log retention, connector, broker partitioning, event envelope, and sink protocol.

Bootstrap must join two histories without a gap: **existing rows from a snapshot** and **new transactions from the log**. Preserve enough source order and schema context for idempotent retries, primary failover, and drift repair; bound consumer lag so slow consumers cannot exhaust source log retention.

---

## 1. Workload and Replication Contract

CDC is a replication mechanism. Start with the state that must be reproduced, not the connector product.

| Contract field | Decision |
|---|---|
| **Authority** | Which database/table is authoritative, and may downstream ever write the same fields? |
| **Capture set** | Included tables, columns, operations, and whether DDL or transaction metadata is required. |
| **Bootstrap** | Consistent snapshot, incremental snapshot, backup restore, or an already complete retained changelog. |
| **Source coordinate** | WAL LSN, binlog file/position, GTID set, oplog token, or product-specific resume token. |
| **Ordering scope** | Per row, per aggregate, per source partition, source commit order, or whole transaction. |
| **Delivery** | Duplicate and replay behavior; how offsets become durable relative to broker publication. |
| **Sink application** | Versioned upsert/delete, append-only audit, transactional group, or compensating repair. |
| **Schema policy** | Before/after image, key changes, DDL events, compatibility, and decoder retention. |
| **Lag objective** | Commit-to-visible freshness and maximum tolerable outage/backlog. |
| **Repair** | Reconciliation cadence, resnapshot boundary, shadow target, and cutover/rollback. |

### Invariants

1. Every captured row’s state at the bootstrap boundary is represented by snapshot state or log replay, with no gap at the handoff.
2. Every committed source change after that boundary is eligible for capture until acknowledged or explicitly declared lost.
3. A duplicate or older event cannot overwrite newer sink state.
4. Delete and primary-key-change semantics are preserved.
5. Source position advances only after the corresponding downstream durability boundary.
6. Schema needed to decode every retained event remains available.
7. Repair runs converge without clearing the live sink first.

---

## 2. Why Application Dual Writes Diverge

An application transaction usually cannot atomically update its database and an unrelated search index, cache, broker, or warehouse:

~~~mermaid
sequenceDiagram
    participant A as Application
    participant D as Primary database
    participant X as External target

    A->>D: Commit order update
    D-->>A: Success
    A->>X: Publish/index update
    Note over A,X: process or network fails
    X-->>A: outcome unknown
~~~

Retries cannot determine whether an ambiguous external call already succeeded unless that target exposes an idempotency protocol. CDC changes the boundary: commit once to the source, then derive downstream state from the source’s durable change history.

This does not make the whole path atomic. It makes the database commit the unambiguous authority and gives the replication path a coordinate from which to retry.

---

## 3. Capture Mechanisms and Their Limits

| Mechanism | Strengths | Failure modes and costs | Appropriate use |
|---|---|---|---|
| **Timestamp/version polling** | Simple; no log privileges | Clock ties and skew, mutable scan boundary, deletes absent unless tombstoned, repeated index scans | Small sources with explicit monotonic version and soft-delete contract |
| **Trigger-written change table** | Custom event shape; transactionally coupled to row write | Adds work and failure surface to source transactions; trigger drift; bulk/DDL edge cases | Controlled database where log decoding is unavailable |
| **Transaction-log capture** | Observes committed row changes and source coordinates; usually avoids application changes | Database-specific decoding, privileges, log retention pressure, failover and schema complexity | General-purpose replication when source support is mature |
| **Transactional outbox capture** | Publishes business intent atomically with application state | Application must maintain outbox schema and cleanup; still needs dispatcher/CDC | Domain events whose meaning is not recoverable from raw row mutation |

Log capture reuses a log the database already produces, but decoding, replication slots, before images, and retained WAL/binlog are not free. Measure primary CPU, storage growth, replica/failover interaction, and transaction latency for the chosen connector and database version.

Polling can capture deletes if the source has durable tombstones. Triggers can be correct when rigorously operated. Log capture can omit tables, columns, unlogged operations, or historical events that were already truncated. Avoid architectural verdicts that ignore the actual source contract.

---

## 4. Data Plane, Control Plane, and Durable State

~~~mermaid
flowchart LR
    subgraph DP["Data plane"]
        DB[("Primary DB<br/>tables + commit log")]
        C["Log decoder / connector"]
        B[("Durable broker<br/>partitioned change stream")]
        A["Sink applier"]
        S[("Search / cache / lakehouse / replica")]
        DB --> C --> B --> A --> S
    end

    subgraph CP["Control plane"]
        CFG["Capture config<br/>tables, schema, policy"]
        OFF[("Offsets and snapshot<br/>chunk progress")]
        SCH[("Schema history")]
        REC["Reconciliation and repair"]
        FAIL["Failover coordination"]
    end

    CFG -.configures.-> C
    C -.persists only after publication.-> OFF
    SCH -.decodes.-> C
    FAIL -.changes source epoch.-> C
    REC -.compares and repairs.-> S
~~~

Durable connector state commonly includes:

- source identity and topology epoch;
- last safely published log coordinate;
- snapshot mode, table, primary-key chunk, and watermarks;
- schema history or references to compatible schemas;
- pending transaction or broker publication state.

If these fields live in different stores, define their crash consistency. Advancing the source offset before broker durability creates loss; publishing then failing before offset durability creates a duplicate. At-least-once publication chooses the second outcome and requires idempotent consumers.

---

## 5. Source Coordinates Are Database-Specific

A coordinate identifies progress in a particular source history; it is not a universal timestamp.

### PostgreSQL

Logical decoding reads WAL through a logical replication slot. Relevant positions include an LSN from which WAL may still be required and a confirmed flush position reported by the consumer. A stalled slot may retain WAL until storage limits or operator action intervene. Failover continuity depends on PostgreSQL version, slot configuration/synchronization, standby position, and connector behavior—promoting a standby does not make every external logical consumer automatically resumable.

### MySQL

Row-based binary logging can be addressed by file/position or GTIDs. A GTID identifies a committed transaction within a replication topology; GTID sets record applied history. Binlog expiration can still remove the event bytes needed for a fresh consumer. Statement, mixed, and row image settings affect what a CDC decoder can reconstruct.

### Other databases

MongoDB resume tokens, SQL Server log sequence coordinates, Oracle redo positions, and managed change-stream tokens have different retention and invalidation rules. Preserve the native coordinate in every event and in sink application metadata.

### Source epochs

A position may only be meaningful with a server, timeline, or topology epoch. After restore from backup, point-in-time recovery, log reset, or failover, a numerically comparable offset may not belong to the same history. Detect epoch change and prove continuity; never “pick the larger offset” across unrelated histories.

---

## 6. Event Envelope and Schema History

A useful change record separates business data from replication provenance:

~~~json
{
  "source": {
    "system": "orders-primary",
    "database": "commerce",
    "table": "orders",
    "epoch": "source-history-7",
    "position": "native-coordinate",
    "transaction_id": "tx-1842",
    "transaction_order": 6,
    "commit_time": "source-reported-time"
  },
  "key": {"order_id": 42},
  "operation": "update",
  "before": {"status": "created"},
  "after": {"status": "paid"},
  "schema_id": "orders-v12",
  "snapshot": false
}
~~~

Not every source supplies every field. Before images may be partial, large unchanged values may be omitted, and DDL may appear on another channel. The contract should distinguish absent, unchanged, unavailable, and null.

Stable field IDs or schema versions are safer than inferring meaning from the latest column name. Retain decoders for at least the maximum source and broker replay horizon. A sink cannot replay an old event if the only surviving schema describes today’s table.

---

## 7. Bootstrap: Proving Snapshot plus Tail Has No Gap

A new target needs existing state and subsequent changes. A naive sequence—scan the table, then start the log—loses transactions committed during the scan. Starting the log first and scanning later can overwrite a newer streamed value with an older snapshot row.

Two families of correct protocol are common.

### Consistent snapshot with a coordinated log boundary

1. Establish a database-consistent snapshot and the exact log position corresponding to that snapshot according to the source’s replication protocol.
2. Retain log history from that position.
3. Scan rows from the consistent snapshot, recording chunk progress.
4. Start or continue decoding from the coordinated position.
5. Apply snapshot and log records with source-aware precedence.
6. Mark bootstrap complete only after all chunks and the tail through the declared handoff coordinate are durable.

How step 1 works is database and connector specific. Merely reading “current LSN” in one session and scanning in unrelated transactions does not prove a consistent cut.

### Incremental watermark snapshot

Systems such as DBLog and connector-specific incremental snapshots interleave chunks with live log traffic:

1. Emit or record a low watermark in the source log.
2. Read a bounded primary-key chunk.
3. Continue buffering or observing changes for keys in that chunk.
4. Emit or record a high watermark.
5. Reconcile snapshot rows with log events between the watermarks so newer changes win.
6. Persist chunk completion and repeat.

This limits source locks and memory while allowing the change stream to progress. The correctness argument is the watermark interval and key reconciliation—not simply that reads are chunked.

### Concrete failure trace: stale snapshot wins

1. Connector begins scanning customer 42 and reads status = bronze.
2. The application commits an update to status = gold at source position P9.
3. CDC applies P9 to the sink: status = gold.
4. The slow snapshot task publishes its earlier row afterward.
5. A blind upsert changes the sink back to bronze.

The sink is now stale even though no event was lost. Fix this by attaching snapshot/log ordering metadata and applying only newer source versions, or by using the connector’s proven watermark reconciliation protocol before publishing the chunk.

### A compacted topic is not automatically a full bootstrap

Log compaction can retain a latest value per key, but completeness depends on compaction progress, delete-tombstone retention, topic retention, key stability, and transaction/read-isolation behavior. Do not discard the source snapshot plan until a test proves the topic can reconstruct the required state—including deletions—over the entire bootstrap duration.

---

## 8. Ordering and Transaction Boundaries

Separate these orders:

- order within a row/key;
- order within a source transaction;
- source commit order;
- broker order within one partition;
- processing and sink-commit order.

Keying broker records by primary key preserves broker partition order for one row when the producer sends them consistently, but events from a multi-row transaction may land in several partitions. A consumer may observe some rows before others. Some connectors emit transaction begin/end metadata or per-transaction order; they still commonly deliver row events, and cross-partition atomic application requires an additional protocol.

Options:

- accept temporary cross-row inconsistency and design materialized views accordingly;
- key by aggregate root so related rows share a partition;
- buffer transaction members until an end marker, with bounds for very large transactions and missing markers;
- stage rows under a transaction ID and atomically publish at the sink;
- publish a domain event through an outbox when the required business boundary differs from table rows.

Large transactions are a first-class capacity case: they can pin WAL, exceed connector buffers, monopolize a broker transaction, or delay visibility. Test them explicitly.

Primary-key changes may appear as delete-old-key plus insert-new-key. A sink that only upserts the new key leaks the old record.

---

## 9. Schema and DDL Evolution

Schema change affects source decoding, event serialization, and sink application independently.

Plan for:

- added nullable/defaulted columns;
- rename versus drop-and-add;
- type widening and incompatible representation changes;
- table or column rename;
- primary-key change;
- generated, large-object, and database-specific types;
- DDL that is transactional in one source and not another;
- source schema changing while a snapshot chunk is in progress.

Use expand/contract:

1. make consumers tolerate the old and new event shape;
2. deploy producers/source DDL;
3. observe both paths through the replay horizon;
4. migrate historical state or backfill if required;
5. remove old fields only after no retained event needs the old decoder.

A schema registry validates serialized messages; it does not prove the sink transformation preserves meaning. Rehearse DDL against a copy with the real connector and retained history.

---

## 10. Sink Application Protocols

### Versioned materialized view

Store the source epoch and position with each sink key. Apply an update or tombstone only if it is newer under the source’s valid ordering relation. The comparison and mutation must be atomic.

This works well for search documents, caches with durable metadata, and keyed tables. It does not automatically preserve a multi-key source transaction.

### Append-only history

Write each change under a unique source identity such as source epoch + transaction + event order. Enforce uniqueness or deduplicate during reads/compaction. Append-only history preserves audit and re-derivation options but requires a separate current-state view.

### Transactional sink application

Stage all events for a declared group, validate completeness, then commit them with the sink’s transaction protocol. Coordinate source offset acknowledgement with sink commit or use a connector protocol that can recover ambiguous commits. Confirm reader isolation hides staging data.

### Non-transactional target

Use idempotency keys and conditional versions if available. Otherwise assume duplicates and partial application, keep a reconciliation ledger, and define compensating repair. “The connector retries” is not an application guarantee.

### Deletes

Represent deletes explicitly through tombstones, versioned deleted flags, or sink delete operations. Retain delete evidence long enough to cover the slowest bootstrap and replay. Physical source deletion without a retained signal makes later reconstruction impossible.

---

## 11. Raw CDC versus Transactional Outbox

Raw CDC describes storage mutation: row inserted, value changed, row deleted. It is appropriate for materialized copies, analytics, cache invalidation, and audit feeds.

An outbox record describes business intent: order accepted, payment authorized, entitlement revoked. The application writes it in the same source transaction as business state, and CDC transports it.

Choose the outbox when:

- downstream must not infer intent from several tables;
- the public event contract differs from internal schema;
- authorization or aggregate version belongs in the event;
- transaction boundaries must match the domain aggregate.

Operate outbox retention and publication lag. Deleting outbox rows too soon can remove the only replay source; keeping them forever can burden the primary. See the [Transactional Outbox](../05-messaging/07-outbox-pattern.md).

---

## 12. Capacity and Cost Model

Let:

- <code>w</code>: peak source log generation bytes/s, including transactions not captured if they share the retained log.
- <code>r</code>: captured change records/s.
- <code>s</code>: average encoded broker bytes/change after envelope overhead.
- <code>a</code>: downstream amplification from one source change to broker, index, and table bytes.
- <code>p_apply</code>: sustainable sink apply records/s with retries and commits enabled.
- <code>T_outage</code>: longest planned connector/sink outage.
- <code>D_snapshot</code>: bytes scanned for bootstrap.
- <code>b_snapshot</code>: source-safe snapshot scan rate.

Source retained-log headroom is approximately:

> retained bytes ≥ w × (outage + detection + repair + catch-up margin)

Use the source’s total log generation, not only captured table volume. For PostgreSQL slots, monitor retained WAL against disk and configured limits; for binlogs, verify expiration exceeds recovery.

Stable catch-up requires <code>p_apply > r</code>. If backlog is <code>L</code> changes:

> catch-up time = L / (p_apply − r)

A lower bound on bootstrap scan time is <code>D_snapshot / b_snapshot</code>, but increasing scan rate competes with production cache, storage, vacuum, and replicas. Choose the rate from a primary-load budget, then ensure log retention covers the resulting snapshot plus tail catch-up.

Cost includes source storage and I/O, connector compute, broker replication/retention, schema storage, sink write amplification, snapshots, reconciliation scans, and repair capacity.

---

## 13. Operations, Failover, and Repair

### Source failover

Maintain a runbook that:

1. identifies the last source coordinate durably published and applied;
2. proves the candidate primary contains that history;
3. verifies logical slot/resume-token readiness where supported;
4. establishes the new source epoch;
5. resumes with duplicate-safe sink application;
6. reconciles the interval around promotion.

If continuity cannot be proven, stop advancing the sink and perform a bounded resnapshot or restore from another authoritative history.

### Drift repair

CDC is not its own audit. Periodically compare:

- row/key counts by stable range;
- checksums or aggregates over canonicalized fields;
- maximum applied source version;
- tombstone/delete counts;
- sampled full records and business invariants.

For repair, snapshot only affected ranges where possible, write into a shadow target or staged generation, reconcile with live tail using source versions, validate, then publish. Preserve the old target for rollback.

### Connector changes

Before upgrading a connector or database:

- capture the exact offset and schema-history formats;
- test restore from existing state;
- test source DDL and large transactions;
- validate event envelopes and ordering;
- run old and new connectors into isolated topics/targets at the same source boundary;
- compare and cut over without resetting source history.

---

## 14. Security and Governance

- Use a dedicated source principal with only replication and table/schema access required.
- Encrypt source connections, broker traffic, offset stores, schema history, and sink traffic.
- Treat before images, deleted rows, logs, snapshots, dead letters, and broker retention as regulated copies.
- Filter columns at the earliest supported point, but preserve keys and versions required for correctness.
- Prevent untrusted consumers from learning table names, schema history, or transaction metadata.
- Audit slot creation, capture-set changes, offset rewinds, snapshots, failovers, and repair publication.
- Reconcile right-to-delete obligations with retained logs, backups, tombstones, and downstream rebuild guarantees.

Masking only the sink is insufficient when the broker contains complete before and after images.

---

## 15. Observability and Verification

### Observe the whole path

- source commit coordinate versus connector-read coordinate;
- connector-published versus broker-durable coordinate;
- consumer-read versus sink-applied coordinate;
- lag in native bytes/positions, records, and wall-clock freshness;
- source retained-log bytes and time headroom;
- snapshot table/chunk progress and tail backlog;
- transaction size, duration, and incomplete-transaction buffers;
- schema/DDL errors, tombstones, duplicates, stale-version rejections;
- reconciliation mismatch by range.

Connector “running” status is not a freshness SLI.

### Failure-injection suite

- crash after broker publish but before offset persistence and verify duplicate handling;
- crash after sink apply but before consumer offset commit;
- update and delete keys while their snapshot chunks are being scanned;
- pause snapshot for longer than normal while live changes continue;
- rotate or expire logs near the recovery boundary in a staging source;
- promote a standby and verify source epoch/position handling;
- inject a primary-key update and a large multi-row transaction;
- apply DDL during snapshot and replay;
- delay one broker partition to expose cross-partition transaction assumptions;
- corrupt or remove schema history in an isolated test and exercise recovery;
- compare a reconstructed target with the source after the full drill.

The test passes only if source-to-sink invariants hold, not merely if the connector restarts.

---

## 16. Decision Framework

| Requirement | Preferred direction |
|---|---|
| Replicate committed row state with low latency | Log-based CDC where the source supports it reliably |
| Publish stable business intent | Transactional outbox transported by CDC |
| Tiny append-only source with durable version column | Polling may be sufficient |
| Need cross-row atomic visibility | Preserve transaction metadata and stage/commit at sink, or publish aggregate outbox event |
| Source cannot retain history through bootstrap/recovery | Backup/snapshot transfer plus coordinated tail, or redesign recovery objective |
| Sink cannot reject stale/duplicate events | Add an application ledger/version layer before claiming correctness |
| Full before image is prohibited or too costly | Capture selected fields and design downstream around partial events |
| Reconciliation cannot query the primary safely | Use a verified replica/backup or retain an independent authoritative snapshot |

Choose the capture mechanism only after proving bootstrap, retention, failover, and sink application. The steady-state happy path is the smallest part of the design.

---

## Primary References

- Debezium, [Stable Documentation](https://debezium.io/documentation/reference/stable/), including connector-specific snapshots, offsets, schema changes, and transaction metadata.
- PostgreSQL, [Logical Decoding](https://www.postgresql.org/docs/current/logicaldecoding.html), [Replication Slots](https://www.postgresql.org/docs/current/view-pg-replication-slots.html), and [Logical Replication Failover](https://www.postgresql.org/docs/current/logical-replication-failover.html).
- MySQL, [Replication with Global Transaction Identifiers](https://dev.mysql.com/doc/refman/8.4/en/replication-gtids.html) and [The Binary Log](https://dev.mysql.com/doc/refman/8.4/en/binary-log.html).
- Andreas Andreakis and Ioannis Papapanagiotou, [DBLog: A Watermark Based Change-Data-Capture Framework](https://arxiv.org/abs/2010.12597).
- Netflix Technology Blog, [DBLog: A Generic Change-Data-Capture Framework](https://netflixtechblog.com/dblog-a-generic-change-data-capture-framework-69351fb9099b).
- Martin Kleppmann, [Designing Data-Intensive Applications](https://dataintensive.net/), chapters on replication, streams, and derived data.

---

**Next:** [Lakehouse Table Formats](05-lakehouse-table-formats.md) shows how immutable files, metadata snapshots, and format-specific commit protocols turn CDC and batch output into transactional analytical tables.
