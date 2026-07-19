# Lakehouse Table Formats: Snapshots, Commits, and Maintenance

## TL;DR

An open table format turns immutable data files in object storage into a table by defining authoritative metadata, snapshots or timeline entries, conflict rules, row-level changes, and garbage collection. Readers plan from a committed metadata state instead of listing a directory while writers are changing it.

Apache Iceberg, Delta Lake, and Apache Hudi share this goal but do **not** use one universal “catalog compare-and-swap” algorithm. Iceberg publishes new table metadata through a catalog or atomic metadata operation; Delta commits ordered actions to its transaction log using storage or a commit coordinator that satisfies the protocol; Hudi publishes actions on a timeline and coordinates file groups and table services. Correctness depends on the exact format version, catalog/storage integration, engine, connector, and enabled features.

Maintenance is part of the write path’s deferred cost. Small files, delete metadata, old snapshots, manifests/log checkpoints, and abandoned objects must be compacted or collected without deleting data still visible to readers or in-flight commits.

---

## 1. Workload and Table Contract

Choose a table format only after defining the table’s workload:

| Contract field | Decision |
|---|---|
| **Writers** | Number of concurrent append, overwrite, merge, compaction, and maintenance writers. |
| **Readers** | Engines, versions, snapshot duration, streaming/incremental reads, and acceptable feature subset. |
| **Mutation** | Append-only, partition replacement, keyed upsert, delete, or mixed workloads. |
| **Isolation** | Required behavior for readers and concurrent writers; single-table versus multi-table atomicity. |
| **Commit authority** | Catalog, transaction-log storage primitive, lock service, or commit coordinator. |
| **Layout** | File format, partition transforms, sort/clustering, target file distribution, and key/index strategy. |
| **Retention** | Time travel, audit, rollback, long readers, streaming offsets, legal holds, and recovery. |
| **Maintenance** | Compaction, clustering, manifest/log optimization, snapshot expiry, cleaning, and orphan detection. |
| **Interoperability** | Which engines must read and write which protocol features—not only recognize the table name. |
| **Recovery** | Commit retry, ambiguous commit resolution, catalog outage, corrupt metadata, and rollback procedure. |

### Invariants

1. Readers resolve one committed table state and ignore uncommitted files.
2. Data files are immutable for the lifetime of every snapshot that references them.
3. Publication is atomic at the format’s documented commit boundary.
4. Concurrent writers either merge compatible changes or detect a conflict; one cannot silently erase another.
5. Schema fields and partition specs retain stable identities where the format requires them.
6. Cleanup deletes only files proven unreachable from every retained reference and in-flight operation.
7. Every enabled feature is supported by every reader or writer allowed to touch the table.

These invariants apply to one table unless a higher-level catalog or transaction system explicitly provides multi-table atomicity.

---

## 2. Why a Directory of Files Is Not a Table

Suppose a writer creates 500 Parquet files under a partition prefix and fails after 317:

- a listing reader may observe a partial result;
- another writer may mix files into the same prefix;
- retry may duplicate objects;
- a partition rename may be copy-plus-delete rather than atomic;
- schema and partition meaning live only in conventions;
- no durable snapshot says which files belong together.

A success marker helps only when every reader honors it and the file set is immutable afterward. It does not provide conflict detection, row-level mutation, schema identity, time travel, or safe garbage collection.

Table formats make **metadata**, not directory listing, authoritative.

---

## 3. Common Metadata Model

The exact files differ, but the logical layers are:

~~~mermaid
flowchart TD
    C["Commit authority<br/>catalog, log primitive, or coordinator"]
    T["Committed table state<br/>snapshot / version / timeline instant"]
    P["Planning metadata<br/>manifests, actions, indexes, statistics"]
    D1["Immutable data files"]
    D2["Delete or delta files"]
    OLD["Older retained table states"]

    C --> T
    T --> P
    P --> D1
    P --> D2
    T --> OLD
~~~

The data plane writes and reads data/delete files. The control plane resolves the current table state, allocates or validates commits, tracks schema and partition evolution, and decides which old files may be removed.

### Reader algorithm

1. Resolve the requested current, tagged, branched, or time-travel state.
2. Pin that state for the query.
3. Read its planning metadata.
4. Prune metadata entries using partition summaries and file statistics.
5. schedule data-file scans and apply relevant delete/delta records;
6. continue using the pinned state even if another writer commits.

The engine must not silently fall back to recursively listing data files, because unreferenced staged and obsolete files may coexist under the table location.

---

## 4. The Formats Differ at the Commit Boundary

### Apache Iceberg

Iceberg table metadata records schemas, partition specs, properties, snapshot history, and the current snapshot. A snapshot references a manifest list; manifests enumerate data and delete files with partition data and statistics. Manifests are immutable and may be reused.

Writers create files and new metadata, then ask the catalog or supported filesystem operation to replace the current table metadata only if the expected base is still current. On conflict, an operation refreshes and validates whether its changes can be retried. Sequence numbers establish relative age for data and delete files. Catalog behavior is part of the protocol: two catalogs with different atomicity or locking capabilities are not operationally identical.

### Delta Lake

Delta represents a table version as an ordered set of actions in the transaction log, including protocol, metadata, add-file, remove-file, transaction, and optional feature actions. Checkpoints compact prior log state for faster reconstruction; they do not replace the authoritative version history required by the protocol.

A writer reads a table version, writes new data files, checks conflicts, and attempts to publish the next valid log commit using a storage primitive or commit coordinator that meets the protocol. The mechanism is not an Iceberg-style catalog pointer. Readers reconstruct a snapshot from committed log actions and checkpoints. Deletion vectors and other table features require declared protocol support.

### Apache Hudi

Hudi organizes records into file groups and publishes actions on an active timeline. Copy-on-write writes new base-file versions. Merge-on-read may append changes to log files and later compact them into base files. Timeline actions distinguish commits, delta commits, replace commits, compaction, cleaning, rollback, and other table services.

Multi-writer and table-service coordination depends on the selected concurrency mode and lock/coordinator configuration. Readers ignore writes that have not reached a completed timeline state. File-group ownership, indexing, clustering, and compaction are more central to the write model than in a generic snapshot-tree description.

### Compatibility is a matrix

| Capability | Iceberg concept | Delta concept | Hudi concept |
|---|---|---|---|
| Committed state | Snapshot in table metadata | Transaction-log version | Completed timeline instant |
| Planning | Manifest list and manifests | Log actions and checkpoints; file statistics | Timeline, file groups, metadata/indexes |
| Row deletion | Position/equality delete files; newer versions add further mechanisms | Remove/add actions and optional deletion vectors | New file slice or merge-on-read log records |
| Concurrency | Optimistic metadata commit and validation through catalog | Optimistic transaction/conflict protocol and commit mechanism | Timeline plus configured OCC/MVCC/NBCC and locks/coordinator |
| Deferred rewrite | Data-file rewrite and manifest optimization | File optimization, log checkpoint/compaction, deletion-vector materialization | MOR compaction, log compaction, clustering, cleaning |

This table maps concepts, not feature parity. Verify protocol version and the exact engine’s read and write implementation. A reader may support snapshots but not a newly enabled delete representation; a writer may read a feature it cannot safely preserve.

---

## 5. Planning, Pruning, and Metadata Scale

Columnar data files carry row-group statistics, but opening every footer to discover which files matter does not scale. Table-level planning metadata can skip files before data access.

For Iceberg, manifest-list summaries prune manifests, and manifest entries contain partition values and column metrics for files. Each manifest is evaluated using the partition spec that wrote its files, allowing old and new specs to coexist.

Delta add-file actions may carry statistics used for data skipping; transaction-log checkpoints reduce the work required to reconstruct a snapshot. Hudi uses timeline/file-group metadata and optional metadata indexes to accelerate discovery and pruning.

Plan for three amplification layers:

> metadata objects read per query
>
> data files scheduled per query
>
> row groups or pages read per data file

Poor clustering can defeat column statistics even when partition pruning works. Excessive commit frequency can fragment metadata even when data files are large. Collect planning latency, metadata bytes, and files selected after each pruning layer.

Do not put high-cardinality raw values directly into physical partitions merely to avoid scanning. Partition choice affects file count, writer conflicts, metadata size, and pruning. Use transforms, clustering/sort order, and statistics based on measured query predicates.

---

## 6. Optimistic Commit and Conflict Validation

A generic optimistic write has these phases, though each format implements them differently:

1. Resolve base table state <code>B</code>.
2. Plan the logical operation and its conflict scope.
3. Write immutable data, delete, delta, and metadata artifacts.
4. Refresh current state <code>C</code>.
5. Validate changes since <code>B</code> against the operation’s assumptions.
6. If compatible, construct a commit rebased on <code>C</code> where the protocol permits.
7. Atomically publish through the format’s commit authority.
8. If publication outcome is ambiguous, resolve whether the commit exists before retrying.

Two appends to disjoint data may be compatible. A partition overwrite can conflict with an append that added a matching file. A row-level merge may conflict with another rewrite of the same files or keys, depending on the format and isolation mode. Compaction must not resurrect data deleted after its plan was created.

### Concrete failure trace: blind retry loses a concurrent write

1. Writers A and B both plan from table state 40.
2. A rewrites partition P and commits state 41.
3. B finishes a stale rewrite of P based on state 40.
4. B sees a commit conflict.
5. A naive retry republishes B’s already-written replacement set without validating state 41.
6. Files added or deletes applied by A disappear from the new logical table.

The correct retry refreshes metadata and validates B’s logical assumptions against changes since state 40. If they conflict, B must replan from current data or fail for operator intervention. Retry count is not a substitute for conflict semantics.

### Commit authority as a control-plane dependency

Catalog outage may prevent new Iceberg commits while already resolved snapshots remain readable. Delta commit behavior depends on the storage/coordinator path used. Hudi writers may depend on a lock provider and timeline coordination. Capacity-plan and protect these services for commit rate, ambiguity resolution, and disaster recovery—not just table discovery.

---

## 7. Updates, Deletes, and Read Amplification

Immutable files require logical indirection or rewriting.

### Copy-on-write

Read affected files, apply mutations, write replacement files, and atomically replace metadata references. This moves cost to writes and usually simplifies reads.

### Merge-on-read and delete metadata

Write deltas, log records, equality deletes, position deletes, or deletion vectors, then merge them during reads until a later rewrite materializes the result. This lowers immediate write amplification but adds lookup, merge, and metadata work to reads.

The names and guarantees are format-specific:

- Iceberg position deletes identify file and row position; equality deletes identify values in declared equality fields and are applied according to partition and sequence rules.
- Delta deletion vectors mark rows removed from a logical data file when the feature is enabled; readers must implement that protocol feature.
- Hudi merge-on-read file slices combine base files with log files according to record keys, pre-combine/order fields, payload or merger semantics, and timeline state.

CDC ingestion must carry a source sequence or version so an older replay cannot overwrite a newer row. A table commit sequence orders commits to the table; it does not by itself tell which of two out-of-order source events is newer.

### Write-amplification model

If a mutation changes <code>U</code> bytes of logical rows and rewrites <code>R</code> bytes of files:

> copy-on-write amplification = R / U

For merge-on-read, estimate:

> read merge bytes = selected base bytes + relevant delete/delta bytes
>
> deferred rewrite bytes = base and delta bytes later compacted

Choose from end-to-end cost and freshness. Fast ingestion that makes every important query merge a deep delta chain is not free.

---

## 8. Schema and Partition Evolution

Do not state that every rename or type change is universally metadata-only.

Iceberg identifies fields and partition specs by IDs, supports safe schema operations defined by its spec, and plans mixed partition specs using the spec associated with each file. Delta and Hudi have their own schema-evolution rules, mapping modes, engine restrictions, and data-file compatibility requirements.

For every change, ask:

- Can old files be interpreted unambiguously under the new logical schema?
- Do all readers honor field identity rather than only name or position?
- Is the type promotion permitted by the table and file-format protocol?
- Can old and new writers overlap without erasing metadata?
- Does a dropped field’s historical data remain visible through time travel?
- Does a partition-transform change apply only to future files, and can planners prune both layouts?

Use expand/contract for engine fleets:

1. inventory every reader and writer plus protocol version;
2. upgrade readers before enabling a feature they must understand;
3. add compatible fields or metadata;
4. dual-read or shadow-write if semantics change;
5. rewrite historical files only when required;
6. remove old behavior after retained snapshots and streaming readers no longer need it.

A protocol upgrade can be irreversible for old clients even when no data file changes.

---

## 9. Small Files, Compaction, and Clustering

Small files increase:

- object requests and file-open latency;
- scheduler tasks and planning entries;
- metadata and checkpoint size;
- low-selectivity footer and delete processing;
- commit conflicts and maintenance overhead.

There is no universal target file size. Derive it from:

- typical query scan range and desired scan parallelism;
- compression ratio and row-group layout;
- object-request and throughput characteristics;
- memory available to writers and readers;
- mutation locality and expected rewrite amplification;
- streaming commit interval and freshness objective.

If daily ingest is <code>D_day</code> bytes and average committed file size is <code>F_avg</code>:

> new files per day ≈ D_day / F_avg

If queries usually select <code>q</code> of the table but clustering selects files at <code>q_eff</code>, data-file read amplification is approximately <code>q_eff / q</code>. Measure this ratio rather than assuming partitioning is effective.

### Safe compaction

1. Resolve and pin a base state.
2. Select candidates using size, delete/delta depth, query heat, and rewrite budget.
3. Rewrite into new immutable files with equivalent logical rows.
4. Validate row counts, key uniqueness, and relevant source versions.
5. Commit replacements with conflict validation against changes since the base.
6. Leave old files for retention/cleanup.

Compaction, clustering, and manifest/log optimization compete with ingestion for I/O and commit bandwidth. Give table services explicit capacity, schedules, and SLOs.

---

## 10. Retention, Garbage Collection, and Reader Safety

Old metadata and files support time travel, rollback, incremental readers, audits, and long-running queries. Cleanup must account for all of them.

Separate:

- **snapshot/version expiry:** remove historical table states according to policy;
- **data-file cleaning/vacuum:** delete files no longer reachable from retained states;
- **orphan cleanup:** delete objects never referenced by a successful commit;
- **metadata optimization:** rewrite manifests, checkpoints, or timelines without changing table rows.

### Concrete failure trace: orphan cleaner races a writer

1. Writer W creates data files at 10:00 and performs a long validation.
2. Files are not yet referenced by a committed table state.
3. Orphan cleanup lists unreferenced files at 10:05.
4. Cleanup deletes W’s files.
5. W successfully publishes metadata referencing the now-deleted files.
6. The commit exists, but readers fail with missing objects.

Correct orphan cleanup uses a safety interval longer than the maximum in-flight commit plus clock/listing uncertainty, excludes registered staging areas or active write leases, and rechecks according to the format’s procedure. It must never treat “not in current snapshot” as “unreferenced”: an older retained snapshot may still need the file.

### Long readers and streaming consumers

A reader that pins a snapshot before cleanup must finish before its files can be removed. Streaming readers may need a chain of versions to discover incremental changes. Retention policy should be:

> maximum query duration + consumer outage/replay window + rollback window + operational margin

Legal holds and reproducible ML datasets may require durable tags/branches or copied snapshots rather than an ever-growing default history.

---

## 11. Capacity and Cost Model

Track data, metadata, commits, and deferred work.

Let:

- <code>D_ingest</code>: compressed bytes ingested per interval.
- <code>F</code>: achieved average data-file size.
- <code>c</code>: commits per interval.
- <code>m_commit</code>: average new metadata bytes per commit.
- <code>d</code>: delete/delta bytes created per interval.
- <code>R_compact</code>: bytes selected for compaction.
- <code>b_maint</code>: sustainable maintenance read-plus-write throughput under production load.
- <code>W_amp</code>: total object bytes written divided by logical new/changed bytes.

Approximate:

> data files created ≈ D_ingest / F
>
> raw metadata growth ≈ c × m_commit before metadata compaction/reuse effects
>
> minimum compaction duration ≈ R_compact / b_maint
>
> storage written ≈ logical bytes × W_amp

Snapshot-retained storage is not simply current table size × snapshots because snapshots share immutable files. Estimate unique live files plus files reachable only from retained historical states, delete/delta files, metadata, and abandoned staging objects.

Commit throughput can become the bottleneck before object bandwidth. Measure conflict rate, commit latency percentiles, retries, catalog/log coordinator saturation, and metadata planning latency as commit frequency grows.

Cost model:

> storage + requests + query scan + metadata planning + write/commit compute + compaction/clustering + cleanup/reconciliation + catalog/control-plane service

Comparing only storage price hides the table’s operational cost.

---

## 12. Operations, Migration, and Disaster Recovery

### Operational ownership

For every table, assign:

- ingest freshness and commit-success SLO;
- maximum small-file and delete/delta backlog;
- snapshot/time-travel retention;
- maintenance owner and capacity budget;
- compatible engine/feature matrix;
- recovery point for catalog and metadata;
- reconciliation and corruption runbook.

### Format or catalog migration

1. Inventory features actually used: deletes, generated columns, partition transforms, branches, indexes, CDC, constraints.
2. Verify target engines preserve those semantics, not only Parquet readability.
3. Freeze or serialize incompatible maintenance.
4. Export a source snapshot and source-to-target file/row reconciliation.
5. Build target metadata in an isolated namespace or dual-write through a tested compatibility layer.
6. Run representative reads and writes, including conflicts and time travel.
7. Catch up incremental changes using a source coordinate.
8. switch readers, then writers, with rollback roots retained.

Copying data files without their transaction metadata may create a directory, not an equivalent table. Copying metadata without catalog state can leave no valid commit root.

### Disaster recovery

Back up or replicate the catalog/transaction metadata with the data files. Test restoration when:

- current metadata is missing or corrupt;
- the latest commit outcome is ambiguous;
- a region contains data files but stale catalog state;
- cleanup ran in one region but not another;
- writer clocks and commit authorities diverged.

Recovery must restore a mutually consistent metadata root and reachable object set.

---

## 13. Security and Governance

- Separate data-file write permission from table commit authority.
- Prevent users from bypassing table policy by reading or mutating raw object paths.
- Enforce least privilege for catalogs, locks, commit coordinators, maintenance, and encryption keys.
- Encrypt data, delete/delta files, metadata statistics, manifests/logs, and backups; statistics may expose sensitive ranges.
- Audit commits, schema/protocol changes, snapshot expiry, rollback, tag/branch changes, and maintenance deletions.
- Propagate row/column classification and deletion requests through current and retained snapshots.
- Define whether time travel is allowed to reveal data a user can no longer access.
- Protect against a malicious writer enabling a protocol feature unsupported by governance readers.

Central policy in a catalog is useful only if direct storage access and alternate catalogs cannot bypass it.

---

## 14. Observability

### Commit path

- commit latency, conflicts, retries, and ambiguous outcomes;
- current snapshot/version/timeline age;
- catalog, lock, or coordinator saturation and availability;
- uncommitted/staged object age and bytes.

### Read path

- planning latency and metadata objects read;
- files and bytes selected before and after pruning;
- delete/delta merge bytes and CPU;
- missing objects, corrupt metadata, unsupported-feature errors;
- query snapshot age and maximum reader duration.

### Maintenance

- file-size and row-count distributions;
- delete ratio and delta/log depth by file group or partition;
- compaction/clustering backlog and write amplification;
- snapshots/versions and metadata growth;
- reachable historical bytes versus orphan candidates;
- cleanup age, deletions, and safety-window violations.

Alert on contract risk, not arbitrary universal thresholds: maintenance completion before backlog consumes its budget, retention headroom before a reader loses history, and conflict rate before writers miss freshness.

---

## 15. Verification and Failure Injection

- run readers continuously while a multi-file write commits and prove each query sees one state;
- race compatible appends and verify both survive;
- race overwrite/merge with append, delete, and compaction and verify conflict handling;
- kill a writer before data files, after data files, during metadata write, and during publication;
- make publication acknowledgement ambiguous and prove retry resolves rather than double-commits;
- run orphan cleanup against a slow in-flight writer;
- expire snapshots while a maximum-duration reader and streaming consumer are pinned;
- read every enabled feature from every approved engine version;
- apply schema and partition evolution with mixed old/new files;
- compare copy-on-write and merge-on-read under real update distribution and query predicates;
- corrupt a non-current metadata artifact and the current root separately; exercise recovery;
- restore catalog and objects to another environment and reconcile file reachability and rows.

Verify logical rows, source versions, snapshot isolation, and object reachability. A successful SQL count alone does not validate a table protocol.

---

## 16. Decision Framework

### Lakehouse table or another store?

| Requirement | Direction |
|---|---|
| Open object storage, multiple analytical engines, large scans | Table format is a strong fit |
| Frequent point updates with strict low-latency transactions | Operational database may remain authoritative |
| Multi-table serializable transaction is mandatory | Verify a higher-level transaction system; do not infer it from per-table ACID |
| All consumers are one managed warehouse and openness has no value | Native warehouse table may be simpler |
| Long-lived reproducible snapshots and direct file access matter | Table format is valuable |

### Choosing among formats

Evaluate, with a compatibility test matrix:

1. read/write engines and the protocol features they actually support;
2. append versus keyed-upsert/delete workload;
3. commit authority and catalog/control-plane operations;
4. concurrent writer and maintenance conflict behavior;
5. partition/schema evolution needed;
6. incremental read and CDC application semantics;
7. compaction, indexing, metadata, and cleanup operational burden;
8. disaster recovery and migration path.

Do not choose by origin story, vendor acquisition, or a claim that every engine can read and write everything. Choose the smallest feature set that all required clients can preserve correctly.

---

## Primary References

- Apache Iceberg, [Table Specification](https://iceberg.apache.org/spec/).
- Apache Iceberg, [Reliability](https://iceberg.apache.org/docs/latest/reliability/) and [Maintenance](https://iceberg.apache.org/docs/latest/maintenance/).
- Delta Lake, [Transaction Log Protocol](https://github.com/delta-io/delta/blob/master/PROTOCOL.md).
- Delta Lake, [Concurrency Control](https://docs.delta.io/latest/concurrency-control.html) and [Table Utility Commands](https://docs.delta.io/latest/delta-utility.html).
- Apache Hudi, [Technical Specification](https://hudi.apache.org/learn/tech-specs/), [Timeline](https://hudi.apache.org/docs/timeline/), and [Concurrency Control](https://hudi.apache.org/docs/concurrency_control/).
- Apache Hudi, [Compaction](https://hudi.apache.org/docs/compaction/) and [Cleaning](https://hudi.apache.org/docs/cleaning/).
- Michael Armbrust et al., [Lakehouse: A New Generation of Open Platforms that Unify Data Warehousing and Advanced Analytics](https://www.vldb.org/cidrdb/2021/lakehouse-a-new-generation-of-open-platforms-that-unify-data-warehousing-and-advanced-analytics.html).

---

**Previous:** [Change Data Capture](04-change-data-capture.md) covers the ordered, versioned change stream that often feeds row-level lakehouse mutations.
