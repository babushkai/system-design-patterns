# Database Schema Migrations

A database schema migration changes the contract shared by stored data, application binaries, database engines, replicas, change streams, and offline consumers. The production risk is not merely whether the DDL statement succeeds. It is whether old and new readers/writers remain compatible during rolling deployment, whether metadata locks and rewrite work stay inside the serving envelope, whether historical rows converge without stale overwrite, and whether rollback still has a lossless representation.

This chapter owns **schema compatibility, engine DDL/lock mechanics, constraints and indexes, in-database representation changes, backfills, and contract cleanup**. [Service and Platform Migration](./06-migration-strategies.md) owns authority transfer between systems, strangler seams, cross-system shadowing, and general migration ledgers. [Change Data Capture](../13-data-pipelines/04-change-data-capture.md) owns log extraction and downstream delivery semantics.

## Primary Evidence and Scope

| Primary evidence | What it establishes | Boundary |
|---|---|---|
| PostgreSQL 18 documentation | `ALTER TABLE` lock levels vary; unspecified forms default to `ACCESS EXCLUSIVE`; concurrent index builds and deferred constraint validation have distinct phases and failure artifacts | PostgreSQL 18 behavior; verify the deployed major/minor version |
| MySQL 8.4 documentation | `INSTANT`, `INPLACE`, and `COPY` differ; online DDL may still need exclusive metadata locks and can cause I/O, replication, and rollback cost | MySQL 8.4/InnoDB behavior, operation-specific |
| GitHub Engineering, gh-ost (August 2016) | A ghost-table migration can copy rows and tail row-based binlog changes, with pause/throttle, test, audit, and controlled cutover | Historical GitHub design and current open-source tool constraints |
| Fowler and Sadalage, evolutionary database design | Application and database changes can be decomposed into backward-compatible evolutionary steps | A method, not an engine guarantee |

Do not copy an operation-safety table across engines or versions. Ask the actual engine for its plan/algorithm where possible, pin the version, inspect locks and space, and rehearse on production-shaped data.

## Migration contract

Before writing DDL, record:

| Field | Required answer |
|---|---|
| **Object** | Database, schema, table, partition, column, index, constraint, view, sequence, trigger, or policy? |
| **Current/target contract** | Which values, nullability, uniqueness, references, types, and query access paths change? |
| **Compatibility set** | Which old/new binaries, jobs, CDC consumers, replicas, and tools coexist with each schema phase? |
| **Engine mechanics** | Required lock, scan/rewrite, algorithm, temporary space, log volume, replication effect, and cancellation behavior? |
| **Data transform** | Mapping, ordering/version token, invalid-source policy, and semantic validation? |
| **Serving budget** | Lock-wait, latency, I/O, replica-lag, log-growth, and failure-headroom budget? |
| **Rollback boundary** | Until which phase can routing/code rollback without losing new writes or values? |
| **Contract evidence** | How is absence of old readers/writers/consumers proven before removal? |
| **Ownership** | Who approves, executes, throttles, aborts, validates, and performs irreversible cleanup? |

“Zero downtime” is not a mechanism. State maximum acquisition pause, allowed latency/lag impact, cutover behavior, and recovery after cancellation or process death.

## State, authority, and invariants

The schema history table is not enough for a multi-phase change. Keep a durable record:

```text
migration identity and immutable definition digest
database/cluster and object identity
source and target schema versions
phase and phase revision
binary/consumer compatibility matrix revision
DDL plan/algorithm and lock expectations
backfill snapshot/range checkpoints and transform version
validation counts, checksums, and exceptions
read/write switch state
rollback deadline and cleanup prerequisites
actor, approvals, timestamps, and engine job identity
```

**Reference-design invariants:**

1. Every active binary and consumer understands the currently exposed schema.
2. One representation is authoritative for reads and writes at each phase, even when two representations are maintained.
3. A historical backfill cannot overwrite a value from a newer application write.
4. A migration retry is idempotent or resumes from durable engine/workflow state.
5. DDL lock acquisition and maintenance work are bounded; timing out does not leave the application blocked indefinitely.
6. Validation covers values, nulls, tombstones, constraints, indexes, and dependent objects—not only row count.
7. Contract/removal begins only after old readers, writers, replicas, CDC consumers, reports, and rollback needs are absent.
8. Schema version never regresses on one database during ordinary rollback; recovery uses a new forward phase.
9. Migration and serving traffic share explicit I/O, log, connection, and storage budgets.
10. Irreversible cleanup records evidence and preserves backup/audit obligations.

## Control path and database data path

~~~mermaid
flowchart LR
    P[Migration definition and policy] --> C[Migration controller]
    C --> DDL[Engine DDL job]
    C --> BF[Backfill workers]
    C --> V[Semantic validator]
    C --> H[(Migration history and evidence)]
    APP[Old and new application cohorts] --> DB[(Primary database)]
    DDL --> DB
    BF --> DB
    DB --> REP[Replicas / CDC / backups]
    DB --> V
    T[Locks, latency, I/O, lag, log, space] --> C
    DB --> T
    REP --> T
~~~

The migration **control path** sequences phases, checks compatibility, admits maintenance work, and decides pause/abort/advance. The database **data path** continues serving transactions. The controller should disappear without leaving an ambiguous phase: engine job identity, cursor, transform version, and evidence are durable.

## Compatibility is a matrix, not “N−1” alone

During a rolling release, several actors coexist:

| Actor | Examples of dependency on schema |
|---|---|
| Old application writer | Writes only old column/type or omits newly required value |
| New application writer | May populate old and new representation |
| Old reader | Selects old names/shape, possibly `SELECT *` |
| New reader | Expects transformed values/index/constraint semantics |
| Replica | Replays DDL/DML under its engine/version and available disk |
| CDC consumer | Decodes column order/type and tombstones from the log |
| Offline/reporting job | Uses stale SQL, snapshots, extracts, views, or ORM metadata |
| Restore tooling | Recreates schema and replays history in order |

For every phase, test every supported actor. “The new binary works” does not prove rollback safety or downstream compatibility.

Prefer additive changes first: new nullable column/table/index/constraint state that old code ignores. Avoid `SELECT *`, positional decoding, and implicit type conversions because additive schema can still change result shape or wire encoding for brittle clients.

## Expand, migrate, switch, contract

The canonical state machine is:

~~~mermaid
stateDiagram-v2
    [*] --> Planned
    Planned --> Expanded: additive schema installed
    Expanded --> Mirroring: new writes maintain target representation
    Mirroring --> Backfilling: historical rows transform
    Backfilling --> Validating: coverage complete
    Validating --> TargetRead: semantic gate passes
    TargetRead --> TargetWrite: rollback path evaluated
    TargetWrite --> ContractReady: old dependency evidence zero
    ContractReady --> Contracted: old representation removed
    Planned --> Aborted
    Expanded --> Aborted
    Mirroring --> Aborted
    Backfilling --> Aborted
    Validating --> Backfilling: repair exceptions
    TargetRead --> Mirroring: read rollback
~~~

### Expand

Create the target representation without breaking existing actors. Examples:

- add a nullable column before requiring values;
- add a new table keyed by stable identity before moving reads;
- build a new index before changing query plans intentionally;
- add a constraint without validating historical data where the engine supports that separation;
- add new enum/value protocol support to readers before writers emit it.

An additive statement can still block or rewrite. Engine/version/table layout decides mechanics, not whether the SQL looks additive.

### Maintain new writes

When old and new fields live in one database transaction, update both atomically and define one as authoritative. Database triggers can centralize coverage but add hidden write cost, recursion/order concerns, and deployment coupling. Application dual-write is explicit but every writer—including admin tools and jobs—must participate.

For cross-system writes, prefer one authoritative commit plus outbox/CDC and use the general [migration protocol](./06-migration-strategies.md). This chapter stays within schema evolution of one logical database contract.

### Backfill historical rows

A production backfill is a durable, throttled [job system](../18-workflow-job-systems/02-background-jobs-worker-pools.md), not DDL glue. Partition by stable keyset or immutable snapshot ranges; persist progress; keep transactions short; make updates conditional and idempotent.

For a target derived from source value plus source version:

```text
backfill(row, source_version):
    update target representation only when
      target is absent OR target_source_version < source_version
```

If current application writes do not expose an ordering token, an `IS NULL` guard can protect values only when “once populated, never recompute” matches semantics. Otherwise a slow scan can overwrite a newer mutation.

Avoid `OFFSET` pagination on a changing table. Use primary-key/keyset ranges, a consistent snapshot, or engine export manifests. Decide how inserts behind the cursor, deletes, nulls, invalid encodings, and changed rows are captured.

### Validate and switch reads

Validation gates include:

- total and per-range coverage, including deleted/invalid/quarantined rows;
- exact or canonicalized value comparison;
- aggregate invariants and domain constraints;
- index usability and query plans under production parameters;
- shadow reads comparing old and new while serving old;
- replica/CDC decoding and lag;
- authorization, row-security, masking, retention, and encryption behavior.

Switch reads by a sticky cohort or feature flag where semantics permit. Keep the old representation current until the rollback window closes.

### Switch writes and contract

Stopping writes to the old representation is the point at which rollback may require reverse transformation. If the mapping is lossy—larger type to smaller, split data to one field, new enum to old—the prior binary may no longer represent new values.

Contract/removal is a separately reviewed deployment. First prove no old query, prepared statement, view, ORM, report, replica, CDC schema, restore script, or rollback artifact refers to the object. Rename-to-tombstone or revoke access can expose hidden dependencies before destructive drop, but even metadata renames can break actors and require locks.

## Engine mechanics: plan the actual operation

### Metadata locks and blocker chains

DDL often needs a strong metadata/catalog lock briefly even when data work is online. A dangerous chain is:

1. a long transaction holds a conflicting lock;
2. DDL queues waiting for its strong lock;
3. later ordinary queries queue behind the waiting DDL under lock fairness;
4. the application exhausts pools while the DDL has changed no data.

Set a short, evidence-based lock-acquisition deadline separately from the statement/runtime deadline. Monitor blockers before execution. A failed attempt should release its queue position and retry with jitter during an approved window; it must not sit ahead of production indefinitely.

**Documented, PostgreSQL 18:** `ALTER TABLE` subforms take different locks, and `ACCESS EXCLUSIVE` is the default when not otherwise documented. Multiple subcommands take the strictest required lock. Inspect the exact deployed documentation and avoid combining a cheap subcommand with a rewrite/strong-lock subcommand.

### Index construction

**Documented, PostgreSQL:** `CREATE INDEX CONCURRENTLY` avoids blocking concurrent inserts/updates/deletes but performs additional work and has caveats: it cannot run inside a transaction block, takes longer/more work, and failure can leave an `INVALID` index requiring inspection and cleanup. Unique concurrent builds can begin enforcing uniqueness before the index is fully valid.

Index build capacity includes table scans, sort/work memory, temporary files, WAL/redo, replicas, storage writes, and changed query-planner choices. Completion does not prove the new plan is better for all parameter distributions.

### Constraint introduction

Separate enforcement for new writes from historical validation when supported. **Documented, PostgreSQL 18:** a foreign-key, check, or not-null constraint created `NOT VALID` can later be validated by scanning, and `VALIDATE CONSTRAINT` takes `SHARE UPDATE EXCLUSIVE` rather than the default `ACCESS EXCLUSIVE` path described for many alterations.

Validation can still saturate I/O and encounter bad history. Quarantine or repair exceptions before declaring the constraint's business invariant true.

### Instant, in-place, and copy are operation-specific

**Documented, MySQL 8.4/InnoDB:** online DDL exposes `INSTANT`, `INPLACE`, and `COPY` algorithms and `LOCK` options, but support depends on the operation. In-place work may still rebuild data, consume substantial resources, and require an exclusive metadata lock at final definition update. Long/inactive transactions can block completion; large online DDL can create replication lag and expensive rollback.

Specify the strongest acceptable algorithm and concurrency contract so the statement fails rather than silently choosing a blocking copy. Rehearse on the exact engine version/table features; generated columns, foreign keys, partitioning, full-text indexes, and old table formats can change support.

### Ghost-table migration

For a rewrite not safely supported in place:

1. create a target/ghost table with the new schema;
2. copy a consistent progression of existing rows;
3. capture concurrent mutations and apply them in order/idempotently;
4. converge and validate source versus ghost;
5. acquire the required metadata lock and atomically switch names/routing;
6. retain the old table through a rollback/verification window.

**Documented, GitHub 2016:** gh-ost tails the MySQL binary log rather than installing triggers, incrementally copies rows, supports pause/throttle and test-on-replica, and postpones cutover. Its constraints and prerequisites remain product-specific; use a released version and read its current limitations.

## Capacity and duration model

Every migration consumes at least:

```text
source reads + target writes + index amplification
+ transaction/change log + replica apply
+ validation reads + temporary/duplicate storage
+ foreground interference and rollback reserve
```

For $D$ bytes, source-read rate $R_s$, network rate $R_n$, target-write rate $R_t$, and migration duty cycle $q$:

$$
\begin{aligned}
T_{\mathrm{base}} &\ge \frac{D}{\min(R_s, R_n, R_t)} \\
T_{\mathrm{elapsed}} &\ge \frac{T_{\mathrm{base}}}{q}
\end{aligned}
$$

**Illustrative assumptions:** a 3 TiB table; source/target path benchmarked at 240 MiB/s, but maintenance receives a 35% duty cycle to preserve serving SLO.

$$
\begin{aligned}
T_{\mathrm{base}} &\ge \frac{3 \times 1{,}048{,}576\ \mathrm{MiB}}{240\ \mathrm{MiB/s}}
= 13{,}107\ \mathrm{s} = 3.64\ \mathrm{h} \\
T_{\mathrm{elapsed}} &\ge \frac{3.64\ \mathrm{h}}{0.35} = 10.4\ \mathrm{h}
\end{aligned}
$$

Continuous changes, secondary indexes, validation, throttling, retries, and cutover extend it. If writes generate change log faster than the migration can apply it, catch-up never completes.

Backfill row-rate planning uses remaining rows $N_{\mathrm{remaining}}$ and effective progress rate $r_{\mathrm{effective}}$:

$$
T_{\mathrm{completion}} \ge \frac{N_{\mathrm{remaining}}}{r_{\mathrm{effective}}}
$$

Effective rate is constrained by source IOPS, target write/redo, replicas, lock conflicts, and user latency—not the worker's configured batch size.

Temporary space must survive original table + ghost/new index + logs/undo + sort/temp + backups/snapshots + free-space threshold. Disk-full during DDL can threaten the database, not merely the migration.

## Specialized failure traces

### “Instant” DDL queues the application

1. A long-running transaction retains a conflicting metadata lock.
2. DDL waits for an exclusive metadata/catalog lock.
3. New application statements line up behind the waiting DDL.
4. Pools saturate and timeouts/retries amplify load.

Preflight blockers, bound lock acquisition, cancel cleanly, and verify the DDL session is gone. Metadata-fast does not mean lock-free.

### Backfill overwrites a fresh write

1. Worker reads old value `A` from a snapshot.
2. Application writes new value `B` to both representations.
3. Delayed worker unconditionally writes `A` into the new column.

Use version-fenced/conditional updates or a transform whose guard proves the target was never populated. Shadow comparison should classify stale-overwrite direction, not just count mismatch.

### Dual-write has an unknown outcome

1. Old representation commits.
2. Target update times out after possibly committing in another system.
3. Caller retries, duplicates, or returns failure after an accepted source write.

Within one database, write both in one transaction. Across systems, use an authoritative commit plus durable outbox/CDC and the general migration protocol.

### Concurrent index build fails halfway

The command exits but leaves an invalid artifact that consumes space and may enforce partial uniqueness behavior per engine phase. Automation sees a migration record and assumes completion. Reconcile engine catalog state with migration history; cleanup/retry is an explicit recovery state.

### Contract runs while one consumer remains

Application telemetry shows zero old-column reads, but a weekly finance export and a lagging CDC consumer still decode it. Drop breaks the export and poisons downstream replay. Dependency evidence must cover observation periods, schemas, prepared statements, and offline schedules—not only live application traces.

### Cutover cannot acquire metadata lock

Ghost copy and change apply are complete, but a long transaction prevents the atomic swap. Change backlog grows while operators repeatedly force cutover, affecting production. Keep source/ghost synchronized, bound each attempt, surface blockers, and allow postpone/abort without discarding hours of progress.

### Rollback binary cannot represent new values

New writers emit values outside the old type/enum domain. Rolling back code while retaining schema does not help; old code truncates, rejects, or misinterprets data. Gate new-value emission separately from new-code rollout and record the irreversible compatibility boundary.

## Security, privacy, and governance

Migration roles are unusually powerful. Grant only required DDL/DML/catalog privileges for named objects and time-bound them. Separate author, approver, and production executor where risk requires it. Sign or digest immutable migration definitions so a reused version cannot execute changed SQL.

Backfill/validation paths must enforce tenant isolation, row-level security expectations, encryption/key domains, masking, retention, legal holds, and audit. Maintenance roles that bypass policies can accidentally copy or log cross-tenant data. Do not emit row values or secrets in mismatch logs.

Temporary tables, snapshots, change logs, old columns, and backups extend the data lifecycle. Cleanup includes indexes, triggers, views, grants, keys, extracts, and ghost tables—not just the visible column.

Protect migration APIs from arbitrary SQL, unbounded batch sizes, broad table targeting, and unauthorized contract/drop. Emergency cancellation and lock termination are audited production actions.

## Operations, rollout, and rollback

Run one phase per deployable change:

1. **preflight:** exact engine version, plan/algorithm, locks, blockers, disk/log/replica capacity, backup/restore, compatibility matrix;
2. **canary:** representative small table/partition/tenant or production clone, then one bounded production object;
3. **expand:** apply additive state and verify every replica/consumer;
4. **mirror/backfill:** throttle below user traffic and persist progress;
5. **validate:** hold at zero unexplained divergence for the declared workload window;
6. **switch:** canary reads, then writes, with explicit rollback gates;
7. **contract:** separate approval after dependency evidence and rollback horizon;
8. **reconcile:** engine catalog, migration ledger, replicas, CDC, backups, and cleanup.

Rollback before target-only writes usually means switching reads back while mirroring continues. After target-only values exist, rollback may require reverse transformation and is another forward migration. Destructive down-migrations are not a recovery plan.

Do not combine engine upgrade, schema rewrite, ORM change, and traffic migration in one release. Each changes compatibility and failure evidence.

Runbooks cover blocked metadata lock, replication/CDC lag, disk/log growth, invalid index, failed constraint validation, stuck/ambiguous engine job, ghost divergence, cutover blocker, old consumer discovered after switch, and accidental contract.

## Observability and verification

Track:

- phase/revision, owner, elapsed time, progress rate, ETA, and last checkpoint;
- lock requested/granted/wait, blocker identity/age, and statements queued behind DDL;
- database latency/errors/connections, CPU, IOPS/throughput, buffer/cache effects;
- WAL/redo/undo/binlog growth, replica/CDC lag, and backup impact;
- original/target/temp/index bytes and free-space reserve;
- backfill rows scanned/changed/skipped/retried/quarantined by range;
- value/checksum/constraint/index mismatches and shadow-read results;
- active old/new reader/writer/consumer versions and dependency evidence;
- cutover pause, retry/error impact, rollback readiness, and cleanup age.

Verification includes engine-plan assertions, migration lint, exact-version integration tests, old/new binary-schema compatibility, realistic-data clones, concurrent writes during scan, duplicate/reordered work, process crash at every phase, blocker/lock timeout, disk pressure, replica lag, CDC schema evolution, restore/replay, invalid source values, cutover and rollback, and old-consumer discovery.

Property-test transformations for nulls, boundaries, encodings, timezone/collation, overflow, and round-trip reversibility. A count match can hide every semantic error that matters.

## Decision framework

1. Which readers, writers, replicas, CDC consumers, and offline jobs share this schema contract?
2. What exact lock, scan/rewrite, temporary-space, log, and cancellation mechanics apply on the deployed engine version?
3. Can the change be additive first, and which phase introduces irreversible values?
4. Which representation is authoritative at every phase, and how are new writes mirrored atomically?
5. What ordering/guard prevents historical backfill from overwriting current data?
6. Which semantic checks—not only counts—gate read and write switches?
7. Can maintenance traffic coexist with peak, failure, compaction, backup, and replica recovery?
8. What evidence proves every old dependency is gone before contract?
9. Until when is rollback lossless, and what forward migration is required afterward?
10. Has crash/timeout/retry recovery been tested for each engine and workflow state?

## Primary references

- [PostgreSQL 18, *ALTER TABLE*](https://www.postgresql.org/docs/current/sql-altertable.html)
- [PostgreSQL 18, *Explicit Locking*](https://www.postgresql.org/docs/current/explicit-locking.html)
- [PostgreSQL 18, *CREATE INDEX*](https://www.postgresql.org/docs/current/sql-createindex.html)
- [MySQL 8.4, *InnoDB and Online DDL*](https://dev.mysql.com/doc/refman/8.4/en/innodb-online-ddl.html)
- [MySQL 8.4, *Online DDL Limitations*](https://dev.mysql.com/doc/refman/8.4/en/innodb-online-ddl-limitations.html)
- [GitHub Engineering, *gh-ost: GitHub's online migration tool for MySQL* (August 2016)](https://github.blog/news-insights/company-news/gh-ost-github-s-online-migration-tool-for-mysql/)
- [GitHub, *gh-ost source and design documentation*](https://github.com/github/gh-ost)
- [Fowler and Sadalage, *Evolutionary Database Design*](https://martinfowler.com/articles/evodb.html)
