# ACID Transactions

## TL;DR

ACID names four properties of a transaction, but the label alone is not a complete contract. The exact isolation level, durability boundary, replication acknowledgement, and failure assumptions vary by database and configuration. Each letter hides real engineering tradeoffs: undo versus redo, flush latency versus durability, and concurrency control versus throughput. Read the product's documented semantics and verify the deployed configuration instead of treating "ACID" as a binary feature flag.

---

## The Problem ACID Solves

Consider a bank transfer: move USD 100 from Account A to Account B.

```text
1. Read balance of A: $500
2. Subtract $100 from A: $400
3. Write new balance to A
4. Read balance of B: $200
5. Add $100 to B: $300
6. Write new balance to B
```

What can go wrong without transactional guarantees?

**Crash failures:**
- Crash after step 3 → A lost USD 100, B gained nothing. Money vanished from the system.
- Crash during step 6 → Disk has partial write. B's balance is corrupted bytes, not USD 200 or USD 300.

**Concurrency failures:**
- Two transfers from A execute concurrently. Both read USD 500, both subtract USD 100, both write USD 400. A should be USD 300, but is USD 400. The bank created USD 100 from nothing.
- A reporting query runs between steps 3 and 6. It sees A debited but B not yet credited. The books don't balance.

**Durability failures:**
- The database says COMMIT succeeded. Power dies. The kernel had the write in its page cache but never called fsync. On restart, the write is gone.
- The device firmware acknowledged the write but the data remained in a volatile cache. Power loss means the "confirmed" write never reached persistent media.

These are not theoretical failure classes. Correctly implemented and configured transactions address them for writes inside the database's transactional boundary. They do not make an external API call, message-broker publish, filesystem write, or incorrectly configured storage device atomic with the database transaction.

---

## Atomicity — Deep Dive

### What It Actually Means

Atomicity does NOT mean "all operations happen instantaneously." That is closer to isolation.

**Atomicity means: all-or-nothing for the database-managed effects in the transaction.** If a transaction commits, all those effects commit. If it aborts, none become committed state. Atomicity says nothing about side effects performed outside that transactional resource.

### Why It Matters

Without atomicity, every multi-statement operation is a potential source of data corruption. Any crash, network timeout, or constraint violation mid-transaction leaves the database in an inconsistent intermediate state. The alternative — writing manual cleanup and rollback logic in application code — is prohibitively error-prone.

### Undo Log vs Redo Log

Undo and redo are two complementary logging techniques used by many databases. They are not the only implementation choices, and a specific engine can combine them with MVCC, shadow paging, copy-on-write, or transaction-status metadata.

**Undo log (rollback log):**
- Records enough prior state to reverse a change and, in MVCC engines such as InnoDB, to reconstruct older visible versions
- On explicit rollback, traversed to reverse the transaction's changes; crash recovery combines redo and undo/transaction-status processing
- Used by InnoDB (MySQL) as the primary mechanism for atomicity
- InnoDB stores undo logs in the system tablespace or dedicated undo tablespaces

**Redo log (write-ahead log / WAL):**
- Records enough information to reproduce page or logical changes; it is not necessarily a copy of the new row value
- The relevant log record must reach the configured durability boundary before the corresponding data page is allowed to be the sole durable copy
- On crash recovery, replay restores the database to a state from which transaction commit/abort visibility can be resolved
- Used by PostgreSQL as the primary mechanism (pg_wal directory)
- PostgreSQL WAL is append-only, sequential I/O — much faster than random page writes

**InnoDB uses both undo and redo:**

```text
InnoDB transaction lifecycle:
1. BEGIN
2. Write old values to undo log (in buffer pool)
3. Generate redo records for the page changes
4. Modify buffer pool pages in memory (dirty pages)
5. On COMMIT with durable flush settings: make the required redo durable → return success
6. Checkpoint: flush dirty pages to tablespace files (async)
7. Purge: clean up undo log entries after no transaction needs them
```

```text
PostgreSQL transaction lifecycle:
1. BEGIN
2. Generate WAL records describing changes
3. For an UPDATE, create a new tuple version and make the old tuple obsolete to later snapshots
4. On COMMIT with synchronous local durability enabled: flush WAL through the commit record before returning success
5. Checkpoint: flush dirty buffers to data files (async, configurable interval)
6. Old row versions cleaned up by autovacuum (async)
```

The key difference: InnoDB needs undo logs for rollback because it updates pages in-place. PostgreSQL uses MVCC — old row versions remain in the heap until vacuumed — so it doesn't need a separate undo log for atomicity.

### How ROLLBACK Works: Undo Chains and Transaction Status

**InnoDB rollback:**

```text
Transaction T1 modifies rows R1, R2, R3:
  u1: undo record for R1, prev_undo_ptr → NULL
  u2: undo record for R2, prev_undo_ptr → u1
  u3: undo record for R3, prev_undo_ptr → u2

ROLLBACK T1:
  1. Find T1's last undo record (u3)
  2. Restore R3 to old value
  3. Follow prev_undo_ptr to u2
  4. Restore R2 to old value
  5. Follow prev_undo_ptr to u1
  6. Restore R1 to old value
  7. Follow prev_undo_ptr to NULL → done
```

Each transaction maintains a linked list of its undo records. Rollback traverses this chain in reverse order. This is why rolling back a transaction that modified millions of rows can take as long as the transaction itself — it must undo each change individually.

**PostgreSQL rollback** normally avoids physically reversing each heap change in the foreground. The transaction is recorded or treated as aborted; its tuple versions are invisible to later transactions and are reclaimed later by vacuum. That can make the client-visible abort fast, but it defers cleanup work and can leave substantial table/index bloat after a large aborted transaction.

### Savepoints and Partial Rollback

Savepoints allow rolling back part of a transaction without aborting the entire thing. This is critical for complex business logic with conditional paths.

```sql
-- PostgreSQL
BEGIN;

INSERT INTO orders (id, customer_id, total) VALUES (1001, 42, 299.99);

SAVEPOINT before_inventory;

UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 7;
-- Suppose this violates a CHECK constraint (quantity >= 0)

ROLLBACK TO SAVEPOINT before_inventory;
-- The order INSERT is still intact
-- Only the inventory UPDATE was undone

-- Try alternative fulfillment
UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 7 AND warehouse = 'secondary';

COMMIT;
```

**Implementation detail:** PostgreSQL implements savepoints with subtransactions and tracks overflowed subtransaction relationships in `pg_subtrans`. InnoDB records a position in the transaction's existing undo history; it does not create a new undo-log segment for every savepoint. `ROLLBACK TO SAVEPOINT` reverses later changes but may retain locks, so a savepoint is not a resource-isolation boundary.

**Warning:** deeply nested savepoints have overhead. PostgreSQL's pg_subtrans can become a bottleneck with thousands of subtransactions. If you need savepoints in a loop, reconsider your transaction design.

### Distributed Atomicity: Two-Phase Commit (2PC)

When a transaction spans multiple database nodes, local undo logs are not enough. The classic solution is the **two-phase commit protocol**.

```text
Coordinator (transaction manager)
├── Participant A (shard holding Account A)
└── Participant B (shard holding Account B)

Phase 1 — Prepare (vote):
  Coordinator → A: "PREPARE transaction T1"
  Coordinator → B: "PREPARE transaction T1"
  A: writes all changes to durable log, acquires locks, responds YES
  B: writes all changes to durable log, acquires locks, responds YES

Phase 2 — Commit (decision):
  Coordinator: all voted YES → writes COMMIT decision to its own durable log
  Coordinator → A: "COMMIT T1"
  Coordinator → B: "COMMIT T1"
  A: commits, releases locks
  B: commits, releases locks
```

```sql
-- PostgreSQL prepared transactions; a separate transaction manager must
-- durably track and complete the global decision.
-- On participant:
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 'A';
PREPARE TRANSACTION 'transfer_1001_partA';

-- Later, coordinator decides:
COMMIT PREPARED 'transfer_1001_partA';
-- or
ROLLBACK PREPARED 'transfer_1001_partA';
```

**The coordinator failure problem:**

The liveness vulnerability of basic 2PC is losing access to the coordinator's decision after participants have prepared:

```text
Timeline:
  t0: Coordinator sends PREPARE to A and B
  t1: A votes YES, B votes YES (both holding locks, changes durable)
  t2: Coordinator crashes before a decision is durably available to participants

  A and B are now "in doubt" — neither can unilaterally choose an outcome:
  - Committing could disagree with an ABORT decision
  - Aborting could disagree with a COMMIT decision
  - Prepared resources remain held until the decision service recovers or an
    operator resolves the transaction from authoritative evidence
```

**In-doubt transactions** are operationally dangerous. They hold locks, block other transactions, and require manual intervention if the coordinator cannot recover.

```sql
-- PostgreSQL: find in-doubt transactions
SELECT gid, prepared, owner, database
FROM pg_prepared_xacts;

-- Manual resolution (ONLY when you've confirmed the correct outcome):
COMMIT PREPARED 'transfer_1001_partA';
```

**Mitigations for coordinator failure:**
- Coordinator writes the decision to a replicated, durable log before phase 2
- Participants timeout and query the coordinator (or its replicas) for the decision
- Three-phase commit adds another state but does not solve consensus in an asynchronous, partitionable network; it is uncommon in database deployments
- Some distributed databases replicate transaction records with a consensus protocol. That removes a single-machine coordinator as the only copy of the decision, but does not make distributed commit free of blocking, retries, or unavailable quorums

**2PC performance cost:** prepare and decision phases add messages, durable metadata, and longer lock retention. The number of round trips and log flushes depends on the database's protocol, replication topology, batching, and parallelism. Measure the implementation you operate; a fixed latency multiplier is not portable.

---

## Consistency — Invariants Are a Shared Responsibility

### What the Database Enforces vs What It Can't

In the ACID acronym, **consistency means that a transaction preserves the invariants the system claims**. It is a property of the transaction program plus the database guarantees, not a promise that the engine understands every business rule. The engine can reject violations only for rules represented in its schema, transaction logic, or trusted stored code.

The database enforces:
- NOT NULL, CHECK constraints
- UNIQUE and PRIMARY KEY
- FOREIGN KEY referential integrity
- EXCLUDE constraints (PostgreSQL)
- Trigger-based invariants

The database does not automatically infer:
- "Account balance should match the sum of all ledger entries"
- "Every submitted order must have at least one line item"
- "The total across all accounts must remain constant"
- Any business rule expressed only in application code

Some of these can be encoded with constraints, triggers, materialized state, or serializable transaction logic; others require a different data model. The important boundary is not "database rule versus impossible rule," but **which invariant is encoded where, under which isolation level, and how it is tested under concurrency**.

### The "C" Overloading Problem

The letter C means completely different things in different contexts:

| Context | "Consistency" means | Enforced by |
|---------|-------------------|-------------|
| ACID | A transaction preserves the system's stated invariants | Schema + transaction program + concurrency control |
| CAP theorem | A read/write object remains linearizable while messages may be lost or delayed | Usually quorum/consensus protocols |
| Replica convergence | Replicas that receive the same updates eventually compute the same state | Replication and merge protocol |

→ see [Consistency Models](04-consistency-models.md) for linearizability, causal consistency, and eventual consistency.

These are three fundamentally different concepts sharing one word. When someone says "this system is consistent," always ask which definition they mean.

### Deferred Constraints

Some constraints can't be checked row-by-row. Consider mutual foreign keys:

```sql
-- PostgreSQL
-- departments references employees.head, employees references departments
-- Inserting either first violates the FK of the other

-- Solution: deferred constraints
ALTER TABLE employees
  ADD CONSTRAINT fk_department
  FOREIGN KEY (department_id) REFERENCES departments(id)
  DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE departments
  ADD CONSTRAINT fk_head
  FOREIGN KEY (head_employee_id) REFERENCES employees(id)
  DEFERRABLE INITIALLY DEFERRED;

BEGIN;
INSERT INTO departments (id, name, head_employee_id) VALUES (1, 'Engineering', 100);
INSERT INTO employees (id, name, department_id) VALUES (100, 'Alice', 1);
-- Constraints checked HERE, at COMMIT time, not at each INSERT
COMMIT;
```

You can also defer constraints per-transaction:

```sql
BEGIN;
SET CONSTRAINTS fk_department DEFERRED;
-- ... operations that temporarily violate the constraint ...
COMMIT;  -- constraint checked here
```

**Use cases for deferred constraints:**
- Circular foreign keys (as above)
- Bulk data loading where intermediate states violate uniqueness
- Graph structures with parent-child self-references
- Schema migrations that reorder data

**Caveat:** deferred unique constraints in PostgreSQL use a different index mechanism and can have performance implications on large tables. Test with production-scale data.

### Foreign Keys Across Shards

Cross-shard referential integrity is possible, but it is distributed work. A distributed SQL database can enforce a foreign key whose rows live on different ranges by performing transactional reads/writes and maintaining the required indexes. Google Spanner and CockroachDB, for example, support enforced foreign keys. In an application-sharded deployment made of independent database instances, the database layer usually has no global transaction or catalog and therefore cannot declare such a foreign key.

```text
Shard A (users 1-1000):
  users table, orders table for these users

Shard B (users 1001-2000):
  users table, orders table for these users

Problem: order on application shard A references a catalog on independent shard B.
  - Neither instance can declare a constraint over the other instance
  - A check-then-insert in the application races a concurrent product delete
  - Strict enforcement requires a transaction protocol or a different placement
    that serializes the referencing insert with referenced-row updates
```

**Practical approaches:**
- **Use database-enforced distributed constraints:** strongest semantics, with extra index, locality, and write-path cost
- **Denormalize:** copy referenced data into the local shard (accept eventual staleness)
- **Application-level enforcement:** check before write, accept race conditions
- **Event-driven cleanup:** detect and repair broken references asynchronously
- **Avoid cross-shard references:** co-locate related data on the same shard

The choice is a data-placement and correctness tradeoff, not proof that distributed databases cannot preserve referential integrity.

### Constraint Enforcement Becomes Distributed Work

When constrained rows or indexes are remote, validation can require:

- Every cross-shard constraint check adds network round-trips
- Distributed deadlock detection is expensive
- Extra index maintenance and distributed transaction metadata

Some systems deliberately omit or relax these constraints for latency, availability, or operational simplicity; distributed SQL systems may preserve them and charge the coordination cost. "ACID" does not imply that constraints are local to one shard, and CAP does not require abandoning invariants during normal operation. → see [CAP Theorem](03-cap-theorem.md)

---

## Isolation — The Expensive Letter

### The Core Challenge

Isolation answers: "What do concurrent transactions see?" The ideal (serializability) means transactions behave as if they ran one-at-a-time. The reality: full isolation is expensive, so databases offer weaker levels.

### Isolation Levels Summary

| Level | Dirty Reads | Non-Repeatable Reads | Phantom Reads | Write Skew |
|-------|-------------|----------------------|---------------|------------|
| Read Uncommitted | Yes | Yes | Yes | Yes |
| Read Committed | No | Yes | Yes | Yes |
| Repeatable Read | No | No | Implementation-dependent; PostgreSQL and InnoDB snapshot implementations prevent them | Often possible under snapshot isolation |
| Serializable | No | No | No | No |

This table is a starting point, not a portable specification. Engines map SQL level names to different mechanisms; PostgreSQL also promotes `READ UNCOMMITTED` to `READ COMMITTED`. Serializable prevents these anomalies for committed transactions, but applications must retry serialization failures.

**Implementation approaches:**
1. **Locking (2PL):** transactions acquire locks, block each other. Used by SQL Server for Serializable.
2. **MVCC:** keep multiple row versions so ordinary reads usually avoid blocking ordinary writes; locks are still required for some writes, constraints, and schema operations. Used by PostgreSQL and InnoDB for most levels.
3. **OCC (Optimistic Concurrency Control):** assume no conflicts, validate at commit time. Used by some in-memory databases.
4. **SSI (Serializable Snapshot Isolation):** MVCC + dependency tracking. PostgreSQL's Serializable implementation since 9.1.

→ see [Isolation Levels](02-isolation-levels.md) for MVCC internals, locking protocols, SSI implementation details, and anomaly deep dives.

### Connection Pool Gotcha: Transaction-Scoped vs Session-Scoped Settings

A common production bug when using connection pools (PgBouncer, HikariCP):

PostgreSQL's `SET TRANSACTION` applies only to the current transaction and must run after `BEGIN` and before the first query. It does **not** leak into later transactions. By contrast, `SET SESSION CHARACTERISTICS AS TRANSACTION ...` changes the default for subsequent transactions on that backend session. A pool that does not reset session state can expose that session default to another borrower.

**Correct approach:**

```sql
-- Preferred: make the scope explicit in one statement.
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT balance FROM accounts WHERE id = 42;
UPDATE accounts SET balance = balance - 100 WHERE id = 42;
COMMIT;
-- Isolation level automatically resets after COMMIT/ROLLBACK

-- Also valid, provided it occurs before the first query:
BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
-- ... statements ...
COMMIT;
```

Prefer a transaction API or `BEGIN ... ISOLATION LEVEL` so scope is visible. If code changes session defaults or any other session state, configure and test the pool's reset behavior rather than assuming transaction pooling makes arbitrary session commands safe.

---

## Durability — The Latency Letter

### Why It Matters

Durability defines which failures a successful commit is promised to survive. A local WAL flush may cover process, OS, and power failure on one host while still not cover device loss, an availability-zone outage, operator error, or regional disaster. The acknowledgement policy and failure model must be stated together.

### fsync Deep Dive

`fsync()` is one common local durability boundary. Understanding its contract and the storage stack beneath it is critical.

```text
Application writes data:
  1. write() → data goes to kernel page cache (RAM) → returns immediately
  2. fsync() → kernel asks the filesystem/device stack to make the file durable
     and waits for completion
  3. The device stack performs the required cache flushes and persistence writes

What fsync actually forces:
  - Complete dirty data required for that file under the filesystem contract
  - Issue cache-flush/barrier operations supported by the device stack
  - Return only after that stack acknowledges the requested durability boundary
```

**Where the durability contract can break:**

```text
Failure point 1: Device or controller acknowledgement
  - Firmware may not honor flushes correctly
  - Volatile caches need power-loss protection or correct flush semantics
  - RAID/controller policy must preserve ordering and flush requests

Failure point 2: Filesystem and error propagation
  - A successful data flush does not automatically make a newly created
    directory entry durable; creation/rename protocols may also fsync the directory
  - Writeback errors must reach the database rather than being ignored or retried

Failure point 3: Virtualized or network storage
  - The guest acknowledgement is only as strong as the provider's documented contract
  - Replication can improve device-failure tolerance without protecting against
    corruption, credentials misuse, or application-level deletion
```

**PostgreSQL and fsync — the 2018 incident:**

PostgreSQL before v12 had a critical bug: if fsync() failed, PostgreSQL retried the fsync, assuming the dirty page was still in the kernel page cache. But some Linux kernels (pre-5.2) removed the dirty page from the page cache on fsync failure. The retry fsync'd a clean page — succeeding without writing anything. This meant PostgreSQL thought data was durable when it wasn't.

PostgreSQL 12+ responds to fsync failure by performing a PANIC (crash recovery) rather than retrying, because the kernel state is untrustworthy.

### WAL Mechanics

The Write-Ahead Log is the cornerstone of durability in PostgreSQL (and redo logs serve the same role in InnoDB).

**WAL segment files (PostgreSQL):**

```text
$PGDATA/pg_wal/
├── 000000010000000000000001   (16 MB segment, default)
├── 000000010000000000000002
├── 000000010000000000000003
└── archive_status/

Segment naming: TimelineID + (LSN >> 24)
Default segment size: 16 MB (configurable at initdb with --wal-segsize)
```

**WAL record structure:**

```text
Each WAL record contains:
  - LSN (Log Sequence Number): unique, monotonically increasing position
  - Resource manager ID (heap, btree, hash, etc.)
  - Record type and data needed to redo that resource-manager operation
  - For a data page's first change after a checkpoint, commonly a full-page image
    when full_page_writes is enabled; later records can contain smaller deltas
  - CRC checksum
```

**Checkpoint frequency and crash recovery:**

Checkpoints write all dirty buffers from shared_buffers to data files, then record the checkpoint LSN. On crash recovery, PostgreSQL only needs to replay WAL from the last checkpoint forward.

```text
Tuning tradeoffs:
  - More frequent checkpoints can reduce redo distance but increase write pressure
  - Less frequent checkpoints can increase redo distance and retained WAL
  - Recovery is not just sequential WAL reading: page access, storage latency,
    full-page images, CPU, and follow-on cache warming all matter

Measure restart time under the production WAL rate and dataset; do not derive an
RTO from WAL bytes divided by advertised sequential bandwidth.
```

**InnoDB redo log:**

```text
InnoDB uses a bounded circular redo-log space. File layout and configuration
variables differ across releases, so reason about the capacity rather than a
particular filename:

  - Circular buffer: head advances as new records are written
  - Tail advances as checkpoints flush dirty pages
  - If reusable space is exhausted, foreground work is throttled while checkpoint
    progress makes space available

Size from measured redo generation, checkpoint throughput, burst duration, and
tested recovery-time objectives. A universal "hours of writes" rule can make a
quiet system wasteful and a bursty system unsafe.
```

### Group Commit: Batching WAL Flushes

Every locally durable commit requires its commit record to be covered by a durable WAL flush, but it does not require a dedicated `fsync` call. Concurrent commits can share one flush; read-only transactions and asynchronous commit policies follow different paths.

**Group commit** batches multiple concurrent commits into a single WAL flush.

```text
Without batching (illustrative):
  T1: write WAL → flush → return
  T2: write WAL → flush → return
  T3: write WAL → flush → return

With group commit:
  T1: write WAL → wait
  T2: write WAL → wait
  T3: write WAL → wait
  Leader: fsync all three → return to T1, T2, T3
  One flush covers all three commit records
```

**PostgreSQL group commit tuning:**

```text
# postgresql.conf

# How long to delay before flushing WAL, hoping more commits arrive
commit_delay = <measured delay in microseconds>

# Only delay if at least this many transactions are active
commit_siblings = <measured concurrency threshold>

# A deliberate delay can increase the batch size, but it also adds latency.
# Benchmark with the actual storage and concurrency distribution.
```

**When to tune group commit:**
- Commit latency is dominated by WAL flush waits under concurrent small transactions
- Storage with high fsync latency (network-attached, cloud volumes)
- Workloads with many small transactions

**InnoDB and binary-log group commit:**

```text
MySQL coordinates InnoDB redo durability with binary-log ordering when the binary
log is enabled. `innodb_flush_log_at_trx_commit` and `sync_binlog` govern different
logs; changing only one can leave a crash window or replication inconsistency.
Verify both, plus whether the deployment enables the binary log, before claiming
a commit durability boundary.
```

### synchronous_commit = off: When Acceptable

PostgreSQL's `synchronous_commit` controls whether COMMIT waits for WAL fsync.

```sql
-- Per-transaction override (PostgreSQL)
SET LOCAL synchronous_commit = off;
-- Subsequent COMMIT returns immediately, WAL fsynced asynchronously
-- The loss window depends on WAL-writer scheduling and configuration.
```

**What you lose:** an OS or database crash before the asynchronous flush can erase transactions that already returned success. PostgreSQL documents the normal maximum delay in relation to `wal_writer_delay`, but scheduler and storage behavior make a fixed millisecond claim inappropriate. The recovered database remains transactionally consistent; acknowledged transactions can be absent.

**When this is acceptable:**
- Logging/analytics inserts where losing a few seconds of data is tolerable
- Session state or cache writes that can be reconstructed
- High-throughput event ingestion with downstream consumers that handle replays

**When this is NOT acceptable:**
- Financial transactions
- Any write whose acknowledgement contract promises survival of a host crash
- Writes that trigger irreversible side effects (sent emails, API calls)

The throughput difference is entirely workload- and storage-dependent. Benchmark commit-latency distributions and the business value of the enlarged acknowledgement window; do not reuse a generic commits-per-second multiplier.

### Cloud Gotchas: Not All fsync Is Equal

Cloud block storage introduces a layer of abstraction that changes durability guarantees.

Provider volume names, limits, and service-level objectives change. At design and deployment time, record:

- whether acknowledgement means one device, replicated storage in one zone, or multiple zones;
- whether an outage makes the volume unavailable versus permanently loses it;
- provisioned and burst IOPS/throughput limits, queue-depth behavior, and measured flush latency;
- snapshot/backup consistency and restore time; and
- the lifecycle of ephemeral local disks on stop, host replacement, and termination.

Tools such as `fio` can measure latency and throttling, but a successful benchmark cannot prove the provider's durability implementation. Use the provider contract for the failure guarantee and fault/restore exercises for operational evidence.

### Replication as Second Tier of Durability

A single disk (or single EBS volume) is not enough for production durability. Disks fail, AZs go offline, and entire regions can have outages.

```text
Durability tiers (PostgreSQL):

Tier 0: synchronous_commit = off
  - COMMIT does not wait for local WAL flush; bytes may be in process or OS buffers
  - Risk: lose the asynchronous-flush window on crash
  - Use: ephemeral data

Tier 1: synchronous_commit = on (default)
  - WAL fsynced to local disk before COMMIT returns
  - Does not by itself cover loss or unavailability of that storage fault domain
  - Use: single-node development, small deployments

Tier 2: Synchronous streaming replication
  - With the appropriate acknowledgement mode, WAL is flushed on a standby
    before COMMIT returns
  - synchronous_standby_names = 'standby1'
  - Risk: simultaneous failure of primary + standby
  - Cost: commit latency and availability include the required standby path
  - Use: production databases requiring durability

Tier 3: Synchronous replication to multiple standbys across AZs
  - synchronous_standby_names = 'FIRST 2 (standby1, standby2, standby3)'
  - Risk: required quorums can make writes unavailable; correlated failure remains
  - Cost: commit latency includes the slowest required acknowledgement
  - Use: critical financial/healthcare systems
```

Replication is not a backup: it also propagates accidental deletes, bad migrations, and some forms of corruption. Point-in-time recovery and regularly tested restores are separate requirements.

---

## Production Failure Modes (Transaction-Specific)

These are failure patterns specific to transaction misuse. → see [Failure Modes](06-failure-modes.md) for the general taxonomy.

### Lost Update Without Proper Isolation

The most common transaction bug in production.

**The pattern (read-then-write):**

```python
# DANGEROUS: Python with psycopg2 (PostgreSQL)
# Two concurrent requests both try to increment a counter

# Request 1                          # Request 2
cur.execute("SELECT count            cur.execute("SELECT count
  FROM counters WHERE id=1")           FROM counters WHERE id=1")
count = cur.fetchone()[0]  # 10      count = cur.fetchone()[0]  # 10
count += 1                            count += 1
cur.execute("UPDATE counters          cur.execute("UPDATE counters
  SET count=%s WHERE id=1",             SET count=%s WHERE id=1",
  (count,))                             (count,))
# Final value: 11 (should be 12)
```

**The fix — atomic UPDATE:**

```sql
-- Correct: single atomic statement, no read-then-write race
UPDATE counters SET count = count + 1 WHERE id = 1;
```

**When you must read-then-write (complex logic):**

```sql
-- Use SELECT FOR UPDATE to acquire a row lock
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
SELECT balance FROM accounts WHERE id = 42 FOR UPDATE;
-- Row is now locked; concurrent transactions block here
-- ... compute new balance in application ...
UPDATE accounts SET balance = 350.00 WHERE id = 42;
COMMIT;
```

### Partial Commit Visibility in Read Committed

Read Committed is the default in PostgreSQL. Each **statement** in a transaction sees a fresh snapshot. This causes subtle bugs in long transactions.

```sql
-- Session 1 (reporting query)
BEGIN;
SELECT sum(balance) FROM accounts WHERE region = 'US';
-- Returns $1,000,000

-- Meanwhile, Session 2 commits: moves $50,000 from US to EU account

SELECT sum(balance) FROM accounts WHERE region = 'EU';
-- This SELECT sees Session 2's commit! Different snapshot than the first SELECT.
-- The report shows $50,000 appearing from nowhere.
COMMIT;
```

**The fix:** use Repeatable Read for reporting queries that must see a consistent snapshot.

```sql
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT sum(balance) FROM accounts WHERE region = 'US';
-- ... even if other transactions commit here ...
SELECT sum(balance) FROM accounts WHERE region = 'EU';
-- Both SELECTs see the same snapshot
COMMIT;
```

### Long-Running Transactions Holding Resources

```text
Symptoms:
  - Lock wait timeouts on unrelated queries
  - Bloated table sizes (PostgreSQL: dead tuples not vacuumed)
  - Replication lag (slot can't advance past long tx)
  - "too many clients already" connection exhaustion

Root causes:
  - BEGIN with no matching COMMIT (idle in transaction)
  - Application exception skipping COMMIT/ROLLBACK
  - Batch jobs running in a single transaction

Monitoring (PostgreSQL):
```

```sql
-- Find long-running transactions
SELECT pid, now() - xact_start AS duration, state, query
FROM pg_stat_activity
WHERE state = 'idle in transaction'
  AND now() - xact_start > interval '5 minutes'
ORDER BY duration DESC;

-- Nuclear option: terminate the session
SELECT pg_terminate_backend(pid);
```

**Prevention:**

```text
# postgresql.conf
idle_in_transaction_session_timeout = '30s'   # kill idle-in-transaction after 30s
statement_timeout = '60s'                      # kill any statement after 60s
lock_timeout = '5s'                            # fail fast on lock waits
```

### Autocommit Misuse

Most database drivers default to autocommit=on, wrapping each statement in its own transaction. This is correct for simple queries but causes problems when developers don't realize multi-statement logic needs explicit transactions.

```python
# DANGEROUS: each statement is a separate transaction
conn.autocommit = True
cur.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
# ← crash here means money vanished
cur.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2")
```

```python
# CORRECT: explicit transaction
conn.autocommit = False
try:
    cur.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
    cur.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2")
    conn.commit()
except Exception:
    conn.rollback()
    raise
```

**SQLAlchemy context manager pattern (recommended):**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

engine = create_engine("postgresql+psycopg2://localhost/mydb")

with Session(engine) as session, session.begin():
    # All operations in this block are a single transaction
    session.execute(text("UPDATE accounts SET balance = balance - 100 WHERE id = 1"))
    session.execute(text("UPDATE accounts SET balance = balance + 100 WHERE id = 2"))
# Automatic COMMIT on clean exit, ROLLBACK on exception
```

---

## Decision Framework

### Isolation Level Selection

| Use Case | Starting Point | Why | What to Measure/Test |
|----------|----------------|-----|----------------------|
| Independent CRUD writes | Read Committed + schema constraints | Avoids unnecessary serialization when writes do not share an invariant | Lost-update behavior and constraint races |
| Transfer over a known row set | Atomic conditional statements or deterministic `SELECT FOR UPDATE` | Serializes modifications to the exact accounts involved | Deadlocks, lock waits, and timeout ambiguity |
| Predicate/global financial invariant | Serializable + retry loop | Database detects executions that cannot be serialized | Abort rate and correctness under generated concurrency |
| Reporting/analytics | Repeatable Read; consider read-only/deferrable Serializable where supported | Stable snapshot across statements | Snapshot age, vacuum impact, replica lag |
| Inventory decrement | Atomic conditional `UPDATE ... WHERE available >= :n` | Check and decrement are one statement | Contention and zero-row retry semantics |
| Counter increment | Atomic `UPDATE` | Removes the application read-then-write window | Hot-row throughput and sharding threshold |

### When to Use 2PC vs Saga vs Outbox

| Pattern | Guarantees | Latency | Complexity | Use When |
|---------|-----------|---------|------------|----------|
| **2PC** | Atomic decision across transactional participants | Prepare/decision coordination and lock retention | Coordinator recovery and in-doubt operations | A transaction manager and every participant implement a compatible prepare protocol |
| **Saga** | Durable sequence of local transactions; compensation is semantic recovery, not rollback | Usually asynchronous step latency | Ambiguous side effects and compensation design | Long-lived workflows and APIs that cannot join one transaction |
| **Outbox** | Atomic local state + publication intent; relay is normally at least once | Relay/CDC delay | Duplicate handling and backlog operations | A local transaction must eventually produce a broker message |

**Decision heuristic:**
- Can all participants be in the same database? → Use a local transaction. No 2PC needed.
- Do all participants support prepare, and can you operate in-doubt recovery? → 2PC may be viable after failure and latency testing.
- Does the workflow involve external services (payment, email, APIs)? → Use a durable workflow/saga with idempotency, reconciliation, and explicit compensation where possible.
- Do you need to publish an event atomically with a database write? → Outbox pattern.

---

## Code Examples

### PostgreSQL: Two Sessions Showing Isolation

Open two `psql` sessions connected to the same PostgreSQL database.

**Setup:**

```sql
CREATE TABLE accounts (id INT PRIMARY KEY, balance NUMERIC NOT NULL);
INSERT INTO accounts VALUES (1, 500), (2, 200);
```

**Demo: Read Committed prevents dirty reads but allows non-repeatable reads:**

```text
Session A (default Read Committed):      Session B:
─────────────────────────────────────     ──────────────────────────────
BEGIN;                                    BEGIN;
UPDATE accounts SET balance = 400
  WHERE id = 1;
                                          SELECT balance FROM accounts
                                            WHERE id = 1;
                                          -- Returns 500 (not 400!)
                                          -- Dirty read prevented ✓
COMMIT;
                                          SELECT balance FROM accounts
                                            WHERE id = 1;
                                          -- Returns 400
                                          -- Non-repeatable read! The
                                          -- value changed within the
                                          -- same transaction.
                                          COMMIT;
```

**Demo: Repeatable Read provides snapshot consistency:**

```text
Session A:                                Session B (Repeatable Read):
─────────────────────────────────────     ──────────────────────────────
                                          BEGIN TRANSACTION ISOLATION
                                            LEVEL REPEATABLE READ;
                                          SELECT balance FROM accounts
                                            WHERE id = 1;
                                          -- Returns 500
BEGIN;
UPDATE accounts SET balance = 400
  WHERE id = 1;
COMMIT;
                                          SELECT balance FROM accounts
                                            WHERE id = 1;
                                          -- Still returns 500!
                                          -- Snapshot is frozen at BEGIN
                                          COMMIT;
```

### Python SQLAlchemy: SELECT FOR UPDATE Pattern

```python
# SQLAlchemy 2.x / PostgreSQL
from decimal import Decimal
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

engine = create_engine(
    "postgresql+psycopg2://user:pass@localhost:5432/mydb",
    pool_size=10,
    pool_pre_ping=True,  # detect stale connections
)

def transfer(from_id: int, to_id: int, amount: Decimal) -> None:
    """Transfer funds between accounts with proper locking.

    Acquires these account-row locks in a consistent order to avoid the
    opposite-transfer deadlock. Other resources can still form a deadlock.
    """
    # Consistent ordering avoids the opposite-transfer cycle for these rows.
    first_id, second_id = sorted([from_id, to_id])

    with Session(engine) as session, session.begin():
        # Acquire row locks in deterministic order
        rows = session.execute(
            text("""
                SELECT id, balance FROM accounts
                WHERE id IN (:id1, :id2)
                ORDER BY id
                FOR UPDATE
            """),
            {"id1": first_id, "id2": second_id},
        ).fetchall()

        balances = {row.id: row.balance for row in rows}

        if balances[from_id] < amount:
            raise ValueError(f"Insufficient funds: {balances[from_id]} < {amount}")

        session.execute(
            text("UPDATE accounts SET balance = balance - :amt WHERE id = :id"),
            {"amt": amount, "id": from_id},
        )
        session.execute(
            text("UPDATE accounts SET balance = balance + :amt WHERE id = :id"),
            {"amt": amount, "id": to_id},
        )
    # COMMIT happens here; ROLLBACK on exception
```

Key details in this example:
- **Lock ordering** (`sorted([from_id, to_id])`) prevents the opposite-transfer cycle for these rows; it does not prove the whole transaction is deadlock-free
- **`FOR UPDATE`** acquires row-level exclusive locks, blocking concurrent modifications
- **`session.begin()` context manager** ensures ROLLBACK on exception
- **`pool_pre_ping=True`** handles connections dropped by PgBouncer or network timeouts

---

## Verify the Deployed Contract

Verify the deployed release and settings. This table lists contract dimensions, not universal defaults.

| Database | Common Default Isolation | Durability Mechanism | Verify Before Claiming a Guarantee |
|----------|--------------------------|----------------------|------------------------------------|
| PostgreSQL | Read Committed | WAL + configured flush policy | `fsync`, `synchronous_commit`, full-page writes, standby acknowledgement, pool/session settings |
| MySQL InnoDB | Repeatable Read | Undo + redo + doublewrite; binary log if enabled | Redo flush and binary-log sync policies, storage flush semantics, replication acknowledgement |
| MongoDB | Read and write semantics depend on `readConcern`/`writeConcern`; transactions can request snapshot reads | WiredTiger journal + replica-set acknowledgement | Explicit concern levels, journaling, majority acknowledgement, and transaction retry labels |
| SQLite | Serializable transactions by serializing writes; WAL and rollback-journal modes have different concurrency | WAL or rollback journal | Filesystem/locking support, `synchronous` mode, journal mode, single-writer contention |
| CockroachDB | Serializable by default; supported levels can vary by release/configuration | Replicated consensus log + storage-engine WAL | Transaction retry contract, selected isolation level, replica placement, quorum availability |
| SQL Server | Read Committed, commonly locking unless read-committed snapshot is enabled | Transaction log | Database-level snapshot settings, delayed durability, availability-group acknowledgement |

### Warning: Check Your Defaults

Defaults vary by product and release, and managed services, containers, or provisioning scripts can override them. After every deployment, verify the settings that define your contract:

```sql
-- PostgreSQL: verify critical durability settings
SHOW synchronous_commit;         -- should be 'on' for critical data
SHOW fsync;                       -- should be 'on' (NEVER disable in production)
SHOW full_page_writes;            -- should be 'on' (prevents torn pages)
SHOW wal_level;                   -- 'replica' or 'logical' for replication

-- MySQL: verify InnoDB settings
SHOW VARIABLES LIKE 'innodb_flush_log_at_trx_commit';  -- should be 1
SHOW VARIABLES LIKE 'innodb_doublewrite';               -- should be ON
SHOW VARIABLES LIKE 'sync_binlog';                      -- should be 1 for durability
```

---

## Key Takeaways

1. **Atomicity has a boundary.** Database effects can commit together; external messages and APIs need outbox, idempotency, or workflow protocols.
2. **Consistency is an invariant contract.** Encode each invariant in schema or transaction logic and test it at the chosen isolation level. Distributed databases can enforce cross-range constraints, at a coordination cost.
3. **Isolation names are implementation-specific.** Select from concrete anomalies and retry behavior, not a universal performance multiplier. Scope transaction settings explicitly.
4. **Durability needs a failure model.** Local WAL flush, synchronous replication, and tested backup/restore protect against different failures.
5. **2PC provides an atomic decision but can hold resources while that decision is unavailable.** Replicating transaction metadata improves availability without eliminating distributed-commit costs.
6. **Group commit amortizes flush cost.** Tune only from measured WAL waits and latency objectives.
7. **Avoid read-then-write races.** Prefer atomic conditional statements; otherwise use explicit locking or Serializable transactions with deterministic retry behavior.

---

## References

- [PostgreSQL: Transaction Isolation](https://www.postgresql.org/docs/current/transaction-iso.html) — level semantics, anomalies, and serialization failures
- [PostgreSQL: SET TRANSACTION](https://www.postgresql.org/docs/current/sql-set-transaction.html) — transaction-scoped versus session-scoped characteristics
- [PostgreSQL: Reliability and the Write-Ahead Log](https://www.postgresql.org/docs/current/wal-reliability.html) — flush, storage-cache, and filesystem assumptions
- [MySQL: SAVEPOINT, ROLLBACK TO SAVEPOINT, and RELEASE SAVEPOINT](https://dev.mysql.com/doc/refman/8.4/en/savepoint.html) — InnoDB savepoint and retained-lock behavior
- [CockroachDB: Foreign Key Constraint](https://www.cockroachlabs.com/docs/stable/foreign-key.html) — enforced referential integrity in a distributed SQL database
- [Spanner: Foreign Keys](https://cloud.google.com/spanner/docs/foreign-keys/overview) — enforced and informational foreign-key semantics
