# ACID Transactions

> The database said "committed" and then the power went out. What exactly were you promised, and which part of the engine keeps that promise?

## TL;DR

ACID names four properties of a transaction, but the label alone is not a complete contract. The exact isolation level, durability boundary, replication acknowledgement, and failure assumptions vary by database and configuration. Each letter hides real engineering tradeoffs: undo versus redo, flush latency versus durability, and concurrency control versus throughput. Read the product's documented semantics and verify the deployed configuration instead of treating "ACID" as a binary feature flag.

---

## The Problem ACID Solves

Before looking at each property, it helps to see what a database is being asked to protect us from. Take the standard example, a bank transfer that moves USD 100 from Account A to Account B. Written as the steps a program would actually perform, it looks like this:

```text
1. Read balance of A: $500
2. Subtract $100 from A: $400
3. Write new balance to A
4. Read balance of B: $200
5. Add $100 to B: $300
6. Write new balance to B
```

Each of these six steps is individually simple, and the sequence is correct if it runs alone, to completion, on hardware that never fails. Production gives us none of those three conditions, and each one that is missing produces its own class of failure.

The first class is the crash. If the process dies after step 3, A has lost USD 100 and B has gained nothing, so money has vanished from the system. If it dies during step 6, the disk may hold a partial write, and B's balance is then neither USD 200 nor USD 300 but a block of corrupted bytes.

The second class is concurrency. Suppose two transfers out of A run at the same time. Both read USD 500, both subtract USD 100, and both write USD 400. A should now hold USD 300, and the bank has created USD 100 from nothing. A milder version involves a reader: a reporting query that runs between steps 3 and 6 sees A debited and B not yet credited, and produces books that do not balance.

The third class is the most uncomfortable, because it happens after the database has said that everything is fine. The database reports that COMMIT succeeded, and then power fails. If the kernel was holding the write in its page cache and nobody had called fsync, the write is gone on restart. Even when fsync was called, a device whose firmware acknowledged the write while the data sat in a volatile cache will lose it in the same way.

None of these is a theoretical failure. A correctly implemented and correctly configured transaction addresses all of them, for writes inside the database's transactional boundary. That qualification matters as much as the guarantee. A transaction does not make an external API call, a message-broker publish, a filesystem write, or an incorrectly configured storage device atomic with the database commit. Most of the difficult bugs involving transactions come from an effect that somebody assumed was inside that boundary and was not.

---

## Atomicity: Deep Dive

### What It Actually Means

Atomicity is often misread as "all the operations happen at the same instant." That idea is closer to isolation. Atomicity makes a narrower promise.

**Atomicity means all-or-nothing for the database-managed effects in the transaction.** If the transaction commits, all of those effects commit. If it aborts, none of them become committed state. The word "atom" here refers to indivisibility with respect to failure: an observer after a crash finds either the whole transaction or no trace of it. Atomicity says nothing about side effects performed outside that transactional resource.

### Why It Matters

Without atomicity, every multi-statement operation is a place where data can be corrupted. A crash, a network timeout, or a constraint violation partway through leaves the database in an intermediate state that no line of the application ever intended to be visible. The alternative is for application code to detect the partial state and clean it up. That code has to work correctly when it is itself interrupted, and has to cover every pair of statements that might be separated by a failure. In practice nobody gets this right, which is why we push the problem down into the database, where it can be solved once.

### Undo Log vs Redo Log

To promise all-or-nothing, the database needs to be able to do two things after a failure: remove the effects of transactions that did not finish, and reinstate the effects of transactions that did. Undo and redo logging are the two complementary techniques most engines use for these. They are not the only implementation choices, and a specific engine can combine them with MVCC, shadow paging, copy-on-write, or transaction-status metadata.

An *undo log* (or rollback log) records enough prior state to reverse a change. In an MVCC engine such as InnoDB it has a second job, which is to let a reader reconstruct the older version of a row that its snapshot is entitled to see. On an explicit rollback the engine walks the transaction's undo records and reverses each change; crash recovery combines redo with undo and transaction-status processing. InnoDB uses undo as its primary mechanism for atomicity and stores it in the system tablespace or in dedicated undo tablespaces.

A *redo log*, also called a write-ahead log or WAL, records enough information to reproduce a page change or a logical change. It is not necessarily a copy of the new row value. The rule that gives the write-ahead log its name is that the relevant log record must reach the configured durability boundary before the corresponding data page is allowed to become the sole durable copy of the change. On crash recovery, replaying the log restores the database to a state from which each transaction's commit or abort status can be resolved. PostgreSQL uses its WAL (the `pg_wal` directory) as the primary mechanism. The log is append-only, so writing it is sequential I/O, which is far cheaper than the random page writes it allows the engine to defer.

InnoDB uses both logs, and the order of operations shows why:

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

The two lifecycles differ because of one design decision. InnoDB updates rows in place, so the old value is overwritten and must be saved somewhere else if the transaction is to be rolled back; that place is the undo log. PostgreSQL does not overwrite. An UPDATE writes a new version of the row and leaves the old one in the heap until vacuum removes it, so the "undo" information is the old row itself, and no separate undo log is needed for atomicity. The price is paid later and elsewhere, in vacuum work and table bloat.

### How ROLLBACK Works: Undo Chains and Transaction Status

InnoDB's rollback follows directly from its undo design. Each transaction keeps a linked list of its undo records, newest first:

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

Rollback walks this chain backward, undoing one change at a time. This is why rolling back a transaction that modified millions of rows can take as long as the transaction itself took to run, and why killing a long-running batch job in MySQL does not free its resources promptly.

PostgreSQL normally avoids physically reversing each heap change in the foreground. The transaction is recorded or treated as aborted, its tuple versions are invisible to every later transaction, and vacuum reclaims them at some later time. The abort the client sees can therefore be fast. The work has been deferred and not avoided, and a large aborted transaction can leave substantial table and index bloat behind it.

### Savepoints and Partial Rollback

All-or-nothing is sometimes coarser than the business logic needs. An order workflow may want to try one fulfillment path, find that it fails, and try another without discarding the order row it has already written. Savepoints provide this by letting us roll back part of a transaction without aborting the whole.

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

The two engines implement this differently. PostgreSQL implements savepoints with subtransactions and tracks overflowed subtransaction relationships in `pg_subtrans`. InnoDB records a position in the transaction's existing undo history; it does not create a new undo-log segment for every savepoint. In both, `ROLLBACK TO SAVEPOINT` reverses the later changes but may retain the locks they acquired, so a savepoint is not a resource-isolation boundary.

Deeply nested savepoints carry a cost that is easy to overlook. With thousands of subtransactions, PostgreSQL's `pg_subtrans` lookups can become a bottleneck that slows unrelated sessions. A savepoint inside a loop is usually a sign that the transaction design should be reconsidered.

### Distributed Atomicity: Two-Phase Commit (2PC)

When a transaction spans several database nodes, a local undo log is not enough. Each node can make its own part all-or-nothing, but nothing yet ensures that the nodes reach the same outcome. If shard A commits the debit and shard B aborts the credit, each node is locally atomic and the transfer as a whole is not. The classic solution is the **two-phase commit protocol**, which separates the question "can you commit?" from the instruction "commit."

```text
Coordinator (transaction manager)
├── Participant A (shard holding Account A)
└── Participant B (shard holding Account B)

Phase 1: Prepare (vote):
  Coordinator → A: "PREPARE transaction T1"
  Coordinator → B: "PREPARE transaction T1"
  A: writes all changes to durable log, acquires locks, responds YES
  B: writes all changes to durable log, acquires locks, responds YES

Phase 2: Commit (decision):
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

The important property of the prepare phase is that a participant that votes YES gives up its right to abort. It has made the changes durable and holds the locks, and from that point it must do whatever the coordinator decides. That is what makes the outcome atomic, and it is also the protocol's weakness. If participants lose access to the coordinator's decision after they have prepared, they cannot make progress:

```text
Timeline:
  t0: Coordinator sends PREPARE to A and B
  t1: A votes YES, B votes YES (both holding locks, changes durable)
  t2: Coordinator crashes before a decision is durably available to participants

  A and B are now "in doubt": neither can unilaterally choose an outcome:
  - Committing could disagree with an ABORT decision
  - Aborting could disagree with a COMMIT decision
  - Prepared resources remain held until the decision service recovers or an
    operator resolves the transaction from authoritative evidence
```

**In-doubt transactions** are operationally dangerous. They hold locks, they block other transactions that touch the same rows, and if the coordinator cannot recover they need manual resolution by someone who can establish the right outcome from authoritative evidence.

```sql
-- PostgreSQL: find in-doubt transactions
SELECT gid, prepared, owner, database
FROM pg_prepared_xacts;

-- Manual resolution (ONLY when you've confirmed the correct outcome):
COMMIT PREPARED 'transfer_1001_partA';
```

Several mitigations exist, and none removes the problem entirely. The coordinator can write its decision to a replicated, durable log before starting phase 2, so that the decision survives the loss of one machine. Participants can time out and ask the coordinator, or its replicas, what was decided. Three-phase commit adds a further state to the protocol, but it does not solve consensus in an asynchronous network that can partition, and it is uncommon in database deployments. Some distributed databases replicate the transaction record itself with a consensus protocol. This removes the single-machine coordinator as the only copy of the decision. It does not make distributed commit free of blocking, retries, or unavailable quorums.

Two-phase commit also has a performance cost in the normal case. The prepare and decision phases add messages and durable metadata, and locks are held for longer than a local transaction would hold them. The number of round trips and log flushes depends on the database's protocol, replication topology, batching, and parallelism, so we measure the implementation we operate. A fixed latency multiplier is not portable from one system to another.

---

## Consistency: Invariants Are a Shared Responsibility

### What the Database Enforces vs What It Can't

Consistency is the odd letter in the acronym, because the database alone cannot provide it. In ACID, **consistency means that a transaction preserves the invariants the system claims**. It is a property of the transaction program together with the database's guarantees, and not a promise that the engine understands every business rule. The engine can reject a violation only for rules that have been expressed in its schema, in transaction logic, or in trusted stored code.

What the engine can enforce is whatever we have declared to it: NOT NULL and CHECK constraints, UNIQUE and PRIMARY KEY constraints, FOREIGN KEY referential integrity, EXCLUDE constraints in PostgreSQL, and invariants implemented in triggers. What it cannot do is infer rules we never stated. It does not know that an account balance should equal the sum of its ledger entries, that every submitted order must have at least one line item, or that the total across all accounts must stay constant. A rule that exists only in application code is invisible to the engine.

Some of those rules can be encoded with constraints, triggers, materialized state, or serializable transaction logic, and others need a different data model. The useful question is therefore not whether a rule is a "database rule" or an impossible one. It is **which invariant is encoded where, under which isolation level, and how it is tested under concurrency**.

### The "C" Overloading Problem

The letter C carries a second difficulty, which is that the same word names different ideas in different parts of the field:

| Context | "Consistency" means | Enforced by |
|---------|-------------------|-------------|
| ACID | A transaction preserves the system's stated invariants | Schema + transaction program + concurrency control |
| CAP theorem | A read/write object remains linearizable while messages may be lost or delayed | Usually quorum/consensus protocols |
| Replica convergence | Replicas that receive the same updates eventually compute the same state | Replication and merge protocol |

These three concepts have little in common beyond the word. → see [Consistency Models](05-consistency-models.md) for linearizability, causal consistency, and eventual consistency. When someone says "this system is consistent," we ask which definition they mean before agreeing or disagreeing.

### Deferred Constraints

Some invariants hold at the end of a transaction and cannot hold at every step within it. Mutual foreign keys are the standard case: whichever row we insert first refers to a row that does not exist yet.

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

A constraint declared deferrable can also be deferred for a single transaction:

```sql
BEGIN;
SET CONSTRAINTS fk_department DEFERRED;
-- ... operations that temporarily violate the constraint ...
COMMIT;  -- constraint checked here
```

Deferral is useful in a small number of situations: circular foreign keys such as the one above, bulk loads in which intermediate states violate uniqueness, graph structures with parent-child self-references, and schema migrations that reorder data.

There is a caveat. A deferred unique constraint in PostgreSQL uses a different index mechanism from an immediate one and can perform differently on large tables, so we test it with data at production scale.

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

There are several ways to handle a reference that crosses shards, and they differ in what they give up. A database-enforced distributed constraint gives the strongest semantics and charges for it in extra index maintenance, locality constraints, and write-path cost. Denormalizing copies the referenced data into the local shard and accepts that the copy can be stale. Application-level enforcement checks before writing and accepts the race. Event-driven cleanup lets broken references occur, then detects and repairs them asynchronously. The simplest option, where the data model allows it, is to avoid the cross-shard reference by placing related data on the same shard.

The choice is a tradeoff between data placement and correctness. It is not evidence that distributed databases cannot preserve referential integrity.

### Constraint Enforcement Becomes Distributed Work

On a single node, checking a constraint is an index lookup. When the constrained rows or indexes live on another node, the same check becomes distributed work. Every cross-shard constraint check adds network round trips. Locks held across nodes make deadlock detection expensive. The system must also maintain the extra indexes and the distributed transaction metadata that the check depends on.

Some systems deliberately omit or relax these constraints for latency, availability, or operational simplicity. Distributed SQL systems may preserve them and charge the coordination cost. "ACID" does not imply that constraints are local to one shard, and CAP does not require abandoning invariants during normal operation. → see [CAP Theorem](06-cap-theorem.md)

---

## Isolation: The Expensive Letter

### The Core Challenge

Isolation answers the question "what do concurrent transactions see of one another?" The ideal answer is serializability: the result is the same as if the transactions had run one at a time, in some order. This is simple to state and expensive to provide, because the database must either prevent conflicting transactions from overlapping or detect the overlap and abort one of them. Both cost throughput. Databases therefore offer weaker levels that permit specific anomalies in exchange for more concurrency, and the level in force is very often a default that nobody on the team chose.

### Isolation Levels Summary

| Level | Dirty Reads | Non-Repeatable Reads | Phantom Reads | Write Skew |
|-------|-------------|----------------------|---------------|------------|
| Read Uncommitted | Yes | Yes | Yes | Yes |
| Read Committed | No | Yes | Yes | Yes |
| Repeatable Read | No | No | Implementation-dependent; PostgreSQL and InnoDB snapshot implementations prevent them | Often possible under snapshot isolation |
| Serializable | No | No | No | No |

This table is a starting point and not a portable specification. Engines map the SQL level names onto different mechanisms, and PostgreSQL, for one, silently promotes `READ UNCOMMITTED` to `READ COMMITTED`. Serializable prevents these anomalies for transactions that commit, but it does so partly by refusing to commit some of them, and the application must retry on a serialization failure.

Four implementation approaches are in common use. *Two-phase locking* has transactions acquire locks and block one another; SQL Server uses it for Serializable. *MVCC* keeps multiple versions of each row so that ordinary reads usually do not block ordinary writes, although locks are still needed for some writes, for constraints, and for schema operations; PostgreSQL and InnoDB use it for most levels. *Optimistic concurrency control* assumes conflicts are rare, lets transactions run without locks, and validates at commit time; some in-memory databases use it. *Serializable snapshot isolation* adds dependency tracking to MVCC and aborts a transaction when the tracked dependencies could form a non-serializable cycle; it has been PostgreSQL's Serializable implementation since 9.1.

→ see [Isolation Levels](02-isolation-levels.md) for MVCC internals, locking protocols, SSI implementation details, and anomaly deep dives.

### Connection Pool Gotcha: Transaction-Scoped vs Session-Scoped Settings

Connection pools such as PgBouncer and HikariCP introduce a production bug that is easy to create and hard to see, because it depends on which connection a request happens to borrow.

PostgreSQL's `SET TRANSACTION` applies only to the current transaction and must run after `BEGIN` and before the first query. It does **not** leak into later transactions. By contrast, `SET SESSION CHARACTERISTICS AS TRANSACTION ...` changes the default for every subsequent transaction on that backend session. If the pool does not reset session state when a connection is returned, the next borrower inherits that default without knowing it.

The safe forms keep the scope visible in the statement itself:

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

We prefer a transaction API or `BEGIN ... ISOLATION LEVEL`, so that the scope is visible where the transaction starts. If code changes session defaults or any other session state, we configure and test the pool's reset behavior. Transaction pooling does not make arbitrary session commands safe.

---

## Durability: The Latency Letter

### Why It Matters

Durability defines which failures a successful commit is promised to survive, and the answer is never "all of them." A local WAL flush may cover a process crash, an OS crash, and a power failure on one host, and still not cover the loss of the device, an availability-zone outage, an operator error, or a regional disaster. Every stronger promise costs latency, because the commit must wait for more things to happen before it can be acknowledged. The acknowledgement policy and the failure model have to be stated together, or the word "durable" means nothing in particular.

### fsync Deep Dive

`fsync()` is the most common local durability boundary, and most misunderstandings about durability are misunderstandings about what it does. A `write()` call does not put data on disk. It copies data into the kernel's page cache and returns. Only `fsync()` asks the kernel to push that data through the filesystem and the device and to wait until they report that it is persistent.

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

That description assumes every layer beneath the database honors the request. There are three places where the contract can break:

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

The second failure point is not hypothetical. In 2018 the PostgreSQL community found that the database had been handling fsync errors unsafely for about twenty years. When `fsync()` failed, PostgreSQL retried it, on the assumption that the dirty page was still in the kernel page cache waiting to be written. On some Linux kernels (before 5.2) that assumption was false: after a failed writeback the kernel marked the page clean and reported the error once. The retry then found nothing dirty to write and succeeded. PostgreSQL concluded that the data was durable when it had never reached the disk.

Since version 12, PostgreSQL responds to an fsync failure with a PANIC and goes through crash recovery, replaying the WAL, because after such an error the state of the kernel's cache cannot be trusted. The lesson applies beyond PostgreSQL: an error on the durability path cannot be retried like an ordinary error, because the retry may succeed without doing the work.

### WAL Mechanics

The write-ahead log is what makes durability affordable. Without it, a commit would have to write every modified page to its place in the data files, which means many random writes per transaction. With it, a commit only has to append a description of the changes to one sequential file and flush that file; the data pages can be written later, in bulk. InnoDB's redo log serves the same role.

In PostgreSQL the WAL is a sequence of fixed-size segment files:

```text
$PGDATA/pg_wal/
├── 000000010000000000000001   (16 MB segment, default)
├── 000000010000000000000002
├── 000000010000000000000003
└── archive_status/

Segment naming: TimelineID + (LSN >> 24)
Default segment size: 16 MB (configurable at initdb with --wal-segsize)
```

Each record in those segments describes one change in enough detail to repeat it:

```text
Each WAL record contains:
  - LSN (Log Sequence Number): unique, monotonically increasing position
  - Resource manager ID (heap, btree, hash, etc.)
  - Record type and data needed to redo that resource-manager operation
  - For a data page's first change after a checkpoint, commonly a full-page image
    when full_page_writes is enabled; later records can contain smaller deltas
  - CRC checksum
```

If the WAL only ever grew, recovery after a crash would have to replay it from the beginning of time. Checkpoints bound that work. A checkpoint writes all dirty buffers from `shared_buffers` to the data files and then records the checkpoint's LSN. Everything before that position is known to be in the data files, so crash recovery only has to replay the WAL from the last checkpoint forward. The checkpoint interval is therefore a tradeoff between steady-state write load and recovery time:

```text
Tuning tradeoffs:
  - More frequent checkpoints can reduce redo distance but increase write pressure
  - Less frequent checkpoints can increase redo distance and retained WAL
  - Recovery is not just sequential WAL reading: page access, storage latency,
    full-page images, CPU, and follow-on cache warming all matter

Measure restart time under the production WAL rate and dataset; do not derive an
RTO from WAL bytes divided by advertised sequential bandwidth.
```

InnoDB makes the same tradeoff with a different structure. Its redo log is a fixed-size ring and not a growing sequence of files:

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

A flush is the slowest step of a commit, often by orders of magnitude, and a naive design would pay for one per transaction. It does not have to. Every locally durable commit requires its commit record to be covered by a durable WAL flush, but it does not require a flush of its own. Because the log is sequential, flushing up to a given position makes every record before that position durable, whichever transaction wrote it. **Group commit** exploits this by letting many concurrent commits share a single flush. Read-only transactions and asynchronous commit policies follow different paths.

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

PostgreSQL forms groups naturally under load, since commits that arrive while a flush is in progress wait for the next one. It can also be told to wait deliberately so that the group is larger:

```text
# postgresql.conf

# How long to delay before flushing WAL, hoping more commits arrive
commit_delay = <measured delay in microseconds>

# Only delay if at least this many transactions are active
commit_siblings = <measured concurrency threshold>

# A deliberate delay can increase the batch size, but it also adds latency.
# Benchmark with the actual storage and concurrency distribution.
```

Tuning group commit is worthwhile when commit latency is dominated by WAL flush waits under many concurrent small transactions, and especially when the storage has high fsync latency, as network-attached and cloud volumes do. On a lightly loaded system the delay only adds latency, because no other commits arrive to share the flush.

MySQL has an additional complication, which is that it may have two logs to keep in step:

```text
MySQL coordinates InnoDB redo durability with binary-log ordering when the binary
log is enabled. `innodb_flush_log_at_trx_commit` and `sync_binlog` govern different
logs; changing only one can leave a crash window or replication inconsistency.
Verify both, plus whether the deployment enables the binary log, before claiming
a commit durability boundary.
```

### synchronous_commit = off: When Acceptable

PostgreSQL lets us give up the flush wait altogether. The `synchronous_commit` setting controls whether COMMIT waits for the WAL to be flushed, and it can be changed per transaction:

```sql
-- Per-transaction override (PostgreSQL)
SET LOCAL synchronous_commit = off;
-- Subsequent COMMIT returns immediately, WAL fsynced asynchronously
-- The loss window depends on WAL-writer scheduling and configuration.
```

With the setting off, an OS or database crash before the asynchronous flush can erase transactions that have already returned success. PostgreSQL documents the normal maximum delay in terms of `wal_writer_delay`, but scheduler and storage behavior make a fixed millisecond figure inappropriate. What we do not lose is consistency. The recovered database is transactionally consistent, and every transaction is either wholly present or wholly absent. Some acknowledged transactions are simply among the absent ones. This is quite different from turning `fsync` off, which can corrupt the database.

That trade is acceptable when the acknowledged write can be lost without harm: logging and analytics inserts where a few seconds of data is tolerable, session state or cache writes that can be reconstructed, and high-throughput event ingestion whose downstream consumers handle replays. It is not acceptable for financial transactions, for any write whose acknowledgement promises survival of a host crash, or for a write that triggers an irreversible side effect such as an email or an external API call, since the outside world would then remember something the database has forgotten.

The throughput gain depends entirely on workload and storage. We benchmark the commit-latency distribution and weigh it against the business cost of the enlarged loss window, and we do not reuse a generic commits-per-second multiplier.

### Cloud Gotchas: Not All fsync Is Equal

On a cloud volume, the "disk" that fsync flushes to is a network service, and what its acknowledgement means is defined by the provider's contract and not by anything the database can observe. Provider volume names, limits, and service-level objectives also change over time. At design and deployment time we therefore record:

- whether acknowledgement means one device, replicated storage in one zone, or multiple zones;
- whether an outage makes the volume unavailable versus permanently loses it;
- provisioned and burst IOPS/throughput limits, queue-depth behavior, and measured flush latency;
- snapshot/backup consistency and restore time; and
- the lifecycle of ephemeral local disks on stop, host replacement, and termination.

Tools such as `fio` can measure latency and throttling, but a successful benchmark cannot prove how the provider implements durability. We rely on the provider's contract for the failure guarantee and on fault and restore exercises for operational evidence.

### Replication as Second Tier of Durability

A single disk, or a single network volume, is not enough for production durability. Disks fail, availability zones go offline, and whole regions have outages. Replication extends the durability boundary past one machine, and each step outward adds to what a commit must wait for:

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

Replication is not a backup. A replica faithfully applies whatever the primary does, including an accidental delete, a bad migration, and some forms of corruption. Point-in-time recovery and regularly tested restores are separate requirements, and they protect against the failures that replication copies.

---

## Production Failure Modes (Transaction-Specific)

The failures in this section come from how applications use transactions, and not from faults in the database. → see [Failure Modes](03-failure-modes.md) for the general taxonomy.

### Lost Update Without Proper Isolation

The lost update is the most common transaction bug in production. It occurs whenever the application reads a value, computes a new one, and writes it back, and two requests do this at the same time:

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

Wrapping these statements in a transaction at the default isolation level does not help, because neither transaction's read conflicts with anything. The simplest fix removes the gap between the read and the write by having the database do the arithmetic in a single statement:

```sql
-- Correct: single atomic statement, no read-then-write race
UPDATE counters SET count = count + 1 WHERE id = 1;
```

When the new value depends on logic that cannot be expressed in one statement, we take a row lock at the time of the read, so that the second transaction waits until the first has committed:

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

Read Committed is the default in PostgreSQL, and it has a property that surprises people: each **statement** in a transaction sees a fresh snapshot. Two queries in the same transaction can therefore observe different committed states of the database.

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

Neither query saw uncommitted data, so no rule of Read Committed was broken, and the report is still wrong. A report that must be internally consistent should run at Repeatable Read, where the snapshot is taken once and used for the whole transaction:

```sql
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT sum(balance) FROM accounts WHERE region = 'US';
-- ... even if other transactions commit here ...
SELECT sum(balance) FROM accounts WHERE region = 'EU';
-- Both SELECTs see the same snapshot
COMMIT;
```

### Long-Running Transactions Holding Resources

An open transaction is not free while it sits idle. It holds its locks, and in an MVCC engine it also pins the oldest snapshot that vacuum must preserve, so one forgotten transaction can cause bloat in tables it never touched.

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

The cure is a limit and not a cleanup script. The database can be told to end transactions and statements that exceed a bound, so that a single forgotten COMMIT cannot degrade the whole system:

```text
# postgresql.conf
idle_in_transaction_session_timeout = '30s'   # kill idle-in-transaction after 30s
statement_timeout = '60s'                      # kill any statement after 60s
lock_timeout = '5s'                            # fail fast on lock waits
```

### Autocommit Misuse

Most database drivers default to autocommit, which wraps each statement in its own transaction. That is correct for single-statement work. It becomes a bug when a developer writes multi-statement logic without realizing that nothing ties the statements together:

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

The explicit form is correct but relies on every code path remembering to commit or roll back. A context manager makes the transaction boundary a property of the block's structure, so that it cannot be forgotten:

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

A short sequence of questions settles most cases. If all participants can live in the same database, we use a local transaction and need nothing else. If they cannot, but every participant supports prepare and we are able to operate in-doubt recovery, 2PC may be viable once we have tested its failure behavior and latency. If the workflow involves external services such as payment, email, or third-party APIs, which cannot join a transaction at all, we use a durable workflow or saga with idempotency, reconciliation, and explicit compensation where compensation is possible. If the need is only to publish an event atomically with a database write, the outbox pattern does that with the least machinery.

---

## Code Examples

### PostgreSQL: Two Sessions Showing Isolation

The behavior of isolation levels is easiest to believe after watching it. Open two `psql` sessions connected to the same PostgreSQL database and create a small table:

```sql
CREATE TABLE accounts (id INT PRIMARY KEY, balance NUMERIC NOT NULL);
INSERT INTO accounts VALUES (1, 500), (2, 200);
```

At Read Committed, session B never sees A's uncommitted change, but it does see the committed change in the middle of its own transaction:

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

At Repeatable Read, session B keeps the snapshot it started with, whatever A commits in the meantime:

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

Four details in this example carry the correctness. Sorting the IDs gives every transfer the same lock order, which prevents the cycle in which one transfer locks A then B while the opposite transfer locks B then A; it does not prove the whole transaction deadlock-free, since other resources can still form a cycle. `FOR UPDATE` takes row-level exclusive locks, so a concurrent transfer on the same accounts waits instead of reading a stale balance. The `session.begin()` context manager guarantees a ROLLBACK on any exception. And `pool_pre_ping=True` detects connections that PgBouncer or a network timeout has dropped, before the transaction begins on one.

---

## Verify the Deployed Contract

The theme of this chapter is that ACID is a set of configurable contracts, so the last step is to check what the running system is actually configured to do. This table lists the dimensions of the contract for common databases. The entries are not universal defaults, and we verify the deployed release and settings.

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

- [PostgreSQL: Transaction Isolation](https://www.postgresql.org/docs/current/transaction-iso.html): level semantics, anomalies, and serialization failures
- [PostgreSQL: SET TRANSACTION](https://www.postgresql.org/docs/current/sql-set-transaction.html): transaction-scoped versus session-scoped characteristics
- [PostgreSQL: Reliability and the Write-Ahead Log](https://www.postgresql.org/docs/current/wal-reliability.html): flush, storage-cache, and filesystem assumptions
- [MySQL: SAVEPOINT, ROLLBACK TO SAVEPOINT, and RELEASE SAVEPOINT](https://dev.mysql.com/doc/refman/8.4/en/savepoint.html): InnoDB savepoint and retained-lock behavior
- [CockroachDB: Foreign Key Constraint](https://www.cockroachlabs.com/docs/stable/foreign-key.html): enforced referential integrity in a distributed SQL database
- [Spanner: Foreign Keys](https://cloud.google.com/spanner/docs/foreign-keys/overview): enforced and informational foreign-key semantics

---

## Related Chapters

- [Isolation Levels](02-isolation-levels.md): isolation is the letter with a market price, and the levels say what anomalies you accept
- [Failure Semantics, Detection, and Recovery Boundaries](03-failure-modes.md): atomicity only matters at the moments when a process stops halfway through a write
- [Distributed Transactions](../03-distributed-databases/10-distributed-transactions.md): what one commit costs once it spans machines and is no longer atomic by luck
- [Write-Ahead Logging](../02-storage-engines/02-write-ahead-logging.md): the mechanism that makes A and D survivable, with its fsync latency attached
