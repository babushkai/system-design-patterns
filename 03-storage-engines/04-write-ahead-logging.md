# Write-Ahead Logging (WAL)

## TL;DR

Write-ahead logging makes durability sequential: append a change record, persist the log, acknowledge, and write scattered data pages later. LSNs make replay idempotent; ARIES separates analysis, redo, and undo; checkpoints bound recovery; group commit amortizes flushes. Correctness still depends on torn-page defenses, honest flush semantics, and bounded log retention for replication and recovery. An abandoned replication slot can pin WAL until disk exhaustion; retention needs explicit ownership, lag budgets, alerts, and a tested emergency policy.

---

## The Problem: Durability Without Random I/O

A committed transaction must survive a crash. Persisting every modified page before acknowledgement turns each commit into scattered I/O; flushing memory later loses acknowledged data. WAL separates durable description from deferred page application.

The WAL resolves the dilemma by separating *describing* a change from *applying* it:

```
Commit path with WAL:
  1. append a log record describing the change   (memory)
  2. fsync the log                               (ONE sequential write)
  3. modify the data page in the buffer pool     (memory)
  4. acknowledge the client

The data page reaches disk later: at a checkpoint, or when the
buffer pool evicts it. If the machine dies first, recovery replays
the log record and rebuilds the page.

The invariant that names the technique: a data page may not be
written to disk before the log records describing its changes are.
Log first, data second: always.
```

One sequential append replaces scattered commit-path writes; background work can batch and schedule page I/O. [LSM trees](./02-lsm-trees.md) extend this append-and-merge approach to primary storage, while WAL-protected [B-trees](./01-b-trees.md) retain update-in-place pages.

The tradeoff is double writing: once to the log and later to the data page. WAL therefore contributes directly to the write amplification described in [B-Trees](./01-b-trees.md).

---

## LSNs: Making Replay Idempotent

Recovery replays the log against pages whose on-disk state is unknown: some got flushed before the crash, some didn't. Applying a change twice would corrupt data as surely as skipping it. The mechanism that makes replay safe is the **log sequence number**: every record gets a monotonically increasing LSN, and every page header records the LSN of the last record applied to it.

```
Log:                                 Page 5 on disk:
  LSN 100: update page 5              page_lsn = 101
  LSN 101: update page 5
  LSN 102: update page 8              Redo walks the log:
  LSN 103: commit T1                    LSN 100 vs page_lsn 101 → skip
                                        LSN 101 vs page_lsn 101 → skip
                                        LSN 102 vs page 8's LSN → apply if newer
```

The comparison `record_lsn > page_lsn` makes replay idempotent, including recovery restarted after another crash. LSNs also coordinate checkpoints, replica apply progress, and the oldest log position still needed for truncation.

---

## What Goes in a Record: Physical vs. Logical

There is a spectrum of what a log record can say, and the choice trades log volume against replay complexity:

**Physical logging** records bytes: "page 5, offset 42, old value `A`, new value `B`." Replay is trivial and fast (copy bytes), but a change that touches many bytes (a B-tree page split rebalancing hundreds of keys) produces enormous records.

**Logical logging** records operations: "execute `UPDATE accounts SET balance = balance - 100 WHERE id = 5`." Records are tiny, but replay must re-execute the operation deterministically (same results, same order), which is fragile in the presence of concurrency, non-determinism (`now()`, random), and code changes between versions.

**Physiological logging** is physical *to* a page and logical *within* it: "on page 5, insert key `abc` at slot 3." Page-oriented transactional engines commonly use it because replay needs no query planning and can be parallelized by page, while records remain compact. It assumes the prior page state is intact; torn-page defenses protect that assumption.

---

## ARIES: Recovery in Three Passes

Nearly every serious database recovers with some variant of **ARIES** (Mohan et al., 1992). Its central design decision sounds strange until you see why: after a crash, first *repeat all of history* (including the changes of transactions that will ultimately be rolled back), and only then undo the losers.

```
1. ANALYSIS: scan from the last checkpoint:
     which transactions were in flight at the crash?
     which pages might have unflushed changes (dirty page table)?

2. REDO: scan forward, reapply every change whose LSN is newer
     than its page (committed or not). The database is now in the
     exact state of the crash instant.

3. UNDO: for each transaction alive at the crash, walk its
     records backward and reverse them, logging a Compensation Log
     Record (CLR) for every reversal.
```

Repeating history first means redo needs no judgment (it is a dumb, fast, page-ordered replay), and undo then operates on a consistent snapshot of the crash state, using the same locking-free logic as a normal rollback. The **CLRs** solve the recursive problem: what if we crash *during* undo? Each CLR says "this reversal happened" and points at the next record to undo, so a second recovery skips completed reversals instead of re-reversing them. Undo, like redo, becomes idempotent; recovery can crash any number of times and still converge.

```
  100: T1 updates P1        Analysis: T2 was alive at crash
  101: T2 updates P3        Redo:     replay 100-103 (yes, including T2)
  102: T1 commit            Undo:     reverse 103, then 101,
  103: T2 updates P4                  writing a CLR for each
  --- CRASH ---             Result:   T1's work stands, T2 vanished
```

---

## Checkpoints: Bounding the Replay

Without checkpoints, recovery replays the log from the beginning of time. A **checkpoint** periodically records "here is a safe starting point": the set of active transactions, the dirty page table, and (implicitly, by flushing) a guarantee that pages older than some LSN are on disk.

Modern engines use **fuzzy checkpoints**: rather than stopping the world to flush every dirty page (a latency catastrophe), the checkpoint records the dirty page *table* and lets background writers flush gradually; redo's LSN comparisons tolerate the imprecision. The checkpoint's cost doesn't disappear, though: it spreads. PostgreSQL's `checkpoint_completion_target` explicitly paces the flushing across the checkpoint interval to avoid an I/O spike, and the interval itself is the fundamental recovery-time knob:

```
Checkpoint interval trade-off:
  frequent  → little log to replay (fast recovery)
              but constant page flushing (foreground I/O impact)
              and in PostgreSQL: more full-page images (see below)
  rare      → cheap steady-state
              but crash recovery replays a huge log (minutes-hours)

Recovery time ≈ log volume since last checkpoint / replay speed.
If you have a recovery-time objective, this is the knob that meets it.
```

Truncation follows from checkpoints: log older than `min(oldest active transaction's first LSN, oldest dirty page's LSN, oldest LSN a replica still needs)` can be recycled. Every term in that `min` is a way the log grows without bound when something stalls: a forgotten open transaction, a stuck background writer, or (most commonly, see below) a dead replication consumer.

---

## Group Commit: The Throughput Engineering

The fsync in the commit path is the entire latency and throughput story. One fsync per commit caps throughput at the device's flush rate:

```
fsync cost:      HDD ~10 ms   SATA SSD ~1 ms   NVMe ~20-100 μs
naive ceiling:   100/s        ~1,000/s          ~10,000-50,000/s

Group commit: while one fsync is in flight, arriving commits queue.
When it returns, ALL queued records flush in the next single fsync.
  20 concurrent committers on a 1 ms device ≈ 20,000 commits/s:
  the batch size self-tunes to concurrency, and each transaction's
  added latency is at most one flush interval.
```

Every serious engine does this (PostgreSQL's WAL writer, InnoDB's redo group commit, RocksDB's write group leader). The remaining choice is what "durable" should mean, and engines expose it as a spectrum:

```
synchronous_commit = on      fsync before ack        lose nothing
synchronous_commit = off     ack, fsync within ~ms   lose last few ms
                             (PostgreSQL: data stays CONSISTENT;
                              you lose recent commits, not integrity)
innodb_flush_log_at_trx_commit = 1 / 2 / 0: same ladder for MySQL
```

Relaxed durability is legitimate engineering for derived or replayable data (event ingestion with an upstream queue, cache-like tables) and indefensible for money. The decision should be per-workload (PostgreSQL lets you set it per *transaction*), not a server-wide default someone chose for a benchmark.

Two more levers matter on the commit path. A **dedicated log device** keeps the log's sequential stream from being interleaved with random data I/O (interleaving turns both into random I/O). And the log buffer size bounds how much batching group commit can do under burst: 16–64 MB is typical; bigger mostly helps bulk loads.

---

## Torn Pages: The Atomicity Gap Under the WAL

The WAL protocol assumes a data-page write either happens or doesn't. Hardware breaks that assumption:

```
Database page: 8 KB (PostgreSQL) / 16 KB (InnoDB)
Device atomic write unit: 4 KB sector (often 512B logically)

Crash mid-page-write → a TORN page: first 4 KB new, last 4 KB old.
Page checksum detects it, but redo may not be able to FIX it:
physiological log records ("insert key at slot 3") assume the page's
prior state is intact. A torn page has no valid prior state.
```

Two production defenses:

```
PostgreSQL: full-page writes (full_page_writes = on):
  the FIRST modification to a page after each checkpoint logs the
  ENTIRE page image into the WAL. Redo restores the image, then
  applies records on top: no dependence on the on-disk page state.
  Cost: WAL volume spikes right after every checkpoint (the FPI
  burst); this is the hidden coupling between checkpoint_timeout
  and WAL bandwidth.

InnoDB: doublewrite buffer:
  pages are first written sequentially to a doublewrite area, synced,
  then written to their final locations. Torn final write → recover
  the page from the doublewrite copy. Cost: ~2× page write volume
  (mitigated by batching; can be disabled ONLY on filesystems/devices
  with guaranteed atomic writes: e.g., ZFS, or NVMe devices exposing
  atomic write units ≥ page size).
```

If you run on storage that genuinely guarantees page-sized atomic writes, both defenses are pure overhead, which is why "can we turn off doublewrite/FPW?" is a real tuning conversation, and why the answer must come from the storage stack's documentation, not optimism.

---

## Does fsync Actually Sync?

The durability of the entire design rests on one syscall telling the truth. It often doesn't, at three layers:

```
1. Volatile drive caches: consumer SSDs/HDDs ack writes into DRAM
   cache. A power cut loses "durable" data unless the OS issues cache
   flush / FUA commands, which filesystem barriers do, but
   misconfigured stacks (some virtualized disks, RAID controllers
   without BBU set to write-back) silently don't.

2. fsync error semantics (fsyncgate, 2018): on Linux, if a background
   writeback fails, fsync() returns EIO ONCE, and marks the pages
   CLEAN. A process that retries fsync gets SUCCESS while the data
   never reached disk. PostgreSQL had assumed retry-until-success was
   safe for ~20 years; the fix (PG 11+) is to PANIC on fsync failure
   and recover from WAL, never retry.

3. fdatasync vs fsync vs directory sync: creating a new WAL segment
   requires fsyncing the DIRECTORY too, or the file itself may vanish
   after crash. Metadata (size changes) needs fsync; fdatasync
   suffices for in-place data and is cheaper.

Verification, not vibes: pull the power plug under load
(diskchecker.pl-style tests) or use dm-flakey/dm-log-writes to
simulate. Storage stacks that "lose" acked fsyncs are common enough
that serious databases treat this as a qualification test.
```

---

## WAL as Replication Substrate

The same byte stream that provides crash recovery is the natural replication feed: a replica is, formally, just a recovery process that never finishes. This dual use creates most WAL operational issues:

```
Physical replication (PostgreSQL streaming, InnoDB redo shipping):
  replica applies page-level records → byte-identical standby.
  Fast, simple; replica must run the same major version and
  architecture.

Logical replication / CDC: decode WAL back into row-level changes
  (pgoutput, Debezium). Enables cross-version, selective, and
  cross-system replication ([CDC pipelines](../13-data-pipelines/04-change-data-capture.md)):
  at the cost of decoding CPU and ordering complexity.

The operational trap: replication slots pin WAL:
  a slot guarantees the WAL a consumer hasn't read yet is retained.
  A dead/abandoned consumer (a decommissioned replica, a stalled
  Debezium connector) pins WAL forever → disk fills → database down.
  This is among the most common self-inflicted PostgreSQL outages.
  Defense: monitor slot lag bytes; set max_slot_wal_keep_size (PG 13+)
  to cap retention and sacrifice the slot instead of the database.

Synchronous replication couples commit latency to the network:
  synchronous_commit = on → wait for local flush
                       remote_write / remote_apply → wait for standby
  Group commit still applies: batches of transactions share both the
  local fsync AND the replication round trip ([Consensus](../02-distributed-databases/08-consensus-algorithms.md)
  makes the same amortization under quorum acks).
```

Log archival extends the same stream through time instead of space. Shipping closed WAL segments to [object storage](./08-object-storage.md) lets a base backup plus a continuous archived-WAL sequence recover to a selected point covered by that archive. PostgreSQL selects the stop point with `recovery_target_time`; this is the standard defense against “we dropped the wrong table at 14:32.”

---

## The WAL in Specific Engines

**PostgreSQL** writes 16 MB WAL segments under `pg_wal/`; the knobs that matter are `synchronous_commit` (the durability ladder, settable per transaction), `max_wal_size`/`checkpoint_timeout` (recovery-time vs. FPI volume), `full_page_writes` (torn pages), and `max_slot_wal_keep_size` (slot protection). `pg_stat_wal` and `pg_stat_replication` expose volume and lag.

**MySQL/InnoDB** uses a circular redo log (`innodb_redo_log_capacity`); `innodb_flush_log_at_trx_commit` is the durability ladder, and the doublewrite buffer covers torn pages. As a structural difference, InnoDB *also* keeps undo logs as first-class MVCC structures, whereas PostgreSQL keeps old row versions in the heap and needs no undo log.

**RocksDB** WALs protect the memtables ([LSM Trees](./02-lsm-trees.md)); a WAL segment is deleted once its memtable flushes to an SSTable. There is no page-oriented redo (recovery is simply "reload the memtable from the log"), which shows how much of ARIES exists specifically to serve *update-in-place* storage. `manual_wal_flush` and per-write `disableWAL` expose the same durability ladder in embedded form.

---

## Failure Modes

**Disk full from WAL growth.** The log grows until *everything* that pins it advances: checkpoints, archival, and every replication slot. The classic incident is an abandoned slot pinning weeks of WAL until the volume fills and the database stops accepting writes. Monitor retained-WAL bytes and slot lag; cap with `max_slot_wal_keep_size`.

**Recovery takes hours.** Nobody notices an oversized `max_wal_size` until the crash. Recovery time is proportional to log-since-checkpoint; if you have an RTO, translate it into a checkpoint interval and *test it* by actually crashing a replica: replay speed (single-threaded in older PostgreSQL versions) is often slower than people assume.

**The FPI burst.** Right after each PostgreSQL checkpoint, every touched page logs a full image: WAL volume can jump 5–10× for a while, saturating replication links and archival. Spreading checkpoints (`checkpoint_completion_target`), `wal_compression = on`, and not scheduling checkpoints to coincide with batch jobs all mitigate.

**Silent durability downgrade.** A migration to new hardware, a VM platform change, or a well-meaning "performance fix" (`synchronous_commit = off`, disabling barriers, an NFS mount) quietly changes what an acknowledged commit means. Treat durability configuration as part of the schema: reviewed, versioned, and re-verified (power-pull test) when the storage stack changes.

**Torn-page defenses disabled on the wrong stack.** `full_page_writes = off` or `skip-innodb_doublewrite` is only safe when the storage genuinely writes pages atomically. The failure is invisible until a crash lands mid-page: then recovery itself fails on a corrupt page.

---

## Decision Framework

| Situation | Do this |
|---|---|
| Default OLTP durability | `synchronous_commit = on` / `innodb_flush_log_at_trx_commit = 1`, group commit does the throughput work |
| Replayable/derived data, ingest-bound | Relax per table/transaction (`synchronous_commit = off`, `= 2`), never server-wide by reflex |
| Recovery-time objective exists | Derive checkpoint interval from it; crash-test a replica to measure real replay speed |
| Commit latency spikes after checkpoints | FPI burst: spread checkpoints, enable `wal_compression`, check WAL bandwidth |
| Using logical replication / CDC | Alert on slot lag bytes; set `max_slot_wal_keep_size`; slots die, databases shouldn't |
| Zero-data-loss failover required | Synchronous replication (`remote_apply`) and accept the RTT in every commit |
| Point-in-time recovery required | Continuous WAL archival to object storage + periodic base backups; rehearse restores |
| New storage stack (cloud disk, ZFS, new NVMe) | Re-verify fsync honesty and page-write atomicity before trusting or disabling defenses |

---

## Key Takeaways

1. **Log first, data second**: one sequential fsync buys durability for arbitrarily scattered page changes; the random I/O moves off the commit path.
2. **LSNs make recovery idempotent**: `record_lsn > page_lsn` is the comparison that lets replay (and undo, via CLRs) crash and rerun safely.
3. **ARIES = repeat history, then undo**: redo is a dumb fast page-ordered replay to the crash instant; undo then rolls back the losers with compensation records.
4. **Checkpoint interval is your recovery-time dial**, and in PostgreSQL it's also the full-page-image volume dial; the two costs trade against each other.
5. **Group commit turns the fsync ceiling into a concurrency game**: batches self-tune to load; the durability ladder below it is a per-workload decision, not a server default.
6. **Page writes aren't atomic**: full-page writes and doublewrite buffers exist for torn pages; disable them only with documented atomic-write guarantees.
7. **fsync can lie**: volatile caches, fsyncgate error semantics, forgotten directory syncs; qualify storage by pulling the plug, not by reading the datasheet.
8. **The WAL is also your replication and PITR substrate**: replicas are unfinished recoveries, archives are recovery through time, and anything that pins the log (slots!) can fill your disk.

---

## References

- Mohan, C., et al. (1992). *ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging*. TODS.
- Gray, J., & Reuter, A. (1992). *Transaction Processing: Concepts and Techniques*. (The durability/logging foundations.)
- Hellerstein, Stonebraker & Hamilton (2007). *Architecture of a Database System*. (Log manager and recovery in context.)
- PostgreSQL documentation: *WAL Configuration*, *Reliability* (fsync/FPW discussion), *Logical Decoding*, `pg_stat_wal`.
- MySQL documentation: *InnoDB Redo Log*, *Doublewrite Buffer*.
- Rebello, A., et al. (2020). *Can Applications Recover from fsync Failures?* USENIX ATC: the systematic follow-up to fsyncgate.
- LWN: *PostgreSQL's fsync() surprise* (2018): the fsyncgate write-up.
