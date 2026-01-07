# Write-Ahead Logging (WAL)

## TL;DR

Write-Ahead Logging ensures durability by writing changes to a sequential log before applying them to data structures. If the system crashes, the log is replayed to recover committed transactions. WAL is fundamental to almost every database system. Key trade-offs: fsync frequency vs durability, log size vs recovery time.

---

## The Durability Problem

### Without WAL

```
Transaction:
  1. Update buffer pool (memory)
  2. Eventually flush to disk
  
Crash between 1 and 2:
  - Data in memory lost
  - Disk has stale data
  - Transaction lost despite "commit"
```

### With WAL

```
Transaction:
  1. Write to log (sequential, fast)
  2. Fsync log (durable)
  3. Update buffer pool (memory)
  4. Return commit to client
  
  [Later: Flush buffer pool to disk]
  [Even later: Truncate log]

Crash at any point:
  - Replay log on recovery
  - All committed transactions restored
```

---

## WAL Protocol

### The WAL Rule

```
Before modifying any data page on disk:
  1. Write log record describing the change
  2. Ensure log record is on stable storage (fsync)
  3. Then (and only then) modify the data page

"Write-Ahead" = Log before Data
```

### Write Path

```
┌────────────────────────────────────────────────────────┐
│ Transaction: UPDATE account SET balance = 500         │
└────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 1. Write log record to WAL buffer                     │
│    <TxnID, PageID, Offset, OldValue, NewValue>        │
└────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 2. On commit: Flush WAL buffer to disk (fsync)        │
└────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 3. Modify data page in buffer pool (memory)           │
│    (Disk write happens later, asynchronously)         │
└────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 4. Return "Commit OK" to client                       │
└────────────────────────────────────────────────────────┘
```

### Log Sequence Numbers (LSN)

```
Every log record has unique, monotonically increasing LSN

Log:
  LSN=100: <Txn1, Update, Page5, ...>
  LSN=101: <Txn1, Update, Page8, ...>
  LSN=102: <Txn1, Commit>
  LSN=103: <Txn2, Update, Page5, ...>
  ...

Page header tracks:
  page_lsn = LSN of last applied log record
  
Recovery:
  If log_lsn > page_lsn: apply log record
  If log_lsn <= page_lsn: skip (already applied)
```

---

## Log Record Types

### Physical Logging

Log exact bytes changed.

```
<LSN=100, TxnID=1, PageID=5, Offset=42, OldValue=100, NewValue=200>

Redo: Write 200 at offset 42 on page 5
Undo: Write 100 at offset 42 on page 5

Pros: Simple, fast recovery
Cons: Large logs for big changes
```

### Logical Logging

Log the operation.

```
<LSN=100, TxnID=1, Operation="UPDATE balance SET balance=200 WHERE id=5">

Redo: Re-execute the operation
Undo: Execute inverse operation

Pros: Compact logs
Cons: Must be deterministic, slower recovery
```

### Physiological Logging

Hybrid: Physical to a page, logical within.

```
<LSN=100, TxnID=1, PageID=5, Op="INSERT key=abc at slot=3">

Page-level physical: Know which page
Slot-level logical: Operation within page

Most databases use this approach
```

---

## ARIES Recovery

### Overview

Algorithms for Recovery and Isolation Exploiting Semantics.
Industry standard, used by most databases.

```
Three phases:
  1. Analysis: Determine what needs to be done
  2. Redo: Replay all logged changes
  3. Undo: Rollback uncommitted transactions
```

### Analysis Phase

```
Scan log from last checkpoint:
  - Build list of active transactions (not committed/aborted)
  - Build dirty page table (pages with unflushed changes)

Input: Log + Last checkpoint
Output: 
  - Redo start point
  - Active transactions to undo
  - Dirty pages
```

### Redo Phase

```
Scan forward from redo start point:
  For each log record:
    if page not in dirty table: skip
    if page LSN >= log record LSN: skip  # Already applied
    else: Apply redo  # Repeat history

Re-applies ALL changes (committed or not)
This brings database to exact crash state
```

### Undo Phase

```
For each active (uncommitted) transaction:
  Scan backward through its log records
  Apply undo for each record
  Write CLR (Compensation Log Record) for each undo

CLR ensures undo is idempotent:
  If crash during undo, CLR prevents re-undoing
```

### Example Recovery

```
Log:
  100: <T1, Update, P1, old=A, new=B>
  101: <T1, Update, P2, old=C, new=D>
  102: <T2, Update, P3, old=E, new=F>
  103: <T1, Commit>
  104: <T2, Update, P4, old=G, new=H>
  [CRASH]

Analysis:
  Active transactions: {T2}
  Need to undo T2

Redo (forward scan):
  Apply all records 100-104 to disk

Undo (backward for T2):
  Undo 104: Set P4 back to G, write CLR
  Undo 102: Set P3 back to E, write CLR
  
Result:
  T1's changes preserved (committed)
  T2's changes undone (was active at crash)
```

---

## Checkpointing

### Purpose

Limit recovery time by recording a known good state.

```
Without checkpoint:
  Must replay entire log from beginning
  Could be gigabytes of log

With checkpoint:
  Only replay from last checkpoint
  Bounded recovery time
```

### Fuzzy Checkpoint

```
1. Pause new transactions briefly
2. Record:
   - Active transactions list
   - Dirty pages table
   - Current LSN
3. Resume transactions
4. [Background: Flush dirty pages]

Called "fuzzy" because:
  - Doesn't wait for all pages to flush
  - Some dirty pages may still be in memory
  - Redo phase handles this
```

### Checkpoint Record

```
<CHECKPOINT, 
  ActiveTxns=[T1, T2, T3],
  DirtyPages=[P5, P8, P12],
  LSN=500>
```

---

## Group Commit

### The Fsync Problem

```
Naive approach:
  Each commit → separate fsync
  Fsync: ~10ms on HDD
  Max throughput: 100 commits/sec
```

### Solution: Group Commit

```
Batch multiple transactions' fsyncs:

Time 0-5ms:  T1, T2, T3 prepare, write to log buffer
Time 5ms:    Single fsync for all three
Time 5-6ms:  All three return "committed"

Amortizes fsync cost across transactions
10,000+ commits/sec possible
```

### Implementation

```python
class GroupCommit:
    def __init__(self):
        self.pending = []
        self.commit_interval = 10  # ms
        
    def request_commit(self, txn):
        # Add to pending batch
        self.pending.append(txn)
        
        # Wait for batch leader to fsync
        event = txn.create_event()
        return event.wait()
    
    def background_flush(self):
        while True:
            sleep(self.commit_interval)
            
            if self.pending:
                batch = self.pending
                self.pending = []
                
                # Single fsync for entire batch
                self.wal.fsync()
                
                # Notify all waiting transactions
                for txn in batch:
                    txn.event.signal()
```

---

## Log Truncation

### When to Truncate

```
Log grows forever without truncation

Can truncate when:
  - All transactions before LSN are committed
  - All dirty pages before LSN are flushed
  - Checkpoint has passed that point

Safe truncation point:
  min(oldest_active_txn_lsn, oldest_dirty_page_lsn)
```

### Archiving

```
For point-in-time recovery:
  1. Don't delete old logs
  2. Archive to cheap storage (S3, tape)
  3. Retain for days/months

Recovery:
  1. Restore base backup
  2. Replay archived logs to desired point
```

---

## WAL Configurations

### Durability Levels

```
Level 1: Fsync every commit
  - Strongest durability
  - Slowest
  - PostgreSQL: synchronous_commit = on
  
Level 2: Fsync every N ms
  - Lose up to N ms on crash
  - Better throughput
  - PostgreSQL: synchronous_commit = off
  
Level 3: OS decides when to flush
  - May lose significant data
  - Fastest
  - Never use for production
```

### Buffer Size

```
Larger WAL buffer:
  + Better batching
  + Higher throughput
  - More data at risk before fsync
  - More memory usage

Typical: 16 MB - 256 MB
```

### Log File Size

```
PostgreSQL: wal_segment_size (16 MB - 1 GB)
MySQL: innodb_log_file_size

Larger files:
  + Fewer file switches
  + Better sequential I/O
  - Longer recovery time
  - More disk space
```

---

## WAL in Different Systems

### PostgreSQL

```
WAL location: pg_wal/
Log format: Binary, 16 MB segments
Replication: Streaming replication uses WAL

Key settings:
  wal_level = replica  # Logging detail
  synchronous_commit = on  # Durability
  checkpoint_timeout = 5min
  max_wal_size = 1GB
```

### MySQL InnoDB

```
Log files: ib_logfile0, ib_logfile1
Circular log with two files

Key settings:
  innodb_log_file_size = 256M
  innodb_flush_log_at_trx_commit = 1  # Fsync each commit
  innodb_log_buffer_size = 16M
```

### RocksDB

```
WAL directory: configurable
Used for MemTable durability

Settings:
  Options::wal_dir
  Options::WAL_ttl_seconds
  Options::WAL_size_limit_MB
  Options::manual_wal_flush
```

---

## Performance Optimization

### Separate WAL Disk

```
Dedicated disk for WAL:
  - Sequential writes only
  - No competition with data reads
  - Consistent latency

NVMe SSD for WAL:
  - High IOPS for fsync
  - Low latency
```

### Compression

```
Compress log records:
  - LZ4 for speed
  - Zstd for ratio

Trade-off:
  + Smaller logs, faster I/O
  - CPU overhead
  - Decompression on recovery
```

### Parallel WAL

```
Multiple WAL partitions:
  - Transactions hashed to partition
  - Parallel writes
  - More complex recovery

Used in high-throughput systems
```

---

## Common Issues

### WAL Full / Disk Full

```
Problem: WAL fills disk
Symptoms: 
  - Writes blocked
  - Database unavailable

Prevention:
  - Monitor disk space
  - Configure max_wal_size
  - Faster checkpointing
  - Archive old WAL files
```

### Replication Lag from WAL

```
Problem: Replica can't keep up with WAL
Causes:
  - Slow replica disk
  - Network bottleneck
  - Large transactions

Solutions:
  - Faster replica
  - More frequent checkpoints (less WAL)
  - Throttle primary writes
```

### Long Recovery Time

```
Problem: Crash recovery takes hours
Causes:
  - Infrequent checkpoints
  - Large dirty page table
  - Huge log to replay

Solutions:
  - More frequent checkpoints
  - Smaller checkpoint_completion_target
  - Archive and truncate logs
```

---

## Key Takeaways

1. **Log before data** - Fundamental WAL rule
2. **LSN tracks progress** - Enables idempotent recovery
3. **ARIES is standard** - Analysis, Redo, Undo phases
4. **Group commit for throughput** - Batch fsync calls
5. **Checkpoint bounds recovery** - Trade checkpoint cost for recovery time
6. **Truncate after checkpoint** - Keep log size bounded
7. **Fsync frequency is key trade-off** - Durability vs performance
8. **Separate disk recommended** - Isolate WAL I/O
