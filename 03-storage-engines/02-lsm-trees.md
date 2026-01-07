# LSM Trees

## TL;DR

Log-Structured Merge Trees trade read performance for write performance. Writes go to an in-memory buffer, then flush to immutable sorted files on disk. Background compaction merges files to maintain read performance. LSM trees excel at write-heavy workloads and are used by LevelDB, RocksDB, Cassandra, and HBase.

---

## The Write Problem

### B-Tree Write Cost

```
B-tree random write:
  1. Find page (random read)
  2. Modify page
  3. Write page back (random write)
  4. Write to WAL (sequential)
  
Random I/O is slow, especially on HDDs
```

### LSM Solution

```
Convert random writes to sequential:
  1. Buffer writes in memory (MemTable)
  2. When full, flush to disk as sorted file
  3. All disk writes are sequential

SSDs: Still faster (no erase-before-write)
HDDs: Much faster (10-100x improvement)
```

---

## LSM Tree Structure

### Overview

```
                  ┌─────────────┐
   Writes ──────► │  MemTable   │  (In-memory, sorted)
                  │  (mutable)  │
                  └──────┬──────┘
                         │ Flush when full
                         ▼
                  ┌─────────────┐
                  │ Immutable   │  (Being flushed)
                  │  MemTable   │
                  └──────┬──────┘
                         │ 
                         ▼
            ┌────────────────────────┐
            │      Level 0           │  (Recent, unsorted between files)
            │  [SST][SST][SST]       │
            └───────────┬────────────┘
                        │ Compact
                        ▼
            ┌────────────────────────┐
            │      Level 1           │  (Sorted, non-overlapping)
            │  [SST][SST][SST][SST]  │
            └───────────┬────────────┘
                        │ Compact
                        ▼
            ┌────────────────────────┐
            │      Level 2           │  (Larger, non-overlapping)
            │  [SST][SST][SST][SST]  │
            └────────────────────────┘
                      ...
```

### MemTable

```
In-memory sorted structure:
  - Red-black tree
  - Skip list (common choice)
  - B-tree

Properties:
  - Fast writes: O(log n)
  - Ordered for efficient flush
  - Size limited (typically 64 MB)
```

### SSTable (Sorted String Table)

```
Immutable file on disk:

┌────────────────────────────────────────────┐
│ Data Block 1 │ Data Block 2 │ ... │ Index │
└────────────────────────────────────────────┘

Data Block:
  [key1, value1][key2, value2]...
  Sorted by key
  Compressed (LZ4, Snappy, Zstd)

Index Block:
  Sparse index: [key → block offset]
  Find block, then binary search within
```

---

## Write Path

### Step by Step

```
1. Write to WAL (sequential, for durability)
2. Write to MemTable (in-memory)
3. Ack to client

4. When MemTable full:
   - Make current MemTable immutable
   - Create new MemTable for writes
   - Background: Flush immutable MemTable to SSTable

5. Background compaction merges SSTables
```

### Code Example

```python
class LSMTree:
    def __init__(self):
        self.wal = WriteAheadLog()
        self.memtable = SkipList()
        self.immutable_memtables = []
        self.levels = [[] for _ in range(MAX_LEVELS)]
        
    def write(self, key, value):
        # Durability: Write to log first
        self.wal.append(key, value)
        
        # Then to memory
        self.memtable.put(key, value)
        
        if self.memtable.size() > MEMTABLE_SIZE:
            self.flush_memtable()
    
    def flush_memtable(self):
        # Make immutable
        self.immutable_memtables.append(self.memtable)
        self.memtable = SkipList()
        
        # Schedule background flush
        schedule(self.flush_to_l0)
    
    def flush_to_l0(self):
        immutable = self.immutable_memtables.pop(0)
        sstable = SSTable.create_from(immutable)
        self.levels[0].append(sstable)
        self.wal.truncate_flushed()
```

---

## Read Path

### Search Order

```
1. Check MemTable
2. Check Immutable MemTables (if any)
3. Check Level 0 SSTables (all of them, might overlap)
4. Check Level 1+ (binary search, non-overlapping)
5. Return value or "not found"
```

### Optimization: Bloom Filters

```
Before reading SSTable from disk:
  Check Bloom filter

if bloom_filter.might_contain(key):
    read_sstable()  # Might be there
else:
    skip()  # Definitely not there (false positive rate ~1%)
    
Reduces disk reads for non-existent keys
```

### Read Amplification

```
Worst case (key doesn't exist):
  MemTable: 1 check
  L0: N SSTables (all overlap)
  L1: 1 SSTable
  L2: 1 SSTable
  ...
  
Total: 1 + N + (L-1) checks

Bloom filters reduce actual disk reads
```

---

## Compaction

### Why Compact?

```
Without compaction:
  - Many overlapping files in L0
  - Same key in multiple SSTables
  - Read performance degrades
  - Disk space wasted (obsolete values)

Compaction:
  - Merges SSTables
  - Removes duplicates (keep latest)
  - Removes tombstones
  - Improves read performance
```

### Compaction Strategies

**Size-Tiered (STCS):**
```
Merge SSTables of similar size

When N SSTables of size S exist:
  Merge into 1 SSTable of size ~N*S

Pros: Simple, good write amplification
Cons: Space amplification (need 2x during compaction)
```

**Leveled (LCS):**
```
Each level has size limit: L(i) = L0 * ratio^i
Files in level are non-overlapping

When level exceeds limit:
  Pick file, merge with overlapping files in next level

Pros: Controlled space, better read performance
Cons: Higher write amplification
```

**FIFO:**
```
Delete oldest SSTables when size limit reached
No merge, just deletion

Use case: Time-series data with TTL
```

---

## Write Amplification

### Definition

```
Write amplification = (Total bytes written to disk) / (Bytes written by user)

Sources:
  1. WAL write
  2. MemTable flush
  3. Compaction (data rewritten multiple times)
```

### Leveled Compaction Math

```
Level ratio = 10
Data moves through ~L levels

At each level:
  Key might be rewritten ~10 times (merge with 10 files)

Total: ~10 * L writes per key
For 1 TB data, 4 levels: ~40x write amplification
```

### Trade-offs

| Strategy | Write Amp | Space Amp | Read Amp |
|----------|-----------|-----------|----------|
| Size-tiered | Low | High | High |
| Leveled | High | Low | Low |
| FIFO | None | None | N/A |

---

## Space Amplification

### Causes

```
1. Obsolete values
   Key updated multiple times, old values not yet compacted

2. Tombstones
   Deleted keys, tombstones not yet garbage collected

3. Compaction temp space
   During compaction, both old and new SSTables exist
```

### Size-Tiered Space

```
Worst case: All SSTables being compacted simultaneously
Space needed: 2x actual data size

Typical: 1.5-2x data size
```

### Leveled Space

```
Bounded by level ratio
Typical: 1.1x data size

Lower because:
  - Non-overlapping files per level
  - Incremental compaction
```

---

## Deletes and Tombstones

### Problem

```
Naive delete:
  Remove key from MemTable
  
But key might exist in SSTables!
Next read might find old value
```

### Tombstone Solution

```
Write special "tombstone" marker:
  write(key, TOMBSTONE)

Read returns "not found" when tombstone seen
Tombstone propagates through compaction
Eventually garbage collected at bottom level
```

### Tombstone Compaction

```
Tombstone can only be removed when:
  - It has reached the bottom level
  - All older versions are removed
  
Long-lived tombstones = space overhead
```

---

## LSM Tree Tuning

### Key Parameters

```
memtable_size:
  Larger: Better write throughput, longer recovery
  Typical: 64 MB - 256 MB

level0_file_num_compaction_trigger:
  Files in L0 before compaction triggers
  Larger: Better write, worse read
  Typical: 4

level_ratio (max_bytes_for_level_multiplier):
  Size ratio between levels
  Larger: Fewer levels, more write amp
  Typical: 10

write_buffer_count:
  Number of MemTables before stalling
  Typical: 2-4
```

### Bloom Filter Sizing

```
Bits per key: 10 = ~1% false positive
Bits per key: 15 = ~0.1% false positive

More bits = Less reads, more memory
Typical: 10 bits per key
```

### Compression

```
Level 0-1: LZ4/Snappy (fast, moderate compression)
Level 2+:  Zstd (better compression, slower)

Trade-off: CPU for I/O
SSDs favor faster compression
```

---

## Systems Using LSM Trees

### LevelDB / RocksDB

```
Google's LevelDB: Original, simple
Facebook's RocksDB: Production-hardened, many features

RocksDB additions:
  - Column families
  - Transactions
  - Multiple compaction styles
  - Statistics and monitoring
```

### Cassandra

```
Each table has its own LSM tree
Compaction strategies configurable per table

Size-tiered: Default (good for write-heavy)
Leveled: Better for read-heavy
Time-window: Time-series data
```

### HBase

```
LSM-based on HDFS
MemStore (MemTable) → HFiles (SSTables)

Major compaction: Merge all files
Minor compaction: Merge some files
```

---

## B-Tree vs LSM Tree

| Aspect | B-Tree | LSM Tree |
|--------|--------|----------|
| Write | Random I/O | Sequential I/O |
| Read | 1 lookup | Multiple lookups |
| Write amp | ~10x | ~10-30x |
| Space amp | ~1.5x | ~1.1-2x |
| Compaction | None | Background |
| Range scan | Excellent | Good |
| Point lookup | Excellent | Good |

### When to Use LSM

```
✓ Write-heavy workloads
✓ Sequential write patterns (time-series)
✓ SSD-based storage (tolerates read amp)
✓ Key-value stores
✓ Need high write throughput

✗ Read-heavy workloads
✗ Complex queries
✗ Latency-sensitive reads
✗ Predictable performance needed
```

---

## Key Takeaways

1. **Writes to memory first** - Sequential disk writes via flush
2. **SSTables are immutable** - Append-only design
3. **Compaction is essential** - Maintains read performance
4. **Trade-off triangle** - Write amp, read amp, space amp
5. **Bloom filters critical** - Reduce disk reads
6. **Leveled for reads** - Size-tiered for writes
7. **Tombstones have cost** - Delayed garbage collection
8. **Tuning is workload-specific** - No universal configuration
