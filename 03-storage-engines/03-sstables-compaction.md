# SSTables and Compaction

## TL;DR

Sorted String Tables (SSTables) are immutable, sorted files that form the on-disk layer of LSM trees. Compaction merges SSTables to reclaim space, remove obsolete data, and maintain read performance. Choosing the right compaction strategy—size-tiered, leveled, or FIFO—depends on your workload's read/write balance and latency requirements.

---

## SSTable Structure

### File Layout

```
┌─────────────────────────────────────────────────────────────┐
│                        SSTable File                        │
├─────────────────────────────────────────────────────────────┤
│  Data Block 1  │  Data Block 2  │  ...  │  Data Block N    │
├─────────────────────────────────────────────────────────────┤
│               Meta Block (Bloom Filter)                    │
├─────────────────────────────────────────────────────────────┤
│                    Index Block                              │
├─────────────────────────────────────────────────────────────┤
│                       Footer                                │
└─────────────────────────────────────────────────────────────┘
```

### Data Blocks

```
Block structure:
┌────────────────────────────────────────┐
│  Entry 1  │  Entry 2  │  ...  │ Entry N │
├────────────────────────────────────────┤
│           Restart Points               │
├────────────────────────────────────────┤
│  Num Restarts (4 bytes)  │  CRC (4)    │
└────────────────────────────────────────┘

Entry format:
  [shared_prefix_len][unshared_len][value_len][unshared_key][value]
  
Prefix compression:
  Keys often share prefixes
  Only store the difference
  "user:1000" → "user:1001" stored as "01" suffix
```

### Index Block

```
Sparse index:
  Every N-th key → block offset

┌───────────────────────────────────┐
│  "aaa" → Block 0 @ offset 0       │
│  "abc" → Block 1 @ offset 4096    │
│  "def" → Block 2 @ offset 8192    │
│  ...                              │
└───────────────────────────────────┘

Lookup:
  1. Binary search index for >= key
  2. Read data block
  3. Binary search within block
```

### Footer

```
┌─────────────────────────────────────────┐
│  Metaindex Handle (offset, size)        │
│  Index Handle (offset, size)            │
│  Padding                                │
│  Magic Number (8 bytes)                 │
└─────────────────────────────────────────┘

Fixed size, always at end of file
Contains pointers to index and meta blocks
```

---

## SSTable Operations

### Creating SSTable (Flush)

```python
def create_sstable(memtable, filename):
    writer = SSTableWriter(filename)
    
    # Iterate memtable in sorted order
    for key, value in memtable.sorted_iterator():
        writer.add(key, value)
    
    # Finalize: write index, bloom filter, footer
    writer.finish()
    
    return SSTable(filename)

class SSTableWriter:
    def __init__(self, filename):
        self.file = open(filename, 'wb')
        self.index = []
        self.bloom_filter = BloomFilter()
        self.current_block = DataBlock()
        self.block_offset = 0
        
    def add(self, key, value):
        self.bloom_filter.add(key)
        
        if self.current_block.size() > BLOCK_SIZE:
            self.flush_block()
        
        self.current_block.add(key, value)
    
    def flush_block(self):
        # Write block
        data = self.current_block.serialize()
        self.file.write(data)
        
        # Record in index
        first_key = self.current_block.first_key
        self.index.append((first_key, self.block_offset))
        
        self.block_offset += len(data)
        self.current_block = DataBlock()
```

### Reading from SSTable

```python
class SSTable:
    def get(self, key):
        # Check bloom filter first
        if not self.bloom_filter.might_contain(key):
            return None  # Definitely not here
        
        # Binary search in index
        block_offset = self.find_block(key)
        
        # Read and search block
        block = self.read_block(block_offset)
        return block.search(key)
    
    def find_block(self, key):
        # Binary search for last index entry <= key
        lo, hi = 0, len(self.index) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.index[mid].key <= key:
                lo = mid
            else:
                hi = mid - 1
        return self.index[lo].offset
```

### Scanning Range

```python
def scan(self, start_key, end_key):
    # Find starting block
    block_offset = self.find_block(start_key)
    
    results = []
    while block_offset < self.data_size:
        block = self.read_block(block_offset)
        
        for key, value in block.entries():
            if key > end_key:
                return results
            if key >= start_key:
                results.append((key, value))
        
        block_offset = self.next_block_offset(block_offset)
    
    return results
```

---

## Compaction Strategies

### Size-Tiered Compaction (STCS)

```
Group SSTables by size, merge when count exceeds threshold.

Before:
  Tier 0 (small): [1MB] [1MB] [1MB] [1MB]  ← 4 files, trigger!
  Tier 1 (medium): [4MB]
  Tier 2 (large): [16MB]

After:
  Tier 0: []
  Tier 1: [4MB] [4MB]  ← merged output
  Tier 2: [16MB]
```

**Algorithm:**
```python
def size_tiered_compaction():
    for tier in tiers:
        if tier.file_count >= MIN_THRESHOLD:
            # Pick similar-sized files
            files = tier.select_files(MIN_THRESHOLD)
            
            # Merge into new file
            merged = merge(files)
            
            # Add to appropriate tier
            next_tier = find_tier_for_size(merged.size)
            next_tier.add(merged)
            
            # Remove old files
            tier.remove(files)
```

**Characteristics:**
| Aspect | Value |
|--------|-------|
| Write amplification | ~3-5x |
| Space amplification | Up to 2x |
| Read amplification | O(tiers × files per tier) |
| Best for | Write-heavy workloads |

### Leveled Compaction (LCS)

```
Organize files into levels with size limits.
Each level (except L0) has non-overlapping key ranges.

Level 0: 4 files max (may overlap)
Level 1: 10 MB total
Level 2: 100 MB total
Level 3: 1000 MB total
...

Compaction:
  When level exceeds limit, pick file and merge with overlapping files in next level.
```

```
Before:
  L0: [a-d] [c-f] [e-h]  ← overlapping
  L1: [a-c] [d-f] [g-i]  ← non-overlapping

L0 exceeds limit, compact [c-f]:
  Overlaps with L1's [d-f]
  Merge them

After:
  L0: [a-d] [e-h]
  L1: [a-c] [c-f'] [g-i]  ← f' is merged result
```

**Algorithm:**
```python
def leveled_compaction():
    for level in range(MAX_LEVELS - 1):
        if level_size(level) > size_limit(level):
            # Pick file to compact (round-robin or by overlap)
            file = pick_file_to_compact(level)
            
            # Find overlapping files in next level
            overlapping = find_overlapping(level + 1, file.key_range)
            
            # Merge
            merged_files = merge(file, overlapping)
            
            # Add to next level
            add_to_level(level + 1, merged_files)
            
            # Remove old
            remove_files([file] + overlapping)
```

**Characteristics:**
| Aspect | Value |
|--------|-------|
| Write amplification | ~10-30x |
| Space amplification | ~1.1x |
| Read amplification | O(levels) |
| Best for | Read-heavy, space-sensitive |

### FIFO Compaction

```
Delete oldest files when total size exceeds limit.
No merging, just time-based deletion.

├── newest_file.sst
├── file_2.sst
├── file_3.sst
├── oldest_file.sst  ← Delete when over limit
```

**Use case:**
- Time-series data with natural TTL
- Logs that expire after N days
- Metrics with fixed retention

### Time-Window Compaction (TWCS)

```
Combine STCS within time windows, FIFO across windows.

Time windows:
  [Today]  [Yesterday]  [2 days ago]  [3 days ago] ...
  
Within each window: Size-tiered compaction
When window expires: Drop entirely

Good for time-series with TTL
Avoids mixing old and new data
```

---

## Merge Process

### N-Way Merge

```python
def merge(sstables):
    # Use min-heap for efficient N-way merge
    heap = MinHeap()
    iterators = [sst.iterator() for sst in sstables]
    
    # Initialize heap with first entry from each
    for i, it in enumerate(iterators):
        if it.valid():
            heap.push((it.key(), it.value(), i))
            it.next()
    
    writer = SSTableWriter()
    last_key = None
    
    while not heap.empty():
        key, value, sst_idx = heap.pop()
        
        # Keep only latest version of each key
        if key != last_key:
            if not is_tombstone(value) or not at_bottom_level:
                writer.add(key, value)
            last_key = key
        
        # Advance that SSTable's iterator
        it = iterators[sst_idx]
        if it.valid():
            heap.push((it.key(), it.value(), sst_idx))
            it.next()
    
    return writer.finish()
```

### Handling Versions

```
Same key in multiple SSTables:
  [key: "a", value: "v1", seq: 100]  ← older
  [key: "a", value: "v2", seq: 150]  ← newer

Merge keeps seq: 150 (latest)
But if snapshot active at seq: 120, must keep both!
```

### Handling Tombstones

```
Tombstone (delete marker):
  [key: "a", TOMBSTONE, seq: 200]

Can only be garbage collected when:
  - At bottom level
  - All older versions are in this compaction
  - No active snapshots before tombstone seq
```

---

## Compaction Scheduling

### When to Compact

```
Triggers:
  1. L0 file count exceeds threshold
  2. Level size exceeds limit
  3. Tombstone ratio too high
  4. Manual trigger

Priority:
  L0 compaction > Other levels
  (L0 blocks writes when too many files)
```

### Rate Limiting

```
Problem: Compaction uses disk I/O
  - Competes with foreground reads/writes
  - Can cause latency spikes

Solutions:
  - Rate limit compaction I/O (bytes/sec)
  - Priority I/O scheduling (ionice)
  - Adaptive rate based on workload
```

### Thread Pool

```
Typical configuration:
  - 1-2 threads for L0→L1 (critical path)
  - Multiple threads for other levels
  - Separate thread pool for flushing
```

---

## Compaction Trade-offs

### Write Amplification Deep Dive

```
Leveled with ratio R=10:

Key written to MemTable: 1 write
Flushed to L0: 1 write
Compacted L0→L1: ~1 write
Compacted L1→L2: ~10 writes (merged with 10 L1 files)
Compacted L2→L3: ~10 writes
...

Total: 1 + 1 + 1 + 10 + 10 + 10 + ... = O(10 * L)

For 4 levels: ~40x write amplification
```

### Space Amplification Deep Dive

```
Size-tiered worst case:
  4 files in tier, about to merge
  + space for merged output
  = 2x space temporarily

Leveled:
  Non-overlapping files
  Only small compaction in progress
  ~1.1x typical
```

### Read Amplification Deep Dive

```
Size-tiered:
  Check all tiers (O(T))
  Check all files in tier (O(F))
  Total: O(T * F)

Leveled:
  L0: Check all files (small count)
  L1+: One file per level (non-overlapping)
  Total: O(L0 files + Levels)
  
Bloom filters reduce actual disk reads significantly
```

---

## Monitoring Compaction

### Key Metrics

```
Compaction pending bytes:
  How much work is queued
  Growing = compaction falling behind

Compaction I/O rate:
  MB/s written by compaction
  Compare to write rate

L0 file count:
  Should stay below stall threshold
  Spikes = write bursts or slow compaction

Write stalls:
  Time spent waiting for compaction
  Should be near zero normally
```

### RocksDB Statistics

```
db.getProperty("rocksdb.compaction-pending")
db.getProperty("rocksdb.num-files-at-level0")
db.getProperty("rocksdb.estimate-pending-compaction-bytes")
db.getProperty("rocksdb.compaction-reason")
```

---

## Optimizations

### Trivial Moves

```
If L(n) file doesn't overlap with L(n+1):
  Just move file pointer, don't rewrite!
  
Huge win for sequential insert patterns
```

### Subcompaction

```
Split large compaction into parallel sub-ranges:

File [a-z] overlaps with 10 files in next level
Split into:
  [a-j] + overlapping → thread 1
  [k-z] + overlapping → thread 2

Parallel merge, faster completion
```

### Dynamic Level Targets

```
RocksDB's dynamic leveling:
  Adjust level targets based on actual data size
  Avoids extra compaction when data is small
  Grows naturally as data grows
```

---

## Key Takeaways

1. **SSTables are immutable** - Write once, read many
2. **Compaction is the core trade-off** - Writes vs reads vs space
3. **Size-tiered favors writes** - But wastes space
4. **Leveled favors reads** - But more write amplification
5. **FIFO for time-series** - When old data expires naturally
6. **L0 is critical** - Too many files stalls writes
7. **Bloom filters are essential** - Reduce disk reads dramatically
8. **Monitor compaction debt** - Falling behind hurts performance
