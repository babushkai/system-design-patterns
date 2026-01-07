# B-Trees

## TL;DR

B-trees are the most widely-used index structure in databases. They maintain sorted data with O(log n) reads, writes, and range queries. B+-trees (the common variant) store all data in leaf nodes, making range scans efficient. Understanding page splits, fill factors, and write amplification is key to optimizing B-tree performance.

---

## Why B-Trees?

### The Disk Access Problem

```
Disk characteristics:
  - Sequential read:  100+ MB/s
  - Random read:      100-200 IOPS (spinning disk)
  - SSD random read:  10,000+ IOPS

Memory access: ~100 nanoseconds
Disk access:   ~10 milliseconds (spinning)

Ratio: 100,000x slower for random disk access

Goal: Minimize disk accesses per operation
```

### B-Tree Solution

```
Store data in large blocks (pages) that match disk I/O
Few levels → few disk reads

Height 3 B-tree with 100 keys/node:
  Level 0: 1 node (100 keys)
  Level 1: 100 nodes (10,000 keys)
  Level 2: 10,000 nodes (1,000,000 keys)
  
  3 disk reads to find any of 1 million keys
```

---

## B-Tree Structure

### Node Layout

```
Each node is a fixed-size page (typically 4-16 KB)

     ┌──────────────────────────────────────────┐
     │  [P0] K1 [P1] K2 [P2] K3 [P3] ... Kn [Pn] │
     └──────────────────────────────────────────┘
     
Ki = Key i
Pi = Pointer to child (or data in leaf)

Invariant:
  All keys in subtree P(i-1) < Ki <= All keys in subtree Pi
```

### B-Tree vs B+-Tree

```
B-Tree:
  - Data stored in all nodes
  - Fewer total nodes
  - Range scan requires tree traversal
  
B+-Tree (most common):
  - Data only in leaf nodes
  - Internal nodes = routing only
  - Leaf nodes linked → efficient range scan
  
     [10 | 20 | 30]        ← Internal node (keys only)
     /    |    |    \
    ↓     ↓    ↓     ↓
   [1-9][10-19][20-29][30-39]  ← Leaf nodes (keys + data)
     ↔     ↔     ↔     ↔       ← Sibling pointers
```

### B+-Tree Properties

```
Order m (max children per node):
  - Internal nodes: ⌈m/2⌉ to m children
  - Leaf nodes: ⌈m/2⌉ to m key-value pairs
  - Root: 2 to m children (or 0 if empty)

Typical m = 100-1000+ depending on page size
```

---

## Operations

### Search

```
def search(node, key):
    if node.is_leaf:
        return binary_search(node.keys, key)
    
    # Find child to descend into
    i = binary_search_position(node.keys, key)
    child = read_page(node.pointers[i])
    return search(child, key)

Complexity: O(log_m n) pages read
           O(log n) total key comparisons
```

### Range Scan

```
def range_scan(start, end):
    # Find start position
    leaf = find_leaf(start)
    position = binary_search(leaf.keys, start)
    
    # Scan leaves using sibling pointers
    results = []
    while leaf and leaf.keys[position] <= end:
        results.append(leaf.values[position])
        position += 1
        if position >= len(leaf.keys):
            leaf = leaf.next_sibling
            position = 0
    return results

Efficient: Sequential access after finding start
```

### Insert

```
def insert(key, value):
    leaf = find_leaf(key)
    
    if leaf.has_space():
        leaf.insert(key, value)
    else:
        # Split the leaf
        new_leaf = split(leaf)
        middle_key = new_leaf.first_key
        
        # Insert middle key into parent
        insert_into_parent(leaf.parent, middle_key, new_leaf)

Split may cascade up to root
```

### Leaf Split

```
Before (full leaf, m=4):
  [10, 20, 30, 40]

Insert 25:
  Split into two leaves
  
  [10, 20] → [30, 40]  (new leaf takes upper half)
                ↓
  [30, 40, 25] → sort → [25, 30, 40]
  
  Actually:
  [10, 20, 25] [30, 40]
  
  Promote 30 to parent:
  Parent: [..., 30, ...]
            ↓    ↓
         [leaf1][leaf2]
```

### Delete

```
def delete(key):
    leaf = find_leaf(key)
    leaf.remove(key)
    
    if leaf.is_underfull():
        if can_borrow_from_sibling(leaf):
            borrow(leaf)
        else:
            merge_with_sibling(leaf)
            # May cascade up

Underflow when: < ⌈m/2⌉ keys
```

---

## Page Layout

### Slotted Page Format

```
┌────────────────────────────────────────────────────┐
│ Header │ Slot 1 │ Slot 2 │ ... │ Free │ ... │Data│
├────────┴────────┴────────┴─────┴──────┴─────┴─────┤
│ ◄─── Slots grow →                 ← Data grows ──►│
└────────────────────────────────────────────────────┘

Header: Page ID, number of slots, free space pointer
Slot: Offset to data, length
Data: Variable-length records

Advantages:
  - Variable-length keys/values
  - Easy deletion (mark slot as empty)
  - Efficient compaction
```

### Key Compression

```
Prefix compression in internal nodes:
  Original: ["application", "apply", "approach"]
  Compressed: ["appli", "appr"]  (minimum to distinguish)

Suffix truncation:
  Don't need full key in internal nodes
  Just enough to route correctly
```

---

## Write Amplification

### The Problem

```
Insert one key-value pair (100 bytes):
  1. Read page (4 KB)
  2. Modify page (add 100 bytes)
  3. Write page (4 KB)
  
Write amplification = 4 KB / 100 bytes = 40x

For update in place:
  WAL write + Page write = 2x amplification minimum
```

### Mitigation

```
1. Larger pages (more data per I/O)
2. Buffer pool (cache hot pages in memory)
3. Batch writes (group modifications)
4. Append-only B-trees (COW for reduced random writes)
```

---

## Concurrency Control

### Page-Level Locking

```
Simple approach:
  Lock page during read/write
  Release when done
  
Problem: 
  Page splits acquire locks bottom-up
  Risk of deadlock
```

### Latch Crabbing

```
Traversal with safe release:

1. Acquire latch on child
2. If child is "safe" (won't split/merge):
   Release latch on all ancestors
3. Continue down

Safe node:
  - For insert: has space for one more key
  - For delete: has more than minimum keys
```

```
Search: Read latch → descend → release parent → repeat
        (crab down the tree)

Insert:
  Acquire write latches top-down
  Release ancestors when child is safe
  
Example (insert, safe child):
  [Parent] ← write latch
      ↓
  [Child is safe] ← write latch, release parent
      ↓
  [Leaf] ← write latch, do insert
```

### Optimistic Locking

```
1. Traverse with read latches only
2. At leaf, try to upgrade to write latch
3. If structure changed (version mismatch):
   Restart traversal

Reduces contention for read-heavy workloads
```

---

## Durability and Recovery

### Write-Ahead Logging (WAL)

```
Before modifying page:
  1. Write log record (page ID, old value, new value)
  2. Fsync log
  3. Modify page in buffer pool
  4. Eventually flush dirty page

Recovery:
  Replay log to reconstruct pages
```

### Crash Recovery

```
WAL ensures:
  - Committed transactions' changes applied
  - Uncommitted transactions' changes undone

B-tree specific:
  - Half-completed splits must be completed or undone
  - Log sufficient info to redo split
```

---

## Copy-on-Write B-Trees

### Concept

Never modify pages in place.

```
Original tree:
     [A]
    /   \
  [B]   [C]
  
Update to [B]:
  1. Create new [B'] with modification
  2. Create new [A'] pointing to [B'] and [C]
  3. Update root pointer to [A']
  
Old pages remain until garbage collected
```

### Advantages

```
+ No WAL needed (old version always valid)
+ Readers never blocked
+ Snapshots are free (just keep old root)
+ Simple crash recovery
```

### Disadvantages

```
- Write amplification (entire path to root)
- Fragmentation (new pages not contiguous)
- Garbage collection needed
- Space amplification during updates
```

### Systems Using COW

```
LMDB:   Copy-on-write B-tree
BoltDB: Copy-on-write B+-tree
btrfs:  Copy-on-write filesystem
```

---

## B-Tree Variants

### B*-Tree

```
More aggressive node filling:
  - Siblings help before splitting
  - Minimum occupancy: 2/3 (not 1/2)
  - Better space utilization
```

### Bᵋ-Tree (B-epsilon Tree)

```
Buffer at each node for pending updates:
  - Insert writes to buffer
  - Buffer flushed when full
  - Reduces write amplification
  
Trade-off: Faster writes, slower reads
```

### Fractal Tree

```
Similar to Bᵋ-tree:
  - Messages buffered at each level
  - Batch flushes down tree
  
Used by TokuDB (MySQL), PerconaFT
```

---

## Performance Characteristics

### Complexity

| Operation | Average | Worst |
|-----------|---------|-------|
| Search | O(log n) | O(log n) |
| Insert | O(log n) | O(log n) |
| Delete | O(log n) | O(log n) |
| Range | O(log n + k) | O(log n + k) |

k = number of results

### Space Utilization

```
Typical fill factor: 50-70%
  - Splits create half-full nodes
  - Random inserts fill non-uniformly

Bulk loading: 90%+ possible
  - Sort data first
  - Build bottom-up
  - Pack leaves fully
```

### I/O Patterns

```
Read:   Random I/O (traverse nodes)
        Sequential within page
        
Write:  Random I/O (update pages)
        WAL is sequential
        
Range:  Sequential after finding start
        (leaf nodes linked)
```

---

## Practical Considerations

### Page Size Selection

```
Larger pages:
  + Fewer levels (faster traversal)
  + Better for range scans
  + Better for HDDs
  - More write amplification
  - More memory per page

Typical: 4 KB (SSD), 8-16 KB (HDD)
```

### Fill Factor

```
CREATE INDEX ... WITH (fillfactor = 70);

Lower fill factor:
  + Room for inserts without splits
  + Better for write-heavy workloads
  - More space, more pages to read

Higher fill factor:
  + Less space, fewer pages
  + Better for read-heavy workloads
  - More splits on insert
```

### Monitoring

```
Key metrics:
  - Tree height (should be stable)
  - Page splits per second
  - Fill factor / space utilization
  - Cache hit ratio for index pages
  - I/O wait time
```

---

## Key Takeaways

1. **B+-trees dominate databases** - Leaf nodes contain data, linked for scans
2. **Log(n) operations** - Few disk accesses for any operation
3. **Page splits cascade** - Insert can modify multiple pages
4. **Write amplification is real** - 40x not unusual
5. **Concurrency is complex** - Latch crabbing for safety
6. **COW simplifies recovery** - At cost of more writes
7. **Fill factor is tunable** - Trade space for write performance
8. **Range scans are efficient** - Sequential access after locate
