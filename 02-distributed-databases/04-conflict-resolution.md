# Conflict Resolution

## TL;DR

Conflicts occur when concurrent updates happen across replicas. Resolution strategies range from simple (last-writer-wins) to complex (custom merge functions). Choose based on your data semantics: LWW for simplicity with data loss risk, CRDTs for automatic convergence, or application-level resolution for full control. The best conflict is one you prevent from happening.

---

## When Conflicts Occur

### Concurrent Updates

```
Replica A                    Replica B
    │                            │
 write(x, 1)                  write(x, 2)
    │                            │
    └────────────────────────────┘
                  │
           Both succeed locally
           Which value is correct?
```

### Network Partition

```
┌─────────────────┐     ┌─────────────────┐
│   Partition A   │     │   Partition B   │
│                 │     │                 │
│   write(x, 1)   │  X  │   write(x, 2)   │
│                 │     │                 │
└─────────────────┘     └─────────────────┘

After partition heals:
  x = 1 or x = 2?
```

### Offline Clients

```
Client A (online):  write(x, 1) → synced
Client B (offline): write(x, 2) → queued

Client B comes online:
  Conflict: x = 1 vs x = 2
```

---

## Conflict Types

### Write-Write Conflict

Same field modified differently.

```
Replica A: user.name = "Alice"
Replica B: user.name = "Bob"

Conflict: Which name?
```

### Read-Modify-Write Conflict

Both replicas read same value, modify, write.

```
Initial: counter = 10

Replica A: read 10, increment, write 11
Replica B: read 10, increment, write 11

Expected: 12
Actual: 11 (lost update)
```

### Delete-Update Conflict

One deletes, one updates.

```
Replica A: DELETE FROM users WHERE id = 1
Replica B: UPDATE users SET name = 'Bob' WHERE id = 1

What happens to the update?
```

### Constraint Violation

Both replicas create conflicting entries.

```
Replica A: INSERT (id=1, email='x@y.com')
Replica B: INSERT (id=2, email='x@y.com')

Unique constraint on email violated
```

---

## Resolution Strategies

### Last-Writer-Wins (LWW)

Highest timestamp wins.

```
Write A: {value: "Alice", timestamp: 1000}
Write B: {value: "Bob", timestamp: 1001}

Resolution: value = "Bob" (higher timestamp)
```

**Implementation:**
```python
def resolve_lww(versions):
    return max(versions, key=lambda v: v.timestamp)
```

**Pros:**
- Simple to implement
- Deterministic
- Automatic convergence

**Cons:**
- Data loss (earlier writes discarded)
- Clock skew can cause wrong winner
- No semantic understanding

**When to use:**
- Cache entries
- Last-modified timestamps
- Data where "most recent" makes sense

### First-Writer-Wins

Lowest timestamp wins (preserve original).

```
Write A: {value: "Alice", timestamp: 1000}
Write B: {value: "Bob", timestamp: 1001}

Resolution: value = "Alice" (lower timestamp)
```

**When to use:**
- Immutable records
- Audit logs
- "Create once" semantics

### Merge Values

Combine conflicting values.

```
Cart at A: [item1, item2]
Cart at B: [item1, item3]

Merge: [item1, item2, item3]
```

**Implementation:**
```python
def merge_sets(versions):
    result = set()
    for v in versions:
        result = result.union(v.items)
    return result
```

**Works for:**
- Sets (union)
- Counters (max or sum)
- Append-only lists

### Application-Level Resolution

Store all versions, let application decide.

```
Read(x) → {
    versions: [
        {value: "Alice", timestamp: 1000, source: "A"},
        {value: "Bob", timestamp: 1001, source: "B"}
    ],
    conflict: true
}

Application: Present UI for user to choose
```

**Implementation:**
```python
def read_with_conflicts(key):
    versions = get_all_versions(key)
    if len(versions) > 1:
        return Conflict(versions)
    return versions[0]

def resolve_conflict(key, chosen_version, discarded_versions):
    write(key, chosen_version)
    for v in discarded_versions:
        mark_as_resolved(v)
```

---

## CRDTs (Conflict-free Replicated Data Types)

### Concept

Data structures designed to always merge without conflict.

```
Property: Merge is:
  - Commutative: merge(A, B) = merge(B, A)
  - Associative: merge(merge(A, B), C) = merge(A, merge(B, C))
  - Idempotent: merge(A, A) = A

Any order of merging produces same result
```

### G-Counter (Grow-only Counter)

```
Each node tracks its own increment:
  Node A: {A: 5, B: 0, C: 0}
  Node B: {A: 3, B: 7, C: 0}

Merge: component-wise max
  Result: {A: 5, B: 7, C: 0}
  
Total: 5 + 7 + 0 = 12
```

**Operations:**
```python
class GCounter:
    def __init__(self, node_id):
        self.node_id = node_id
        self.counts = {}
    
    def increment(self):
        self.counts[self.node_id] = self.counts.get(self.node_id, 0) + 1
    
    def value(self):
        return sum(self.counts.values())
    
    def merge(self, other):
        for node, count in other.counts.items():
            self.counts[node] = max(self.counts.get(node, 0), count)
```

### PN-Counter (Positive-Negative Counter)

Supports increment and decrement:
```
P (positive): G-Counter for increments
N (negative): G-Counter for decrements

Value = P.value() - N.value()

Merge: merge P and N separately
```

### G-Set (Grow-only Set)

Elements can only be added, never removed.

```
Set A: {1, 2, 3}
Set B: {2, 3, 4}

Merge: union = {1, 2, 3, 4}
```

### OR-Set (Observed-Remove Set)

Supports add and remove with "add wins" semantics.

```
Each element has unique tags:
  add(x) → x with new tag
  remove(x) → remove all current tags

Concurrent add and remove:
  add(x) creates new tag, remove sees old tags
  Result: x exists (add wins)
```

**Implementation:**
```python
class ORSet:
    def __init__(self):
        self.elements = {}  # element → set of tags
    
    def add(self, element):
        tag = unique_tag()
        self.elements.setdefault(element, set()).add(tag)
    
    def remove(self, element):
        self.elements[element] = set()  # Remove all known tags
    
    def contains(self, element):
        return len(self.elements.get(element, set())) > 0
    
    def merge(self, other):
        for elem, tags in other.elements.items():
            self.elements[elem] = self.elements.get(elem, set()).union(tags)
```

### LWW-Register

Last-writer-wins register as a CRDT.

```python
class LWWRegister:
    def __init__(self):
        self.value = None
        self.timestamp = 0
    
    def write(self, value, timestamp):
        if timestamp > self.timestamp:
            self.value = value
            self.timestamp = timestamp
    
    def merge(self, other):
        if other.timestamp > self.timestamp:
            self.value = other.value
            self.timestamp = other.timestamp
```

### LWW-Map

Map where each key is an LWW-Register.

```
Node A: {"name": ("Alice", t=100), "age": (30, t=100)}
Node B: {"name": ("Bob", t=101), "city": ("NYC", t=100)}

Merge:
  "name": ("Bob", t=101)  ← higher timestamp
  "age": (30, t=100)
  "city": ("NYC", t=100)
```

---

## Version Vectors

### Tracking Causality

```
Version vector: {A: 3, B: 2, C: 1}

Meaning:
  - Incorporates 3 updates from A
  - Incorporates 2 updates from B
  - Incorporates 1 update from C
```

### Comparing Versions

```python
def compare(vv1, vv2):
    less_or_equal = all(vv1.get(k, 0) <= vv2.get(k, 0) for k in set(vv1) | set(vv2))
    greater_or_equal = all(vv1.get(k, 0) >= vv2.get(k, 0) for k in set(vv1) | set(vv2))
    
    if less_or_equal and not greater_or_equal:
        return "vv1 < vv2"  # vv1 is ancestor
    elif greater_or_equal and not less_or_equal:
        return "vv1 > vv2"  # vv2 is ancestor
    elif less_or_equal and greater_or_equal:
        return "equal"
    else:
        return "concurrent"  # Neither is ancestor → conflict
```

### Detecting Conflicts

```
Write 1: value="A", vv={A:1, B:0}
Write 2: value="B", vv={A:0, B:1}

Compare:
  {A:1, B:0} vs {A:0, B:1}
  Neither dominates → concurrent → conflict!
```

### Resolving with Version Vectors

```python
def read_repair(versions):
    # Find all versions that are not dominated by another
    non_dominated = []
    for v in versions:
        if not any(dominates(other.vv, v.vv) for other in versions if other != v):
            non_dominated.append(v)
    
    if len(non_dominated) == 1:
        return non_dominated[0]  # Clear winner
    else:
        return Conflict(non_dominated)  # Need resolution
```

---

## Semantic Conflict Resolution

### Three-Way Merge

Use common ancestor to intelligently merge.

```
Original (base): "The quick brown fox"
Version A:       "The quick red fox"    (changed brown → red)
Version B:       "The fast brown fox"   (changed quick → fast)

Three-way merge:
  - "quick" → "fast" (B's change)
  - "brown" → "red" (A's change)
  Result: "The fast red fox"
```

**Implementation:**
```python
def three_way_merge(base, a, b):
    a_changes = diff(base, a)
    b_changes = diff(base, b)
    
    for change in a_changes + b_changes:
        if conflicts_with(change, a_changes + b_changes):
            return Conflict(a, b)
    
    return apply_changes(base, a_changes + b_changes)
```

### Operational Transformation (OT)

Transform operations to apply in any order.

```
Base: "abc"
Op A: insert('X', position=1) → "aXbc"
Op B: insert('Y', position=2) → "abYc"

If A applied first:
  B needs transformation: position 2 → position 3
  "aXbc" + insert('Y', 3) = "aXbYc"

If B applied first:
  A remains: position 1
  "abYc" + insert('X', 1) = "aXbYc"

Same result regardless of order
```

---

## Delete Handling

### Tombstones

Mark deleted items instead of removing.

```
Before: {id: 1, name: "Alice", deleted: false}
Delete: {id: 1, name: "Alice", deleted: true, deleted_at: 1000}

Why tombstones:
  - Replicas need to know item was deleted
  - Otherwise they might resurrect it from their version
```

### Tombstone Cleanup

```
Problem: Tombstones accumulate forever

Solutions:
1. Time-based garbage collection
   Delete tombstones older than X days
   Risk: old replica comes back, resurrects data

2. Version vector garbage collection
   Delete when all replicas have seen tombstone
   Requires coordination

3. Compaction
   Periodically merge and remove tombstones
```

### Soft Delete vs Hard Delete

```
Soft delete: Keep record, mark as deleted
  + Can undelete
  + Preserves audit trail
  - Storage overhead

Hard delete: Remove record
  + No storage overhead
  - Can cause resurrection
  - Loses history
```

---

## Real-World Approaches

### Amazon Dynamo (Riak)

```
Strategy: Return all conflicting versions to client

Read → [{value: "A", clock: {A:1}}, {value: "B", clock: {B:1}}]

Client merges, writes back with merged clock
  Write(merged, clock: {A:1, B:1, Client:1})
```

### CouchDB

```
Store all revisions in conflict
  _id: "doc1"
  _conflicts: ["2-abc123", "2-def456"]

Application picks winner or merges
Losing revisions become historical
```

### Cassandra

```
LWW by default per column

CREATE TABLE users (
  id uuid,
  name text,
  email text,
  PRIMARY KEY (id)
);

-- Each column resolved independently
-- Highest timestamp wins per column
```

### Git

```
Three-way merge for file contents
Conflict markers for unresolvable conflicts

<<<<<<< HEAD
my changes
=======
their changes
>>>>>>> branch

User manually resolves
```

---

## Choosing a Strategy

### Decision Matrix

| Scenario | Strategy | Reasoning |
|----------|----------|-----------|
| Cache | LWW | Stale data acceptable |
| Counter | CRDT G-Counter | Accurate aggregation |
| Shopping cart | OR-Set CRDT | Add wins, merge items |
| User profile | LWW or merge fields | Field-level resolution |
| Document editing | OT or CRDT | Real-time collaboration |
| Financial transaction | Prevent conflicts | Use single leader |
| Social feed | LWW | Staleness OK |

### Prevention Over Resolution

Best conflict strategy is preventing conflicts:

```
1. Single leader for conflicting data
2. Partition data so conflicts impossible
3. Use optimistic locking with retries
4. Serialize operations through queue

Prevention is simpler than resolution
```

---

## Key Takeaways

1. **Conflicts are inevitable** in multi-leader/leaderless systems
2. **LWW is simple but lossy** - acceptable for some data
3. **CRDTs guarantee convergence** - no conflicts by design
4. **Version vectors detect concurrency** - essential for conflict detection
5. **Application knows best** - sometimes let user decide
6. **Tombstones are necessary** - but need cleanup strategy
7. **Prevent when possible** - single leader for critical data
8. **Choose by data semantics** - counters, sets, registers need different strategies
