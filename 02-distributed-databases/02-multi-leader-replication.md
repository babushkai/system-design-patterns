# Multi-Leader Replication

## TL;DR

Multi-leader (master-master) replication allows writes at multiple nodes, each replicating to others. It enables low-latency writes from any location and tolerates datacenter failures. The price: write conflicts are possible and must be resolved. Use when you need multi-region writes or high write availability; avoid when strong consistency is required.

---

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ┌────────────┐         ┌────────────┐         ┌────────────┐
│   │  Leader A  │◄───────►│  Leader B  │◄───────►│  Leader C  │
│   │   (US)     │         │  (Europe)  │         │   (Asia)   │
│   └─────┬──────┘         └─────┬──────┘         └─────┬──────┘
│         │                      │                      │
│         ▼                      ▼                      ▼
│   ┌──────────┐           ┌──────────┐           ┌──────────┐
│   │ Followers│           │ Followers│           │ Followers│
│   └──────────┘           └──────────┘           └──────────┘
│                                                             │
└─────────────────────────────────────────────────────────────┘
                         Clients write to nearest leader
```

### Write Flow

```
1. Client in Europe writes to Leader B
2. Leader B accepts write, responds to client
3. Leader B asynchronously replicates to A and C
4. Leaders A and C apply the change

Write succeeds without waiting for cross-datacenter round-trip
```

### Replication Topologies

**Circular:**
```
    A ───► B
    ▲      │
    │      ▼
    D ◄─── C
    
Each node replicates to next; failures break the ring
```

**Star (Hub and Spoke):**
```
        A
       /│\
      / │ \
     ▼  ▼  ▼
    B   C   D
    
Central hub coordinates; hub failure is critical
```

**All-to-All:**
```
    A ◄───► B
    ▲ \   / ▲
    │  \ /  │
    │   X   │
    │  / \  │
    ▼ /   \ ▼
    C ◄───► D
    
Most resilient; conflicts more complex
```

---

## Use Cases

### Multi-Datacenter Operation

```
US Datacenter:
  - Users write locally
  - <10ms write latency
  - Survives Europe/Asia outage

Europe Datacenter:
  - Users write locally
  - <10ms write latency
  - Survives US/Asia outage

Cross-DC replication: 100-200ms (async)
```

### Collaborative Editing

```
User A (laptop): types "Hello"
User B (phone): types "World" (same document, same position)

Both succeed locally
Conflict resolution determines final state
```

### Offline Clients

```
Mobile app (disconnected):
  Write changes locally (local leader)
  Queue for sync
  
When connected:
  Sync with server
  Resolve conflicts
  
Each device is essentially a leader
```

---

## Conflict Handling

### When Conflicts Occur

```
Timeline:
  
  Leader A                    Leader B
     │                           │
  write(x, 1)                 write(x, 2)
     │                           │
     └───────────────────────────┘
                   │
            Both succeed locally
            Replication reveals conflict
```

### Types of Conflicts

**Write-Write:**
Same field updated differently.
```
Leader A: user.email = "a@example.com"
Leader B: user.email = "b@example.com"
```

**Delete-Update:**
One deletes, one updates.
```
Leader A: DELETE user WHERE id=1
Leader B: UPDATE user SET name='Bob' WHERE id=1
```

**Uniqueness violation:**
Both create records with same unique value.
```
Leader A: INSERT (id=auto, email='x@y.com')
Leader B: INSERT (id=auto, email='x@y.com')
```

### Conflict Avoidance

Prevent conflicts by routing related writes to same leader.

```
Strategy: All writes for a user go to their "home" datacenter

User 123 → always Leader A
User 456 → always Leader B

Conflicts impossible for per-user data
Cross-user operations may still conflict
```

---

## Conflict Resolution Strategies

### Last-Writer-Wins (LWW)

Highest timestamp wins; discard other writes.

```
Write at A: {value: 1, timestamp: 100}
Write at B: {value: 2, timestamp: 105}

Resolution: value = 2 (higher timestamp)

Problem: Write at A is silently lost
Problem: Clock skew can choose "wrong" winner
```

### Merge Values

Combine conflicting values.

```
Shopping cart at A: [item1, item2]
Shopping cart at B: [item1, item3]

Merge: [item1, item2, item3]
```

### Custom Resolution

Application-specific logic.

```
// For document editing
func resolve_conflict(version_a, version_b):
  merged = three_way_merge(base, version_a, version_b)
  if has_semantic_conflict(merged):
    return create_conflict_marker(version_a, version_b)
  return merged
```

### Application-Level Resolution

Store all versions; let user decide.

```
Read returns: {
  versions: [
    {value: "Alice", timestamp: 100, origin: "A"},
    {value: "Bob", timestamp: 105, origin: "B"}
  ],
  conflict: true
}

UI: "Multiple versions found. Which is correct?"
```

### CRDTs

Conflict-free Replicated Data Types - mathematically guaranteed to converge.

```
G-Counter (grow-only counter):
  Node A: {A: 5, B: 3}
  Node B: {A: 4, B: 7}
  
  Merge: {A: max(5,4), B: max(3,7)} = {A: 5, B: 7}
  Total: 12
  
  Always converges, never conflicts
```

---

## Handling Causality

### The Problem

Without tracking causality, operations may be applied in wrong order.

```
User 1 at Leader A:
  1. INSERT message(id=1, text="Hello")
  2. INSERT message(id=2, text="World", reply_to=1)

Replication to Leader B might arrive:
  Message 2 arrives before Message 1
  reply_to=1 references non-existent message
```

### Version Vectors

Track causality across leaders.

```
Version vector: {A: 3, B: 5, C: 2}

Meaning: 
  - Seen 3 operations from A
  - Seen 5 operations from B
  - Seen 2 operations from C

Comparing:
  {A:3, B:5} vs {A:4, B:4}
  Neither dominates → concurrent, potential conflict
```

### Detecting Causality

```
Write at A: attached vector {A:10, B:5, C:7}
Write at B: attached vector {A:10, B:6, C:7}

A's write precedes B's? 
  Check if A's vector ≤ B's vector
  {A:10, B:5, C:7} ≤ {A:10, B:6, C:7}? 
  Yes: A ≤ B in all components

Apply A's write before B's
```

---

## Replication Lag and Ordering

### Causality Anomalies

```
Leader A: User posts message (seq 1)
Leader B: User edits profile (seq 1)

Without ordering:
  Follower might see edit before message
  Or message before edit
  
With logical clocks:
  Total order preserved across leaders
```

### Conflict-Free Operations

Some operations don't conflict even if concurrent:

```
Concurrent but safe:
  Leader A: UPDATE users SET last_login = now() WHERE id = 1
  Leader B: UPDATE users SET email_count = email_count + 1 WHERE id = 1

Different columns → merge both changes
```

---

## Implementation Considerations

### Primary Key Generation

Avoid conflicts on auto-increment IDs.

```
Strategy 1: Range allocation
  Leader A: IDs 1-1000000
  Leader B: IDs 1000001-2000000
  
Strategy 2: Composite keys
  ID = (leader_id, sequence_number)
  
Strategy 3: UUIDs
  Globally unique, no coordination needed
```

### Uniqueness Constraints

How to enforce unique email across leaders?

```
Option 1: Check before write (racy)
  Check locally → might conflict with other leader

Option 2: Conflict detection
  Accept write, detect duplicate on sync
  Application handles de-duplication

Option 3: Deterministic routing
  All writes for email domain → specific leader
```

### Foreign Key Constraints

Cross-leader FK enforcement is hard.

```
Leader A: INSERT order (user_id = 123)
Leader B: DELETE user WHERE id = 123 (concurrent)

Results:
  Order references non-existent user
  
Solutions:
  - Soft deletes
  - Application-level referential integrity
  - Accept inconsistency, repair later
```

---

## Real-World Systems

### CouchDB

```
// Writes go to any node
PUT /db/doc123
{
  "_id": "doc123",
  "_rev": "1-abc123",
  "name": "Alice"
}

// Conflict detection on sync
GET /db/doc123?conflicts=true
{
  "_id": "doc123",
  "_rev": "2-def456",
  "_conflicts": ["2-xyz789"],
  "name": "Alice Smith"
}

// Application resolves by deleting losing revisions
```

### MySQL Group Replication

```sql
-- Enable multi-primary mode
SET GLOBAL group_replication_single_primary_mode = OFF;

-- All members accept writes
-- Certification-based conflict detection
-- Conflicting transactions rolled back on one node
```

### Galera Cluster

```
wsrep_provider = /usr/lib/galera/libgalera_smm.so
wsrep_cluster_address = gcomm://node1,node2,node3

-- Synchronous replication with certification
-- Conflicts detected before commit
-- "Optimistic locking" - most transactions succeed
```

---

## Monitoring Multi-Leader

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Replication lag | Time behind other leaders | > 1 minute |
| Conflict rate | Conflicts per second | Increasing trend |
| Conflict resolution time | Time to resolve | > 1 second avg |
| Cross-DC latency | Replication RTT | > 500ms |
| Queue depth | Pending replication ops | Growing |

### Health Checks

```python
def check_multi_leader_health():
  for leader in leaders:
    for other in leaders:
      if leader == other:
        continue
      
      # Check replication is flowing
      lag = get_replication_lag(leader, other)
      if lag > threshold:
        alert(f"{leader} → {other} lag: {lag}")
      
      # Check connectivity
      if not can_connect(leader, other):
        alert(f"{leader} cannot reach {other}")
```

---

## When to Use Multi-Leader

### Good Fit

- Multi-datacenter deployment with local writes
- Offline-first applications
- Collaborative editing
- High write availability requirements
- Tolerance for eventual consistency

### Poor Fit

- Strong consistency requirements
- Complex transactions across datacenters
- Low conflict tolerance
- Simple single-region deployments
- Applications unable to handle conflict resolution

---

## Key Takeaways

1. **Writes anywhere** - Low latency, but conflicts possible
2. **Conflicts are inevitable** - Must have resolution strategy
3. **LWW is simple but lossy** - Consider merge or CRDTs
4. **Topology matters** - All-to-all most resilient
5. **Causality tracking is complex** - Version vectors help
6. **Avoid conflicts when possible** - Route related writes together
7. **Unique constraints are hard** - Application-level handling often needed
8. **Great for multi-region** - Primary use case is geo-distribution
