# Consistency Models

## TL;DR

Consistency models define what guarantees a distributed system provides about the order and visibility of operations. Stronger models (linearizability) are easier to reason about but expensive. Weaker models (eventual consistency) offer better performance but require careful application design.

---

## Why Consistency Models Matter

In a single-node system, operations happen in a clear order. In distributed systems:
- Nodes have different views of data at any moment
- Network delays cause operations to arrive out of order
- Failures mean some nodes miss updates

A consistency model is a contract between the system and application:
- **System promise**: "Here's what ordering guarantees you can rely on"
- **Application requirement**: "Here's what ordering I need for correctness"

---

## The Consistency Spectrum

```
Strongest
    │
    ▼
┌─────────────────────┐
│  Strict Consistency │  (Theoretical - requires instantaneous global updates)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Linearizability   │  (Single object, real-time ordering)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│    Sequential       │  (Global order, but not real-time)
│    Consistency      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│     Causal          │  (Only causally related ops ordered)
│     Consistency     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Read-Your-Writes  │  (See your own writes)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│    Monotonic Reads  │  (Never go backwards)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│     Eventual        │  (Eventually all see same value)
│     Consistency     │
└─────────────────────┘
    │
    ▼
Weakest
```

---

## Linearizability

### Definition

Every operation appears to take effect atomically at some point between its start and end. All operations have a global order respecting real-time.

```
Timeline:
              ┌─────────────┐
    Client A: │  write(x,1) │
              └─────────────┘
                    ▲
                    │ linearization point
                    │
              ┌─────┴─────────────┐
    Client B: │     read(x) → 1   │
              └───────────────────┘
```

### Properties

1. **Recency**: Reads return most recent write
2. **Real-time ordering**: If op A completes before op B starts, A appears before B
3. **Single-copy illusion**: System behaves as if there's one copy

### Implementation Approaches

**Single leader with synchronous replication:**
```
Client → Leader → [sync write to followers] → ack to client
```

**Consensus (Raft, Paxos):**
```
Client → Leader → [majority agreement] → commit → ack
```

**Compare-and-swap registers:**
```
CAS(expected, new) → atomic read-modify-write
```

### Cost of Linearizability

| Aspect | Impact |
|--------|--------|
| Latency | Must wait for coordination |
| Availability | Cannot respond during partition |
| Throughput | Single serialization point |

### When You Need It

- Distributed locks
- Leader election
- Unique constraint enforcement
- Financial transactions

---

## Sequential Consistency

### Definition

All operations appear to execute in some sequential order, and each processor's operations appear in program order. But this order doesn't need to match real-time.

```
Actual execution:
  Time →
  Node A: write(x,1) ─────────────────────
  Node B: ───────────────── write(x,2) ───
  Node C: ────────── read(x)→? ───────────

Sequential consistency allows:
  read(x) → 1  (order: write(1), read, write(2))
  read(x) → 2  (order: write(2), read, write(1))
  
But NOT: reading 1, then 2, then 1 again
```

### Difference from Linearizability

Linearizability: Real-time order matters
Sequential: Only program order per process matters

```
Real time:
  Process 1: write(x,1) completes at t=10
  Process 2: read(x) starts at t=15

Linearizable: read must return 1
Sequential: read might return old value if "read" is ordered before "write"
```

### Use Cases

- Total order broadcast
- Multi-threaded programming model
- Replicated state machines

---

## Causal Consistency

### Definition

Operations that are causally related appear in the same order to all nodes. Concurrent (unrelated) operations may appear in different orders.

### Causality Defined

Operation A causally precedes B if:
1. Same process: A happens before B in program order
2. Reads-from: A is a write, B is a read that returns A's value
3. Transitivity: A precedes C, C precedes B → A precedes B

```
Causal chain:
  User 1: write("Hello")          [message 1]
          ↓ reads
  User 2: write("Reply to Hello") [message 2]

All nodes must see message 1 before message 2
```

### Concurrent Operations

```
User A: write("Post A")
User B: write("Post B")   ← concurrent, no causal relation

Node 1 might show: Post A, Post B
Node 2 might show: Post B, Post A
Both are valid under causal consistency
```

### Implementation: Vector Clocks

```
Vector clock: [A:3, B:2, C:1]

Each node maintains clock for every node
On local event: increment own counter
On send: attach vector clock
On receive: merge (max each component), then increment own
```

**Comparing vector clocks:**
```
V1 = [2, 3, 1]
V2 = [2, 2, 2]

V1 < V2?  No (3 > 2)
V2 < V1?  No (2 > 1)
Concurrent? Yes (neither dominates)
```

### Causal+ Consistency

Causal consistency plus convergence: concurrent writes resolve to same value everywhere.

Resolution strategies:
- Last-writer-wins (LWW)
- Multi-value (return all concurrent values)
- Application-specific merge

---

## Session Guarantees

Weaker consistency models that are often "good enough."

### Read Your Writes

After writing, you see your own writes.

```
✓ Correct:
  write(x, 1)
  read(x) → 1

✗ Violation:
  write(x, 1)
  read(x) → old_value  (stale replica)
```

**Implementation:**
- Sticky sessions (always same node)
- Include write timestamp, wait if replica behind
- Read from leader after writing

### Monotonic Reads

Once you've seen a value, you never see older values.

```
✓ Correct:
  read(x) → 5
  read(x) → 5 or higher

✗ Violation:
  read(x) → 5
  read(x) → 3  (went backwards)
```

**Implementation:**
- Track high-water mark per client
- Sticky sessions
- Version vectors

### Monotonic Writes

Writes by a process are seen in order by all nodes.

```
✓ Correct:
  write(x, 1)
  write(x, 2)
  All nodes eventually have: 1 → 2

✗ Violation:
  Node A sees: 2, then 1 (wrong order)
```

### Writes Follow Reads

If you read a value and then write, your write is ordered after the read.

```
Process reads x = 5, then writes y = 10

All nodes see: write(x, 5) happens before write(y, 10)
```

---

## Eventual Consistency

### Definition

If no new updates are made, eventually all replicas converge to the same value.

```
write(x, 1) at node A

Time →
  Node A: x=1 ─────────────────────────────
  Node B: x=0 ──── x=1 (propagated) ───────
  Node C: x=0 ────────────── x=1 ──────────
                              ↑
                    Eventual convergence
```

### What Eventual Consistency Does NOT Guarantee

- How long "eventually" takes
- What value you'll read before convergence
- Which write "wins" if concurrent

### Conflict Resolution

When concurrent writes exist:

**Last-Writer-Wins (LWW):**
```
write(x, 1) at t=10
write(x, 2) at t=15
Result: x = 2 (higher timestamp wins)

Problem: Clock skew can discard writes
```

**Multi-Value (Siblings):**
```
write(x, 1) at Node A
write(x, 2) at Node B (concurrent)
Result: x = {1, 2} (application must resolve)
```

**CRDTs (Conflict-free Replicated Data Types):**
```
G-Counter: only increment, merge = max per node
LWW-Register: last-writer-wins with logical clock
OR-Set: add wins over concurrent remove
```

---

## Tunable Consistency

Many systems allow per-operation consistency choice.

### Quorum Parameters

```
N = total replicas
W = write quorum (replicas that must ack write)
R = read quorum (replicas to read from)
```

**Guarantees:**
```
W + R > N  → Strong consistency (overlap guarantees seeing latest)
W + R ≤ N  → Eventual consistency (might miss latest)
```

**Common configurations:**

| Config | W | R | Consistency | Use Case |
|--------|---|---|-------------|----------|
| Strong | N | 1 | Strong | Writes slow, reads fast |
| Strong | ⌈N/2⌉+1 | ⌈N/2⌉+1 | Strong | Balanced |
| Eventual | 1 | 1 | Eventual | Maximum performance |
| Write-heavy | 1 | N | Eventual+ | Tolerate write loss |

### Example: Cassandra

```cql
-- Strong consistency
SELECT * FROM users WHERE id = 123 
USING CONSISTENCY QUORUM;

-- Eventual consistency (faster)
SELECT * FROM users WHERE id = 123 
USING CONSISTENCY ONE;
```

---

## Consistency in Practice

### Choosing a Model

| Requirement | Minimum Model |
|-------------|---------------|
| Distributed lock | Linearizable |
| Counter with exact count | Linearizable |
| User sees own posts | Read-your-writes |
| Chat message ordering | Causal |
| Social feed | Eventual |
| Shopping cart | Eventual + CRDT |
| Configuration | Linearizable |

### Mixing Consistency Levels

Most applications use multiple levels:

```
User profile updates: Eventual (staleness OK)
Password changes: Read-your-writes (must see new password)
Account balance: Linearizable (must be accurate)
```

### Testing Consistency

**Jepsen** - Black-box consistency testing:
1. Perform operations against cluster
2. Record history of operations
3. Check if history matches consistency model

**Linearizability checker:**
```
History:
  [invoke write(1)]
  [invoke read]
  [ok write(1)]
  [ok read → 0]  ← Violation! Read should see 1

Check: Is there a linearization? No.
```

---

## Key Takeaways

1. **Stronger isn't always better** - Pay for what you need
2. **Linearizability is expensive** - Requires coordination, hurts availability
3. **Causal consistency is often sufficient** - Preserves intuitive ordering
4. **Eventual consistency requires conflict handling** - CRDTs or application logic
5. **Session guarantees help** - Read-your-writes often enough for good UX
6. **Tune per-operation** - Different data has different requirements
7. **Test your assumptions** - Use tools like Jepsen to verify
