# Consensus Algorithms

## TL;DR

Consensus enables distributed nodes to agree on a value despite failures. Paxos is theoretically elegant but hard to implement. Raft is designed for understandability and is widely used. Both tolerate f failures with 2f+1 nodes. Consensus is expensive—use it only for coordination, not for all data operations.

---

## The Consensus Problem

### Definition

Get a group of nodes to agree on a single value.

```
Nodes: A, B, C, D, E
Proposals: A proposes "X", B proposes "Y"

Requirements:
  1. Agreement: All nodes decide the same value
  2. Validity: Decided value was proposed by some node
  3. Termination: All correct nodes eventually decide
```

### Why It's Hard

```
Scenario: 5 nodes, A proposes "X"

Time 1: A sends proposal to all
Time 2: B, C receive proposal, agree
Time 3: A crashes
Time 4: D, E haven't received proposal

What should D, E decide?
  - They don't know about "X"
  - Can't contact A (crashed)
  - Might receive conflicting proposal from B
```

### FLP Impossibility

```
Fischer, Lynch, Paterson (1985):

In an asynchronous system with even ONE faulty node,
no consensus protocol can guarantee termination.

Implication:
  - Can't distinguish slow node from crashed node
  - Must use timeouts (gives up pure asynchrony)
  - All practical systems use partial synchrony
```

---

## Paxos

### Basic Paxos

Agrees on a single value.

**Roles:**
```
Proposers: Propose values
Acceptors: Accept/reject proposals
Learners:  Learn decided values

(Same node can play multiple roles)
```

**Phase 1: Prepare**
```
Proposer:
  1. Choose proposal number n (unique, increasing)
  2. Send Prepare(n) to majority of acceptors

Acceptor on receiving Prepare(n):
  If n > highest_seen:
    highest_seen = n
    Promise: won't accept proposals < n
    Reply with any previously accepted value
```

**Phase 2: Accept**
```
Proposer (if majority promised):
  If any acceptor replied with accepted value:
    Use that value (can't choose own value)
  Else:
    Use own proposed value
  Send Accept(n, value) to majority

Acceptor on receiving Accept(n, value):
  If n >= highest_seen:
    Accept proposal
    Reply with acceptance
```

**Decision:**
```
When proposer receives majority acceptances:
  Value is decided
  Notify learners
```

### Paxos Example

```
Proposers P1, P2; Acceptors A, B, C

P1: Prepare(1) → A, B, C
  A: Promise(1), no prior value
  B: Promise(1), no prior value
  C: Promise(1), no prior value

P1: Accept(1, "X") → A, B, C
  A: Accepted(1, "X")
  B: Accepted(1, "X")
  C: (delayed, hasn't responded)

P1 has majority → "X" is decided

Meanwhile, P2: Prepare(2) → A, B, C
  A: Promise(2), previously accepted (1, "X")
  B: Promise(2), previously accepted (1, "X")
  
P2 must propose "X", not own value!
```

### Multi-Paxos

Optimize for multiple decisions in sequence.

```
Basic Paxos: 2 round-trips per decision (expensive)

Multi-Paxos:
  1. Elect stable leader (one Prepare phase)
  2. Leader issues Accept for many values
  3. Re-elect only if leader fails

Amortizes Prepare phase across many decisions
```

### Paxos Challenges

```
1. Complex state machine
   - Many edge cases
   - Hard to implement correctly

2. Livelock possible
   - P1 prepares, P2 prepares with higher number
   - P1 retries higher, P2 retries higher
   - No progress

3. No distinguished leader by default
   - Multi-Paxos needs leader election on top
```

---

## Raft

### Design Goals

```
"Understandability first"

Key simplifications:
  1. Strong leader: all writes through leader
  2. Clear separation of concerns
  3. Reduced state space
```

### Raft Roles

```
Leader:   Handles all client requests, replicates log
Follower: Passively replicate leader's log
Candidate: Trying to become leader

State transitions:
  Follower → Candidate (election timeout)
  Candidate → Leader (wins election)
  Candidate → Follower (discovers current leader)
  Leader → Follower (discovers higher term)
```

### Terms

```
Term: Logical clock, monotonically increasing

Term 1: Leader A
Term 2: Leader B (A failed, B elected)
Term 3: Leader B (re-elected)
Term 4: Leader C (B failed)

Each term has at most one leader
Terms used to detect stale leaders
```

### Leader Election

```
1. Follower times out (no heartbeat from leader)
2. Increments term, becomes Candidate
3. Votes for self, sends RequestVote to all
4. Others reply with vote (if not already voted in term)
5. Candidate with majority becomes Leader
6. Leader sends heartbeats to maintain authority
```

```
Node A          Node B          Node C
   │               │               │
   │  (timeout)    │               │
   │───►Candidate  │               │
   │               │               │
   │──RequestVote─►│               │
   │──RequestVote──┼──────────────►│
   │               │               │
   │◄──Vote────────│               │
   │◄──Vote────────┼───────────────│
   │               │               │
   │───►Leader     │               │
   │               │               │
   │──Heartbeat───►│               │
   │──Heartbeat────┼──────────────►│
```

### Log Replication

```
Client → Leader: Write "X"
Leader:
  1. Append "X" to local log
  2. Send AppendEntries to followers
  3. Wait for majority acknowledgment
  4. Commit entry
  5. Apply to state machine
  6. Respond to client
```

```
Leader log:     [1:A][2:B][3:C][4:D]
                              ↑
                          committed

Follower 1 log: [1:A][2:B][3:C][4:D]  ← up to date
Follower 2 log: [1:A][2:B][3:C]       ← lagging
Follower 3 log: [1:A][2:B]            ← more lagging

Commit when entry in log of majority (Leader + F1 + F2 = 3/5)
```

### Log Matching

```
Property: If two logs contain entry with same index and term,
          all preceding entries are identical.

Enforcement:
  - Leader includes previous entry (index, term) in AppendEntries
  - Follower rejects if doesn't match
  - Leader decrements and retries until match found
  - Follower overwrites conflicting entries
```

### Raft vs Paxos

| Aspect | Paxos | Raft |
|--------|-------|------|
| Understandability | Complex | Simple |
| Leader | Optional (Multi-Paxos) | Required |
| Log gaps | Allowed | Not allowed |
| Membership change | Complex | Joint consensus |
| Industry adoption | Low | High |

---

## Practical Consensus Systems

### etcd (Raft)

```
Distributed key-value store

Client → Leader → Replicate → Majority ack → Commit

Usage: Kubernetes configuration, service discovery

Operations:
  Put(key, value)
  Get(key)
  Watch(prefix)  // Stream changes
  Transaction    // Atomic compare-and-swap
```

### ZooKeeper (ZAB)

```
ZooKeeper Atomic Broadcast

Similar to Raft:
  - Leader-based
  - Log replication
  - Majority commit

Differences:
  - Designed before Raft
  - Different recovery protocol
  - Hierarchical namespace (like filesystem)
```

### Consul (Raft)

```
Service mesh and configuration

Raft for:
  - Catalog (service registry)
  - KV store
  - Session management

Each datacenter has Raft cluster
Cross-DC uses gossip + WAN Raft (optional)
```

---

## Performance Considerations

### Latency

```
Consensus write latency:
  1. Client → Leader (network)
  2. Leader → Followers (network, parallel)
  3. Disk write at majority nodes
  4. Leader → Client (network)

Minimum: 2 × network RTT + disk flush

Optimization: Pipeline writes, batch entries
```

### Throughput

```
Bottlenecks:
  1. Network bandwidth (replication)
  2. Leader CPU (processing all writes)
  3. Disk I/O (fsync on commit)

Scaling:
  - Can't add nodes (more replication overhead)
  - Batch entries
  - Pipeline replication
  - Use consensus for metadata, not all data
```

### Leader Bottleneck

```
All writes through leader = single point of throughput

Solutions:
  1. Partition data, separate Raft groups per partition
  2. Use consensus for coordination only
  3. Offload reads to followers (stale reads acceptable)
```

---

## Membership Changes

### The Problem

```
Old config: A, B, C (majority = 2)
New config: A, B, C, D, E (majority = 3)

Danger: During transition, two majorities might exist

Old majority: A, B
New majority: C, D, E

Could elect two leaders!
```

### Joint Consensus (Raft)

```
1. Leader creates joint config: C_old + C_new
2. Replicates joint config
3. Once committed, both majorities needed
4. Leader creates C_new config
5. Replicates C_new
6. Once committed, old members can leave

Safety: Never two independent majorities
```

### Single-Server Changes

```
Simpler: Add/remove one server at a time

Adding D to {A, B, C}:
  1. D syncs log from leader
  2. Once caught up, add to config
  3. Now {A, B, C, D}

One at a time guarantees overlap between old and new quorums
```

---

## Common Patterns

### Replicated State Machine

```
Same log → Same state

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Log:      │    │   Log:      │    │   Log:      │
│ [A][B][C]   │    │ [A][B][C]   │    │ [A][B][C]   │
├─────────────┤    ├─────────────┤    ├─────────────┤
│ State: X    │    │ State: X    │    │ State: X    │
│ (from ABC)  │    │ (from ABC)  │    │ (from ABC)  │
└─────────────┘    └─────────────┘    └─────────────┘
   Node 1            Node 2             Node 3

All nodes apply same operations in same order
All arrive at same state
```

### Linearizable Reads

```
Option 1: Read through leader
  - Leader confirms still leader (heartbeat)
  - Then responds
  
Option 2: ReadIndex
  - Record commit index
  - Wait for commit index to be applied
  - Then respond
  
Option 3: Lease-based
  - Leader has time-based lease
  - Responds if within lease
  - No network round-trip if lease valid
```

### Snapshots

```
Problem: Log grows forever

Solution: Periodic snapshots
  1. Serialize state machine to disk
  2. Truncate log up to snapshot point
  3. New followers can bootstrap from snapshot

Snapshot = State at log index N
Resume replication from index N+1
```

---

## When to Use Consensus

### Good Use Cases

```
✓ Configuration management (who is the leader)
✓ Distributed locks
✓ Coordination (barriers, leader election)
✓ Metadata storage
✓ Sequence number generation
```

### Avoid For

```
✗ All user data (too expensive)
✗ High-throughput writes (leader bottleneck)
✗ Latency-sensitive reads (can use caching instead)
✗ Data that can tolerate eventual consistency
```

### Architecture Pattern

```
                    ┌──────────────┐
                    │   Metadata   │
                    │ (Raft/etcd)  │
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │  Data     │    │   Data    │    │   Data    │
   │  Shard 1  │    │  Shard 2  │    │  Shard 3  │
   │ (primary- │    │ (primary- │    │ (primary- │
   │  backup)  │    │  backup)  │    │  backup)  │
   └───────────┘    └───────────┘    └───────────┘

Consensus for: Who owns which shard, leader election
Data: Simpler replication (cheaper)
```

---

## Key Takeaways

1. **Consensus solves agreement** - One value, all nodes agree
2. **FLP impossibility limits guarantees** - Need timeouts in practice
3. **Paxos is correct but complex** - Hard to implement
4. **Raft prioritizes clarity** - Widely adopted
5. **Majority needed (2f+1 nodes)** - Tolerates f failures
6. **Consensus is expensive** - Use for coordination, not all data
7. **Leader is bottleneck** - Partition data for scale
8. **Snapshots prevent log growth** - Essential for long-running systems
