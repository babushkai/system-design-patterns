# CAP Theorem

## TL;DR

CAP theorem states: A distributed system can provide at most 2 of 3 guarantees: Consistency, Availability, Partition tolerance. But this framing is misleading. In practice, partitions happen, so you choose between C and A during partitions.

---

## The Three Properties

### Consistency (C)

Every read receives the most recent write or an error.

This is **linearizability** - the strongest consistency model:
- All nodes see the same data at the same time
- Operations appear to execute atomically
- If write W completes before read R starts, R sees W

Not to be confused with ACID consistency (invariants).

### Availability (A)

Every request receives a non-error response, without guarantee of most recent data.

- Every non-failing node must respond
- No "sorry, try again later"
- Response time not specified (could be slow)

### Partition Tolerance (P)

System continues to operate despite network partitions.

A partition means: some nodes cannot communicate with others.

```
     ┌─────────────────┐
     │                 │
   ┌─┴─┐             ┌─┴─┐
   │ A │──────X──────│ B │
   └───┘   Partition └───┘
```

---

## Why "Pick 2" Is Misleading

### Partitions Are Not Optional

In a distributed system, network partitions will happen:
- Network equipment fails
- Cables get cut
- Datacenters lose connectivity

You cannot choose "CA" (Consistency + Availability, no Partition tolerance) because you don't control the network.

### The Real Choice

When a partition occurs, you must choose:

**CP (Consistency + Partition tolerance):**
- Refuse to respond if uncertain about latest data
- Return error rather than stale data
- Example: Leader can't reach followers, rejects writes

**AP (Availability + Partition tolerance):**
- Always respond, even with potentially stale data
- Accept writes on both sides of partition
- Deal with conflicts later
- Example: Eventual consistency

### When There's No Partition

When the network is healthy, you can have both C and A:
- Synchronous replication provides consistency
- All nodes available

CAP only forces a choice during partitions.

---

## CAP in Practice

### CP Systems

Prioritize consistency when partition occurs:

| System | Behavior During Partition |
|--------|---------------------------|
| ZooKeeper | Minority partition becomes unavailable |
| etcd | Same - Raft requires majority |
| HBase | Regions with unreachable master unavailable |
| Spanner | Uses TrueTime, can wait out uncertainty |

**When to choose CP:**
- Financial transactions
- Coordination services (locks, leader election)
- Any case where stale reads cause serious harm

### AP Systems

Prioritize availability when partition occurs:

| System | Behavior During Partition |
|--------|---------------------------|
| Cassandra | Accept writes on any node |
| DynamoDB | Accept writes, resolve with LWW |
| CouchDB | Accept writes, resolve on merge |
| DNS | Serve cached records |

**When to choose AP:**
- Shopping carts (merge items later)
- Social media feeds (slight staleness OK)
- Caching systems
- Any case where availability matters more than immediate consistency

---

## Beyond CAP: PACELC

CAP only describes behavior during partitions. What about normal operation?

**PACELC** extends CAP:
- If Partition: choose between Availability and Consistency
- Else: choose between Latency and Consistency

```
PACELC:
  P     A/C     E     L/C
  │      │      │      │
  └──────┴──────┴──────┘
  Partition? → A or C
  Normal?    → L or C
```

Examples:
- DynamoDB: PA/EL (available during partition, low latency normally)
- MongoDB (default): PA/EC (available during partition, consistent normally)
- Spanner: PC/EC (consistent always, pays latency cost)

---

## Consistency Models Spectrum

CAP's "Consistency" is linearizability, but there are weaker models:

```
Strongest
    │
    ▼
┌─────────────────┐
│ Linearizability │ ← CAP "Consistency"
└────────┬────────┘
         │
┌────────▼────────┐
│ Sequential      │
│ Consistency     │
└────────┬────────┘
         │
┌────────▼────────┐
│ Causal          │
│ Consistency     │
└────────┬────────┘
         │
┌────────▼────────┐
│ Eventual        │
│ Consistency     │
└─────────────────┘
    │
    ▼
Weakest
```

Many systems offer tunable consistency - choose per-operation.

---

## Common Misconceptions

### "We need a CP database"

Maybe not. Questions to ask:
- What happens if user sees stale data?
- Can application handle retries?
- Is eventual consistency with conflict resolution acceptable?

### "AP means no consistency"

AP means no consistency guarantee during partitions. Systems are usually consistent when healthy.

### "CAP means distributed systems are limited"

CAP describes extreme cases. Most systems:
- Experience few partitions
- Offer tunable consistency
- Can handle limited staleness

### "Network partitions are rare"

In large systems, partitions happen regularly:
- Google: ~5 partitions per year per cluster
- More common in geo-distributed systems
- Also: partial partitions, asymmetric partitions

---

## Practical Guidelines

### For Most Applications

1. Use AP with conflict resolution for user data
2. Use CP for coordination (locks, sequences)
3. Implement idempotency to handle retries

### Hybrid Approaches

Different data has different requirements:

| Data Type | Choice | Reasoning |
|-----------|--------|-----------|
| User sessions | AP | Can regenerate |
| Inventory count | CP | Overselling is costly |
| Social feed | AP | Staleness acceptable |
| Payment ledger | CP | Must be accurate |
| Shopping cart | AP | Merge on checkout |
| Configuration | CP | All nodes need same view |

### Implementation Pattern

```
Read path:
  if (strong_consistency_needed):
    read_from_leader()
  else:
    read_from_any_replica()  // faster, maybe stale

Write path:
  if (consistency_critical):
    synchronous_replication()  // wait for ack
  else:
    async_replication()  // faster, risk of loss
```

---

## Key Takeaways

1. CAP is about partition behavior, not normal operation
2. Partitions happen - you can't choose "CA"
3. The real choice: consistency or availability during partition
4. Most systems are consistent when healthy, differ during partitions
5. PACELC extends CAP to normal operation latency tradeoffs
6. Hybrid approaches: different consistency for different data
7. Linearizability is expensive; weaker models often sufficient
