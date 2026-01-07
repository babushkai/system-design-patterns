# Distributed Time

## TL;DR

There is no global clock in distributed systems. Physical clocks drift and can't provide ordering guarantees. Use logical clocks (Lamport, Vector) for causality tracking and hybrid clocks (HLC) when you need both causality and wall-clock time. Physical time is good for humans; logical time is good for machines.

---

## The Problem With Physical Time

### Clock Drift

Every computer has a quartz crystal oscillator. They're cheap but imprecise:
- Typical drift: 10-200 ppm (parts per million)
- 100 ppm = 8.6 seconds/day
- After a week: ~1 minute off

```
True time:    |────────────────────────────────|
              0                               1 day

Server A:     |─────────────────────────────────|
              0                               1 day + 17 sec

Server B:     |───────────────────────────────|
              0                               1 day - 8 sec
```

### NTP Synchronization

Network Time Protocol synchronizes clocks, but imperfectly:
- LAN accuracy: 1-10 ms typical
- WAN accuracy: 10-100 ms
- Spikes during network issues

```
NTP correction:
  Before: Server clock 500ms ahead
  After:  Clock slewed/stepped back
  
  t=1000ms  t=1001ms  t=1002ms  t=1001ms  t=1002ms
                                    ↑
                             Time went backwards!
```

### Leap Seconds

UTC occasionally adds leap seconds. Clocks might:
- Jump forward (missing a second)
- Repeat a second (same timestamp twice)
- "Smear" over hours (Google's approach)

---

## Why Ordering Matters

### The Timestamp Ordering Problem

```
Server A (clock fast):   write(x, "A") at 10:00:05.000
Server B (clock slow):   write(x, "B") at 10:00:03.000

Actual wall-clock order: B happened first
Timestamp order:         A appears first

If using Last-Writer-Wins: wrong value wins!
```

### Causality Violations

```
Message ordering failure:

Alice → Bob: "Want to grab lunch?" [t=10:00:01.000]
Bob → Alice: "Sure, where?"        [t=10:00:00.500 - clock behind!]

Displayed to Alice:
  Bob: "Sure, where?"
  Alice: "Want to grab lunch?"
  
  ↑ Nonsensical order
```

---

## Lamport Clocks

### Definition

A logical clock that provides a partial ordering of events.

**Rules:**
1. Each process maintains a counter
2. On local event: increment counter
3. On send: attach counter, then increment
4. On receive: counter = max(local, received) + 1

### Example

```
Process A          Process B          Process C
    │                  │                  │
    1 (internal)       │                  │
    │                  │                  │
    2 ─────────────────►3                 │
    │              (received 2,          │
    │               max(0,2)+1=3)        │
    │                  │                  │
    │                  4 ─────────────────►5
    │                  │                  │
    3                  │                  │
    │                  │                  │
```

### Lamport Clock Properties

**Provides:**
- If A → B (A causally precedes B), then L(A) < L(B)

**Does NOT provide:**
- If L(A) < L(B), we cannot conclude A → B
- They might be concurrent

```
L(A) = 5, L(B) = 7

Possible interpretations:
1. A caused B (A → B)
2. A and B are concurrent, happened to get these values
3. B happened before A in real time, but didn't cause it
```

### Total Ordering with Lamport Clocks

Break ties with process ID:

```
Event ordering: (timestamp, process_id)

Event at A: (5, A)
Event at B: (5, B)

Order: (5, A) < (5, B)  (assuming A < B alphabetically)
```

This gives a consistent total order, but it's arbitrary for concurrent events.

---

## Vector Clocks

### Definition

A vector of counters, one per process. Tracks causality precisely.

**Rules:**
1. Each process has a vector of N counters (N = number of processes)
2. On local event: increment own position
3. On send: attach entire vector, increment own position
4. On receive: merge vectors (component-wise max), then increment own

### Example

```
Process A              Process B              Process C
[A:1, B:0, C:0]            │                      │
    │                      │                      │
[A:2, B:0, C:0]──────►[A:2, B:1, C:0]             │
    │                      │                      │
    │              [A:2, B:2, C:0]─────────►[A:2, B:2, C:1]
    │                      │                      │
[A:3, B:0, C:0]            │                      │
    │                      │                      │
```

### Comparing Vector Clocks

```
V1 ≤ V2  iff  ∀i: V1[i] ≤ V2[i]
V1 < V2  iff  V1 ≤ V2 and V1 ≠ V2
V1 || V2 iff  ¬(V1 ≤ V2) and ¬(V2 ≤ V1)  (concurrent)
```

**Examples:**
```
[2, 3, 1] < [2, 4, 1]  ✓  (causally before)
[2, 3, 1] < [3, 3, 1]  ✓  (causally before)
[2, 3, 1] || [2, 2, 2]    (concurrent: 3>2 but 1<2)
[2, 3, 1] || [1, 4, 1]    (concurrent: 2>1 but 3<4)
```

### Vector Clock Properties

**Provides:**
- V(A) < V(B) ⟺ A → B (causally precedes)
- V(A) || V(B) ⟺ A and B are concurrent

**Limitations:**
- O(N) space per event
- Problematic with many processes
- Need to know all processes upfront

---

## Optimizations for Vector Clocks

### Version Vectors (Dotted)

Used in Dynamo-style systems. Track per-replica, not per-event.

```
Object version: {A:3, B:2}

Client reads from A, writes to B:
  New version: {A:3, B:3}

Concurrent write at A:
  Version: {A:4, B:2}

Conflict detected: {A:4, B:2} || {A:3, B:3}
```

### Interval Tree Clocks

For dynamic systems where processes join/leave:
- Split clock on fork
- Merge on join
- More complex but O(log N) typical size

### Bloom Clocks

Probabilistic: may say "concurrent" for causally related events, but never misses true concurrency.

---

## Hybrid Logical Clocks (HLC)

### Motivation

We want:
1. Causality tracking (like logical clocks)
2. Correlation with wall-clock time (for humans/debugging)
3. Bounded divergence from physical time

### Design

HLC = (physical_time, logical_counter)

```
HLC: (pt, l)
  pt = physical time component (wall clock)
  l  = logical component (counter)
```

**Rules:**
```
On local/send event:
  pt' = max(pt, physical_clock())
  if pt' == pt:
    l' = l + 1
  else:
    l' = 0
  
On receive(remote_pt, remote_l):
  pt' = max(pt, remote_pt, physical_clock())
  if pt' == pt == remote_pt:
    l' = max(l, remote_l) + 1
  elif pt' == pt:
    l' = l + 1
  elif pt' == remote_pt:
    l' = remote_l + 1
  else:
    l' = 0
```

### Example

```
Physical clocks: A=100, B=100, C=98 (C is behind)

Event at A: (100, 0)
Send A→B:   B receives, sees (100, 0)
            B's clock is 100, so: (100, 1)
Send B→C:   C receives (100, 1)
            C's clock is 98 (behind!)
            pt' = max(98, 100) = 100
            Result: (100, 2)
```

### HLC Properties

- Monotonic: always moves forward
- Bounded: l is bounded by max clock skew × message rate
- Causal: if A → B, then HLC(A) < HLC(B)
- Close to real time: pt tracks physical time

---

## TrueTime (Google Spanner)

### The Approach

Instead of pretending clocks are accurate, expose uncertainty.

```
TrueTime API:
  TT.now() → [earliest, latest]
  
Example:
  TT.now() = [10:00:05.003, 10:00:05.009]
  
  Meaning: actual time is somewhere in that interval
```

### Infrastructure

- GPS receivers + atomic clocks in datacenters
- Uncertainty typically 1-7ms
- After GPS outage, uncertainty grows

```
Uncertainty sources:
  ├── GPS receiver jitter: ~1ms
  ├── Network delay to GPS: ~1ms
  └── Oscillator drift since sync: grows over time
```

### Commit Wait

For serializable transactions:

```
Transaction T1:
  1. Acquire locks
  2. Get commit timestamp s = TT.now().latest
  3. Wait until TT.now().earliest > s  ("commit wait")
  4. Release locks

Transaction T2 starting after T1 completes:
  Gets timestamp > s guaranteed
  
Result: Real-time ordering of transactions
```

### Trade-off

Commit wait = latency cost:
- ~7ms added to writes
- Worth it for global serializability

---

## Time in Practice

### Use Cases by Clock Type

| Use Case | Clock Type | Why |
|----------|------------|-----|
| Log timestamps | Physical (NTP) | Human readability |
| Event ordering | Lamport | Simple, sufficient for total order |
| Conflict detection | Vector | Detect concurrent writes |
| Database transactions | HLC | Causality + time correlation |
| Global transactions | TrueTime | Real-time ordering guarantee |
| Cache TTL | Physical | Approximate is OK |
| Lease expiration | Physical + margin | Add safety buffer |

### Common Patterns

**Lease safety margin:**
```
Lease holder: refresh every 30s, lease valid 60s
Other nodes: assume lease valid until 60s + clock_skew
```

**Timestamp generation:**
```
// Ensure monotonicity despite clock adjustments
last_timestamp = 0

function get_timestamp():
  now = physical_clock()
  if now <= last_timestamp:
    now = last_timestamp + 1
  last_timestamp = now
  return now
```

**Distributed unique ID with time:**
```
// Snowflake-style ID
ID = [timestamp_ms (41 bits)][machine_id (10 bits)][sequence (12 bits)]

// 4096 IDs per millisecond per machine
// Ordered by time (mostly)
```

---

## Handling Clock Issues

### Detecting Clock Problems

```
On message receive:
  sender_time = message.timestamp
  receiver_time = local_clock()
  
  if receiver_time < sender_time - threshold:
    // Receiver clock behind
    log_warning("Clock skew detected")
    
  if sender_time < last_message_from_sender:
    // Sender clock went backward
    log_error("Clock regression detected")
```

### Defensive Design

1. **Never trust equality:** `if time_a == time_b` is dangerous
2. **Use ranges:** "happened between T1 and T2"
3. **Include sequence numbers:** for ordering within same timestamp
4. **Prefer logical time for internal events**
5. **Reserve physical time for human interfaces**

---

## Key Takeaways

1. **Physical clocks drift** - Don't rely on them for ordering
2. **NTP helps but isn't perfect** - Millisecond accuracy typical
3. **Lamport clocks give total order** - Simple, low overhead
4. **Vector clocks detect concurrency** - But expensive at scale
5. **HLC combines benefits** - Causality + wall-clock correlation
6. **TrueTime exposes uncertainty** - Enables global transactions
7. **Choose based on requirements** - More guarantees = more cost
8. **Design defensively** - Assume clocks will misbehave
