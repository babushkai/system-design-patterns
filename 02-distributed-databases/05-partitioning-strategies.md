# Partitioning Strategies

## TL;DR

Partitioning (sharding) splits data across multiple nodes to scale beyond a single machine. Key strategies: hash partitioning for uniform distribution, range partitioning for efficient range queries. Consistent hashing minimizes rebalancing. Hotspots are the enemy. Choose partitioning key carefully—it determines what queries are efficient and which require scatter-gather.

---

## Why Partition?

### Single Node Limits

```
Data volume:    > 10 TB (exceeds disk)
Query load:     > 100K QPS (CPU bound)
Write load:     > 50K WPS (disk I/O bound)
Memory:         > 1 TB (RAM limit)
```

### Benefits of Partitioning

```
Before:
  [Single Node] ← All queries, all data

After:
  [Node 1] ← data A-M, queries for A-M
  [Node 2] ← data N-Z, queries for N-Z
  
Capacity: 2x data, 2x queries
Add more nodes → linear scaling
```

---

## Partitioning Methods

### Range Partitioning

Assign consecutive key ranges to partitions.

```
Partition 1: A-F    [aardvark, apple, ... , fox]
Partition 2: G-L    [giraffe, house, ... , lion]
Partition 3: M-R    [monkey, nest, ... , rabbit]
Partition 4: S-Z    [snake, tree, ... , zebra]
```

**Advantages:**
- Range queries efficient (scan one partition)
- Keys are naturally ordered

**Disadvantages:**
- Prone to hotspots (common prefixes)
- Manual rebalancing often needed

**Example: Time-series data**
```
Partition by date:
  2024-01: Partition 1
  2024-02: Partition 2
  2024-03: Partition 3  ← hot (current month)

Problem: Current month gets all writes
Solution: Compound key (sensor_id, timestamp)
```

### Hash Partitioning

Hash the key, assign to partition based on hash.

```
partition = hash(key) mod N

Example:
  hash("user_123") = 8742
  8742 mod 4 = 2
  → Partition 2
```

**Advantages:**
- Even distribution (good hash = uniform)
- No hotspots from sequential keys

**Disadvantages:**
- Range queries inefficient (all partitions)
- Adding nodes reshuffles many keys

**Hash function properties:**
```
Requirements:
  - Deterministic (same key → same hash)
  - Uniform distribution
  - Fast computation

Examples:
  - MD5 (slow, cryptographic)
  - MurmurHash (fast, good distribution)
  - xxHash (very fast)
```

### Consistent Hashing

Map both keys and nodes to a ring.

```
        0°
        │
   ┌────┼────┐
   │    │    │
270°────┼────90°
   │    │    │
   └────┼────┘
        │
       180°

Nodes: N1 at 30°, N2 at 120°, N3 at 250°
Key K: hash(K) = 100°
  → Goes to next node clockwise: N2 (120°)
```

**Adding a node:**
```
Before: N1(30°), N2(120°), N3(250°)
Add N4 at 200°

Only keys between 120°-200° move to N4
Other partitions unchanged
```

**Virtual nodes:**
```
Each physical node → multiple positions on ring
  N1: [30°, 100°, 220°]
  N2: [50°, 130°, 280°]
  
Benefits:
  - More even distribution
  - Smoother rebalancing
  - Handle heterogeneous hardware
```

---

## Choosing a Partition Key

### Single-Key Partitioning

```
Users table → partition by user_id

Query: SELECT * FROM users WHERE user_id = 123
  → Goes to one partition ✓

Query: SELECT * FROM users WHERE email = 'x@y.com'
  → Scatter to all partitions ✗
```

### Compound Keys

```
CREATE TABLE posts (
  user_id INT,
  post_id INT,
  content TEXT,
  PRIMARY KEY ((user_id), post_id)
);

user_id = partition key
post_id = clustering key (sort within partition)

Query: SELECT * FROM posts WHERE user_id = 123
  → One partition, sorted by post_id ✓

Query: SELECT * FROM posts WHERE user_id = 123 AND post_id > 100
  → One partition, range scan ✓
```

### Key Design Guidelines

| Access Pattern | Good Key | Bad Key |
|----------------|----------|---------|
| User's data | user_id | email |
| Time-series | (device_id, date) | timestamp |
| Orders | (customer_id, order_id) | order_date |
| Chat messages | (room_id, message_time) | sender_id |

---

## Handling Hotspots

### The Problem

```
Celebrity user: 10M followers
  - All posts by this user → one partition
  - All reads of their posts → one partition
  - That partition is overwhelmed

Sequential key: order_id auto-increment
  - All new orders → highest partition
  - Write hotspot
```

### Mitigation Strategies

**Add random prefix:**
```
Original key: user_123
Prefixed key: {0-9}_user_123  (random prefix)

Reads: scatter to 10 partitions, aggregate
Writes: distributed across 10 partitions

Trade-off: Single-key queries become scatter-gather
```

**Time bucketing:**
```
Instead of: partition by user_id
Use: partition by (user_id, time_bucket)

time_bucket = hour or day

Hot user's data spread across time buckets
Recent data in few buckets (queryable)
Old data in many buckets (archived)
```

**Read replicas per partition:**
```
Hot partition → more read replicas
Route reads to replicas
Writes still go to primary
```

---

## Rebalancing

### When to Rebalance

```
Triggers:
  - Node added
  - Node removed
  - Load imbalance detected
  - Data growth uneven
```

### Fixed Partition Count

```
Create more partitions than nodes:
  100 partitions for 10 nodes (10 each)

Add node:
  Move some partitions to new node
  (10 partitions → new node)

Partition boundaries never change
Simple, predictable
```

### Dynamic Partitioning

```
Partition grows too large → split
Partition shrinks → merge with neighbor

Example (HBase):
  Region grows > 10 GB → split
  Parent: [A-M]
  Children: [A-G], [H-M]
```

### Partition-Proportional Nodes

```
Each node gets fixed number of partitions
New node → steal partitions from existing nodes
More nodes → smaller partitions

Cassandra approach:
  Each node: 256 virtual nodes
  Add node: 256 new vnodes, take data from neighbors
```

### Minimizing Movement

```
Goal: Move minimum data when rebalancing

Consistent hashing: O(K/N) keys move
  K = total keys
  N = number of nodes

Naive hash mod: O(K) keys move
  Almost everything moves!
```

---

## Query Routing

### Client-Side Routing

```
Client knows partition map
Client sends request directly to correct partition

┌────────┐
│ Client │──────knows partition map
└────┬───┘
     │  partition_for(user_123) = Node 2
     ▼
┌─────────┐
│ Node 2  │
└─────────┘

Pros: No extra hop
Cons: Client must track partition changes
```

### Routing Tier

```
All requests → Router → Correct partition

┌────────┐     ┌────────┐     ┌─────────┐
│ Client │ ──► │ Router │ ──► │ Node N  │
└────────┘     └────────┘     └─────────┘

Pros: Clients are simple
Cons: Extra network hop, router can be bottleneck
```

### Coordinator Node

```
Any node can receive request
Node forwards to correct partition (or handles locally)

┌────────┐     ┌─────────┐     ┌─────────┐
│ Client │ ──► │ Node 1  │ ──► │ Node 3  │
└────────┘     │(coordinator)  │(partition owner)
               └─────────┘     └─────────┘

Pros: Any node is entry point
Cons: Extra hop if wrong node
```

### Partition Discovery

```
Approach 1: ZooKeeper / etcd
  - Nodes register partition ownership
  - Clients/routers watch for changes

Approach 2: Gossip protocol
  - Nodes share partition knowledge
  - Eventually consistent
  
Approach 3: Central metadata service
  - Dedicated service tracks partitions
  - Single source of truth
```

---

## Cross-Partition Operations

### Scatter-Gather Queries

```
Query: SELECT COUNT(*) FROM users WHERE age > 30

All partitions:
  [P1] → count: 1000
  [P2] → count: 1500
  [P3] → count: 800
  [P4] → count: 1200
  
Coordinator aggregates: 4500
```

**Performance:**
```
Latency = max(partition latencies) + aggregation
Throughput limited by slowest partition

One slow partition → slow query
```

### Cross-Partition Joins

```
Orders partitioned by customer_id
Products partitioned by product_id

SELECT o.*, p.name
FROM orders o
JOIN products p ON o.product_id = p.id
WHERE o.customer_id = 123

Strategy 1: Broadcast join
  Send all products to order partition
  
Strategy 2: Shuffle join
  Repartition both tables by join key
  
Strategy 3: Denormalize
  Store product name in orders table
```

### Cross-Partition Transactions

```
Transfer $100 from Account A (Partition 1) to Account B (Partition 2)

Requires distributed transaction:
  1. Start transaction on both partitions
  2. Debit A, credit B
  3. Two-phase commit

Expensive and complex
Consider: same-partition transfers only
```

---

## Partitioning in Practice

### PostgreSQL (Declarative Partitioning)

```sql
-- Range partitioning
CREATE TABLE orders (
    id SERIAL,
    order_date DATE,
    customer_id INT
) PARTITION BY RANGE (order_date);

CREATE TABLE orders_2024_q1 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

-- Hash partitioning
CREATE TABLE users (
    id INT,
    name TEXT
) PARTITION BY HASH (id);

CREATE TABLE users_p0 PARTITION OF users
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);
```

### Cassandra

```sql
CREATE TABLE posts (
    user_id uuid,
    post_time timestamp,
    content text,
    PRIMARY KEY ((user_id), post_time)
) WITH CLUSTERING ORDER BY (post_time DESC);

-- user_id is partition key (hashed)
-- post_time is clustering key (sorted within partition)
```

### MongoDB

```javascript
// Enable sharding
sh.enableSharding("mydb")

// Hash-based sharding
sh.shardCollection("mydb.users", { _id: "hashed" })

// Range-based sharding
sh.shardCollection("mydb.logs", { timestamp: 1 })
```

---

## Anti-Patterns

### Uneven Partition Sizes

```
Problem:
  Partition A: 100 GB
  Partition B: 10 GB
  Partition C: 500 GB ← overloaded

Causes:
  - Poor key distribution
  - Natural data skew
  
Solutions:
  - Better key choice
  - Salting keys
  - Dynamic splitting
```

### Monotonic Keys

```
Problem:
  key = timestamp or auto_increment
  All new data → last partition

Solutions:
  - Hash the key
  - Prepend random bytes
  - Use compound key with better distribution
```

### Too Few Partitions

```
Problem:
  4 partitions, want 10 nodes
  Cannot distribute evenly
  
Solution:
  Create many partitions upfront (e.g., 256)
  Distribute across available nodes
  Room to grow
```

### Cross-Partition Access as Primary Pattern

```
Problem:
  Most queries span all partitions
  No benefit from partitioning

Solutions:
  - Reconsider partition key
  - Denormalize data
  - Accept scatter-gather cost
```

---

## Key Takeaways

1. **Hash for distribution, range for queries** - Choose based on access patterns
2. **Consistent hashing reduces movement** - Essential for large-scale systems
3. **Partition key is critical** - Determines query efficiency
4. **Hotspots kill performance** - Use salting, time bucketing
5. **Rebalancing is expensive** - Plan partition count upfront
6. **Scatter-gather has overhead** - Design to minimize cross-partition queries
7. **Cross-partition transactions are hard** - Avoid if possible
8. **More partitions = more flexibility** - But more coordination overhead
