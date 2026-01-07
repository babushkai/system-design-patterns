# Secondary Indexes in Distributed Databases

## TL;DR

Primary key determines partition location. Secondary indexes enable queries on non-partition-key columns but add complexity. Two approaches: local indexes (fast writes, scatter reads) and global indexes (fast reads, slow writes). Choose based on read/write ratio and query patterns. Secondary indexes are expensive in distributed systems—use sparingly.

---

## The Problem

### Partition by Primary Key

```
Users table, partitioned by user_id:

Partition 1: user_id 1-1000
Partition 2: user_id 1001-2000
Partition 3: user_id 2001-3000

Query: SELECT * FROM users WHERE user_id = 1500
  → Goes to Partition 2 only ✓
```

### Query by Non-Partition Column

```
Query: SELECT * FROM users WHERE email = 'alice@example.com'

Problem: email is not the partition key
  - Which partition has this email?
  - Must check ALL partitions (scatter-gather)
  
Without secondary index: O(N) partitions scanned
With secondary index: O(1) or O(few) partitions
```

---

## Local Secondary Indexes

### Concept

Each partition maintains its own index for local data.

```
Partition 1:                    Partition 2:
┌─────────────────────────┐    ┌─────────────────────────┐
│ Data:                   │    │ Data:                   │
│  user_id=1, email=a@x   │    │  user_id=1001, email=d@x│
│  user_id=2, email=b@x   │    │  user_id=1002, email=e@x│
│  user_id=3, email=c@x   │    │  user_id=1003, email=f@x│
│                         │    │                         │
│ Local Index (email):    │    │ Local Index (email):    │
│  a@x → user_id=1        │    │  d@x → user_id=1001     │
│  b@x → user_id=2        │    │  e@x → user_id=1002     │
│  c@x → user_id=3        │    │  f@x → user_id=1003     │
└─────────────────────────┘    └─────────────────────────┘
```

### Write Path

```
INSERT user (id=1500, email='new@example.com')

1. Route to Partition 2 (based on id=1500)
2. Insert data row
3. Update local email index on Partition 2

Single partition operation ✓
```

### Read Path

```
SELECT * FROM users WHERE email = 'alice@example.com'

1. Don't know which partition has this email
2. Query ALL partitions' local indexes
3. Aggregate results

Scatter-gather to all partitions ✗
```

### Trade-offs

| Aspect | Local Index |
|--------|-------------|
| Write performance | Fast (single partition) |
| Read performance | Slow (all partitions) |
| Consistency | Strong (same partition) |
| Index maintenance | Simple |
| Hotspot risk | None (distributed) |

### Use Cases

- Write-heavy workloads
- Queries often include partition key
- Analytics queries (expect scatter-gather)
- Low-cardinality columns (few matches per partition)

---

## Global Secondary Indexes

### Concept

Index partitioned separately from data.

```
Data Partitions:                Index Partitions (by email hash):
┌───────────────┐              ┌───────────────────────────┐
│ Partition 1   │              │ Index Partition 1 (a-m)   │
│ user_id 1-1000│              │  alice@x → user_id=5      │
└───────────────┘              │  bob@x → user_id=1500     │
┌───────────────┐              │  carol@x → user_id=2500   │
│ Partition 2   │              └───────────────────────────┘
│ user_id 1001- │              ┌───────────────────────────┐
│         2000  │              │ Index Partition 2 (n-z)   │
└───────────────┘              │  ned@x → user_id=42       │
┌───────────────┐              │  zoe@x → user_id=999      │
│ Partition 3   │              └───────────────────────────┘
│ user_id 2001- │
│         3000  │
└───────────────┘
```

### Write Path

```
INSERT user (id=1500, email='bob@example.com')

1. Write data to Partition 2 (based on id)
2. Write to Index Partition 1 (based on email hash)

Two partitions involved
May need distributed transaction or async update
```

### Read Path

```
SELECT * FROM users WHERE email = 'bob@example.com'

1. Hash 'bob@example.com' → Index Partition 1
2. Lookup in Index Partition 1 → user_id=1500
3. Fetch from Data Partition 2

Two partitions, but targeted (not scatter) ✓
```

### Trade-offs

| Aspect | Global Index |
|--------|--------------|
| Write performance | Slow (multi-partition) |
| Read performance | Fast (targeted lookup) |
| Consistency | Async = eventual, Sync = slow |
| Index maintenance | Complex (distributed update) |
| Hotspot risk | Possible (popular index values) |

### Consistency Options

**Synchronous update:**
```
Transaction:
  1. Write data
  2. Write index
  3. Commit both

Guarantees: Read-your-writes
Cost: 2PC overhead, higher latency
```

**Asynchronous update:**
```
1. Write data (committed)
2. Queue index update
3. Apply index update (eventually)

Guarantees: Eventually consistent
Cost: May read stale index
```

---

## Partitioning the Global Index

### By Index Value (Term-Partitioned)

```
Index partitioned by the indexed column value:

email starting with a-m → Index Partition 1
email starting with n-z → Index Partition 2

Query: WHERE email = 'alice@x'
  → Only Index Partition 1
  
Good for: Single-value lookups
Bad for: Range queries across partition boundaries
```

### By Document ID

```
Index entries for same document → same partition

user_id 1-1000: all indexes in Index Partition 1
user_id 1001-2000: all indexes in Index Partition 2

Write: Single partition for all indexes
Read: May need multiple index partitions

Similar to local index but separated
```

---

## Implementation Examples

### DynamoDB Global Secondary Index

```
Table: Users
  Primary Key: user_id (partition key)
  Attributes: email, name, city

GSI: email-index
  Partition Key: email
  Projection: ALL  (copies all attributes)
  
Query:
  aws dynamodb query \
    --table-name Users \
    --index-name email-index \
    --key-condition-expression "email = :e" \
    --expression-attribute-values '{":e":{"S":"alice@x"}}'
```

**GSI Characteristics:**
- Eventually consistent reads
- Provisioned capacity separate from table
- Writes to table propagate async to GSI

### Cassandra Materialized Views

```sql
CREATE TABLE users (
    user_id uuid PRIMARY KEY,
    email text,
    name text
);

-- Materialized view for email lookups
CREATE MATERIALIZED VIEW users_by_email AS
    SELECT * FROM users
    WHERE email IS NOT NULL
    PRIMARY KEY (email, user_id);

-- Query by email
SELECT * FROM users_by_email WHERE email = 'alice@example.com';
```

**MV Characteristics:**
- Synchronous update
- Base table write waits for MV update
- Strongly consistent

### Elasticsearch

```json
// Index with multiple searchable fields
PUT /users
{
  "mappings": {
    "properties": {
      "user_id": { "type": "keyword" },
      "email": { "type": "keyword" },
      "name": { "type": "text" },
      "tags": { "type": "keyword" }
    }
  }
}

// Query by any field
GET /users/_search
{
  "query": {
    "term": { "email": "alice@example.com" }
  }
}
```

**ES Characteristics:**
- Inverted index for all fields
- Near real-time indexing
- Scatter-gather for distributed search

---

## Scatter-Gather Optimization

### Parallel Queries

```
Query all partitions simultaneously:

Coordinator:
  for partition in partitions:
    async_query(partition)
  
  results = await_all()
  return merge(results)

Latency = max(partition latencies) + merge time
```

### Short-Circuit Evaluation

```
Query: SELECT * FROM users WHERE email = 'x' LIMIT 1

Scatter to all partitions
First partition to return match → return immediately
Cancel other queries

Optimization for existence checks
```

### Bloom Filters

```
Each partition maintains Bloom filter for indexed values

Query: WHERE email = 'alice@example.com'

1. Check Bloom filter on each partition (local operation)
2. Only query partitions where Bloom filter says "maybe"
3. Skip partitions where Bloom filter says "definitely not"

Reduces scatter-gather to likely partitions
```

---

## Covering Indexes

### Include All Needed Columns

```
Index includes:
  - Indexed column (email)
  - Primary key (user_id)
  - Additional columns (name, city)

Query: SELECT name, city FROM users WHERE email = 'alice@x'

1. Lookup in index
2. Return directly from index (no data fetch needed)

Avoids second lookup to data partition
```

### Index-Only Scans

```sql
-- PostgreSQL example
CREATE INDEX idx_users_email_name ON users(email) INCLUDE (name);

EXPLAIN SELECT name FROM users WHERE email = 'alice@x';
-- Index Only Scan using idx_users_email_name
```

### Trade-off

```
+ Faster reads (no secondary fetch)
- Larger index (stores more data)
- More writes (update index on any included column change)
```

---

## Secondary Index Alternatives

### Denormalization

```
Instead of index on orders.customer_email:

Store customer_email directly in orders table:
  orders: {order_id, customer_id, customer_email, ...}

Query by email → query orders directly

Trade-off:
  + No index maintenance
  - Data duplication
  - Update anomalies
```

### Materialized Views

```
Pre-compute query results as a new table:

CREATE TABLE orders_by_customer_email AS
  SELECT * FROM orders
  JOIN customers ON orders.customer_id = customers.id;

Refresh periodically or on change

Trade-off:
  + Fast reads
  - Storage overhead
  - Staleness or refresh cost
```

### External Search System

```
PostgreSQL (source of truth)
        │
        ▼ (CDC or batch sync)
   Elasticsearch (search)

Query flow:
  1. Search Elasticsearch for IDs
  2. Fetch full records from PostgreSQL

Trade-off:
  + Purpose-built search
  + Complex query support
  - Operational complexity
  - Eventual consistency
```

---

## Choosing an Approach

### Decision Matrix

| Scenario | Recommendation |
|----------|----------------|
| Write-heavy, occasional reads | Local index |
| Read-heavy, rare writes | Global index |
| Full-text search | External search system |
| Analytics queries | Local index + scatter-gather |
| Single-value lookups | Global index |
| Point queries with partition key | No secondary index needed |

### Questions to Ask

1. **What's the read/write ratio?**
   - High reads → global index
   - High writes → local index

2. **Is eventual consistency acceptable?**
   - Yes → async global index
   - No → sync global index or local

3. **Do queries include partition key?**
   - Yes → local index might suffice
   - No → global index or scatter-gather

4. **How selective is the index?**
   - Low cardinality → local index OK
   - High cardinality → global index better

---

## Key Takeaways

1. **Local indexes are write-friendly** - But require scatter-gather for reads
2. **Global indexes are read-friendly** - But complicate writes
3. **Async global indexes** - Fast writes, eventual consistency
4. **Sync global indexes** - Strong consistency, slower writes
5. **Covering indexes reduce lookups** - At cost of larger indexes
6. **Bloom filters optimize scatter** - Skip partitions that don't match
7. **Consider alternatives** - Denormalization, materialized views, external search
8. **Indexes are expensive** - Use sparingly, measure performance
