# Cache Strategies

## TL;DR

Caching reduces latency and database load by storing frequently accessed data in fast storage. Key strategies: Cache-aside (lazy loading), Read-through, Write-through, Write-behind, and Write-around. Choose based on read/write ratios, consistency requirements, and failure tolerance. Cache hit rate is the primary metric—optimize for your access patterns.

---

## Why Cache?

### The Latency Problem

```
Access times:
  L1 cache:         1 ns
  L2 cache:         4 ns
  RAM:              100 ns
  SSD:              100 μs
  Network (DC):     500 μs
  Disk:             10 ms

Database query: 1-100 ms (with network)
Cache hit: 1-10 ms (in-memory)

10-100x improvement possible
```

### Benefits

```
1. Reduced latency
   Cache hit: 1ms vs DB query: 50ms

2. Reduced database load
   1000 QPS → 100 QPS to DB (90% hit rate)

3. Cost reduction
   Cache memory cheaper than DB scaling

4. Improved availability
   Can serve from cache during DB issues
```

---

## Cache-Aside (Lazy Loading)

### How It Works

```
Read path:
┌────────┐     ┌───────┐     ┌──────────┐
│ Client │────►│ Cache │     │ Database │
└────────┘     └───┬───┘     └────┬─────┘
                   │              │
    1. Check cache │              │
    2. Miss? ──────┼──────────────►
    3. Fetch from DB              │
                   │◄─────────────┤
    4. Store in cache             │
                   │              │
    5. Return     ◄┘              │

Application manages cache explicitly
```

### Implementation

```python
def get_user(user_id):
    # Check cache first
    cached = cache.get(f"user:{user_id}")
    if cached:
        return cached
    
    # Cache miss - fetch from database
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # Store in cache for next time
    cache.set(f"user:{user_id}", user, ttl=3600)
    
    return user
```

### Pros & Cons

```
Pros:
  + Only requested data is cached
  + Cache failures don't break reads
  + Simple to implement
  + Works with any data store

Cons:
  - First request always misses (cold start)
  - Data can become stale
  - Application must handle cache logic
  - "Cache stampede" on cold cache
```

---

## Read-Through

### How It Works

```
┌────────┐     ┌─────────────────┐     ┌──────────┐
│ Client │────►│      Cache      │────►│ Database │
└────────┘     │ (handles fetch) │     └──────────┘
               └─────────────────┘

Cache transparently loads from DB on miss
Application only talks to cache
```

### Implementation

```python
class ReadThroughCache:
    def __init__(self, cache, db):
        self.cache = cache
        self.db = db
    
    def get(self, key):
        value = self.cache.get(key)
        if value is None:
            # Cache handles the fetch
            value = self.db.query_by_key(key)
            self.cache.set(key, value)
        return value

# Application code is simpler
user = cache.get(f"user:{user_id}")
```

### Pros & Cons

```
Pros:
  + Simpler application code
  + Cache logic centralized
  + Consistent caching behavior

Cons:
  - Cache must understand data schema
  - Harder to customize per-query
  - Cache failure = read failure
```

---

## Write-Through

### How It Works

```
Write path:
┌────────┐     ┌───────┐     ┌──────────┐
│ Client │────►│ Cache │────►│ Database │
└────────┘     └───────┘     └──────────┘

1. Write to cache
2. Cache synchronously writes to DB
3. Return success after both complete
```

### Implementation

```python
class WriteThroughCache:
    def set(self, key, value):
        # Write to database first (must succeed)
        self.db.write(key, value)
        
        # Then update cache
        self.cache.set(key, value)
        
        return True

# Every write goes to both
cache.set(f"user:{user_id}", user_data)
```

### Pros & Cons

```
Pros:
  + Cache always consistent with DB
  + No stale reads
  + Simple mental model

Cons:
  - Write latency increased (cache + DB)
  - Every write hits cache (may cache unused data)
  - Cache failure blocks writes
```

---

## Write-Behind (Write-Back)

### How It Works

```
┌────────┐     ┌───────┐            ┌──────────┐
│ Client │────►│ Cache │───async───►│ Database │
└────────┘     └───────┘            └──────────┘

1. Write to cache immediately
2. Return success
3. Asynchronously persist to DB (batched)
```

### Implementation

```python
class WriteBehindCache:
    def __init__(self):
        self.dirty_keys = set()
        self.flush_interval = 1000  # ms
        
    def set(self, key, value):
        self.cache.set(key, value)
        self.dirty_keys.add(key)
        # Return immediately
    
    async def flush_loop(self):
        while True:
            await sleep(self.flush_interval)
            if self.dirty_keys:
                batch = list(self.dirty_keys)
                self.dirty_keys.clear()
                # Batch write to DB
                for key in batch:
                    self.db.write(key, self.cache.get(key))
```

### Pros & Cons

```
Pros:
  + Very fast writes (only cache)
  + Batch DB writes (efficient)
  + Absorbs write spikes

Cons:
  - Data loss risk (cache crash before flush)
  - Complex failure handling
  - Inconsistency window
  - Hard to debug
```

---

## Write-Around

### How It Works

```
Write path:
┌────────┐     ┌───────┐
│ Client │  ─┐ │ Cache │
└────────┘   │ └───────┘
             │
             └─────────►┌──────────┐
                        │ Database │
                        └──────────┘

Writes go directly to DB, skip cache
Cache populated only on read
```

### Implementation

```python
def write_user(user_id, data):
    # Write directly to database
    db.write(f"user:{user_id}", data)
    # Optionally invalidate cache
    cache.delete(f"user:{user_id}")

def read_user(user_id):
    # Cache-aside for reads
    cached = cache.get(f"user:{user_id}")
    if cached:
        return cached
    user = db.query(user_id)
    cache.set(f"user:{user_id}", user)
    return user
```

### Pros & Cons

```
Pros:
  + Avoids caching infrequently read data
  + No cache pollution from writes
  + Simple write path

Cons:
  - Recent writes not in cache
  - Higher read latency after writes
  - Need cache invalidation strategy
```

---

## Eviction Policies

### LRU (Least Recently Used)

```
Access pattern: A, B, C, D, A, E (capacity: 4)

[A]           → A accessed
[A, B]        → B accessed  
[A, B, C]     → C accessed
[A, B, C, D]  → D accessed (full)
[B, C, D, A]  → A accessed (move to end)
[C, D, A, E]  → E accessed, B evicted (least recent)
```

### LFU (Least Frequently Used)

```
Track access count per item
Evict item with lowest count

Better for skewed access patterns
More memory overhead (counters)
```

### TTL (Time To Live)

```
Each entry has expiration time
Evict when expired

set("key", value, ttl=3600)  # Expires in 1 hour

Good for:
  - Data that changes periodically
  - Bounding staleness
```

### Random Eviction

```
Randomly select items to evict
Surprisingly effective for uniform access
Very simple to implement
Redis uses approximated LRU (random sampling)
```

---

## Cache Sizing

### Hit Rate Formula

```
Hit rate = Hits / (Hits + Misses)

Working set: Frequently accessed data
If cache > working set → high hit rate

Diminishing returns:
  10% cache: 80% hit rate
  20% cache: 90% hit rate
  50% cache: 95% hit rate
```

### Memory Calculation

```
Per-item overhead:
  Key: ~50 bytes avg
  Value: varies
  Metadata: ~50 bytes (pointers, TTL, etc.)
  
Example:
  1 million items
  100 bytes avg value
  Total: 1M × (50 + 100 + 50) = 200 MB
```

### Monitoring

```
Key metrics:
  - Hit rate (target: >90%)
  - Eviction rate
  - Memory usage
  - Latency percentiles
  
Alert on:
  - Hit rate drop
  - Memory pressure
  - High eviction rate
```

---

## Comparison

| Strategy | Read Latency | Write Latency | Consistency | Complexity |
|----------|--------------|---------------|-------------|------------|
| Cache-aside | Low (hit) | N/A | Eventual | Low |
| Read-through | Low | N/A | Eventual | Medium |
| Write-through | Low | High | Strong | Medium |
| Write-behind | Low | Very Low | Weak | High |
| Write-around | Medium | Low | Eventual | Low |

---

## Choosing a Strategy

### Decision Tree

```
Is write latency critical?
  Yes → Write-behind (if data loss acceptable)
      → Write-around (if not)
  No  → Write-through (if consistency critical)
      → Cache-aside (otherwise)

Is read latency critical?
  Yes → Any caching helps
  No  → May not need cache

Is consistency critical?
  Yes → Write-through or no cache
  No  → Cache-aside or write-behind
```

### Common Patterns

```
User profiles: Cache-aside + TTL
  - Read-heavy
  - Staleness OK for seconds
  
Session data: Write-through
  - Consistency important
  - Lost sessions = bad UX
  
Analytics: Write-behind
  - High write volume
  - Batch aggregation OK
  
Inventory: Write-around + invalidation
  - Writes change rarely-read data
  - Fresh data on read
```

---

## Key Takeaways

1. **Cache-aside is most common** - Simple, resilient, flexible
2. **Write-through for consistency** - At cost of write latency
3. **Write-behind for write speed** - Accept data loss risk
4. **Hit rate is king** - Optimize for your access patterns
5. **Size for working set** - Not total dataset
6. **TTL provides freshness bounds** - Eventual consistency guarantee
7. **Monitor everything** - Hit rate, latency, evictions
8. **Failure modes matter** - Cache down shouldn't mean app down
