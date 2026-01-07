# Distributed Caching

## TL;DR

Distributed caching spreads cache data across multiple nodes for scale and availability. Key challenges: partitioning data, maintaining consistency, handling node failures. Redis Cluster and Memcached are popular choices. Use consistent hashing to minimize rebalancing. Design for partial availability—cache failures shouldn't crash your application.

---

## Why Distributed Cache?

### Single Node Limits

```
Single Redis instance:
  Memory: ~100 GB practical limit
  Throughput: ~100K ops/sec
  Availability: Single point of failure

When you need:
  - More memory (TB of cached data)
  - More throughput (millions ops/sec)
  - High availability (no single point of failure)

→ Distributed cache
```

### Scaling Options

```
Vertical: Bigger machine
  - Simple
  - Has limits
  - Still single point of failure

Horizontal: More machines
  - Partition data across nodes
  - Replicate for availability
  - More complex
  - Unlimited scale
```

---

## Partitioning

### Hash-Based Partitioning

```python
# Simple hash mod
node_id = hash(key) % num_nodes

key = "user:123"
hash("user:123") = 7429
7429 % 3 = 1
→ Node 1

Problem: Adding/removing node reshuffles most keys
```

### Consistent Hashing

```
Hash ring:
     0°
     │
  ┌──┼──┐
  │     │ Node A (45°)
270°   90°
  │     │ Node B (180°)
  └──┼──┘ Node C (270°)
    180°

Key hash = 100° → next node clockwise = Node B

Add Node D at 135°:
  Only keys between 90° and 135° move to D
  Other nodes unaffected
```

### Virtual Nodes

```
Without vnodes:
  Node A: 1 position on ring
  Uneven distribution likely

With vnodes (e.g., 150 per node):
  Node A: 150 positions on ring
  Better distribution
  Smoother rebalancing
  
  Also handles heterogeneous hardware:
    Powerful node: 200 vnodes
    Smaller node: 100 vnodes
```

---

## Replication

### Primary-Replica

```
Write → Primary → Replicas (async or sync)
Read  → Primary or any Replica

┌─────────┐
│ Primary │◄───writes
└────┬────┘
     │ replication
┌────┴────┬─────────┐
▼         ▼         ▼
[Replica] [Replica] [Replica]
    ▲         ▲         ▲
    └─────────┴─────────┴─── reads
```

### Replication Trade-offs

```
Synchronous:
  + No data loss on primary failure
  - Higher write latency
  - Replica failure blocks writes

Asynchronous:
  + Fast writes
  - Data loss window
  - Stale reads possible

Common: 1 sync replica + N async replicas
```

---

## Redis Cluster

### Architecture

```
16384 hash slots distributed across masters

Master 1: slots 0-5460
Master 2: slots 5461-10922
Master 3: slots 10923-16383

Each master has replicas for failover

┌────────────────────────────────────────────┐
│ Slot 0-5460    │ Slot 5461-10922 │ Slot 10923-16383 │
│ [Master 1]     │ [Master 2]      │ [Master 3]       │
│     ↓          │     ↓           │     ↓            │
│ [Replica 1a]   │ [Replica 2a]    │ [Replica 3a]     │
└────────────────────────────────────────────┘
```

### Slot Assignment

```python
def key_slot(key):
    # If key contains {}, hash only that part
    # Allows co-locating related keys
    if "{" in key and "}" in key:
        hash_part = key[key.index("{")+1:key.index("}")]
    else:
        hash_part = key
    
    return crc16(hash_part) % 16384

# Examples:
key_slot("user:123")         # Based on "user:123"
key_slot("{user:123}:profile") # Based on "user:123"
key_slot("{user:123}:orders")  # Same slot as above
```

### Failover

```
1. Replica detects master failure (no heartbeat)
2. Replica promotes itself to master
3. Cluster updates routing
4. Old master rejoins as replica (if recovers)

Automatic failover: No manual intervention
Typical failover time: 1-2 seconds
```

### Client Configuration

```python
from redis.cluster import RedisCluster

rc = RedisCluster(
    host="redis-cluster.example.com",
    port=7000,
    # Client maintains slot mapping
    # Automatically routes to correct node
)

rc.set("user:123", "Alice")  # Routes to correct slot
```

---

## Memcached

### Architecture

```
No replication (by design)
Clients partition data using consistent hashing

Client ─────► [Memcached 1]
       └────► [Memcached 2]
       └────► [Memcached 3]

Client is responsible for:
  - Deciding which node to query
  - Handling node failures
```

### Client-Side Sharding

```python
import pylibmc

servers = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
client = pylibmc.Client(
    servers,
    behaviors={
        "ketama": True,  # Consistent hashing
        "dead_timeout": 60,  # Mark dead for 60s
    }
)

client.set("user:123", "Alice")
# Client hashes key, picks server
```

### Comparison: Redis Cluster vs Memcached

| Aspect | Redis Cluster | Memcached |
|--------|--------------|-----------|
| Replication | Built-in | None |
| Data types | Rich (lists, sets, etc.) | String only |
| Persistence | Optional | None |
| Sharding | Server-side | Client-side |
| Failover | Automatic | Manual/client |
| Memory efficiency | Lower | Higher |

---

## Consistency Challenges

### Read-After-Write

```
Client writes to Node A (primary)
Client reads from Node B (replica)
Replica hasn't received update yet

Solutions:
  - Read from primary after write
  - Read-your-writes guarantee (sticky sessions)
  - Wait for replication before ack
```

### Split-Brain

```
Network partition:
  Partition 1: Master A, Replica B
  Partition 2: Replica C, Replica D

C or D might be promoted to master
Two masters accepting writes

Prevention:
  - Require quorum for writes
  - Fencing tokens
  - Redis: min-replicas-to-write
```

### Cache Coherence

```
Multiple app servers, each with local + distributed cache

App Server 1: Local cache: user:123 = v1
App Server 2: Local cache: user:123 = v1
Distributed:  Redis: user:123 = v2

Local caches are stale!

Solutions:
  - No local cache (always distributed)
  - Short TTL on local cache
  - Publish invalidation events
```

---

## Handling Node Failures

### Graceful Degradation

```python
def get_with_fallback(key):
    try:
        value = distributed_cache.get(key)
        if value:
            return value
    except CacheConnectionError:
        log.warn("Cache unavailable, falling back to DB")
    
    # Fallback to database
    return database.get(key)
```

### Rehashing on Node Removal

```
With consistent hashing:
  Node B removed
  Only keys that were on B need to move
  ~1/N of keys affected

Without consistent hashing:
  Almost all keys rehash to different nodes
  Cache becomes effectively empty
```

### Hot Standby

```
For critical caches:
  Active: Redis Cluster (3 masters, 3 replicas)
  Standby: Cold replica in another DC
  
  On cluster failure:
    Promote standby
    Redirect traffic
```

---

## Performance Optimization

### Connection Pooling

```python
# Bad: New connection per request
def get_user(user_id):
    conn = redis.Redis()  # New connection
    return conn.get(f"user:{user_id}")

# Good: Reuse connections
pool = redis.ConnectionPool(max_connections=50)

def get_user(user_id):
    conn = redis.Redis(connection_pool=pool)
    return conn.get(f"user:{user_id}")
```

### Pipelining

```python
# Bad: Round-trip per command
for id in user_ids:
    users.append(redis.get(f"user:{id}"))  # 100 round-trips

# Good: Batch commands
pipe = redis.pipeline()
for id in user_ids:
    pipe.get(f"user:{id}")
users = pipe.execute()  # 1 round-trip
```

### Local Caching

```
Two-tier:
  L1: Local in-memory (per-process)
  L2: Distributed cache (Redis)

Read path:
  1. Check L1 (microseconds)
  2. Check L2 (milliseconds)
  3. Check database (tens of milliseconds)

Write path:
  1. Update database
  2. Invalidate L2
  3. Broadcast invalidation to L1s
```

---

## Monitoring

### Key Metrics

```
Hit rate:
  hits / (hits + misses)
  Target: >90%

Latency:
  p50, p95, p99
  Watch for outliers

Memory usage:
  Used vs max
  Eviction rate

Connections:
  Current vs max
  Connection errors

Replication lag:
  Seconds behind master
```

### Alerting

```yaml
alerts:
  - name: CacheHitRateLow
    condition: hit_rate < 80%
    for: 5m
    
  - name: CacheLatencyHigh
    condition: p99_latency > 100ms
    for: 1m
    
  - name: CacheMemoryHigh
    condition: memory_usage > 90%
    for: 5m
    
  - name: ReplicationLag
    condition: lag_seconds > 10
    for: 1m
```

---

## Common Patterns

### Caching with Fallback

```python
def get_user(user_id):
    # Try cache
    user = cache.get(f"user:{user_id}")
    if user:
        return deserialize(user)
    
    # Cache miss or error
    user = db.get_user(user_id)
    
    # Populate cache (best effort)
    try:
        cache.set(f"user:{user_id}", serialize(user), ex=3600)
    except:
        pass  # Don't fail the request
    
    return user
```

### Circuit Breaker for Cache

```python
class CacheCircuitBreaker:
    def __init__(self, threshold=5, reset_time=60):
        self.failures = 0
        self.threshold = threshold
        self.reset_time = reset_time
        self.last_failure = 0
        
    def call(self, func):
        if self.is_open():
            raise CacheBypassException()
        
        try:
            result = func()
            self.failures = 0
            return result
        except:
            self.failures += 1
            self.last_failure = time.time()
            raise
    
    def is_open(self):
        if self.failures >= self.threshold:
            if time.time() - self.last_failure < self.reset_time:
                return True
            self.failures = 0
        return False
```

---

## Key Takeaways

1. **Consistent hashing minimizes reshuffling** - Use for node additions/removals
2. **Redis Cluster for rich features** - Replication, data types, persistence
3. **Memcached for simplicity** - Pure cache, high memory efficiency
4. **Plan for node failures** - Graceful degradation to database
5. **Connection pooling is essential** - Don't create connections per request
6. **Pipeline for batches** - Reduce round-trips dramatically
7. **Monitor hit rate and latency** - Primary health indicators
8. **Cache is not critical path** - Failures should never crash the app
