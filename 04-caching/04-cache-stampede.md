# Cache Stampede

## TL;DR

A cache stampede (thundering herd) occurs when many requests simultaneously try to repopulate an expired cache entry, overwhelming the database. Solutions include locking (only one request regenerates), probabilistic early expiration, and request coalescing. Prevention is critical for popular, expensive-to-compute cache entries.

---

## The Problem

### Normal Operation

```
Time 0: 1000 requests/sec, cache hit
  Cache: [user:123] → data (TTL: 5min remaining)
  Database: idle
  
All requests served from cache ✓
```

### Stampede Scenario

```
Time T: Cache entry expires
  Cache: [user:123] → MISS
  
  Request 1: Cache miss → Query DB
  Request 2: Cache miss → Query DB
  Request 3: Cache miss → Query DB
  ...
  Request 1000: Cache miss → Query DB

  Database: 1000 simultaneous queries!
  Response time: 10x slower or timeout
  Possible cascade failure
```

### Visualizing the Problem

```
Requests over time:
        
        │ Cache expires
        │     ↓
────────┼─────╱╲───────────
        │    ╱  ╲
        │   ╱    ╲
        │  ╱      ╲─────────
        │ ╱
────────┴─────────────────────
             │
        Stampede window
```

---

## Solution 1: Locking

### External Lock

```python
def get_with_lock(key):
    value = cache.get(key)
    if value:
        return value
    
    # Try to acquire lock
    lock_key = f"lock:{key}"
    if cache.set(lock_key, "1", nx=True, ex=30):
        try:
            # Won the lock - fetch from DB
            value = database.get(key)
            cache.set(key, value, ex=3600)
            return value
        finally:
            cache.delete(lock_key)
    else:
        # Another process is refreshing
        # Wait and retry
        sleep(0.1)
        return get_with_lock(key)  # Retry
```

### Problems with Simple Locking

```
1. Retry storms
   1000 requests waiting, all retry simultaneously

2. Lock expiration
   Lock expires before DB query completes
   
3. Deadlock potential
   Lock holder crashes without releasing
```

### Better Locking: Wait and Return Stale

```python
def get_with_stale_fallback(key):
    value, ttl = cache.get_with_ttl(key)
    
    if value and ttl > 0:
        return value  # Fresh data
    
    lock_key = f"lock:{key}"
    if cache.set(lock_key, "1", nx=True, ex=30):
        try:
            # Refresh in background
            value = database.get(key)
            cache.set(key, value, ex=3600)
        finally:
            cache.delete(lock_key)
    
    if value:
        return value  # Return stale data while refreshing
    
    # No stale data - must wait
    sleep(0.1)
    return cache.get(key)
```

---

## Solution 2: Probabilistic Early Expiration

### Concept

Randomly refresh before expiration, spreading the load.

```python
import random
import math

def should_recompute(key, ttl_remaining, beta=1):
    """
    XFetch algorithm:
    Probability of recompute increases as TTL decreases
    """
    if ttl_remaining <= 0:
        return True
    
    # Probability increases exponentially as TTL approaches 0
    expiry_gap = beta * math.log(random.random())
    return -expiry_gap >= ttl_remaining
```

### Implementation

```python
def get_with_probabilistic_refresh(key, compute_func, ttl=3600, beta=60):
    value, remaining_ttl = cache.get_with_ttl(key)
    
    if value is None or should_recompute(key, remaining_ttl, beta):
        # Recompute (either expired or probabilistically chosen)
        value = compute_func()
        cache.set(key, value, ex=ttl)
    
    return value
```

### Visualization

```
TTL timeline:
  |────────────────────────────────|
  Full TTL                         Expiry
  
Refresh probability:
  |░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓██|
  Low                    Medium   High
  
Some requests refresh early, spreading the load
```

---

## Solution 3: Request Coalescing

### Concept

Combine duplicate requests into one database query.

```
Before coalescing:
  Request 1 ─────► DB Query
  Request 2 ─────► DB Query
  Request 3 ─────► DB Query

After coalescing:
  Request 1 ─────┐
  Request 2 ─────┼─► DB Query ─► All requests
  Request 3 ─────┘
```

### Implementation

```python
import threading
from concurrent.futures import Future

class RequestCoalescer:
    def __init__(self):
        self.pending = {}  # key → Future
        self.lock = threading.Lock()
    
    def get(self, key, fetch_func):
        with self.lock:
            if key in self.pending:
                # Another request is fetching - wait for it
                return self.pending[key].result()
            
            # First request - create future
            future = Future()
            self.pending[key] = future
        
        try:
            # Fetch the data
            value = fetch_func(key)
            future.set_result(value)
            return value
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            with self.lock:
                del self.pending[key]

# Usage
coalescer = RequestCoalescer()

def get_user(user_id):
    cached = cache.get(f"user:{user_id}")
    if cached:
        return cached
    
    # Coalesce concurrent requests
    user = coalescer.get(
        f"user:{user_id}",
        lambda k: database.get_user(user_id)
    )
    
    cache.set(f"user:{user_id}", user)
    return user
```

### Go: singleflight

```go
import "golang.org/x/sync/singleflight"

var group singleflight.Group

func GetUser(userID string) (*User, error) {
    key := fmt.Sprintf("user:%s", userID)
    
    // Check cache
    if cached, ok := cache.Get(key); ok {
        return cached.(*User), nil
    }
    
    // Coalesce concurrent requests
    result, err, _ := group.Do(key, func() (interface{}, error) {
        user, err := database.GetUser(userID)
        if err != nil {
            return nil, err
        }
        cache.Set(key, user, time.Hour)
        return user, nil
    })
    
    if err != nil {
        return nil, err
    }
    return result.(*User), nil
}
```

---

## Solution 4: Background Refresh

### Concept

Never let cache expire. Refresh in background before expiration.

```python
import threading

def get_with_background_refresh(key, ttl=3600, refresh_threshold=300):
    value, remaining_ttl = cache.get_with_ttl(key)
    
    if value and remaining_ttl > refresh_threshold:
        return value  # Fresh enough
    
    if value and remaining_ttl > 0:
        # Getting stale - trigger background refresh
        trigger_background_refresh(key)
        return value  # Return current value
    
    # Expired or missing - must fetch synchronously
    value = database.get(key)
    cache.set(key, value, ex=ttl)
    return value

def trigger_background_refresh(key):
    # Non-blocking refresh
    def refresh():
        value = database.get(key)
        cache.set(key, value, ex=3600)
    
    thread = threading.Thread(target=refresh)
    thread.start()
```

### With Refresh Tokens

```python
def set_with_refresh(key, value, ttl=3600, refresh_at=300):
    """Store value with metadata for refresh"""
    data = {
        "value": value,
        "refresh_at": time.time() + ttl - refresh_at
    }
    cache.set(key, json.dumps(data), ex=ttl)

def get_with_refresh(key, fetch_func):
    raw = cache.get(key)
    if not raw:
        return fetch_sync(key, fetch_func)
    
    data = json.loads(raw)
    
    if time.time() > data["refresh_at"]:
        # Time to refresh in background
        trigger_async_refresh(key, fetch_func)
    
    return data["value"]
```

---

## Solution 5: Lease-Based Approach

### Concept

First requester gets a "lease" to refresh. Others wait or get stale data.

```python
def get_with_lease(key, stale_ttl=60):
    # Try to get current value
    value = cache.get(key)
    if value:
        return value
    
    # Try to acquire lease
    lease_key = f"lease:{key}"
    lease_acquired = cache.set(lease_key, "1", nx=True, ex=30)
    
    if lease_acquired:
        # We have the lease - refresh
        value = database.get(key)
        cache.set(key, value, ex=3600)
        cache.delete(lease_key)
        return value
    else:
        # Someone else is refreshing
        # Check for stale value
        stale_value = cache.get(f"stale:{key}")
        if stale_value:
            return stale_value
        
        # No stale value - wait briefly
        sleep(0.1)
        return cache.get(key)

# Keep stale copy for fallback
def set_with_stale(key, value, ttl=3600, stale_ttl=86400):
    cache.set(key, value, ex=ttl)
    cache.set(f"stale:{key}", value, ex=stale_ttl)
```

---

## Prevention Strategies

### Pre-warming

```python
def warm_cache_on_startup():
    """Pre-populate cache before taking traffic"""
    popular_keys = get_popular_keys()
    for key in popular_keys:
        value = database.get(key)
        cache.set(key, value, ex=3600)
    
    log.info(f"Warmed {len(popular_keys)} cache entries")
```

### Staggered TTLs

```python
import random

def set_with_jitter(key, value, base_ttl=3600, jitter_percent=10):
    """Add random jitter to prevent synchronized expiration"""
    jitter = base_ttl * jitter_percent / 100
    actual_ttl = base_ttl + random.uniform(-jitter, jitter)
    cache.set(key, value, ex=int(actual_ttl))

# Instead of all keys expiring at exactly 1 hour:
# Keys expire between 54 and 66 minutes
```

### Never-Expire with Async Refresh

```python
def setup_cache_refresh_worker():
    """Background worker that refreshes cache entries"""
    while True:
        # Get keys approaching expiration
        keys = get_keys_expiring_soon(threshold=300)
        
        for key in keys:
            try:
                value = database.get(key)
                cache.set(key, value, ex=3600)
            except Exception as e:
                log.error(f"Failed to refresh {key}: {e}")
        
        sleep(60)
```

---

## Comparison

| Solution | Complexity | Latency | DB Load | Stale Data |
|----------|------------|---------|---------|------------|
| Locking | Medium | Higher | Lowest | Possible |
| Probabilistic | Low | Normal | Low | No |
| Coalescing | Medium | Normal | Low | No |
| Background refresh | High | Lowest | Medium | Yes |
| Lease + stale | High | Low | Low | Yes |

---

## Key Takeaways

1. **Stampedes kill databases** - A single popular key can cause outage
2. **Locking prevents redundant queries** - But adds latency
3. **Probabilistic refresh spreads load** - Simple and effective
4. **Coalescing combines requests** - Best for concurrent requests
5. **Background refresh keeps cache warm** - Never truly expire
6. **Stale data is often acceptable** - Serve old data while refreshing
7. **Stagger TTLs** - Prevent synchronized expiration
8. **Pre-warm critical keys** - Don't start cold
