# Rate Limiting

## TL;DR

Rate limiting controls the number of requests a client can make within a time window, protecting services from abuse, ensuring fair usage, and preventing resource exhaustion. Common algorithms include token bucket, leaky bucket, fixed window, and sliding window. Implementation can be done at the API gateway, application layer, or using distributed stores like Redis.

---

## Why Rate Limiting?

Without rate limiting:

```
                    ┌─────────────────────────────────┐
                    │          API Server             │
                    │                                 │
   Legitimate ──────┤  CPU: 100%                      │
   Users            │  Memory: 95%                    │
                    │  Connections: EXHAUSTED         │
   Abusive   ───────┤                                 │
   Client           │  Status: OVERWHELMED            │
   (10000 req/s)    │                                 │
                    └─────────────────────────────────┘
                              │
                              ▼
                    All users experience failures
```

With rate limiting:

```
                    ┌─────────────────────────────────┐
                    │         Rate Limiter            │
                    └─────────────┬───────────────────┘
                                  │
   Legitimate ────────────────────┼────► Allowed (100 req/s each)
   Users                          │
                                  │
   Abusive   ─────────────────────┴────► Rejected (429 Too Many Requests)
   Client                               (exceeds 100 req/s limit)
   (10000 req/s)
```

---

## Rate Limiting Algorithms

### 1. Token Bucket

```
Token Bucket Visualization:

    ┌─────────────────────────────────────┐
    │              BUCKET                  │
    │  ┌─────────────────────────────────┐│
    │  │ 🪙 🪙 🪙 🪙 🪙 🪙 🪙 🪙          ││  Capacity: 10 tokens
    │  │       (8 tokens)                ││
    │  └─────────────────────────────────┘│
    │                 ▲                    │
    │                 │                    │
    │    Refill: 2 tokens/second          │
    └─────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────┐
    │           REQUEST                    │
    │   Takes 1 token (if available)      │
    │   Rejected if no tokens             │
    └─────────────────────────────────────┘
```

```python
import time
from threading import Lock
from dataclasses import dataclass

@dataclass
class TokenBucketConfig:
    capacity: int      # Maximum tokens in bucket
    refill_rate: float # Tokens added per second

class TokenBucket:
    def __init__(self, config: TokenBucketConfig):
        self.capacity = config.capacity
        self.refill_rate = config.refill_rate
        self.tokens = config.capacity
        self.last_refill = time.time()
        self.lock = Lock()
    
    def _refill(self):
        """Add tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now
    
    def allow(self, tokens: int = 1) -> bool:
        """Check if request is allowed"""
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """Calculate wait time until tokens available"""
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                return 0
            
            needed = tokens - self.tokens
            return needed / self.refill_rate

# Usage
limiter = TokenBucket(TokenBucketConfig(
    capacity=100,      # Burst up to 100 requests
    refill_rate=10     # 10 requests per second sustained
))

if limiter.allow():
    process_request()
else:
    raise RateLimitExceeded(retry_after=limiter.wait_time())
```

### 2. Leaky Bucket

```
Leaky Bucket Visualization:

         Incoming Requests
              │ │ │
              ▼ ▼ ▼
    ┌─────────────────────────────────────┐
    │              BUCKET                  │
    │  ┌─────────────────────────────────┐│
    │  │ ● ● ● ● ●                       ││  Queue: 5 requests
    │  │ ● ● ● ●                         ││  Capacity: 10
    │  └─────────────────────────────────┘│
    └────────────────┬────────────────────┘
                     │
                     ▼ Constant leak rate
              (process 2 req/sec)
                     │
                     ▼
              ┌──────────┐
              │ Process  │
              └──────────┘

Overflow → Request rejected (queue full)
```

```python
from collections import deque
import time
import threading

class LeakyBucket:
    def __init__(self, capacity: int, leak_rate: float):
        self.capacity = capacity
        self.leak_rate = leak_rate  # requests per second
        self.queue = deque()
        self.lock = threading.Lock()
        self.last_leak = time.time()
    
    def _leak(self):
        """Process queued requests at constant rate"""
        now = time.time()
        elapsed = now - self.last_leak
        leaked = int(elapsed * self.leak_rate)
        
        for _ in range(min(leaked, len(self.queue))):
            self.queue.popleft()
        
        self.last_leak = now
    
    def allow(self) -> bool:
        """Add request to queue if space available"""
        with self.lock:
            self._leak()
            
            if len(self.queue) < self.capacity:
                self.queue.append(time.time())
                return True
            return False
    
    def queue_position(self) -> int:
        """Get current position in queue"""
        with self.lock:
            self._leak()
            return len(self.queue)

# Leaky bucket smooths out traffic
# Even if 100 requests arrive at once,
# they're processed at constant rate (e.g., 10/sec)
```

### 3. Fixed Window

```
Fixed Window Visualization:

Window 1 (12:00:00 - 12:01:00)    Window 2 (12:01:00 - 12:02:00)
┌─────────────────────────────┐  ┌─────────────────────────────┐
│  ████████████████░░░░░░░░░░ │  │  ████░░░░░░░░░░░░░░░░░░░░░░ │
│  80 requests (limit: 100)   │  │  20 requests                │
└─────────────────────────────┘  └─────────────────────────────┘

Problem: Boundary burst
       12:00:30              12:01:30
          │                     │
Window 1: │█████████████████████│
          │   80 requests       │ 
                                │
Window 2:                       │█████████████████████
                                │   80 requests

Within 1 minute (12:00:30 - 12:01:30): 160 requests! (exceeds 100 limit)
```

```python
import time
from collections import defaultdict

class FixedWindowLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.counters = defaultdict(int)
        self.windows = {}
    
    def _get_window(self) -> int:
        """Get current window identifier"""
        return int(time.time() // self.window_seconds)
    
    def allow(self, key: str) -> bool:
        """Check if request allowed for given key"""
        window = self._get_window()
        window_key = f"{key}:{window}"
        
        # Reset counter if new window
        if self.windows.get(key) != window:
            self.counters[key] = 0
            self.windows[key] = window
        
        if self.counters[key] < self.limit:
            self.counters[key] += 1
            return True
        return False
    
    def remaining(self, key: str) -> int:
        """Get remaining requests in current window"""
        window = self._get_window()
        if self.windows.get(key) != window:
            return self.limit
        return max(0, self.limit - self.counters[key])
    
    def reset_time(self) -> int:
        """Seconds until window resets"""
        return self.window_seconds - (int(time.time()) % self.window_seconds)

# Simple but has burst issue at window boundaries
limiter = FixedWindowLimiter(limit=100, window_seconds=60)
```

### 4. Sliding Window Log

```
Sliding Window Log:

Current Time: 12:01:30
Window: Last 60 seconds (12:00:30 - 12:01:30)

Request Log:
┌──────────────────────────────────────────────────────────────┐
│  12:00:25  ✗ (outside window)                                │
│  12:00:35  ✓ (inside window)  ────┐                          │
│  12:00:45  ✓ (inside window)      │                          │
│  12:01:00  ✓ (inside window)      │  Count these            │
│  12:01:15  ✓ (inside window)      │                          │
│  12:01:28  ✓ (inside window)  ────┘                          │
└──────────────────────────────────────────────────────────────┘
                                     
Count in window: 5
Limit: 100
→ Allow request
```

```python
import time
from collections import defaultdict
import bisect

class SlidingWindowLogLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.logs = defaultdict(list)  # key -> sorted list of timestamps
    
    def _cleanup(self, key: str, now: float):
        """Remove timestamps outside window"""
        cutoff = now - self.window_seconds
        logs = self.logs[key]
        
        # Find first timestamp in window
        idx = bisect.bisect_left(logs, cutoff)
        self.logs[key] = logs[idx:]
    
    def allow(self, key: str) -> bool:
        now = time.time()
        self._cleanup(key, now)
        
        if len(self.logs[key]) < self.limit:
            bisect.insort(self.logs[key], now)
            return True
        return False
    
    def get_count(self, key: str) -> int:
        now = time.time()
        self._cleanup(key, now)
        return len(self.logs[key])

# Accurate but memory-intensive (stores every timestamp)
# O(n) space where n = requests in window
```

### 5. Sliding Window Counter

```
Sliding Window Counter:

Current Time: 12:01:30 (30 seconds into window 2)

Window 1 (12:00:00 - 12:01:00): 70 requests
Window 2 (12:01:00 - 12:02:00): 20 requests (so far)

Weighted count = 
    (Window 1 count × overlap %) + Window 2 count
    = 70 × 50% + 20
    = 35 + 20
    = 55 requests

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Window 1        │         Window 2                         │
│  ████████████████│█████████░░░░░░░░░░░░░░░░░░░░░           │
│        70        │   20                                     │
│           │◄────────────────────────────────►│              │
│           │     Sliding Window (60 sec)      │              │
│           │                                  │              │
│           12:00:30                    12:01:30              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
import time
from dataclasses import dataclass

@dataclass
class WindowData:
    count: int
    start_time: float

class SlidingWindowCounterLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.windows = {}  # key -> {current: WindowData, previous: WindowData}
    
    def _get_window_start(self, now: float) -> float:
        return (now // self.window_seconds) * self.window_seconds
    
    def allow(self, key: str) -> bool:
        now = time.time()
        window_start = self._get_window_start(now)
        
        if key not in self.windows:
            self.windows[key] = {
                'current': WindowData(0, window_start),
                'previous': WindowData(0, window_start - self.window_seconds)
            }
        
        data = self.windows[key]
        
        # Check if we need to slide windows
        if data['current'].start_time < window_start:
            data['previous'] = data['current']
            data['current'] = WindowData(0, window_start)
        
        # Calculate weighted count
        elapsed_ratio = (now - window_start) / self.window_seconds
        previous_weight = 1 - elapsed_ratio
        
        weighted_count = (
            data['previous'].count * previous_weight + 
            data['current'].count
        )
        
        if weighted_count < self.limit:
            data['current'].count += 1
            return True
        return False

# Best of both worlds: accurate + memory efficient
# O(1) space per key
```

---

## Distributed Rate Limiting with Redis

```python
import redis
import time
from typing import Tuple

class RedisRateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def token_bucket(self, key: str, capacity: int, 
                     refill_rate: float, tokens: int = 1) -> Tuple[bool, dict]:
        """
        Distributed token bucket using Redis
        """
        now = time.time()
        bucket_key = f"ratelimit:bucket:{key}"
        
        # Lua script for atomic operation
        lua_script = """
        local bucket_key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local tokens_requested = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        
        -- Get current state
        local bucket = redis.call('HMGET', bucket_key, 'tokens', 'last_refill')
        local current_tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now
        
        -- Calculate tokens to add
        local elapsed = now - last_refill
        local new_tokens = math.min(capacity, current_tokens + (elapsed * refill_rate))
        
        -- Check if we can consume tokens
        if new_tokens >= tokens_requested then
            new_tokens = new_tokens - tokens_requested
            redis.call('HMSET', bucket_key, 'tokens', new_tokens, 'last_refill', now)
            redis.call('EXPIRE', bucket_key, math.ceil(capacity / refill_rate) * 2)
            return {1, new_tokens, capacity}
        else
            redis.call('HMSET', bucket_key, 'tokens', new_tokens, 'last_refill', now)
            return {0, new_tokens, capacity}
        end
        """
        
        result = self.redis.eval(
            lua_script, 1, bucket_key,
            capacity, refill_rate, tokens, now
        )
        
        allowed, remaining, limit = result
        return bool(allowed), {
            'remaining': remaining,
            'limit': limit,
            'reset_after': (tokens - remaining) / refill_rate if not allowed else 0
        }
    
    def sliding_window(self, key: str, limit: int, 
                       window_seconds: int) -> Tuple[bool, dict]:
        """
        Distributed sliding window counter using Redis
        """
        now = time.time()
        window_start = int(now // window_seconds) * window_seconds
        current_key = f"ratelimit:window:{key}:{window_start}"
        previous_key = f"ratelimit:window:{key}:{window_start - window_seconds}"
        
        pipe = self.redis.pipeline()
        pipe.get(current_key)
        pipe.get(previous_key)
        results = pipe.execute()
        
        current_count = int(results[0] or 0)
        previous_count = int(results[1] or 0)
        
        # Calculate weighted count
        elapsed_ratio = (now - window_start) / window_seconds
        previous_weight = 1 - elapsed_ratio
        weighted_count = previous_count * previous_weight + current_count
        
        if weighted_count < limit:
            # Increment and set expiry
            pipe = self.redis.pipeline()
            pipe.incr(current_key)
            pipe.expire(current_key, window_seconds * 2)
            pipe.execute()
            
            return True, {
                'remaining': int(limit - weighted_count - 1),
                'limit': limit,
                'reset': window_start + window_seconds
            }
        
        return False, {
            'remaining': 0,
            'limit': limit,
            'reset': window_start + window_seconds
        }
```

---

## Rate Limit Response Headers

```python
from flask import Flask, request, jsonify, make_response
from functools import wraps

app = Flask(__name__)
limiter = RedisRateLimiter(redis.Redis())

def rate_limit(limit: int, window: int):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Use IP or API key as identifier
            key = request.headers.get('X-API-Key') or request.remote_addr
            
            allowed, info = limiter.sliding_window(
                key=f"{f.__name__}:{key}",
                limit=limit,
                window_seconds=window
            )
            
            # Set rate limit headers
            headers = {
                'X-RateLimit-Limit': str(limit),
                'X-RateLimit-Remaining': str(info['remaining']),
                'X-RateLimit-Reset': str(info['reset']),
            }
            
            if not allowed:
                response = make_response(
                    jsonify({
                        'error': 'Rate limit exceeded',
                        'retry_after': info['reset'] - int(time.time())
                    }),
                    429
                )
                headers['Retry-After'] = str(info['reset'] - int(time.time()))
                for header, value in headers.items():
                    response.headers[header] = value
                return response
            
            response = make_response(f(*args, **kwargs))
            for header, value in headers.items():
                response.headers[header] = value
            return response
        return wrapper
    return decorator

@app.route('/api/resource')
@rate_limit(limit=100, window=60)
def get_resource():
    return jsonify({'data': 'resource'})
```

```
HTTP Response Headers:

HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1609459200

HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1609459200
Retry-After: 45

{
  "error": "Rate limit exceeded",
  "retry_after": 45
}
```

---

## Multi-Tier Rate Limiting

```
Rate Limiting Tiers:

┌─────────────────────────────────────────────────────────────────────┐
│                          Global Tier                                 │
│                  100,000 requests/second total                       │
│                  (protects entire system)                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Per-API Tier                                │
│                                                                      │
│   /api/search: 10,000 req/s    /api/write: 1,000 req/s              │
│   (expensive operation)         (database writes)                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Per-User Tier                                │
│                                                                      │
│   Free: 100 req/min            Pro: 1000 req/min                    │
│   Enterprise: 10000 req/min                                          │
└─────────────────────────────────────────────────────────────────────┘
```

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Tier(Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

@dataclass
class RateLimitRule:
    name: str
    key_template: str  # e.g., "user:{user_id}" or "global"
    limit: int
    window_seconds: int

class TieredRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.limiter = RedisRateLimiter(redis_client)
        
        # Define tier limits
        self.tier_limits = {
            Tier.FREE: {'per_minute': 60, 'per_hour': 1000, 'per_day': 10000},
            Tier.PRO: {'per_minute': 600, 'per_hour': 10000, 'per_day': 100000},
            Tier.ENTERPRISE: {'per_minute': 6000, 'per_hour': 100000, 'per_day': 1000000},
        }
        
        # Endpoint-specific limits (override tier limits)
        self.endpoint_limits = {
            '/api/search': {'limit': 10, 'window': 1},  # 10/sec
            '/api/export': {'limit': 5, 'window': 60},   # 5/min
            '/api/batch': {'limit': 1, 'window': 10},    # 1/10sec
        }
    
    def check(self, user_id: str, tier: Tier, endpoint: str) -> tuple[bool, dict]:
        """Check all applicable rate limits"""
        
        results = []
        
        # 1. Global rate limit
        allowed, info = self.limiter.sliding_window(
            key="global",
            limit=100000,
            window_seconds=1
        )
        if not allowed:
            return False, {'reason': 'global_limit', **info}
        results.append(('global', info))
        
        # 2. Endpoint-specific limit
        if endpoint in self.endpoint_limits:
            config = self.endpoint_limits[endpoint]
            allowed, info = self.limiter.sliding_window(
                key=f"endpoint:{endpoint}:{user_id}",
                limit=config['limit'],
                window_seconds=config['window']
            )
            if not allowed:
                return False, {'reason': f'endpoint_limit:{endpoint}', **info}
            results.append(('endpoint', info))
        
        # 3. User tier limits (check multiple windows)
        limits = self.tier_limits[tier]
        
        for window_name, limit in limits.items():
            window_seconds = {
                'per_minute': 60,
                'per_hour': 3600,
                'per_day': 86400
            }[window_name]
            
            allowed, info = self.limiter.sliding_window(
                key=f"user:{user_id}:{window_name}",
                limit=limit,
                window_seconds=window_seconds
            )
            if not allowed:
                return False, {'reason': f'user_limit:{window_name}', **info}
            results.append((window_name, info))
        
        # All limits passed
        return True, {
            'limits_checked': len(results),
            'details': results
        }
```

---

## Rate Limiting Strategies

### By IP Address

```python
def get_client_ip(request) -> str:
    """Extract real client IP considering proxies"""
    # Check forwarded headers
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        # First IP is the original client
        return forwarded.split(',')[0].strip()
    
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip
    
    return request.remote_addr

# Problem: Shared IPs (NAT, proxies)
# Solution: Combine with other identifiers
```

### By API Key

```python
def rate_limit_by_api_key(request):
    api_key = request.headers.get('X-API-Key')
    
    if not api_key:
        # Anonymous requests get stricter limits
        return rate_limit_by_ip(request)
    
    # Get tier from API key
    key_info = get_api_key_info(api_key)
    tier = key_info.tier
    
    return check_rate_limit(
        key=f"api_key:{api_key}",
        tier=tier
    )
```

### By User with Quotas

```python
@dataclass
class UserQuota:
    requests_remaining: int
    requests_total: int
    reset_at: datetime
    overage_allowed: bool
    overage_rate: float  # Cost per request over quota

class QuotaRateLimiter:
    def check_quota(self, user_id: str) -> tuple[bool, dict]:
        quota = self.get_user_quota(user_id)
        
        if quota.requests_remaining > 0:
            self.decrement_quota(user_id)
            return True, {
                'remaining': quota.requests_remaining - 1,
                'total': quota.requests_total,
                'reset_at': quota.reset_at.isoformat()
            }
        
        if quota.overage_allowed:
            # Allow but charge overage
            self.record_overage(user_id)
            return True, {
                'remaining': 0,
                'overage': True,
                'overage_rate': quota.overage_rate
            }
        
        return False, {
            'remaining': 0,
            'reset_at': quota.reset_at.isoformat()
        }
```

---

## Algorithm Comparison

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| Token Bucket | Allows bursts, smooth average rate | Slightly complex state | APIs with burst traffic |
| Leaky Bucket | Smooth output rate, prevents bursts | Requests may wait | Backend protection |
| Fixed Window | Simple, low memory | Boundary burst problem | Simple use cases |
| Sliding Log | Accurate | High memory usage | When accuracy critical |
| Sliding Window | Accurate, low memory | Slightly complex | Production rate limiting |

---

## Rate Limiting at Different Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CDN/Edge                                   │
│        • IP-based DDoS protection                                    │
│        • Geographic rate limiting                                    │
│        • Bot detection                                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         API Gateway                                  │
│        • API key validation                                          │
│        • Tier-based limits                                           │
│        • Endpoint-specific limits                                    │
│        • Request counting                                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Application Layer                               │
│        • Business logic limits                                       │
│        • User quotas                                                 │
│        • Resource-specific limits                                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Database                                     │
│        • Connection pooling                                          │
│        • Query rate limiting                                         │
│        • Read/write quotas                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Choose the right algorithm**: Token bucket for APIs (allows bursts), sliding window for accurate limits, leaky bucket for smoothing

2. **Use distributed storage**: Redis or similar for rate limiting across multiple application instances

3. **Implement multiple tiers**: Global → API → User limits provide defense in depth

4. **Return proper headers**: `X-RateLimit-*` headers help clients implement backoff strategies

5. **Consider legitimate use cases**: Set limits that allow legitimate use while preventing abuse

6. **Monitor and adjust**: Track rate limit hits, false positives, and adjust limits based on real usage patterns

7. **Graceful degradation**: Consider allowing degraded service instead of hard blocking (e.g., slower responses, reduced features)
