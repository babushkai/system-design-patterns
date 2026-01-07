# Circuit Breakers

## TL;DR

A circuit breaker prevents cascading failures in distributed systems by monitoring calls to external services and "opening" the circuit when failure rates exceed a threshold. When open, requests fail immediately without attempting the downstream call, giving the failing service time to recover. After a timeout, the circuit moves to "half-open" to test if the service has recovered.

---

## Why Circuit Breakers?

Without circuit breaker:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Cascade Failure                               │
│                                                                      │
│   Service A          Service B          Service C                   │
│   ┌───────┐         ┌───────┐          ┌───────┐                   │
│   │       │────────►│       │─────────►│       │                   │
│   │       │ waiting │       │ waiting  │  ╳    │ DOWN!             │
│   │       │◄────────│       │◄─────────│       │                   │
│   │       │ timeout │       │ timeout  │       │                   │
│   └───────┘         └───────┘          └───────┘                   │
│       │                 │                                           │
│       │                 │                                           │
│       ▼                 ▼                                           │
│   Thread Pool       Thread Pool                                     │
│   EXHAUSTED         EXHAUSTED                                       │
│       │                 │                                           │
│       ▼                 ▼                                           │
│   Service A          Service B                                      │
│     DOWN              DOWN           ← Cascade failure!             │
└─────────────────────────────────────────────────────────────────────┘
```

With circuit breaker:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Circuit Breaker Protection                        │
│                                                                      │
│   Service A          Service B          Service C                   │
│   ┌───────┐         ┌───────┐          ┌───────┐                   │
│   │       │         │   CB  │          │       │                   │
│   │       │────────►│  ──── │    ╳     │  ╳    │ DOWN!             │
│   │       │         │ OPEN! │          │       │                   │
│   │       │◄────────│       │          │       │                   │
│   │       │ fallback│       │          │       │                   │
│   └───────┘         └───────┘          └───────┘                   │
│       │                                                             │
│       │                                                             │
│       ▼                                                             │
│   Returns cached    Service B stays healthy                         │
│   or default data   (doesn't wait for C)                            │
│                                                                      │
│   All services remain operational!                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Circuit Breaker States

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│                          CLOSED                                      │
│                     (Normal Operation)                               │
│                                                                      │
│                    Requests pass through                             │
│                    Failures are counted                              │
│                                                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           │ Failure threshold exceeded
                           │ (e.g., 5 failures in 10 seconds)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│                           OPEN                                       │
│                    (Failing Fast)                                    │
│                                                                      │
│                    All requests fail immediately                     │
│                    No calls to downstream service                    │
│                                                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           │ Timeout expires
                           │ (e.g., 30 seconds)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│                        HALF-OPEN                                     │
│                    (Testing Recovery)                                │
│                                                                      │
│                    Limited requests pass through                     │
│                    Testing if service recovered                      │
│                                                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          │ Success                         │ Failure
          ▼                                 ▼
     ┌─────────┐                      ┌──────────┐
     │ CLOSED  │                      │   OPEN   │
     └─────────┘                      └──────────┘
```

---

## Basic Implementation

```python
import time
from enum import Enum
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Optional, Any
from functools import wraps

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5       # Failures before opening
    success_threshold: int = 3       # Successes to close from half-open
    timeout: float = 30.0            # Seconds before trying half-open
    half_open_max_calls: int = 3     # Max calls in half-open state

class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open"""
    def __init__(self, circuit_name: str, retry_after: float):
        self.circuit_name = circuit_name
        self.retry_after = retry_after
        super().__init__(f"Circuit '{circuit_name}' is open. Retry after {retry_after:.1f}s")

class CircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self.lock = Lock()
    
    def _should_attempt_reset(self) -> bool:
        """Check if timeout has passed to try half-open"""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.config.timeout
    
    def _handle_success(self):
        """Handle successful call"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._close()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    def _handle_failure(self):
        """Handle failed call"""
        with self.lock:
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._open()
            elif self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.config.failure_threshold:
                    self._open()
    
    def _open(self):
        """Transition to open state"""
        self.state = CircuitState.OPEN
        self.success_count = 0
        self.half_open_calls = 0
        print(f"Circuit '{self.name}' OPENED")
    
    def _close(self):
        """Transition to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        print(f"Circuit '{self.name}' CLOSED")
    
    def _half_open(self):
        """Transition to half-open state"""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
        print(f"Circuit '{self.name}' HALF-OPEN")
    
    def can_execute(self) -> bool:
        """Check if request can proceed"""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._half_open()
                    return True
                return False
            
            # Half-open: allow limited calls
            if self.half_open_calls < self.config.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if not self.can_execute():
            retry_after = self.config.timeout - (time.time() - self.last_failure_time)
            raise CircuitBreakerOpen(self.name, max(0, retry_after))
        
        try:
            result = func(*args, **kwargs)
            self._handle_success()
            return result
        except Exception as e:
            self._handle_failure()
            raise

# Decorator usage
def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    breaker = CircuitBreaker(name, config)
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.execute(func, *args, **kwargs)
        wrapper.circuit_breaker = breaker
        return wrapper
    return decorator

# Example usage
@circuit_breaker("payment_service", CircuitBreakerConfig(
    failure_threshold=3,
    timeout=60
))
def call_payment_service(order_id: str):
    # Make external call
    response = requests.post(
        "https://payment.example.com/charge",
        json={"order_id": order_id}
    )
    response.raise_for_status()
    return response.json()
```

---

## Advanced Circuit Breaker with Metrics

```python
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Deque
import time
import threading

@dataclass
class CallMetrics:
    timestamp: float
    success: bool
    duration_ms: float
    error_type: Optional[str] = None

@dataclass
class CircuitBreakerStats:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    average_response_time_ms: float = 0.0
    error_percentage: float = 0.0
    state: CircuitState = CircuitState.CLOSED
    last_state_change: Optional[float] = None

class SlidingWindowCircuitBreaker:
    """Circuit breaker with sliding window failure detection"""
    
    def __init__(
        self,
        name: str,
        window_size: int = 10,            # Number of calls to track
        failure_rate_threshold: float = 0.5,  # 50% failure rate
        slow_call_threshold_ms: float = 5000,  # Calls over 5s are "slow"
        slow_call_rate_threshold: float = 0.5,  # 50% slow calls
        wait_duration_seconds: float = 30,
        permitted_calls_in_half_open: int = 3
    ):
        self.name = name
        self.window_size = window_size
        self.failure_rate_threshold = failure_rate_threshold
        self.slow_call_threshold_ms = slow_call_threshold_ms
        self.slow_call_rate_threshold = slow_call_rate_threshold
        self.wait_duration = wait_duration_seconds
        self.permitted_calls_in_half_open = permitted_calls_in_half_open
        
        self.state = CircuitState.CLOSED
        self.metrics: Deque[CallMetrics] = deque(maxlen=window_size)
        self.half_open_calls = 0
        self.half_open_successes = 0
        self.opened_at: Optional[float] = None
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = CircuitBreakerStats()
    
    def _calculate_failure_rate(self) -> float:
        if len(self.metrics) < self.window_size:
            return 0.0
        failures = sum(1 for m in self.metrics if not m.success)
        return failures / len(self.metrics)
    
    def _calculate_slow_call_rate(self) -> float:
        if len(self.metrics) < self.window_size:
            return 0.0
        slow_calls = sum(
            1 for m in self.metrics 
            if m.duration_ms > self.slow_call_threshold_ms
        )
        return slow_calls / len(self.metrics)
    
    def _should_open(self) -> bool:
        failure_rate = self._calculate_failure_rate()
        slow_call_rate = self._calculate_slow_call_rate()
        
        return (
            failure_rate >= self.failure_rate_threshold or
            slow_call_rate >= self.slow_call_rate_threshold
        )
    
    def _transition_to_open(self):
        self.state = CircuitState.OPEN
        self.opened_at = time.time()
        self.stats.last_state_change = time.time()
        self._notify_state_change(CircuitState.OPEN)
    
    def _transition_to_half_open(self):
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.half_open_successes = 0
        self.stats.last_state_change = time.time()
        self._notify_state_change(CircuitState.HALF_OPEN)
    
    def _transition_to_closed(self):
        self.state = CircuitState.CLOSED
        self.metrics.clear()
        self.stats.last_state_change = time.time()
        self._notify_state_change(CircuitState.CLOSED)
    
    def _notify_state_change(self, new_state: CircuitState):
        """Hook for monitoring/alerting"""
        print(f"Circuit breaker '{self.name}' transitioned to {new_state.value}")
    
    def record_success(self, duration_ms: float):
        with self.lock:
            self.metrics.append(CallMetrics(
                timestamp=time.time(),
                success=True,
                duration_ms=duration_ms
            ))
            self.stats.total_calls += 1
            self.stats.successful_calls += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1
                if self.half_open_successes >= self.permitted_calls_in_half_open:
                    self._transition_to_closed()
    
    def record_failure(self, duration_ms: float, error_type: str):
        with self.lock:
            self.metrics.append(CallMetrics(
                timestamp=time.time(),
                success=False,
                duration_ms=duration_ms,
                error_type=error_type
            ))
            self.stats.total_calls += 1
            self.stats.failed_calls += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self.state == CircuitState.CLOSED:
                if self._should_open():
                    self._transition_to_open()
    
    def allow_request(self) -> bool:
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            if self.state == CircuitState.OPEN:
                if time.time() - self.opened_at >= self.wait_duration:
                    self._transition_to_half_open()
                    self.half_open_calls = 1
                    return True
                self.stats.rejected_calls += 1
                return False
            
            # Half-open
            if self.half_open_calls < self.permitted_calls_in_half_open:
                self.half_open_calls += 1
                return True
            
            self.stats.rejected_calls += 1
            return False
    
    def get_stats(self) -> CircuitBreakerStats:
        with self.lock:
            stats = CircuitBreakerStats(
                total_calls=self.stats.total_calls,
                successful_calls=self.stats.successful_calls,
                failed_calls=self.stats.failed_calls,
                rejected_calls=self.stats.rejected_calls,
                error_percentage=self._calculate_failure_rate() * 100,
                state=self.state,
                last_state_change=self.stats.last_state_change
            )
            
            if self.metrics:
                stats.average_response_time_ms = sum(
                    m.duration_ms for m in self.metrics
                ) / len(self.metrics)
            
            return stats
```

---

## Circuit Breaker with Fallback

```python
from typing import TypeVar, Generic, Callable, Optional

T = TypeVar('T')

class CircuitBreakerWithFallback(Generic[T]):
    def __init__(
        self,
        name: str,
        primary: Callable[..., T],
        fallback: Callable[..., T],
        config: CircuitBreakerConfig = None
    ):
        self.name = name
        self.primary = primary
        self.fallback = fallback
        self.circuit = SlidingWindowCircuitBreaker(name)
        self.config = config or CircuitBreakerConfig()
    
    def call(self, *args, **kwargs) -> T:
        if not self.circuit.allow_request():
            return self._execute_fallback(*args, **kwargs)
        
        start = time.time()
        try:
            result = self.primary(*args, **kwargs)
            duration = (time.time() - start) * 1000
            self.circuit.record_success(duration)
            return result
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.circuit.record_failure(duration, type(e).__name__)
            return self._execute_fallback(*args, **kwargs)
    
    def _execute_fallback(self, *args, **kwargs) -> T:
        try:
            return self.fallback(*args, **kwargs)
        except Exception as e:
            raise FallbackFailedException(
                f"Both primary and fallback failed for {self.name}"
            ) from e

# Fallback strategies
class FallbackStrategies:
    @staticmethod
    def cached(cache: dict, key_func: Callable):
        """Return cached value"""
        def fallback(*args, **kwargs):
            key = key_func(*args, **kwargs)
            if key in cache:
                return cache[key]
            raise CacheMissException(f"No cached value for {key}")
        return fallback
    
    @staticmethod
    def default(value: T):
        """Return default value"""
        def fallback(*args, **kwargs):
            return value
        return fallback
    
    @staticmethod
    def fail():
        """Fail explicitly"""
        def fallback(*args, **kwargs):
            raise ServiceUnavailableException("Service unavailable")
        return fallback
    
    @staticmethod
    def queue_for_later(queue: 'AsyncQueue'):
        """Queue request for later processing"""
        def fallback(*args, **kwargs):
            queue.enqueue({
                'args': args,
                'kwargs': kwargs,
                'timestamp': time.time()
            })
            return {'status': 'queued', 'message': 'Request queued for processing'}
        return fallback

# Usage example
product_cache = {}

product_service = CircuitBreakerWithFallback(
    name="product_service",
    primary=lambda product_id: fetch_product_from_api(product_id),
    fallback=FallbackStrategies.cached(
        product_cache, 
        lambda product_id: f"product:{product_id}"
    )
)

# Or with default fallback
recommendation_service = CircuitBreakerWithFallback(
    name="recommendations",
    primary=lambda user_id: get_personalized_recommendations(user_id),
    fallback=FallbackStrategies.default([
        {"id": 1, "name": "Popular Item 1"},
        {"id": 2, "name": "Popular Item 2"},
    ])
)
```

---

## Circuit Breaker Registry

```python
from typing import Dict
import threading

class CircuitBreakerRegistry:
    """Centralized management of circuit breakers"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.breakers: Dict[str, CircuitBreaker] = {}
                    cls._instance.configs: Dict[str, CircuitBreakerConfig] = {}
        return cls._instance
    
    def register(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        if name in self.breakers:
            return self.breakers[name]
        
        config = config or self.configs.get(name, CircuitBreakerConfig())
        breaker = SlidingWindowCircuitBreaker(name)
        self.breakers[name] = breaker
        return breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        return self.breakers.get(name)
    
    def get_all_stats(self) -> Dict[str, CircuitBreakerStats]:
        return {
            name: breaker.get_stats()
            for name, breaker in self.breakers.items()
        }
    
    def reset_all(self):
        for breaker in self.breakers.values():
            breaker._transition_to_closed()
    
    def force_open(self, name: str):
        if name in self.breakers:
            self.breakers[name]._transition_to_open()
    
    def force_close(self, name: str):
        if name in self.breakers:
            self.breakers[name]._transition_to_closed()

# Usage
registry = CircuitBreakerRegistry()

@circuit_breaker_from_registry("user_service")
def get_user(user_id: str):
    return requests.get(f"http://user-service/users/{user_id}").json()

def circuit_breaker_from_registry(name: str, config: CircuitBreakerConfig = None):
    def decorator(func):
        breaker = registry.register(name, config)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not breaker.allow_request():
                raise CircuitBreakerOpen(name, breaker.wait_duration)
            
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                breaker.record_success(duration)
                return result
            except Exception as e:
                duration = (time.time() - start) * 1000
                breaker.record_failure(duration, type(e).__name__)
                raise
        
        return wrapper
    return decorator

# Health endpoint
@app.route('/health/circuits')
def circuit_health():
    registry = CircuitBreakerRegistry()
    stats = registry.get_all_stats()
    
    all_closed = all(s.state == CircuitState.CLOSED for s in stats.values())
    
    return {
        'status': 'healthy' if all_closed else 'degraded',
        'circuits': {
            name: {
                'state': s.state.value,
                'error_rate': f"{s.error_percentage:.1f}%",
                'total_calls': s.total_calls,
                'rejected_calls': s.rejected_calls
            }
            for name, s in stats.items()
        }
    }, 200 if all_closed else 503
```

---

## Bulkhead Pattern with Circuit Breaker

```python
import concurrent.futures
from dataclasses import dataclass
from typing import Dict

@dataclass
class BulkheadConfig:
    max_concurrent: int = 10
    max_wait_seconds: float = 1.0

class BulkheadCircuitBreaker:
    """Combines bulkhead (concurrency limit) with circuit breaker"""
    
    def __init__(
        self,
        name: str,
        bulkhead_config: BulkheadConfig,
        circuit_config: CircuitBreakerConfig = None
    ):
        self.name = name
        self.bulkhead = bulkhead_config
        self.circuit = SlidingWindowCircuitBreaker(name)
        self.semaphore = threading.Semaphore(bulkhead_config.max_concurrent)
        self.active_calls = 0
        self.lock = threading.Lock()
    
    def execute(self, func: Callable, *args, **kwargs):
        # Check circuit breaker first
        if not self.circuit.allow_request():
            raise CircuitBreakerOpen(self.name, self.circuit.wait_duration)
        
        # Try to acquire bulkhead slot
        acquired = self.semaphore.acquire(
            timeout=self.bulkhead.max_wait_seconds
        )
        
        if not acquired:
            raise BulkheadFullException(
                f"Bulkhead '{self.name}' is full ({self.bulkhead.max_concurrent} concurrent calls)"
            )
        
        with self.lock:
            self.active_calls += 1
        
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = (time.time() - start) * 1000
            self.circuit.record_success(duration)
            return result
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.circuit.record_failure(duration, type(e).__name__)
            raise
        finally:
            with self.lock:
                self.active_calls -= 1
            self.semaphore.release()

# Usage: Isolate different services
class ServiceCaller:
    def __init__(self):
        # Each service has its own bulkhead + circuit breaker
        self.payment_bulkhead = BulkheadCircuitBreaker(
            "payment",
            BulkheadConfig(max_concurrent=20),
            CircuitBreakerConfig(failure_threshold=5)
        )
        
        self.inventory_bulkhead = BulkheadCircuitBreaker(
            "inventory",
            BulkheadConfig(max_concurrent=50),
            CircuitBreakerConfig(failure_threshold=10)
        )
    
    def process_order(self, order):
        # Payment failure won't exhaust inventory threads
        payment = self.payment_bulkhead.execute(
            self.call_payment_service, order
        )
        
        # Inventory failure won't exhaust payment threads
        inventory = self.inventory_bulkhead.execute(
            self.call_inventory_service, order
        )
        
        return payment, inventory
```

```
Bulkhead Isolation:

┌────────────────────────────────────────────────────────────────────┐
│                        Application Thread Pool                      │
└────────────────────────────────────────────────────────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│    Payment     │      │   Inventory    │      │   Shipping     │
│   Bulkhead     │      │   Bulkhead     │      │   Bulkhead     │
│   (20 slots)   │      │   (50 slots)   │      │   (30 slots)   │
├────────────────┤      ├────────────────┤      ├────────────────┤
│ Circuit Breaker│      │ Circuit Breaker│      │ Circuit Breaker│
└───────┬────────┘      └───────┬────────┘      └───────┬────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   Payment API            Inventory API           Shipping API

If Payment API is slow/down:
- Only Payment bulkhead is affected
- Inventory and Shipping continue working
- No cascading thread exhaustion
```

---

## Distributed Circuit Breaker

```python
import redis
import json
from typing import Optional

class DistributedCircuitBreaker:
    """Circuit breaker state shared across instances via Redis"""
    
    def __init__(
        self,
        name: str,
        redis_client: redis.Redis,
        config: CircuitBreakerConfig = None
    ):
        self.name = name
        self.redis = redis_client
        self.config = config or CircuitBreakerConfig()
        self.key_prefix = f"circuit_breaker:{name}"
    
    def _state_key(self) -> str:
        return f"{self.key_prefix}:state"
    
    def _metrics_key(self) -> str:
        return f"{self.key_prefix}:metrics"
    
    def _get_state(self) -> CircuitState:
        state_data = self.redis.get(self._state_key())
        if state_data:
            data = json.loads(state_data)
            return CircuitState(data['state'])
        return CircuitState.CLOSED
    
    def _set_state(self, state: CircuitState, opened_at: float = None):
        data = {
            'state': state.value,
            'opened_at': opened_at or time.time(),
            'updated_at': time.time()
        }
        self.redis.set(self._state_key(), json.dumps(data))
    
    def allow_request(self) -> bool:
        # Use Lua script for atomicity
        lua_script = """
        local state_key = KEYS[1]
        local config = cjson.decode(ARGV[1])
        local now = tonumber(ARGV[2])
        
        local state_data = redis.call('GET', state_key)
        if not state_data then
            return 1  -- Allow (CLOSED assumed)
        end
        
        local state = cjson.decode(state_data)
        
        if state.state == 'closed' then
            return 1  -- Allow
        elseif state.state == 'open' then
            -- Check if timeout passed
            if now - state.opened_at >= config.timeout then
                -- Transition to half-open
                state.state = 'half_open'
                state.half_open_calls = 1
                redis.call('SET', state_key, cjson.encode(state))
                return 1  -- Allow
            end
            return 0  -- Reject
        elseif state.state == 'half_open' then
            if state.half_open_calls < config.half_open_max_calls then
                state.half_open_calls = state.half_open_calls + 1
                redis.call('SET', state_key, cjson.encode(state))
                return 1  -- Allow
            end
            return 0  -- Reject
        end
        
        return 1  -- Default allow
        """
        
        result = self.redis.eval(
            lua_script,
            1,
            self._state_key(),
            json.dumps({
                'timeout': self.config.timeout,
                'half_open_max_calls': self.config.half_open_max_calls
            }),
            time.time()
        )
        
        return bool(result)
    
    def record_success(self):
        lua_script = """
        local state_key = KEYS[1]
        local metrics_key = KEYS[2]
        local config = cjson.decode(ARGV[1])
        local now = tonumber(ARGV[2])
        
        -- Record metric
        redis.call('LPUSH', metrics_key, cjson.encode({success=true, ts=now}))
        redis.call('LTRIM', metrics_key, 0, config.window_size - 1)
        
        local state_data = redis.call('GET', state_key)
        if not state_data then return end
        
        local state = cjson.decode(state_data)
        
        if state.state == 'half_open' then
            state.success_count = (state.success_count or 0) + 1
            if state.success_count >= config.success_threshold then
                state.state = 'closed'
                state.success_count = 0
            end
            redis.call('SET', state_key, cjson.encode(state))
        end
        """
        
        self.redis.eval(
            lua_script,
            2,
            self._state_key(),
            self._metrics_key(),
            json.dumps({
                'window_size': 100,
                'success_threshold': self.config.success_threshold
            }),
            time.time()
        )
    
    def record_failure(self):
        lua_script = """
        local state_key = KEYS[1]
        local metrics_key = KEYS[2]
        local config = cjson.decode(ARGV[1])
        local now = tonumber(ARGV[2])
        
        -- Record metric
        redis.call('LPUSH', metrics_key, cjson.encode({success=false, ts=now}))
        redis.call('LTRIM', metrics_key, 0, config.window_size - 1)
        
        -- Get current metrics
        local metrics = redis.call('LRANGE', metrics_key, 0, config.window_size - 1)
        local failures = 0
        for _, m in ipairs(metrics) do
            local metric = cjson.decode(m)
            if not metric.success then
                failures = failures + 1
            end
        end
        
        local failure_rate = failures / #metrics
        
        local state_data = redis.call('GET', state_key)
        local state = state_data and cjson.decode(state_data) or {state='closed'}
        
        if state.state == 'half_open' then
            -- Any failure in half-open opens circuit
            state.state = 'open'
            state.opened_at = now
        elseif state.state == 'closed' and #metrics >= config.window_size then
            if failure_rate >= config.failure_rate_threshold then
                state.state = 'open'
                state.opened_at = now
            end
        end
        
        redis.call('SET', state_key, cjson.encode(state))
        """
        
        self.redis.eval(
            lua_script,
            2,
            self._state_key(),
            self._metrics_key(),
            json.dumps({
                'window_size': 10,
                'failure_rate_threshold': 0.5
            }),
            time.time()
        )
```

---

## Monitoring and Alerting

```python
from prometheus_client import Counter, Gauge, Histogram

class CircuitBreakerMetrics:
    def __init__(self, name: str):
        self.name = name
        
        # Prometheus metrics
        self.calls_total = Counter(
            'circuit_breaker_calls_total',
            'Total number of calls',
            ['circuit', 'result']
        )
        
        self.state_gauge = Gauge(
            'circuit_breaker_state',
            'Current state (0=closed, 1=open, 2=half-open)',
            ['circuit']
        )
        
        self.call_duration = Histogram(
            'circuit_breaker_call_duration_seconds',
            'Call duration in seconds',
            ['circuit'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        )
    
    def record_call(self, result: str, duration: float):
        self.calls_total.labels(circuit=self.name, result=result).inc()
        self.call_duration.labels(circuit=self.name).observe(duration)
    
    def update_state(self, state: CircuitState):
        state_value = {
            CircuitState.CLOSED: 0,
            CircuitState.OPEN: 1,
            CircuitState.HALF_OPEN: 2
        }[state]
        self.state_gauge.labels(circuit=self.name).set(state_value)

# Alert rules (Prometheus format)
"""
groups:
  - name: circuit_breaker_alerts
    rules:
      - alert: CircuitBreakerOpen
        expr: circuit_breaker_state == 1
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Circuit breaker {{ $labels.circuit }} is open"
          
      - alert: HighCircuitBreakerFailureRate
        expr: |
          rate(circuit_breaker_calls_total{result="failure"}[5m]) /
          rate(circuit_breaker_calls_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High failure rate for circuit {{ $labels.circuit }}"
"""
```

---

## Key Takeaways

1. **Fail fast**: When a service is down, fail immediately instead of waiting for timeouts

2. **Give services time to recover**: Open circuit prevents hammering a struggling service

3. **Use fallbacks**: Return cached data, defaults, or gracefully degraded responses when circuit is open

4. **Monitor circuit state**: Track when circuits open—it's an early warning of system problems

5. **Tune thresholds carefully**: Too sensitive = false positives; too lenient = not protective enough

6. **Combine with bulkheads**: Isolate services to prevent one slow service from exhausting all resources

7. **Distributed state for microservices**: Share circuit state across instances for consistent behavior
