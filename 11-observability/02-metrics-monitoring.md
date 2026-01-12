# Metrics and Monitoring

## TL;DR

Metrics are numerical measurements collected over time. They answer "what is happening?" through aggregated data, unlike logs (what happened?) and traces (how did it flow?). The RED method (Rate, Errors, Duration) and USE method (Utilization, Saturation, Errors) provide frameworks for comprehensive monitoring.

---

## Metrics vs. Logs vs. Traces

```
                    Metrics              Logs                 Traces
─────────────────────────────────────────────────────────────────────────
Question            What's happening?    What happened?       How did it flow?
Data type           Numbers              Text                 Spans
Cardinality         Low                  High                 Medium
Storage cost        Low                  High                 Medium
Query pattern       Aggregate            Search               Lookup by ID
Example             error_rate=0.02      "User 123 failed"    Request path

Use when you need:
- Alerting          ✓ Primary           Sometimes            Rarely
- Dashboards        ✓ Primary           Sometimes            ✓
- Debugging         Sometimes            ✓ Primary           ✓ Primary
- Capacity          ✓ Primary           Rarely               Sometimes
```

---

## Metric Types

### Counter

Cumulative value that only increases (or resets to zero).

```
http_requests_total

Time:    T0      T1      T2      T3      T4
Value:   100     150     225     310     400
Delta:   -       +50     +75     +85     +90

Use for:
- Request counts
- Bytes transferred
- Tasks completed
- Errors occurred
```

```python
from prometheus_client import Counter

http_requests = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# Increment
http_requests.labels(method='GET', endpoint='/api/users', status='200').inc()
```

### Gauge

Value that can go up or down.

```
active_connections

Time:    T0      T1      T2      T3      T4
Value:   10      25      15      30      20

Use for:
- Current connections
- Queue depth
- Temperature
- Memory usage
- Active users
```

```python
from prometheus_client import Gauge

active_connections = Gauge(
    'active_connections',
    'Current active connections',
    ['service']
)

# Set absolute value
active_connections.labels(service='api').set(42)

# Increment/decrement
active_connections.labels(service='api').inc()
active_connections.labels(service='api').dec()

# Context manager for tracking in-progress
with active_connections.labels(service='api').track_inprogress():
    process_request()
```

### Histogram

Distribution of values across configurable buckets.

```
http_request_duration_seconds

Buckets: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, +Inf]

Observations: 0.02, 0.03, 0.08, 0.15, 0.5, 2.0, 0.04

Bucket counts:
  le="0.01":  0
  le="0.05":  3  (0.02, 0.03, 0.04)
  le="0.1":   4  (+ 0.08)
  le="0.5":   6  (+ 0.15, 0.5)
  le="1.0":   6
  le="5.0":   7  (+ 2.0)
  le="+Inf":  7

Enables:
- Percentile calculation (p50, p95, p99)
- Distribution analysis
- SLO tracking
```

```python
from prometheus_client import Histogram

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Observe a value
request_duration.labels(method='GET', endpoint='/api').observe(0.25)

# Time a function
@request_duration.labels(method='GET', endpoint='/api').time()
def handle_request():
    pass
```

### Summary

Similar to histogram but calculates quantiles client-side.

```python
from prometheus_client import Summary

request_duration = Summary(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method'],
    # Pre-calculated quantiles (cannot aggregate across instances!)
    objectives={0.5: 0.05, 0.9: 0.01, 0.99: 0.001}
)
```

**Histogram vs. Summary:**

| Aspect | Histogram | Summary |
|--------|-----------|---------|
| Aggregation | ✓ Can aggregate | ✗ Cannot aggregate |
| Quantile accuracy | Approximate | Exact |
| Server load | Lower | Higher |
| Bucket config | Must pre-define | N/A |
| Recommendation | Use histogram | Rarely needed |

---

## Naming Conventions

### Prometheus Naming Best Practices

```python
# Format: <namespace>_<subsystem>_<name>_<unit>

# GOOD
http_requests_total                    # Counter
http_request_duration_seconds          # Histogram
process_memory_bytes                   # Gauge
database_connections_active            # Gauge

# BAD
requests                               # Too vague
http_requests_count                    # Use _total for counters
requestDurationMilliseconds            # Wrong format, wrong unit
HttpRequestDuration                    # Wrong case
```

### Label Best Practices

```python
# GOOD - Low cardinality
http_requests_total{method="GET", status="200", endpoint="/api/users"}

# BAD - High cardinality (unbounded)
http_requests_total{user_id="12345"}  # Millions of unique values!
http_requests_total{request_id="..."}  # Unique per request!

# Rule of thumb: 
# Unique label combinations < 10,000
# Each label value should have < 100 unique values
```

---

## RED Method (Request-Driven)

For services (APIs, microservices):

```
R - Rate:     Request throughput (requests/second)
E - Errors:   Failed requests (count or rate)
D - Duration: Response time distribution

Dashboard Layout:
┌─────────────────────────────────────────────────────────────┐
│  Service: user-api                                          │
│                                                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐ │
│  │ Request Rate    │ │ Error Rate      │ │ Latency p99   │ │
│  │     523/s       │ │     0.02%       │ │     245ms     │ │
│  │     ↑ 12%       │ │     ↓ 50%       │ │     ↑ 15%     │ │
│  └─────────────────┘ └─────────────────┘ └───────────────┘ │
│                                                             │
│  [Request Rate Over Time Graph]                             │
│  [Latency Distribution Heatmap]                             │
│  [Error Rate by Endpoint]                                   │
└─────────────────────────────────────────────────────────────┘
```

### Prometheus Queries (PromQL)

```promql
# Rate: Requests per second
rate(http_requests_total[5m])

# Rate by endpoint
sum by (endpoint) (rate(http_requests_total[5m]))

# Errors: Error rate percentage
sum(rate(http_requests_total{status=~"5.."}[5m])) 
/ 
sum(rate(http_requests_total[5m])) * 100

# Duration: p99 latency
histogram_quantile(0.99, 
  sum by (le) (rate(http_request_duration_seconds_bucket[5m]))
)

# Duration: Average latency
sum(rate(http_request_duration_seconds_sum[5m])) 
/ 
sum(rate(http_request_duration_seconds_count[5m]))
```

---

## USE Method (Resource-Oriented)

For resources (CPU, memory, disk, network):

```
U - Utilization: Percentage of resource capacity in use
S - Saturation:  Degree of queuing/backlog
E - Errors:      Error events

Resource Analysis:
┌──────────────┬─────────────────┬─────────────────┬─────────────────┐
│   Resource   │   Utilization   │   Saturation    │     Errors      │
├──────────────┼─────────────────┼─────────────────┼─────────────────┤
│ CPU          │ % busy          │ Run queue depth │ -               │
│ Memory       │ % used          │ Swap usage      │ OOM kills       │
│ Disk I/O     │ % time busy     │ Queue length    │ I/O errors      │
│ Network      │ % bandwidth     │ Socket backlog  │ Packet errors   │
│ Connection   │ % pool used     │ Wait queue      │ Timeouts        │
└──────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### Key Resource Metrics

```promql
# CPU
# Utilization: CPU usage percentage
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Saturation: Load average / CPU count
node_load1 / count by (instance) (node_cpu_seconds_total{mode="idle"})

# Memory
# Utilization
(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100

# Saturation (swap usage indicates memory pressure)
node_memory_SwapTotal_bytes - node_memory_SwapFree_bytes

# Disk
# Utilization
rate(node_disk_io_time_seconds_total[5m]) * 100

# Saturation
node_disk_io_time_weighted_seconds_total

# Network
# Utilization (requires knowing interface speed)
rate(node_network_receive_bytes_total[5m]) * 8 / 1e9  # Gbps
```

---

## Golden Signals (Google SRE)

```
Latency:     Time to service a request (successful vs. failed)
Traffic:     Demand on your system (requests/sec, transactions/sec)
Errors:      Rate of failed requests
Saturation:  How "full" your service is (capacity utilization)

Relationship to RED/USE:
├── Latency   ≈ RED Duration
├── Traffic   ≈ RED Rate
├── Errors    ≈ RED Errors / USE Errors
└── Saturation ≈ USE Saturation
```

---

## Instrumentation Patterns

### Middleware Instrumentation

```python
import time
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'Request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

REQUESTS_IN_PROGRESS = Gauge(
    'http_requests_in_progress',
    'Requests currently being processed',
    ['method', 'endpoint']
)

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    def __call__(self, environ, start_response):
        method = environ['REQUEST_METHOD']
        endpoint = self.normalize_endpoint(environ['PATH_INFO'])
        
        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
        start_time = time.time()
        
        status_code = '500'  # Default if exception
        
        def custom_start_response(status, headers, exc_info=None):
            nonlocal status_code
            status_code = status.split()[0]
            return start_response(status, headers, exc_info)
        
        try:
            response = self.app(environ, custom_start_response)
            return response
        finally:
            duration = time.time() - start_time
            
            REQUEST_COUNT.labels(
                method=method, 
                endpoint=endpoint, 
                status=status_code
            ).inc()
            
            REQUEST_LATENCY.labels(
                method=method, 
                endpoint=endpoint
            ).observe(duration)
            
            REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
    
    def normalize_endpoint(self, path):
        # /users/123 → /users/{id}
        # Prevents cardinality explosion
        import re
        path = re.sub(r'/\d+', '/{id}', path)
        path = re.sub(r'/[a-f0-9-]{36}', '/{uuid}', path)
        return path
```

### Business Metrics

```python
# Business-relevant metrics beyond technical ones

orders_total = Counter(
    'orders_total',
    'Total orders processed',
    ['status', 'payment_method']
)

order_value = Histogram(
    'order_value_dollars',
    'Order value in dollars',
    buckets=[10, 25, 50, 100, 250, 500, 1000, 5000]
)

active_users = Gauge(
    'active_users',
    'Currently active users'
)

# In application code
def process_order(order):
    orders_total.labels(
        status='completed',
        payment_method=order.payment_method
    ).inc()
    
    order_value.observe(order.total)
```

---

## Alerting

### Alert Design Principles

```yaml
# Good alert characteristics:
# - Actionable: Someone needs to do something
# - Relevant: Indicates real user impact
# - Specific: Clear what's wrong
# - Timely: Neither too sensitive nor too delayed

# BAD: Alert on symptoms of normal operation
- alert: HighCPU
  expr: cpu_usage > 70    # Normal for busy server

# GOOD: Alert on user-facing impact
- alert: HighErrorRate
  expr: |
    sum(rate(http_requests_total{status=~"5.."}[5m])) 
    / sum(rate(http_requests_total[5m])) > 0.01
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High error rate (> 1%)"
    description: "Error rate is {{ $value | humanizePercentage }}"
```

### SLO-Based Alerting

```yaml
# Alert when burning through error budget too fast

# SLO: 99.9% availability = 0.1% error budget
# 30-day budget = 43.2 minutes of errors

# Multi-window, multi-burn-rate alerting
groups:
- name: slo-alerts
  rules:
  # Fast burn: 14.4x burn rate for 1 hour = 2% of monthly budget
  - alert: HighBurnRate_Fast
    expr: |
      (
        sum(rate(http_requests_total{status=~"5.."}[1h]))
        / sum(rate(http_requests_total[1h]))
      ) > (14.4 * 0.001)
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Burning error budget 14x too fast"
  
  # Slow burn: 3x burn rate for 6 hours = 3% of monthly budget
  - alert: HighBurnRate_Slow
    expr: |
      (
        sum(rate(http_requests_total{status=~"5.."}[6h]))
        / sum(rate(http_requests_total[6h]))
      ) > (3 * 0.001)
    for: 15m
    labels:
      severity: warning
```

### Alert Routing

```yaml
# alertmanager.yml
route:
  receiver: 'default'
  group_by: ['alertname', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
  - match:
      severity: critical
    receiver: 'pagerduty-critical'
    continue: true
  - match:
      severity: warning
    receiver: 'slack-warnings'
  - match:
      team: database
    receiver: 'database-team'

receivers:
- name: 'pagerduty-critical'
  pagerduty_configs:
  - service_key: '<key>'
    
- name: 'slack-warnings'
  slack_configs:
  - channel: '#alerts'
    send_resolved: true
```

---

## Dashboard Design

### Layout Principles

```
┌─────────────────────────────────────────────────────────────────────┐
│  Service Overview Dashboard                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TOP ROW: Key indicators (current state at a glance)               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Requests │ │  Errors  │ │ Latency  │ │ Success  │ │Saturation│  │
│  │  1.2k/s  │ │   0.1%   │ │  42ms    │ │  99.9%   │ │   34%    │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                                                     │
│  MIDDLE: Time series (trends and patterns)                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Request Rate & Error Rate (overlaid)                       │   │
│  │  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁                                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Latency Distribution (heatmap or percentiles)              │   │
│  │  p99: ████████████████████                                  │   │
│  │  p95: ████████████                                          │   │
│  │  p50: █████                                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  BOTTOM: Breakdown (drill-down details)                            │
│  ┌────────────────────────┐ ┌────────────────────────────────┐     │
│  │ Errors by Endpoint     │ │ Latency by Endpoint            │     │
│  │ /api/orders    45%     │ │ /api/search     250ms          │     │
│  │ /api/users     30%     │ │ /api/orders     120ms          │     │
│  │ /api/products  25%     │ │ /api/users       45ms          │     │
│  └────────────────────────┘ └────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### Essential Dashboards

```
1. Service Overview
   - RED metrics for each service
   - Dependency status
   - Recent deployments marked

2. Infrastructure
   - USE metrics for all resources
   - Cluster health
   - Node status

3. Business KPIs
   - Orders/Revenue
   - Active users
   - Conversion rates

4. SLO Dashboard
   - Current error budget
   - Burn rate
   - Historical SLO performance
```

---

## Anti-Patterns

### Cardinality Explosion

```python
# BAD: Unbounded label values
metrics.labels(
    user_id=user.id,        # Millions of users
    request_id=request.id,  # Unique per request
    timestamp=str(time.time())  # Infinite
)

# Impact:
# - Memory exhaustion
# - Query performance degradation
# - Storage costs explode

# GOOD: Bounded, low-cardinality labels
metrics.labels(
    user_tier=user.tier,    # free, pro, enterprise
    endpoint="/api/users",  # ~100 endpoints
    status_class="2xx"      # 2xx, 3xx, 4xx, 5xx
)
```

### Monitoring What's Easy, Not What Matters

```
BAD (easy to measure):
- CPU usage
- Memory usage
- Uptime

These don't tell you if users are happy!

GOOD (what matters):
- Request success rate
- Request latency (user-perceived)
- Error rate by type
- Business transactions/sec
```

---

## References

- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [RED Method](https://grafana.com/blog/2018/08/02/the-red-method-how-to-instrument-your-services/)
- [USE Method](https://www.brendangregg.com/usemethod.html)
- [Practical Monitoring](https://www.oreilly.com/library/view/practical-monitoring/9781491957349/)
