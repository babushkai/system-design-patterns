# Distributed Tracing

## TL;DR

Distributed tracing tracks requests as they flow through multiple services, creating a complete picture of a transaction's journey. Each trace consists of spans representing individual operations, connected by context propagation. Essential for debugging latency issues and understanding system behavior.

---

## The Problem Tracing Solves

In a microservices architecture, a single user request touches many services:

```
User Request
     │
     ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│   API   │───►│  Auth   │───►│  User   │───►│  Cache  │
│ Gateway │    │ Service │    │ Service │    │         │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
     │                             │
     │                             ▼
     │                        ┌─────────┐
     │                        │   DB    │
     │                        └─────────┘
     ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│  Order  │───►│ Payment │───►│ Notif.  │
│ Service │    │ Service │    │ Service │
└─────────┘    └─────────┘    └─────────┘

Without tracing:
- "Request took 2 seconds" - but where?
- "Payment failed" - but what was the user context?
- Logs scattered across 8 different services
```

---

## Tracing Concepts

### Trace

A trace represents the entire journey of a request through the system.

```
Trace ID: abc123

┌─────────────────────────────────────────────────────────────────┐
│                          Time →                                 │
│                                                                 │
│ ├────────────── API Gateway (500ms) ───────────────────────────┤│
│ │    ├────── Auth Service (50ms) ──────┤                       ││
│ │    │                                  │                       ││
│ │    │    ├── User Service (200ms) ────┤                       ││
│ │    │    │   ├─ DB Query (150ms) ─┤   │                       ││
│ │    │    │                            │                       ││
│ │    ├────────── Order Service (400ms) ───────────────────────┤││
│ │    │         │    ├── Payment (250ms) ──┤                   │││
│ │    │         │    │                      │                   │││
│ │    │         │    │    ├─ Notify (30ms) ─┤                  │││
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Span

A span represents a single unit of work within a trace.

```python
{
    "trace_id": "abc123",
    "span_id": "span456",
    "parent_span_id": "span123",  # None for root span
    "operation_name": "HTTP GET /users/{id}",
    "service_name": "user-service",
    "start_time": "2024-01-01T10:00:00.000Z",
    "duration_ms": 200,
    "status": "OK",
    "tags": {
        "http.method": "GET",
        "http.url": "/users/123",
        "http.status_code": 200,
        "user.id": "123"
    },
    "logs": [
        {"timestamp": "...", "message": "Cache miss, querying database"},
        {"timestamp": "...", "message": "User found"}
    ]
}
```

### Context Propagation

Trace context must be passed between services:

```
Service A                        Service B
    │                                │
    │  HTTP Request                  │
    │  Headers:                      │
    │    traceparent: 00-abc123-...  │
    │    tracestate: vendor=value    │
    │ ──────────────────────────────►│
    │                                │
    │                                │ Extracts trace context
    │                                │ Creates child span
    │                                │ with same trace_id
```

### W3C Trace Context Standard

```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
             │  │                                │                  │
             │  │                                │                  └─ Flags (sampled)
             │  │                                └─ Parent Span ID
             │  └─ Trace ID
             └─ Version

tracestate: congo=t61rcWkgMzE,rojo=00f067aa0ba902b7
            Vendor-specific key-value pairs
```

---

## Implementation

### Manual Instrumentation

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

def process_order(order_id: str):
    # Start a new span
    with tracer.start_as_current_span("process_order") as span:
        # Add attributes
        span.set_attribute("order.id", order_id)
        
        try:
            # Child span for validation
            with tracer.start_as_current_span("validate_order") as child:
                order = validate_order(order_id)
                child.set_attribute("order.total", order.total)
            
            # Child span for payment
            with tracer.start_as_current_span("process_payment") as child:
                result = payment_service.charge(order)
                child.set_attribute("payment.method", result.method)
                
                if not result.success:
                    child.set_status(Status(StatusCode.ERROR))
                    child.record_exception(result.error)
                    raise PaymentError(result.error)
            
            span.set_status(Status(StatusCode.OK))
            return order
            
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
            raise
```

### Automatic Instrumentation

```python
# Most frameworks have auto-instrumentation libraries
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

# Automatically instruments Flask routes
FlaskInstrumentor().instrument_app(app)

# Automatically traces outgoing HTTP requests
RequestsInstrumentor().instrument()

# Automatically traces database queries
SQLAlchemyInstrumentor().instrument(engine=engine)
```

### Context Propagation in HTTP

```python
from opentelemetry import trace
from opentelemetry.propagate import inject, extract
import requests

def call_downstream_service(url: str, data: dict):
    headers = {}
    
    # Inject current trace context into headers
    inject(headers)
    
    # Headers now contain: traceparent, tracestate
    response = requests.post(url, json=data, headers=headers)
    return response

# On the receiving service
from flask import request

@app.route('/api/endpoint', methods=['POST'])
def handle_request():
    # Extract trace context from incoming request
    context = extract(request.headers)
    
    # Create span with extracted context as parent
    with tracer.start_as_current_span("handle_request", context=context):
        # Process request
        pass
```

### Context Propagation in Message Queues

```python
# Producer
def publish_message(queue: str, message: dict):
    headers = {}
    inject(headers)  # Inject trace context
    
    kafka_producer.send(
        queue,
        value=message,
        headers=[(k, v.encode()) for k, v in headers.items()]
    )

# Consumer
def consume_message(message):
    # Extract trace context from message headers
    headers = {k: v.decode() for k, v in message.headers}
    context = extract(headers)
    
    with tracer.start_as_current_span("process_message", context=context):
        process(message.value)
```

---

## Sampling Strategies

Tracing everything is expensive. Sampling reduces overhead:

### Head-Based Sampling

Decision made at trace start, propagated to all spans.

```python
from opentelemetry.sdk.trace.sampling import (
    TraceIdRatioBased,
    ParentBased,
    ALWAYS_ON,
    ALWAYS_OFF
)

# Sample 10% of traces
sampler = TraceIdRatioBased(0.10)

# Respect parent's sampling decision
sampler = ParentBased(root=TraceIdRatioBased(0.10))

# Configuration
trace.set_tracer_provider(
    TracerProvider(sampler=sampler)
)
```

### Tail-Based Sampling

Decision made after trace completes, based on full trace data.

```
Collector receives all spans
         │
         ▼
┌─────────────────────┐
│  Tail-Based Sampler │
│                     │
│  Rules:             │
│  - Keep all errors  │
│  - Keep > 2s        │
│  - Keep 1% of rest  │
└─────────────────────┘
         │
         ▼
    Store/Discard

Pros:
- Can sample based on outcome (errors, latency)
- More intelligent decisions

Cons:
- Must buffer complete traces
- Higher resource usage
- More complex
```

### Adaptive Sampling

```python
class AdaptiveSampler:
    """Adjust sampling rate based on traffic volume"""
    
    def __init__(self, target_traces_per_second: int):
        self.target_tps = target_traces_per_second
        self.current_tps = 0
        self.sample_rate = 1.0
    
    def should_sample(self, trace_id: str) -> bool:
        # Adjust sample rate to hit target
        if self.current_tps > self.target_tps:
            self.sample_rate = max(0.01, self.sample_rate * 0.9)
        elif self.current_tps < self.target_tps * 0.8:
            self.sample_rate = min(1.0, self.sample_rate * 1.1)
        
        # Hash-based consistent sampling
        hash_value = hash(trace_id) % 100
        return hash_value < (self.sample_rate * 100)
```

---

## Tracing Systems Architecture

### OpenTelemetry Collector

```
┌─────────────────────────────────────────────────────────────────┐
│                    Applications                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │Service A│  │Service B│  │Service C│  │Service D│            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │                   │
└───────┼────────────┼────────────┼────────────┼───────────────────┘
        │            │            │            │
        └────────────┼────────────┼────────────┘
                     │            │
                     ▼            ▼
              ┌─────────────────────────┐
              │   OTel Collector        │
              │                         │
              │  ┌─────────────────┐    │
              │  │    Receivers    │    │  OTLP, Jaeger, Zipkin
              │  └────────┬────────┘    │
              │           │             │
              │  ┌────────▼────────┐    │
              │  │   Processors    │    │  Batch, Filter, Sample
              │  └────────┬────────┘    │
              │           │             │
              │  ┌────────▼────────┐    │
              │  │    Exporters    │    │  Jaeger, Tempo, X-Ray
              │  └─────────────────┘    │
              └───────────┬─────────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
      ┌─────────┐   ┌─────────┐   ┌─────────┐
      │  Jaeger │   │ Grafana │   │ AWS     │
      │         │   │  Tempo  │   │ X-Ray   │
      └─────────┘   └─────────┘   └─────────┘
```

### Collector Configuration

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024
  
  # Filter out health check spans
  filter:
    spans:
      exclude:
        match_type: regexp
        attributes:
          - key: http.url
            value: .*/health.*
  
  # Tail-based sampling
  tail_sampling:
    decision_wait: 10s
    policies:
      - name: errors
        type: status_code
        status_code: {status_codes: [ERROR]}
      - name: slow-traces
        type: latency
        latency: {threshold_ms: 2000}
      - name: percentage
        type: probabilistic
        probabilistic: {sampling_percentage: 10}

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true
  
  otlp:
    endpoint: tempo:4317

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, filter, tail_sampling]
      exporters: [jaeger, otlp]
```

---

## Best Practices

### Span Naming

```python
# BAD - Too specific, causes cardinality explosion
span_name = f"GET /users/{user_id}"  # Millions of unique names

# GOOD - Parameterized
span_name = "GET /users/{id}"

# BAD - Too generic
span_name = "database_query"

# GOOD - Descriptive
span_name = "SELECT users by id"
```

### Useful Attributes

```python
# HTTP spans
span.set_attribute("http.method", "POST")
span.set_attribute("http.url", "/api/orders")
span.set_attribute("http.status_code", 200)
span.set_attribute("http.request_content_length", 1024)

# Database spans
span.set_attribute("db.system", "postgresql")
span.set_attribute("db.name", "users")
span.set_attribute("db.statement", "SELECT * FROM users WHERE id = ?")
span.set_attribute("db.operation", "SELECT")

# Business context
span.set_attribute("user.id", "123")
span.set_attribute("order.id", "ord_456")
span.set_attribute("tenant.id", "acme-corp")
```

### Error Handling

```python
with tracer.start_as_current_span("operation") as span:
    try:
        result = do_something()
    except ValidationError as e:
        # Expected error - set status but don't record exception
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    except Exception as e:
        # Unexpected error - record full exception
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(e)
        raise
```

### Correlation with Logs

```python
import logging
from opentelemetry import trace

class TraceContextFilter(logging.Filter):
    def filter(self, record):
        span = trace.get_current_span()
        ctx = span.get_span_context()
        
        record.trace_id = format(ctx.trace_id, '032x') if ctx.trace_id else None
        record.span_id = format(ctx.span_id, '016x') if ctx.span_id else None
        
        return True

# Log format includes trace context
formatter = logging.Formatter(
    '%(asctime)s [%(trace_id)s:%(span_id)s] %(levelname)s %(message)s'
)

# Now logs are correlated with traces
# 2024-01-01 10:00:00 [abc123...:def456...] INFO Processing order
```

---

## Analyzing Traces

### Finding Bottlenecks

```
Trace Timeline View:
├── API Gateway (total: 2000ms)
│   ├── Auth (50ms) ✓
│   ├── Get User (1500ms) ← BOTTLENECK
│   │   ├── Cache Lookup (5ms)
│   │   ├── DB Query (1400ms) ← ROOT CAUSE
│   │   └── Serialize (10ms)
│   └── Send Response (50ms)

Investigation:
1. DB Query taking 1400ms
2. Check db.statement attribute
3. Query: SELECT * FROM users WHERE email = ?
4. Missing index on email column!
```

### Trace Comparison

```
Normal Trace (200ms):                Slow Trace (5000ms):
├── Service A (50ms)                 ├── Service A (50ms)
│   └── Cache HIT (5ms)              │   └── Cache MISS (5ms)
├── Service B (100ms)                ├── Service B (4800ms) ← Different
│   └── DB Query (80ms)              │   ├── DB Query (80ms)
└── Service C (50ms)                 │   └── Retry x3 (4500ms) ← Retries!
                                     └── Service C (50ms)
```

---

## Trade-offs

| Aspect | Consideration |
|--------|---------------|
| Overhead | 1-5% latency, storage costs |
| Sampling | Miss important traces vs. cost |
| Cardinality | Too many unique tags = expensive |
| Completeness | Uninstrumented services break trace |
| Complexity | Learning curve, operational burden |

---

## References

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Google Dapper Paper](https://research.google/pubs/pub36356/)
- [Distributed Tracing in Practice](https://www.oreilly.com/library/view/distributed-tracing-in/9781492056621/)
