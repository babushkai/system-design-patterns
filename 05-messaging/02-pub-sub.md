# Publish-Subscribe (Pub/Sub)

## TL;DR

Pub/Sub decouples message producers from consumers through topics. Publishers send messages to topics without knowing subscribers. Subscribers receive copies of all messages from topics they subscribe to. Enables event-driven architectures, real-time updates, and loose coupling. Key considerations: fan-out cost, ordering, filtering, and backpressure.

---

## Core Concepts

### Architecture

```
             ┌─────────────────┐
             │      Topic      │
             │   "user.events" │
             └────────┬────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌────────┐       ┌────────┐       ┌────────┐
│ Sub A  │       │ Sub B  │       │ Sub C  │
│ Email  │       │Analytics│       │ Audit  │
│Service │       │Service │       │Service │
└────────┘       └────────┘       └────────┘

Publisher doesn't know about subscribers
Each subscriber gets a copy of every message
```

### vs Point-to-Point

```
Point-to-Point (Queue):
  Message ──► [Queue] ──► ONE consumer
  Work distribution

Pub/Sub (Topic):
  Message ──► [Topic] ──► ALL subscribers
  Event broadcasting
```

### Message Flow

```
1. Publisher sends event to topic
2. Broker stores message
3. Broker fans out to all subscribers
4. Each subscriber processes independently
5. Subscribers acknowledge individually
```

---

## Subscription Models

### Push vs Pull

**Push (Broker pushes to subscriber):**
```
Broker ──push──► Subscriber endpoint

Pros:
  - Low latency
  - Simple subscriber

Cons:
  - Subscriber must handle load
  - Need webhook endpoint
  
Example: Google Pub/Sub push subscriptions
```

**Pull (Subscriber pulls from broker):**
```
Subscriber ──pull──► Broker

Pros:
  - Subscriber controls rate
  - Works behind firewalls

Cons:
  - Polling overhead
  - Higher latency possible

Example: Kafka consumer groups
```

### Durable vs Ephemeral

**Durable Subscription:**
```
Subscriber disconnects at T=0
Messages arrive at T=1, T=2, T=3
Subscriber reconnects at T=4

Gets all messages (T=1, T=2, T=3)
Broker stored them during disconnect
```

**Ephemeral Subscription:**
```
Only receives messages while connected
Missed messages during disconnect are lost

Use for: Real-time displays, live updates
```

---

## Topic Design

### Hierarchical Topics

```
events.user.created
events.user.updated
events.user.deleted
events.order.placed
events.order.shipped

Wildcards:
  events.user.*     → All user events
  events.*.created  → All creation events
  events.#          → Everything under events
```

### Topic Naming Conventions

```
Pattern: <domain>.<entity>.<action>

Examples:
  payment.transaction.completed
  inventory.stock.low
  user.profile.updated
  
Benefits:
  - Clear ownership
  - Easy filtering
  - Logical grouping
```

### Single vs Multiple Topics

```
Single topic (events):
  All events in one place
  Consumers filter by type
  Simpler infrastructure

Multiple topics (events.user, events.order):
  Natural partitioning
  Subscribe to relevant topics only
  Better access control
  
Recommendation: Start with fewer topics, split when needed
```

---

## Fan-Out Patterns

### Simple Fan-Out

```
One message → N copies

Publisher ──► [Topic] ──┬──► Sub 1 (email)
                        ├──► Sub 2 (push)
                        └──► Sub 3 (analytics)

Each gets same message
Process independently
```

### Fan-Out with Filtering

```
Not all subscribers want all messages

Publisher ──► [Topic: user.events]
                   │
    Filter: type=signup ──► Email Service
    Filter: type=purchase ──► Analytics
    Filter: country=US ──► US Team
```

### Implementation

```python
# Google Cloud Pub/Sub with filter
subscriber.create_subscription(
    name="email-signups",
    topic="user-events",
    filter='attributes.type = "signup"'
)

# Kafka: Consumer reads all, filters in code
for message in consumer:
    if message.value['type'] == 'signup':
        process_signup(message)
```

---

## Ordering Guarantees

### No Ordering (Default)

```
Published: A, B, C
Subscriber 1 sees: B, A, C
Subscriber 2 sees: A, C, B

No guarantee between subscribers or even for one subscriber
```

### Per-Publisher Ordering

```
Publisher 1: A1, B1, C1 → Subscriber sees A1, B1, C1
Publisher 2: A2, B2, C2 → Subscriber sees A2, B2, C2

But A1 and A2 may interleave arbitrarily
```

### Partition-Based Ordering

```
Messages with same key → same partition → ordered

user_123 events: login, view, purchase
  All go to partition 3
  Subscriber sees in order

Different users may interleave
```

### Total Ordering

```
All messages in strict global order
Very expensive (single bottleneck)
Rarely needed
```

---

## Implementations

### Apache Kafka

```
Topics → Partitions → Consumer Groups

Producer ──► Topic ──► Partition 0 ──► Consumer Group A
                   ──► Partition 1     (each partition to one consumer)
                   ──► Partition 2

Features:
  - Log-based (replay possible)
  - Consumer groups for scaling
  - Ordered within partition
  - High throughput
```

### Google Cloud Pub/Sub

```
Topic → Subscriptions

Publisher ──► Topic ──► Subscription A ──► Subscriber 1
                    ──► Subscription B ──► Subscriber 2

Features:
  - Managed service
  - Push and pull
  - Message filtering
  - At-least-once (exactly-once preview)
```

### Amazon SNS + SQS

```
SNS Topic → SQS Queues

Publisher ──► SNS Topic ──► SQS Queue A ──► Consumer A
                       ──► SQS Queue B ──► Consumer B
                       ──► Lambda (direct)

Features:
  - SNS for fan-out
  - SQS for durability and processing
  - Multiple protocols (HTTP, email, SMS)
```

### Redis Pub/Sub

```
Simple in-memory pub/sub

PUBLISH user-events '{"type":"login"}'
SUBSCRIBE user-events

Features:
  - Very fast
  - No persistence (ephemeral)
  - No consumer groups
  - Good for real-time
```

---

## Handling Backpressure

### The Problem

```
Publisher: 10,000 msg/sec
Subscriber A: Can handle 10,000 msg/sec ✓
Subscriber B: Can handle 1,000 msg/sec ✗

Subscriber B falls behind
Queue grows unbounded
Eventually: OOM or dropped messages
```

### Solutions

**Per-Subscriber Queues:**
```
Topic ──► Queue A (for fast subscriber)
     ──► Queue B (for slow subscriber)

Each queue buffers independently
Slow subscriber doesn't affect fast one
```

**Backpressure Signals:**
```
Subscriber signals "slow down"
Broker reduces send rate
Or: Subscriber pulls at own pace
```

**Dead Letter after Timeout:**
```
Message unacked for > 1 hour
Move to dead letter queue
Alert and manual handling
```

---

## Exactly-Once Challenges

### Duplicate Delivery

```
Scenario:
  1. Broker sends message to subscriber
  2. Subscriber processes
  3. Ack lost in network
  4. Broker re-sends (thinks it failed)
  5. Subscriber processes again

Result: Processed twice
```

### Solutions

```
1. Idempotent subscriber
   Track processed message IDs
   Skip if already seen

2. Transactional processing
   Process + ack in same transaction
   (Not always possible)

3. Deduplication at broker
   Broker tracks delivered message IDs
   (Limited time window)
```

---

## Event Schema Evolution

### The Challenge

```
Version 1:
  {user_id: 123, name: "Alice"}

Version 2 (add field):
  {user_id: 123, name: "Alice", email: "..."}

Old subscribers must handle new fields
New subscribers must handle missing fields
```

### Best Practices

```
1. Only add optional fields
2. Never remove or rename fields
3. Use schema registry
4. Version in message (or use schema ID)

{
  "schema_version": 2,
  "user_id": 123,
  "name": "Alice",
  "email": "alice@example.com"  // Optional
}
```

### Schema Registry

```
Publisher:
  1. Register schema with registry
  2. Get schema ID
  3. Include schema ID in message

Subscriber:
  1. Get schema ID from message
  2. Fetch schema from registry
  3. Deserialize with correct schema
```

---

## Use Cases

### Event-Driven Architecture

```
User signs up
  → UserCreated event published

Subscribed services:
  - Email service: Send welcome email
  - Analytics: Track signup
  - Billing: Create account
  - Recommendations: Initialize profile

Services evolve independently
Add new subscriber without changing publisher
```

### Real-Time Updates

```
Stock price changes
  → PriceUpdated event

Subscribers:
  - Trading dashboards (WebSocket push)
  - Alert service (check thresholds)
  - Historical database (record)
```

### Log Aggregation

```
All services ──► Log topic ──► Aggregator ──► Elasticsearch
                          ──► Metrics ──► Prometheus
                          ──► Archive ──► S3
```

---

## Monitoring

### Key Metrics

```
Publication rate:
  Messages published per second

Fan-out latency:
  Time from publish to subscriber receive

Subscriber lag:
  Messages pending per subscription

Acknowledgment rate:
  Acks per second (subscriber health)

Dead letter rate:
  Failed messages per time
```

### Health Checks

```python
def check_pubsub_health():
    # Check broker connectivity
    assert can_connect_to_broker()
    
    # Check subscription lag
    for sub in subscriptions:
        lag = get_subscription_lag(sub)
        if lag > threshold:
            alert(f"Subscription {sub} lagging: {lag}")
    
    # Check dead letter queue
    dlq_size = get_dlq_size()
    if dlq_size > 0:
        alert(f"Dead letter queue has {dlq_size} messages")
```

---

## Key Takeaways

1. **Pub/Sub decouples producers from consumers** - Neither knows the other
2. **Each subscriber gets every message** - Fan-out pattern
3. **Durable subscriptions survive disconnection** - Messages queued
4. **Ordering is expensive** - Use partition keys when needed
5. **Backpressure is critical** - Slow subscribers can cause problems
6. **Idempotency handles duplicates** - At-least-once is common
7. **Schema evolution needs planning** - Use registry, add-only changes
8. **Monitor subscriber lag** - Early warning of processing issues
