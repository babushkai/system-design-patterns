# Message Queues

## TL;DR

Message queues decouple producers and consumers, enabling asynchronous communication, load leveling, and fault tolerance. Key concepts: producers, consumers, brokers, acknowledgments. Choose based on ordering needs, throughput, durability, and exactly-once requirements. Popular options: RabbitMQ, Amazon SQS, Apache Kafka (log-based).

---

## Why Message Queues?

### Synchronous Problems

```
Direct HTTP call:
  Service A ──HTTP──► Service B
  
Problems:
  - A waits for B (latency)
  - If B is down, A fails
  - Spikes in A overwhelm B
  - Tight coupling
```

### Queue Benefits

```
With message queue:
  Service A ──► [Queue] ──► Service B

Benefits:
  - A doesn't wait (async)
  - If B is down, messages wait in queue
  - Queue absorbs traffic spikes
  - A and B don't know about each other
```

---

## Core Concepts

### Components

```
┌──────────┐     ┌───────────┐     ┌──────────┐
│ Producer │────►│   Queue   │────►│ Consumer │
│          │     │  (Broker) │     │          │
└──────────┘     └───────────┘     └──────────┘

Producer: Creates and sends messages
Queue/Broker: Stores messages durably
Consumer: Receives and processes messages
```

### Message Lifecycle

```
1. Producer sends message
2. Broker acknowledges receipt (producer-side)
3. Broker stores message durably
4. Consumer fetches message
5. Consumer processes message
6. Consumer acknowledges (consumer-side)
7. Broker removes message
```

### Message Structure

```json
{
  "id": "msg-12345",
  "timestamp": "2024-01-15T10:30:00Z",
  "headers": {
    "content-type": "application/json",
    "correlation-id": "req-67890"
  },
  "body": {
    "user_id": 123,
    "action": "signup",
    "email": "user@example.com"
  }
}
```

---

## Queue Types

### Point-to-Point

```
One message, one consumer

Producer ──► [Queue] ──► Consumer A
                    └──► Consumer B  (different message)
                    
Each message delivered to exactly one consumer
Used for: Task distribution, work queues
```

### Publish-Subscribe

```
One message, many consumers

Producer ──► [Topic] ──┬──► Consumer A (copy)
                       ├──► Consumer B (copy)
                       └──► Consumer C (copy)
                       
Each message delivered to all subscribers
Used for: Event broadcasting, notifications
```

### Competing Consumers

```
Multiple consumers share work from one queue

             ┌──► Consumer 1
Producer ──► [Queue] ──┼──► Consumer 2
             └──► Consumer 3

Each message goes to one consumer
Consumers process in parallel
Used for: Load distribution, scaling
```

---

## Delivery Semantics

### At-Most-Once

```
Producer: Send message, don't wait for ack
Consumer: Process, don't ack

Possible outcomes:
  - Message delivered and processed ✓
  - Message lost (never delivered) ✗
  
Use case: Metrics, logs where loss is acceptable
```

### At-Least-Once

```
Producer: Send, retry until acked
Consumer: Process, then ack

Possible outcomes:
  - Message delivered once ✓
  - Message delivered multiple times (retry after timeout)
  
Consumer must be idempotent!
Use case: Most applications
```

### Exactly-Once

```
Very hard to achieve truly
Usually: At-least-once + idempotent consumer

Techniques:
  - Deduplication by message ID
  - Transactional outbox
  - Kafka transactions

Use case: Financial transactions
```

---

## Acknowledgments

### Producer Acknowledgments

```python
# Fire and forget (at-most-once)
producer.send(message)

# Wait for broker ack (at-least-once)
producer.send(message).get()  # Blocks until acked

# Wait for replication (stronger durability)
producer.send(message, acks='all').get()
```

### Consumer Acknowledgments

```python
# Auto-ack (dangerous - message may be lost)
message = queue.get(auto_ack=True)
process(message)  # If this fails, message lost

# Manual ack (safer)
message = queue.get(auto_ack=False)
try:
    process(message)
    queue.ack(message)
except Exception:
    queue.nack(message)  # Requeue or dead letter
```

### Ack Timeout

```
Consumer gets message at T=0
Timeout = 30 seconds

If no ack by T=30:
  Broker assumes consumer died
  Message redelivered to another consumer
  
Choose timeout > max processing time
```

---

## Queue Patterns

### Work Queue

```python
# Producer: Distribute tasks
for task in tasks:
    queue.send(task)

# Consumers: Process in parallel
while True:
    task = queue.get()
    result = process(task)
    queue.ack(task)
```

### Request-Reply

```
Request queue: client → service
Reply queue: service → client

Client:
  1. Create temp reply queue
  2. Send request with reply_to = temp queue
  3. Wait on temp queue

Service:
  1. Get request from request queue
  2. Process
  3. Send response to reply_to queue
```

### Priority Queue

```python
# High priority messages processed first
queue.send(critical_task, priority=10)
queue.send(normal_task, priority=5)
queue.send(low_task, priority=1)

# Consumer always gets highest priority first
```

---

## Popular Message Queues

### RabbitMQ

```
Protocol: AMQP
Model: Broker-centric, exchanges + queues

Features:
  - Flexible routing (direct, topic, fanout)
  - Message TTL
  - Dead letter exchanges
  - Plugins ecosystem

Best for: Complex routing, enterprise messaging
```

### Amazon SQS

```
Model: Managed queue service

Standard Queue:
  - At-least-once delivery
  - Best-effort ordering
  - Unlimited throughput

FIFO Queue:
  - Exactly-once processing
  - Strict ordering (within group)
  - 3,000 msg/sec limit

Best for: AWS-native, managed simplicity
```

### Apache Kafka

```
Model: Distributed log

Features:
  - Persistent storage (replay)
  - Partitioned for parallelism
  - Consumer groups
  - High throughput

Best for: Event streaming, large scale
```

### Redis Streams

```
Model: Append-only log in Redis

Features:
  - Consumer groups
  - Message IDs
  - Trimming by size/time
  - Fast (in-memory)

Best for: Simple streaming, already using Redis
```

---

## Sizing and Capacity

### Throughput Planning

```
Expected load:
  Peak messages: 10,000/sec
  Avg message size: 1 KB
  Retention: 7 days

Calculations:
  Throughput: 10,000 × 1 KB = 10 MB/sec
  Daily storage: 10 MB/sec × 86,400 = 864 GB/day
  Total storage: 864 × 7 = 6 TB
```

### Consumer Scaling

```
Processing time per message: 100ms
Required throughput: 1,000 msg/sec

Consumers needed:
  1000 msg/sec × 0.1 sec = 100 concurrent
  With 10 consumers: 10 parallel each
  
Add consumers until throughput met
```

---

## Monitoring

### Key Metrics

```
Queue depth:
  Number of messages waiting
  Growing = consumers too slow

Consumer lag:
  Time/messages behind producer
  Growing = falling behind

Message age:
  Oldest unprocessed message
  High = potential SLA breach

Throughput:
  Messages/second in and out
  Compare to capacity

Error rate:
  Failed processing / total
  Trigger alerts > threshold
```

### Alerting Rules

```yaml
alerts:
  - name: QueueDepthHigh
    condition: queue_depth > 10000
    for: 5m
    
  - name: ConsumerLagHigh
    condition: consumer_lag > 1h
    for: 10m
    
  - name: MessageAgeOld
    condition: oldest_message_age > 30m
    for: 5m
    
  - name: ProcessingErrors
    condition: error_rate > 1%
    for: 5m
```

---

## Error Handling

### Retry Strategies

```python
def process_with_retry(message, max_retries=3):
    for attempt in range(max_retries):
        try:
            process(message)
            return True
        except TransientError:
            delay = exponential_backoff(attempt)
            time.sleep(delay)
        except PermanentError:
            send_to_dead_letter(message)
            return False
    
    # Max retries exceeded
    send_to_dead_letter(message)
    return False
```

### Dead Letter Queue

```
Main Queue ──► Consumer ──► Success
                  │
                  └──► Failure (after retries)
                         │
                         ▼
                  Dead Letter Queue
                         │
                         ▼
               Manual review / alerting
```

---

## Best Practices

### Message Design

```
1. Include correlation ID for tracing
2. Add timestamp for debugging
3. Keep messages small (< 256 KB typically)
4. Use schema versioning
5. Include message type for routing
```

### Idempotent Consumers

```python
def process_message(message):
    # Check if already processed
    if is_processed(message.id):
        return  # Skip duplicate
    
    # Process
    result = do_work(message)
    
    # Mark as processed (atomically with work if possible)
    mark_processed(message.id)
```

### Graceful Shutdown

```python
def shutdown_handler(signal, frame):
    # Stop accepting new messages
    consumer.stop_consuming()
    
    # Wait for in-flight messages
    consumer.wait_for_current()
    
    # Cleanup
    consumer.close()
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown_handler)
```

---

## Key Takeaways

1. **Queues decouple systems** - Async, resilient, scalable
2. **At-least-once is common** - Requires idempotent consumers
3. **Ack after processing** - Not before
4. **Monitor queue depth** - Early warning of problems
5. **Use dead letter queues** - Handle permanent failures
6. **Size for peak load** - Queues absorb spikes
7. **Plan message schema** - Include ID, timestamp, type
8. **Graceful shutdown** - Don't lose in-flight messages
