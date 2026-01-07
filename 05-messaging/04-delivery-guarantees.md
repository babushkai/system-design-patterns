# Delivery Guarantees

## TL;DR

Delivery guarantees define how many times a message will be delivered: at-most-once (may lose), at-least-once (may duplicate), exactly-once (ideal but hard). True exactly-once is extremely difficult; most systems achieve it through at-least-once + idempotent consumers. Understand the guarantees of your messaging system and design consumers accordingly.

---

## The Three Guarantees

### At-Most-Once

```
Message delivered 0 or 1 time

Send ──► Broker ──► Consumer
           │
     (no retry on failure)

Possible outcomes:
  ✓ Delivered once
  ✗ Never delivered (lost)

Never duplicated
May be lost
```

### At-Least-Once

```
Message delivered 1 or more times

Send ──► Broker ──► Consumer
           │           │
     (retry if no ack) │
           │◄────(ack)─┘

Possible outcomes:
  ✓ Delivered once
  ✓ Delivered multiple times (retries)

Never lost (if producer retries)
May be duplicated
```

### Exactly-Once

```
Message delivered exactly 1 time

Requires:
  - Deduplication at broker or consumer
  - Transactional processing
  - Or: At-least-once + idempotency

Ideal but extremely difficult
Often simulated rather than true
```

---

## Failure Scenarios

### Producer Failures

```
Scenario 1: Message lost before broker
  Producer ──X──► Broker
  
  At-most-once: Lost
  At-least-once: Lost (unless producer retries)

Scenario 2: Ack lost
  Producer ──► Broker ──X──► Producer
  
  At-most-once: Producer thinks it failed, doesn't retry
  At-least-once: Producer retries, duplicate at broker
```

### Broker Failures

```
Scenario: Broker crashes after receive, before persist

  Producer ──► Broker (memory) ──X── (disk)
  
  At-most-once: Message lost
  At-least-once: Producer retries (if no ack received)

Solution: Sync to disk before ack, or replicate first
```

### Consumer Failures

```
Scenario: Consumer crashes after processing, before ack

  Broker ──► Consumer (processed) ──X── (ack)
  
  At-most-once: N/A (no ack expected)
  At-least-once: Broker redelivers, processed twice
```

---

## Implementing At-Most-Once

### Fire and Forget

```python
# Producer: Don't wait for ack
producer.send(message)
# Continue immediately, don't care if it arrived

# Consumer: Auto-ack before processing
def consume():
    message = queue.get(auto_ack=True)  # Ack immediately
    process(message)  # If this fails, message lost
```

### Use Cases

```
✓ Metrics and telemetry (loss OK)
✓ Logging (best effort)
✓ Real-time displays (stale data acceptable)
✗ Financial transactions
✗ State changes
✗ Anything requiring reliability
```

---

## Implementing At-Least-Once

### Producer Retries

```python
def send_with_retry(message, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Wait for broker acknowledgment
            ack = producer.send(message, timeout=5000)
            if ack.success:
                return True
        except TimeoutError:
            if attempt < max_retries - 1:
                sleep(exponential_backoff(attempt))
    
    raise MessageDeliveryError("Failed after retries")
```

### Consumer Ack After Processing

```python
def consume():
    while True:
        message = queue.get(auto_ack=False)
        
        try:
            process(message)
            queue.ack(message)  # Only ack after success
        except Exception as e:
            queue.nack(message)  # Requeue for retry
            log.error(f"Processing failed: {e}")
```

### Handling Duplicates

```python
# Consumer must be idempotent
def process(message):
    message_id = message.id
    
    # Check if already processed
    if redis.sismember('processed_messages', message_id):
        log.info(f"Duplicate message {message_id}, skipping")
        return
    
    # Process
    do_work(message)
    
    # Mark as processed
    redis.sadd('processed_messages', message_id)
    redis.expire('processed_messages', 86400)  # 24h TTL
```

---

## Implementing Exactly-Once

### Approach 1: Deduplication

```python
class DeduplicatingConsumer:
    def __init__(self):
        self.seen = set()  # Or external store
    
    def process(self, message):
        if message.id in self.seen:
            return  # Skip duplicate
        
        do_work(message)
        self.seen.add(message.id)

# Limitation: Seen set must persist, has memory limits
```

### Approach 2: Idempotent Operations

```python
# Instead of: counter += 1
# Use: counter = specific_value

# Instead of: INSERT
# Use: UPSERT

def process_payment(payment):
    # Idempotent: Same payment_id always results in same state
    db.execute("""
        INSERT INTO payments (id, amount, status)
        VALUES (%s, %s, 'completed')
        ON CONFLICT (id) DO NOTHING
    """, payment.id, payment.amount)
```

### Approach 3: Transactional Outbox

```python
def process(message):
    with db.transaction():
        # Check if processed
        if is_processed(message.id):
            return
        
        # Do work
        update_state(message)
        
        # Mark processed (same transaction)
        mark_processed(message.id)
    
    # Only ack after transaction commits
    queue.ack(message)
```

### Approach 4: Kafka Transactions

```python
producer.init_transactions()

try:
    producer.begin_transaction()
    
    # Consume
    records = consumer.poll()
    
    # Process and produce
    for record in records:
        result = process(record)
        producer.send(output_topic, result)
    
    # Commit offsets and produced messages atomically
    producer.send_offsets_to_transaction(
        consumer.position(), 
        consumer_group
    )
    producer.commit_transaction()
    
except Exception:
    producer.abort_transaction()
```

---

## Kafka Delivery Semantics

### Producer Settings

```python
# At-most-once
producer = KafkaProducer(
    acks=0  # Don't wait for ack
)

# At-least-once
producer = KafkaProducer(
    acks='all',  # Wait for all replicas
    retries=3,
    retry_backoff_ms=100
)

# Exactly-once (idempotent producer)
producer = KafkaProducer(
    acks='all',
    enable_idempotence=True,  # Broker deduplicates
    transactional_id='my-producer'  # For transactions
)
```

### Consumer Settings

```python
# At-most-once
consumer = KafkaConsumer(
    enable_auto_commit=True,
    auto_commit_interval_ms=100  # Commit often
)

# At-least-once
consumer = KafkaConsumer(
    enable_auto_commit=False  # Manual commit after processing
)

# Exactly-once (with transactions)
consumer = KafkaConsumer(
    isolation_level='read_committed'  # Only see committed
)
```

---

## RabbitMQ Delivery Semantics

### Publisher Confirms

```python
# At-least-once with publisher confirms
channel.confirm_delivery()

try:
    channel.basic_publish(
        exchange='',
        routing_key='queue',
        body=message,
        properties=pika.BasicProperties(delivery_mode=2)  # Persistent
    )
except pika.exceptions.UnroutableError:
    # Message was not delivered
    handle_failure()
```

### Consumer Acknowledgments

```python
# At-least-once
def callback(ch, method, properties, body):
    try:
        process(body)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception:
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

channel.basic_consume(queue='queue', on_message_callback=callback)
```

---

## SQS Delivery Semantics

### Standard Queue

```
At-least-once delivery
Best-effort ordering

Messages may be delivered more than once
Order not guaranteed
High throughput
```

### FIFO Queue

```
Exactly-once processing
Strict ordering (within message group)

Deduplication by:
  - MessageDeduplicationId (5-minute window)
  - Content-based (hash of body)

Lower throughput (300-3000 msg/sec)
```

### Visibility Timeout

```python
# Message invisible while processing
sqs.receive_message(
    QueueUrl=queue_url,
    VisibilityTimeout=30  # seconds
)

# If not deleted within 30s, becomes visible again
# Another consumer might process it (duplicate)

# After processing:
sqs.delete_message(
    QueueUrl=queue_url,
    ReceiptHandle=receipt_handle
)
```

---

## Testing Delivery Guarantees

### Chaos Testing

```python
def test_at_least_once():
    # Send message
    message_id = producer.send(message)
    
    # Kill consumer mid-processing
    consumer.start()
    wait_for_processing_start()
    consumer.kill()
    
    # Restart consumer
    consumer.start()
    
    # Verify message processed (possibly twice)
    assert is_processed(message_id)

def test_no_message_loss():
    # Send many messages
    sent_ids = [producer.send(m) for m in messages]
    
    # Process all
    process_until_empty()
    
    # Verify all processed
    for id in sent_ids:
        assert is_processed(id)
```

### Duplicate Detection Testing

```python
def test_duplicate_handling():
    message = create_message()
    
    # Send same message twice
    producer.send(message)
    producer.send(message)
    
    # Process
    process_all()
    
    # Verify processed only once
    assert get_process_count(message.id) == 1
```

---

## Choosing a Guarantee

### Decision Matrix

| Requirement | Guarantee |
|-------------|-----------|
| Maximum throughput, loss OK | At-most-once |
| No message loss | At-least-once |
| No duplicates | Exactly-once or idempotent |
| Financial transactions | Exactly-once preferred |
| Event logging | At-least-once |
| Metrics | At-most-once OK |

### Cost Comparison

| Guarantee | Latency | Throughput | Complexity |
|-----------|---------|------------|------------|
| At-most-once | Lowest | Highest | Lowest |
| At-least-once | Medium | Medium | Medium |
| Exactly-once | Highest | Lowest | Highest |

---

## Key Takeaways

1. **At-most-once is fastest** - But may lose messages
2. **At-least-once is most common** - Requires idempotent consumers
3. **Exactly-once is hard** - Usually simulated via deduplication
4. **Ack after processing** - Not before
5. **Idempotency is your friend** - Makes duplicates harmless
6. **Test failure scenarios** - Crash consumers, drop acks
7. **Know your system's guarantees** - Kafka vs SQS vs RabbitMQ differ
8. **Design for duplicates** - They will happen
