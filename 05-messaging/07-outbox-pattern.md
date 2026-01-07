# Outbox Pattern

## TL;DR

The outbox pattern ensures reliable message publishing by writing messages to a database table (outbox) in the same transaction as business data. A separate process reads from the outbox and publishes to the message broker. This guarantees atomicity between database writes and message publishing, solving the dual-write problem.

---

## The Dual-Write Problem

### Naive Approach

```python
def create_order(order):
    # Step 1: Save to database
    db.save(order)
    
    # Step 2: Publish event
    message_queue.publish(OrderCreated(order))
```

### Failure Scenarios

```
Scenario 1: DB succeeds, publish fails
  db.save(order)     ✓ (committed)
  mq.publish(event)  ✗ (failed)
  
  Result: Order exists, but no event
  Downstream systems never know

Scenario 2: Publish succeeds, DB fails
  db.save(order)     (pending)
  mq.publish(event)  ✓ (published)
  db.commit()        ✗ (rolled back)
  
  Result: Event exists, but no order
  Downstream systems process phantom order
```

### Why Distributed Transactions Don't Help

```
XA/2PC:
  - Not supported by most message brokers
  - Slow (blocks on coordinator)
  - Complex failure handling
  
Need simpler, more reliable approach
```

---

## The Outbox Solution

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Database                           │
│  ┌─────────────┐    ┌─────────────────────────────┐    │
│  │   Orders    │    │         Outbox              │    │
│  │  ┌───────┐  │    │  ┌─────────────────────┐    │    │
│  │  │ Order │  │◄───┼──│ id, payload, status │    │    │
│  │  └───────┘  │    │  └─────────────────────┘    │    │
│  └─────────────┘    └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
         ▲                      │
         │                      │ Poll
         │                      ▼
┌────────┴──────┐    ┌─────────────────────┐    ┌────────┐
│  Application  │    │ Outbox Publisher    │───►│ Broker │
└───────────────┘    └─────────────────────┘    └────────┘
```

### How It Works

```
1. Application writes business data AND outbox record
   in SAME transaction

2. Transaction commits atomically
   Both order and outbox record exist, or neither

3. Background process polls outbox
   Reads unpublished messages

4. Publisher sends to message broker
   Message delivered to queue/topic

5. Publisher marks outbox record as published
   Prevents duplicate publishing
```

---

## Implementation

### Outbox Table Schema

```sql
CREATE TABLE outbox (
    id UUID PRIMARY KEY,
    aggregate_type VARCHAR(255) NOT NULL,
    aggregate_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(255) NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    published_at TIMESTAMP NULL,
    
    INDEX idx_outbox_unpublished (published_at) WHERE published_at IS NULL
);
```

### Writing to Outbox

```python
def create_order(order_data):
    with db.transaction():
        # Create order
        order = Order(**order_data)
        db.add(order)
        
        # Write to outbox (same transaction)
        outbox_entry = OutboxEntry(
            id=uuid4(),
            aggregate_type="Order",
            aggregate_id=str(order.id),
            event_type="OrderCreated",
            payload=json.dumps({
                "order_id": str(order.id),
                "customer_id": order.customer_id,
                "total": order.total
            })
        )
        db.add(outbox_entry)
    
    # Transaction commits atomically
    return order
```

### Outbox Publisher (Polling)

```python
class OutboxPublisher:
    def __init__(self, db, broker):
        self.db = db
        self.broker = broker
    
    def run(self):
        while True:
            self.publish_pending()
            sleep(100)  # Poll interval
    
    def publish_pending(self):
        # Get unpublished messages
        entries = self.db.query("""
            SELECT * FROM outbox 
            WHERE published_at IS NULL 
            ORDER BY created_at 
            LIMIT 100
            FOR UPDATE SKIP LOCKED
        """)
        
        for entry in entries:
            try:
                # Publish to broker
                self.broker.publish(
                    topic=f"{entry.aggregate_type}.{entry.event_type}",
                    message=entry.payload,
                    headers={"event_id": str(entry.id)}
                )
                
                # Mark as published
                self.db.execute("""
                    UPDATE outbox 
                    SET published_at = NOW() 
                    WHERE id = %s
                """, entry.id)
                
            except BrokerError:
                # Will retry on next poll
                log.error(f"Failed to publish {entry.id}")
```

---

## CDC-Based Outbox

### Using Change Data Capture

```
Instead of polling, use database log

Database ──► CDC (Debezium) ──► Kafka

Outbox table changes captured from binlog/WAL
Lower latency than polling
No separate publisher process
```

### Debezium Configuration

```json
{
  "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
  "database.hostname": "db.example.com",
  "database.dbname": "myapp",
  "table.include.list": "public.outbox",
  "transforms": "outbox",
  "transforms.outbox.type": "io.debezium.transforms.outbox.EventRouter",
  "transforms.outbox.table.field.event.type": "event_type",
  "transforms.outbox.table.field.event.payload": "payload"
}
```

### CDC Benefits

```
+ Lower latency (near real-time)
+ No polling load on database
+ Guaranteed ordering (from log)
+ No missed messages

- More infrastructure (Debezium, Kafka Connect)
- CDC setup complexity
- Database must support log access
```

---

## Handling Duplicates

### Why Duplicates Happen

```
Scenario:
  1. Publisher reads message from outbox
  2. Publisher sends to broker ✓
  3. Publisher crashes before marking published
  4. New publisher instance starts
  5. Same message published again

Consumer receives duplicate message
```

### Idempotent Consumers

```python
class OrderEventConsumer:
    def handle(self, event):
        event_id = event.headers["event_id"]
        
        # Check if already processed
        if self.is_processed(event_id):
            log.info(f"Duplicate event {event_id}, skipping")
            return
        
        # Process event
        self.process(event)
        
        # Mark as processed
        self.mark_processed(event_id)
    
    def is_processed(self, event_id):
        return redis.sismember("processed_events", event_id)
    
    def mark_processed(self, event_id):
        redis.sadd("processed_events", event_id)
        redis.expire("processed_events", 86400)  # 24h
```

### Transactional Deduplication

```python
def handle(event):
    event_id = event.headers["event_id"]
    
    with db.transaction():
        # Try to insert processing record
        try:
            db.execute("""
                INSERT INTO processed_events (event_id, processed_at)
                VALUES (%s, NOW())
            """, event_id)
        except UniqueViolation:
            # Already processed
            return
        
        # Process event (same transaction)
        process(event)
```

---

## Ordering Guarantees

### Per-Aggregate Ordering

```sql
-- Outbox entries ordered by aggregate
SELECT * FROM outbox 
WHERE published_at IS NULL 
ORDER BY aggregate_id, created_at
FOR UPDATE SKIP LOCKED
```

### Partition by Aggregate

```python
def publish(entry):
    broker.publish(
        topic="order-events",
        key=entry.aggregate_id,  # Same aggregate → same partition
        value=entry.payload
    )
```

### Handling Out-of-Order

```
If strict ordering required:
  1. Single publisher per aggregate type
  2. Or: Sequence numbers in messages
  3. Or: Consumer reordering buffer
```

---

## Cleanup Strategies

### Delete After Publishing

```python
# Immediately delete after successful publish
db.execute("DELETE FROM outbox WHERE id = %s", entry.id)
```

### Soft Delete with Cleanup

```python
# Mark as published
db.execute("""
    UPDATE outbox SET published_at = NOW() WHERE id = %s
""", entry.id)

# Separate cleanup job
@scheduled(cron="0 * * * *")  # Hourly
def cleanup_outbox():
    db.execute("""
        DELETE FROM outbox 
        WHERE published_at < NOW() - INTERVAL '7 days'
    """)
```

### Archive Before Delete

```python
@scheduled(cron="0 0 * * *")  # Daily
def archive_outbox():
    # Move to archive table
    db.execute("""
        INSERT INTO outbox_archive
        SELECT * FROM outbox 
        WHERE published_at < NOW() - INTERVAL '7 days'
    """)
    
    # Delete from main table
    db.execute("""
        DELETE FROM outbox 
        WHERE published_at < NOW() - INTERVAL '7 days'
    """)
```

---

## Monitoring

### Key Metrics

```
Outbox lag:
  Count of unpublished messages
  Should stay low

Publish latency:
  Time from created_at to published_at
  Indicates processing speed

Publish failures:
  Rate of failed publish attempts
  Indicates broker issues

Outbox size:
  Total table size
  Should be bounded
```

### Alerting

```yaml
alerts:
  - name: OutboxLagHigh
    condition: count(unpublished) > 1000
    for: 5m
    
  - name: OutboxLatencyHigh
    condition: avg(publish_latency) > 30s
    for: 5m
    
  - name: OutboxPublishFailing
    condition: publish_error_rate > 0.01
    for: 5m
```

### Health Check

```python
def outbox_health():
    oldest_unpublished = db.query("""
        SELECT MIN(created_at) 
        FROM outbox 
        WHERE published_at IS NULL
    """)
    
    if oldest_unpublished:
        age = now() - oldest_unpublished
        if age > timedelta(minutes=5):
            return Health.DEGRADED
    
    return Health.HEALTHY
```

---

## Variations

### Inbox Pattern (Idempotent Consumer)

```
Mirror of outbox for consumers

Message arrives → Write to inbox → Process → Mark processed

Inbox table:
  id, message_id, payload, processed_at

Guarantees idempotency at consumer
```

### Transactional Inbox

```python
def handle_message(message):
    with db.transaction():
        # Check/insert inbox record
        result = db.execute("""
            INSERT INTO inbox (message_id, received_at)
            VALUES (%s, NOW())
            ON CONFLICT (message_id) DO NOTHING
            RETURNING id
        """, message.id)
        
        if not result:
            return  # Already processed
        
        # Process in same transaction
        process(message)
```

---

## Key Takeaways

1. **Solves dual-write problem** - Atomic database + message
2. **Same transaction is key** - Business data + outbox together
3. **Polling or CDC** - Choose based on latency needs
4. **Duplicates will happen** - Consumers must be idempotent
5. **Order by aggregate** - Preserve per-entity ordering
6. **Clean up regularly** - Don't let outbox grow unbounded
7. **Monitor lag** - Detect publishing problems early
8. **Inbox for consumers** - Same pattern on receive side
