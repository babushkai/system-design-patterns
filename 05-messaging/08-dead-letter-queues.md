# Dead Letter Queues

## TL;DR

Dead Letter Queues (DLQ) capture messages that cannot be processed successfully. Instead of losing failed messages or blocking the queue, they're moved to a separate queue for investigation. Essential for debugging, compliance, and preventing data loss. Configure retry limits, monitor DLQ depth, and establish procedures for handling dead letters.

---

## Why Dead Letter Queues?

### The Problem

```
Message arrives → Processing fails → What now?

Options without DLQ:
  1. Retry forever (blocks queue)
  2. Discard (lose data)
  3. Crash consumer (disrupts service)

None are good!
```

### DLQ Solution

```
Message arrives → Processing fails → Retry N times → Move to DLQ

Main Queue ──► Consumer ──► Success
                  │
              Failure (after retries)
                  │
                  ▼
              Dead Letter Queue
                  │
                  ▼
          Manual investigation
```

---

## How DLQs Work

### Basic Flow

```
1. Consumer receives message
2. Processing fails
3. Message returned to queue (nack)
4. Retry counter incremented
5. After N retries, move to DLQ
6. Original queue continues processing
7. DLQ monitored and investigated
```

### Message Metadata

```json
{
  "original_message": {
    "body": "...",
    "headers": {...}
  },
  "dlq_metadata": {
    "original_queue": "orders",
    "failure_reason": "ValidationError: Invalid product ID",
    "failure_timestamp": "2024-01-15T10:30:00Z",
    "retry_count": 3,
    "stack_trace": "..."
  }
}
```

---

## Configuration

### RabbitMQ

```python
# Declare DLQ
channel.queue_declare(
    queue='orders-dlq',
    durable=True
)

# Declare main queue with DLQ binding
channel.queue_declare(
    queue='orders',
    durable=True,
    arguments={
        'x-dead-letter-exchange': '',
        'x-dead-letter-routing-key': 'orders-dlq',
        'x-message-ttl': 60000,  # Optional: TTL before DLQ
        'x-max-retries': 3  # Requires plugin
    }
)
```

### Amazon SQS

```python
import boto3

sqs = boto3.client('sqs')

# Create DLQ
dlq = sqs.create_queue(QueueName='orders-dlq')
dlq_arn = sqs.get_queue_attributes(
    QueueUrl=dlq['QueueUrl'],
    AttributeNames=['QueueArn']
)['Attributes']['QueueArn']

# Create main queue with redrive policy
sqs.create_queue(
    QueueName='orders',
    Attributes={
        'RedrivePolicy': json.dumps({
            'deadLetterTargetArn': dlq_arn,
            'maxReceiveCount': '3'  # After 3 failures → DLQ
        })
    }
)
```

### Kafka

```python
# Kafka doesn't have native DLQ
# Implement in consumer

from kafka import KafkaConsumer, KafkaProducer

consumer = KafkaConsumer('orders')
producer = KafkaProducer()

for message in consumer:
    try:
        process(message)
    except Exception as e:
        # Send to DLQ topic
        producer.send(
            'orders-dlq',
            key=message.key,
            value=message.value,
            headers=[
                ('original-topic', b'orders'),
                ('failure-reason', str(e).encode()),
                ('retry-count', str(get_retry_count(message)).encode())
            ]
        )
        consumer.commit()
```

---

## Retry Strategies

### Immediate Retry

```python
def process_with_retry(message, max_retries=3):
    for attempt in range(max_retries):
        try:
            process(message)
            return True
        except RetryableError:
            if attempt < max_retries - 1:
                continue
    
    # Max retries exceeded
    send_to_dlq(message)
    return False
```

### Exponential Backoff

```python
def process_with_backoff(message, max_retries=3):
    for attempt in range(max_retries):
        try:
            process(message)
            return True
        except RetryableError:
            if attempt < max_retries - 1:
                delay = min(2 ** attempt, 60)  # Cap at 60 seconds
                sleep(delay)
    
    send_to_dlq(message)
    return False
```

### Delayed Retry Queue

```
Instead of immediate retry, use delay queue

Main Queue → Failure → Delay Queue (5 min) → Main Queue

Delay Queue implementation:
  - Message TTL + DLQ routing back to main queue
  - Or: Scheduled re-delivery
```

---

## Handling Dead Letters

### Investigation Workflow

```
1. Alert on DLQ messages
2. View message content and failure reason
3. Determine root cause
   - Bug in consumer?
   - Invalid message format?
   - External dependency failure?
4. Fix root cause
5. Replay or discard messages
```

### Message Inspection

```python
def inspect_dlq():
    messages = dlq.receive_messages(max_messages=10)
    
    for msg in messages:
        print(f"Message ID: {msg.id}")
        print(f"Failed at: {msg.attributes['failure_timestamp']}")
        print(f"Reason: {msg.attributes['failure_reason']}")
        print(f"Retry count: {msg.attributes['retry_count']}")
        print(f"Body: {msg.body}")
        print("---")
```

### Replay Messages

```python
def replay_dlq_messages():
    """Move messages from DLQ back to main queue"""
    while True:
        messages = dlq.receive_messages(max_messages=10)
        if not messages:
            break
        
        for msg in messages:
            # Send back to original queue
            main_queue.send(
                body=msg.body,
                headers=msg.headers
            )
            
            # Delete from DLQ
            dlq.delete(msg)
    
    log.info("DLQ replay complete")
```

### Selective Replay

```python
def replay_if_fixed(message):
    """Only replay if we've fixed the issue"""
    
    failure_reason = message.attributes['failure_reason']
    
    if "ValidationError" in failure_reason:
        # Skip - message itself is invalid
        archive_dlq_message(message)
    elif "ServiceUnavailable" in failure_reason:
        # Retry - service might be back
        replay_message(message)
    else:
        # Unknown - manual review
        flag_for_review(message)
```

---

## DLQ per Error Type

### Separate DLQs

```
orders-dlq-validation  → Invalid message format
orders-dlq-external    → External service failures
orders-dlq-unknown     → Unknown errors

Benefits:
  - Different handling per type
  - Easier investigation
  - Different retention policies
```

### Routing Implementation

```python
def send_to_appropriate_dlq(message, error):
    if isinstance(error, ValidationError):
        dlq = "orders-dlq-validation"
    elif isinstance(error, ExternalServiceError):
        dlq = "orders-dlq-external"
    else:
        dlq = "orders-dlq-unknown"
    
    send_to_dlq(dlq, message, error)
```

---

## Monitoring

### Key Metrics

```
DLQ depth:
  Number of messages in DLQ
  Should be near zero normally

DLQ arrival rate:
  Messages arriving per minute
  Spike indicates processing issue

DLQ age:
  Age of oldest message
  Stale messages indicate neglect

Failure categories:
  Breakdown by error type
  Identify systemic issues
```

### Alerting

```yaml
alerts:
  - name: DLQDepthHigh
    condition: dlq_depth > 100
    severity: warning
    
  - name: DLQDepthCritical
    condition: dlq_depth > 1000
    severity: critical
    
  - name: DLQArrivalSpike
    condition: dlq_arrival_rate > 10/min
    for: 5m
    
  - name: DLQStaleMessages
    condition: oldest_dlq_message_age > 24h
    severity: warning
```

### Dashboard

```
DLQ Dashboard:
  - Current depth (gauge)
  - Arrival rate (time series)
  - Top failure reasons (pie chart)
  - Age distribution (histogram)
  - Recent messages (table)
```

---

## Retention and Cleanup

### Retention Policy

```
Consider:
  - Compliance requirements (must keep N days)
  - Investigation time (allow time to debug)
  - Storage costs (don't keep forever)
  
Typical: 7-30 days
```

### Automatic Cleanup

```python
@scheduled(cron="0 0 * * *")  # Daily
def cleanup_old_dlq_messages():
    cutoff = now() - timedelta(days=30)
    
    while True:
        messages = dlq.receive_messages(
            max_messages=100,
            attributes=['sent_timestamp']
        )
        
        if not messages:
            break
        
        for msg in messages:
            if msg.sent_timestamp < cutoff:
                archive_message(msg)  # Optional: archive first
                dlq.delete(msg)
```

### Archive Before Delete

```python
def archive_message(message):
    s3.put_object(
        Bucket='dlq-archive',
        Key=f'{date.today()}/{message.id}.json',
        Body=json.dumps({
            'body': message.body,
            'attributes': message.attributes
        })
    )
```

---

## Common Patterns

### Poison Message Detection

```python
# Message that always fails - detect and sideline quickly

def process_with_poison_detection(message):
    retry_count = get_retry_count(message)
    
    if retry_count > 10:
        # Poison message - don't even try
        send_to_poison_queue(message)
        return
    
    try:
        process(message)
    except Exception as e:
        increment_retry_count(message)
        if retry_count >= 3:
            send_to_dlq(message, e)
        else:
            requeue(message)
```

### DLQ Consumer

```python
# Dedicated service to process DLQ

class DLQConsumer:
    def run(self):
        for message in self.dlq:
            try:
                self.handle_dead_letter(message)
            except Exception:
                # Even DLQ processing can fail!
                log.exception(f"Failed to handle DLQ message: {message.id}")
    
    def handle_dead_letter(self, message):
        # Attempt auto-fix
        if self.can_auto_fix(message):
            fixed = self.auto_fix(message)
            self.main_queue.send(fixed)
            self.dlq.delete(message)
        else:
            # Create ticket for manual review
            self.create_ticket(message)
```

### Circuit Breaker Integration

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def call_external_service(data):
    return external_api.process(data)

def process_message(message):
    try:
        result = call_external_service(message.body)
        return result
    except CircuitBreakerError:
        # Service unhealthy - delay processing
        delay_message(message, seconds=300)
        raise
```

---

## Key Takeaways

1. **DLQs prevent message loss** - Failed messages preserved
2. **Configure retry limits** - Don't retry forever
3. **Include failure metadata** - Reason, timestamp, retry count
4. **Monitor DLQ depth** - Alert on accumulation
5. **Establish handling procedures** - Investigation, replay, archive
6. **Different DLQs for different errors** - Easier categorization
7. **Retention policies matter** - Compliance and storage costs
8. **Automate where possible** - Replay, cleanup, alerting
