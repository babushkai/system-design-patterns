# Message Ordering

## TL;DR

Message ordering determines whether messages are delivered in the order they were sent. Options range from no ordering (best performance) to total ordering (worst performance). Most systems use partition-based ordering: messages with the same key are ordered, different keys may interleave. Choose based on business requirements—true total ordering is rarely needed and expensive.

---

## Why Ordering Matters

### The Problem

```
User actions (sent in this order):
  1. Create account
  2. Update profile
  3. Delete account

If delivered out of order:
  Delete arrives first → "Account not found" error
  Update arrives → Creates orphaned data
  Create arrives → Account exists again

Result: Corrupted state
```

### When Ordering Matters

```
Critical:
  - Financial transactions (credit before debit)
  - State machine transitions
  - Log aggregation
  - Replication

Less Critical:
  - Analytics events (can be reordered later)
  - Notifications (slight reorder OK)
  - Independent operations
```

---

## Ordering Levels

### No Ordering Guarantee

```
Messages may arrive in any order

Producer sends: A, B, C
Consumer sees:  C, A, B (any permutation)

Advantages:
  - Maximum throughput
  - Easy scaling
  - No coordination

Use when:
  - Operations are independent
  - Consumer can handle any order
```

### FIFO Within Producer

```
Each producer's messages arrive in order
Different producers may interleave

Producer 1: A1, B1, C1 → arrive in order
Producer 2: A2, B2, C2 → arrive in order

But overall: A1, A2, B1, C2, B2, C1 (interleaved)

Use when:
  - Events from same source must be ordered
  - Different sources are independent
```

### FIFO Within Partition/Key

```
Messages with same key are ordered
Different keys may interleave

Key=user1: login, update, logout → ordered
Key=user2: login, purchase → ordered

But: user1.login, user2.login, user1.update... (interleaved)

Most common approach
Kafka, SQS FIFO use this
```

### Total Ordering

```
ALL messages in strict global order

Send: A, B, C, D, E
Receive: A, B, C, D, E (exactly)

Requires:
  - Single partition/queue
  - Or distributed consensus

Expensive, limits throughput
Rarely truly needed
```

---

## Kafka Ordering

### Partition-Based

```
Topic with 3 partitions:
  Partition 0: [A, D, G]
  Partition 1: [B, E, H]
  Partition 2: [C, F, I]

Within partition: Strictly ordered
Across partitions: No ordering

Producer:
  - Key = null: Round-robin to partitions
  - Key = "user123": Hash to consistent partition
```

### Consumer Groups

```
Consumer Group A:
  Consumer 1 ← Partition 0
  Consumer 2 ← Partition 1
  Consumer 3 ← Partition 2

Each partition processed by one consumer
Ordering preserved within partition

If consumer fails:
  Partition reassigned
  Continues from last committed offset
```

### Ordering Guarantees

```python
# Producer: Same key = same partition = ordered
producer.send(topic='events', key='user123', value=event1)
producer.send(topic='events', key='user123', value=event2)
# event1 always before event2 for user123

# Consumer: Process in order
for message in consumer:
    process(message)
    consumer.commit()  # Commit offset
```

---

## SQS FIFO Ordering

### Message Group ID

```python
# Messages with same group ID are ordered
sqs.send_message(
    QueueUrl=queue_url,
    MessageBody='{"action": "create"}',
    MessageGroupId='user-123',
    MessageDeduplicationId='msg-001'
)

sqs.send_message(
    QueueUrl=queue_url,
    MessageBody='{"action": "update"}',
    MessageGroupId='user-123',
    MessageDeduplicationId='msg-002'
)

# Consumer receives in order for user-123
```

### Deduplication

```
FIFO queues deduplicate by:
  - MessageDeduplicationId (explicit)
  - Content hash (if content-based dedup enabled)

Window: 5 minutes
Same ID in window → message dropped
```

### Throughput Limits

```
Standard SQS: Unlimited throughput
FIFO SQS: 300 msg/sec (3000 with batching)

Per message group: 300 msg/sec max
Use multiple groups to scale
```

---

## Implementing Ordering

### Sequence Numbers

```python
class OrderedProducer:
    def __init__(self):
        self.sequence = {}  # key → last sequence
    
    def send(self, key, message):
        seq = self.sequence.get(key, 0) + 1
        self.sequence[key] = seq
        
        message['_seq'] = seq
        queue.send(key=key, message=message)

class OrderedConsumer:
    def __init__(self):
        self.expected_seq = {}  # key → expected next
        self.buffer = {}  # key → out-of-order messages
    
    def process(self, key, message):
        seq = message['_seq']
        expected = self.expected_seq.get(key, 1)
        
        if seq == expected:
            # In order - process
            handle(message)
            self.expected_seq[key] = seq + 1
            
            # Check buffer for next messages
            self.process_buffered(key)
        elif seq > expected:
            # Out of order - buffer
            self.buffer.setdefault(key, {})[seq] = message
        # seq < expected: Duplicate, ignore
```

### Resequencing Buffer

```
Incoming (out of order): 3, 1, 4, 2, 5

Buffer state:
  Receive 3: buffer=[3], wait for 1
  Receive 1: process 1, buffer=[3], wait for 2
  Receive 4: buffer=[3,4], wait for 2
  Receive 2: process 2,3,4, buffer=[], wait for 5
  Receive 5: process 5

Considerations:
  - Buffer size limit
  - Timeout for missing sequences
  - Gap detection
```

### Handling Gaps

```python
def handle_potential_gap(key, expected, received):
    gap_start = expected
    gap_end = received - 1
    
    # Wait for gap to fill
    wait_until = time.time() + GAP_TIMEOUT
    
    while time.time() < wait_until:
        if gap_filled(key, gap_start, gap_end):
            return True
        sleep(0.1)
    
    # Gap timeout - decide action
    if GAP_POLICY == 'skip':
        log.warn(f"Skipping gap {gap_start}-{gap_end}")
        return True
    elif GAP_POLICY == 'fail':
        raise GapError(f"Gap detected: {gap_start}-{gap_end}")
```

---

## Scaling with Ordering

### Partition Strategies

```
By entity ID:
  user-123 → partition 0
  user-456 → partition 1
  All events for user-123 ordered ✓

By time bucket:
  Events 00:00-00:05 → partition 0
  Events 00:05-00:10 → partition 1
  Time-ordered within bucket

By hash:
  hash(key) % num_partitions
  Uniform distribution
```

### Increasing Partitions

```
Initial: 4 partitions
  Key A → partition 1
  Key B → partition 3

After adding partitions: 8 partitions
  Key A → partition 5 (different!)
  Key B → partition 3 (might change)

Problem: Key-partition mapping changes

Solutions:
  - Over-partition initially (100+ partitions)
  - Use consistent hashing
  - Coordinate partition increase with consumers
```

### Parallel Processing Limits

```
Strictly ordered queue:
  Max parallelism = number of keys
  
  1000 unique keys = 1000 parallel operations
  
If single key has high volume:
  That key becomes bottleneck
  Consider time-windowing or sub-keys
```

---

## Common Patterns

### Ordered by Entity

```python
# All events for an entity go to same partition
def get_partition_key(event):
    return event.entity_id

# Examples:
# Order events → key = order_id
# User events → key = user_id
# Session events → key = session_id
```

### Ordered by Causality

```
If event B depends on event A:
  Use same partition key

User creates order → Order events
  Key for both: order_id
  Creation before updates guaranteed

But: User profile update doesn't need order ordering
  Different partition key OK
```

### Hybrid Ordering

```
Critical path: FIFO queue (ordered, slower)
Best-effort: Standard queue (fast, unordered)

Create/Update/Delete → FIFO (order matters)
Analytics events → Standard (order doesn't matter)
```

---

## Trade-offs

| Ordering Level | Throughput | Latency | Complexity |
|----------------|------------|---------|------------|
| None | Highest | Lowest | Lowest |
| Per-producer | High | Low | Low |
| Per-key | Medium | Medium | Medium |
| Total | Lowest | Highest | Highest |

### Decision Framework

```
Question 1: Do messages affect shared state?
  No → No ordering needed
  Yes → Continue

Question 2: Is state partitioned by key?
  Yes → Per-key ordering sufficient
  No → Continue

Question 3: Is total ordering truly required?
  Usually no → Reconsider design
  Yes → Accept performance penalty
```

---

## Debugging Ordering Issues

### Out-of-Order Detection

```python
def detect_out_of_order(messages):
    issues = []
    last_seq = {}
    
    for msg in messages:
        key = msg.partition_key
        seq = msg.sequence
        
        if key in last_seq:
            if seq <= last_seq[key]:
                issues.append({
                    'key': key,
                    'expected': last_seq[key] + 1,
                    'got': seq
                })
        
        last_seq[key] = seq
    
    return issues
```

### Logging for Ordering

```python
logger.info(f"Received message",
    extra={
        'message_id': msg.id,
        'partition': msg.partition,
        'offset': msg.offset,
        'key': msg.key,
        'sequence': msg.sequence,
        'timestamp': msg.timestamp
    }
)

# Enables post-hoc ordering analysis
```

---

## Key Takeaways

1. **Total ordering is expensive** - Avoid unless truly needed
2. **Per-key ordering is usually enough** - Partition by entity ID
3. **Same key → same partition → same consumer** - Ordering chain
4. **Sequence numbers enable verification** - Detect gaps and duplicates
5. **Buffer out-of-order messages** - With timeout for gaps
6. **More partitions = more parallelism** - But per-partition ordering
7. **Scaling affects key mapping** - Over-partition initially
8. **Design for independent keys** - Maximize parallelism
