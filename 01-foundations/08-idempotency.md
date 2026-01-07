# Idempotency

## TL;DR

An operation is idempotent if executing it multiple times produces the same result as executing it once. In distributed systems with retries, timeouts, and partial failures, idempotency prevents duplicate effects. Implement idempotency using idempotency keys, deduplication, and careful API design. Without idempotency, retries can cause double-charges, duplicate emails, and corrupt data.

---

## Why Idempotency Matters

### The Fundamental Problem

```
Client → Server: "Charge $100"
Server: Process payment ✓
Server → Client: "Success"
[network drops response]
Client: No response, retry?

Client → Server: "Charge $100" (retry)
Server: Process payment ✓ (again!)

Result: Customer charged $200 for one purchase
```

### When Retries Happen

- Network timeout (response lost)
- Client crashed, restarted, retries
- Load balancer retry on backend failure
- Message queue redelivery
- User double-click
- Kubernetes pod restart during request

**Assume every operation will be executed multiple times.**

---

## Idempotent vs Non-Idempotent Operations

### Naturally Idempotent

```
SET x = 5          ✓ Idempotent (same result every time)
DELETE user:123    ✓ Idempotent (already deleted = no-op)
PUT /users/123     ✓ Idempotent (replace entire resource)
GET /users/123     ✓ Idempotent (read-only)
```

### NOT Naturally Idempotent

```
x = x + 1          ✗ Each execution adds 1
INSERT row         ✗ Creates duplicate rows
POST /orders       ✗ Creates new order each time
send_email()       ✗ Sends email each time
charge_card()      ✗ Charges each time
```

### HTTP Methods

| Method | Idempotent? | Notes |
|--------|-------------|-------|
| GET | Yes | Read-only |
| HEAD | Yes | Read-only |
| PUT | Yes | Replace entire resource |
| DELETE | Yes | Delete is idempotent |
| OPTIONS | Yes | Read-only |
| POST | **No** | Creates new resource |
| PATCH | **No** | May not be idempotent |

---

## Implementing Idempotency

### Pattern 1: Idempotency Keys

Client generates unique key for each logical operation.

```
Request 1:
  POST /payments
  Idempotency-Key: abc123
  Body: {amount: 100}
  
  Server: Process payment, store key
  Response: 201 Created

Request 2 (retry, same key):
  POST /payments
  Idempotency-Key: abc123
  Body: {amount: 100}
  
  Server: Key exists, return cached response
  Response: 201 Created (same as before, no new payment)
```

**Storage schema:**
```sql
CREATE TABLE idempotency_keys (
  key VARCHAR(255) PRIMARY KEY,
  request_hash VARCHAR(64),
  response_code INT,
  response_body JSONB,
  created_at TIMESTAMP,
  expires_at TIMESTAMP
);
```

### Pattern 2: Request Deduplication

Server detects and ignores duplicates.

```
// Message queue consumer
func process_message(msg):
  if seen_before(msg.id):
    return ack()  // Already processed
  
  process(msg)
  mark_seen(msg.id)
  return ack()
```

**Deduplication storage:**
```
// Simple: in-memory set with TTL
seen_ids = ExpiringSet(ttl=24h)

// Scalable: Bloom filter (probabilistic)
// False positives OK (skip legitimate message)
// False negatives NOT OK (never miss duplicate)
bloom_filter.add(msg_id)
if bloom_filter.contains(msg_id): skip
```

### Pattern 3: Conditional Operations

Make non-idempotent operations conditional.

```sql
-- Instead of: UPDATE balance SET amount = amount - 100
-- Use conditional update:

UPDATE balance 
SET amount = amount - 100, version = version + 1
WHERE user_id = 123 AND version = 5;

-- If version changed (already processed), 0 rows affected
```

```
// Compare-and-swap style
func transfer(from, to, amount, expected_version):
  if from.version != expected_version:
    return AlreadyProcessed
  
  from.balance -= amount
  to.balance += amount
  from.version += 1
```

### Pattern 4: Natural Idempotency Keys

Use business identifiers that are naturally unique.

```
// Payment for order 12345
// Order can only be paid once
// Order ID is the idempotency key

func pay_order(order_id, amount):
  order = get_order(order_id)
  if order.payment_status == 'paid':
    return order.payment  // Already done
  
  payment = process_payment(amount)
  order.payment_status = 'paid'
  order.payment = payment
  return payment
```

---

## Idempotency Key Design

### Key Generation

```
// UUID - universally unique
key = uuid.v4()  // "550e8400-e29b-41d4-a716-446655440000"

// ULID - sortable, timestamp-based
key = ulid.new()  // "01ARZ3NDEKTSV4RRFFQ69G5FAV"

// Composite - include context
key = f"{user_id}:{action}:{timestamp}:{nonce}"
```

### Key Scope

```
Narrow scope (per-action):
  key = "create-order-{uuid}"
  
Broad scope (per-request):
  key = "request-{uuid}"  // Covers entire request
  
Semantic scope (per-intent):
  key = "user:123:pay-invoice:456"  // One payment per invoice
```

### Key Storage Considerations

```
Questions:
  - How long to store? (TTL)
  - What to store? (Key only? Full response?)
  - Where to store? (DB, cache, both?)
  - Consistency with operation? (Same transaction?)

Typical answers:
  - TTL: 24 hours to 7 days
  - Store: Key + response code + response body
  - Where: Database (durable) with cache layer
  - Transaction: Same transaction as operation
```

---

## Handling Concurrent Requests

### The Race Condition

```
Time →
Request A (key=X):  [────lock────][process][store]
Request B (key=X):       [──lock──][process][store]
                              ↑
                    Both got through!
```

### Solution: Lock Before Check

```
func process_with_idempotency(key, request):
  // Acquire lock on key
  lock = acquire_lock(key, timeout=30s)
  
  try:
    // Check if already processed
    existing = lookup(key)
    if existing:
      return existing.response
    
    // Process the request
    response = do_work(request)
    
    // Store result
    store(key, response)
    return response
  finally:
    release_lock(lock)
```

### Database-Level Locking

```sql
-- Use advisory lock
SELECT pg_advisory_lock(hashtext('idempotency:' || key));

-- Or use unique constraint
INSERT INTO idempotency_keys (key, status)
VALUES ('abc123', 'processing')
ON CONFLICT (key) DO NOTHING
RETURNING *;

-- If no rows returned, another request is processing
```

---

## Idempotency at Different Layers

### API Layer

```
Client:
  Include Idempotency-Key header
  Retry with same key on failure

Server:
  Check key before processing
  Store response with key
  Return cached response on duplicate
```

### Message Queue Layer

```
Producer:
  Include unique message ID
  
Consumer:
  Track processed message IDs
  Skip duplicates
  
Queue (built-in):
  Some queues deduplicate (SQS FIFO, Kafka exactly-once)
```

### Database Layer

```
-- Use UPSERT for idempotent writes
INSERT INTO events (id, data)
VALUES ('event-123', '{"type": "click"}')
ON CONFLICT (id) DO NOTHING;

-- Use optimistic locking
UPDATE accounts
SET balance = balance - 100, version = version + 1
WHERE id = 'acc-123' AND version = 5;
```

### Application Layer

```
// State machine prevents duplicate transitions
func complete_order(order_id):
  order = get_order(order_id)
  
  match order.status:
    'pending' -> 
      process()
      order.status = 'completed'
    'completed' ->
      return ok()  // Already done
    'cancelled' ->
      return error("Cannot complete cancelled order")
```

---

## Real-World Examples

### Stripe Payments

```http
POST /v1/charges
Idempotency-Key: unique-charge-key-123
Content-Type: application/json

{
  "amount": 1000,
  "currency": "usd",
  "source": "tok_visa"
}
```

- Keys stored for 24 hours
- Same key + same parameters = cached response
- Same key + different parameters = error
- Retries are safe

### AWS SQS FIFO

```
Message:
  MessageDeduplicationId: "unique-id-123"
  MessageGroupId: "group-1"

SQS deduplicates messages with same ID within 5-minute window
```

### Kafka Exactly-Once

```
Producer:
  enable.idempotence = true
  transactional.id = "my-producer-1"

Broker:
  Tracks producer sequence numbers
  Rejects duplicate messages
  Supports transactions across partitions
```

---

## Common Pitfalls

### Pitfall 1: Storing Key After Processing

```
// WRONG
response = process(request)
store_key(key, response)  // Crash here = key not stored, will retry

// RIGHT
begin_transaction()
store_key(key, 'processing')
response = process(request)
update_key(key, response)
commit_transaction()
```

### Pitfall 2: Not Validating Request

```
Request 1: POST /pay {amount: 100, key: "abc"}
Request 2: POST /pay {amount: 200, key: "abc"}  // Same key, different amount!

// WRONG: Just return cached response
// RIGHT: Return error - request mismatch

func check_idempotency(key, request):
  existing = lookup(key)
  if existing:
    if hash(request) != existing.request_hash:
      return error("Request mismatch for idempotency key")
    return existing.response
```

### Pitfall 3: Side Effects Outside Transaction

```
// WRONG
begin_transaction()
  create_order()
commit_transaction()
send_email()  // If this fails after retry, email sent twice

// RIGHT
begin_transaction()
  create_order()
  queue_email()  // Idempotent queue with dedup
commit_transaction()
// Email worker handles deduplication
```

### Pitfall 4: Using Timestamps as Keys

```
// WRONG
key = f"user:{user_id}:payment:{timestamp}"
// Clock skew, timing variance = different keys for retry

// RIGHT
key = f"user:{user_id}:payment:{client_generated_uuid}"
// Client generates consistent key
```

---

## Testing Idempotency

### Unit Tests

```python
def test_idempotent_charge():
  key = "test-key-123"
  
  # First request
  response1 = charge(amount=100, key=key)
  assert response1.status == "success"
  
  # Duplicate request (retry)
  response2 = charge(amount=100, key=key)
  assert response2.status == "success"
  assert response1.charge_id == response2.charge_id
  
  # Only charged once
  assert get_total_charges() == 100
```

### Integration Tests

```python
def test_concurrent_idempotent_requests():
  key = "concurrent-key"
  
  # Send 10 concurrent requests with same key
  responses = parallel_execute([
    lambda: charge(100, key) for _ in range(10)
  ])
  
  # All should return same response
  charge_ids = set(r.charge_id for r in responses)
  assert len(charge_ids) == 1
  
  # Only one charge created
  assert count_charges() == 1
```

### Chaos Testing

```
1. Start operation
2. Kill process mid-operation
3. Restart and retry
4. Verify single execution

Test scenarios:
- Crash before processing
- Crash during processing
- Crash after processing, before response
- Network timeout (response lost)
```

---

## Key Takeaways

1. **Assume multiple executions** - Network is unreliable, retries will happen
2. **Use idempotency keys** - Client-generated, unique per logical operation
3. **Store before processing** - Prevent race conditions
4. **Same transaction** - Key storage and operation atomically
5. **Validate request match** - Same key must have same parameters
6. **Handle side effects** - Queue, deduplicate, or make idempotent
7. **Set appropriate TTL** - Balance storage vs. retry window
8. **Test explicitly** - Concurrent requests, crash scenarios
