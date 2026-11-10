# 冪等性

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../01-foundations/08-idempotency.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

ある操作が冪等であるとは、それを複数回実行しても1回実行した場合と同じ結果になることを意味します。リトライ、タイムアウト、部分障害が存在する分散システムにおいて、冪等性は重複した副作用を防止します。冪等性キー、重複排除、慎重なAPI設計を用いて冪等性を実装してください。冪等性がなければ、リトライにより二重課金、重複メール、データ破損が発生する可能性があります。

---

## なぜ冪等性が重要なのか

### 根本的な問題

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

### リトライが発生する場面

- ネットワークタイムアウト（応答の喪失）
- クライアントのクラッシュ、再起動、リトライ
- ロードバランサーのバックエンド障害時のリトライ
- メッセージキューの再配信
- ユーザーのダブルクリック
- リクエスト処理中のKubernetes Podの再起動

**すべての操作は複数回実行されると想定してください。**

---

## 冪等な操作 vs 非冪等な操作

### 本質的に冪等

```
SET x = 5          ✓ Idempotent (same result every time)
DELETE user:123    ✓ Idempotent (already deleted = no-op)
PUT /users/123     ✓ Idempotent (replace entire resource)
GET /users/123     ✓ Idempotent (read-only)
```

### 本質的に非冪等

```
x = x + 1          ✗ Each execution adds 1
INSERT row         ✗ Creates duplicate rows
POST /orders       ✗ Creates new order each time
send_email()       ✗ Sends email each time
charge_card()      ✗ Charges each time
```

---

## 冪等性の実装

### パターン1：冪等性キー

クライアントが各論理操作に対してユニークなキーを生成します。

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

**ストレージスキーマ：**
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

### パターン2：リクエストの重複排除

サーバーが重複を検出して無視します。

```
// Message queue consumer
func process_message(msg):
  if seen_before(msg.id):
    return ack()  // Already processed

  process(msg)
  mark_seen(msg.id)
  return ack()
```

**重複排除ストレージ：**
```
// Simple: in-memory set with TTL
seen_ids = ExpiringSet(ttl=24h)

// Scalable: Bloom filter (probabilistic)
// False positives OK (skip legitimate message)
// False negatives NOT OK (never miss duplicate)
bloom_filter.add(msg_id)
if bloom_filter.contains(msg_id): skip
```

### パターン3：条件付き操作

非冪等な操作を条件付きにします。

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

### パターン4：自然な冪等性キー

本質的にユニークなビジネス識別子を使用します。

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

## 各レイヤーでの冪等性

```
API Layer:         Idempotency-Key header, response caching (see HTTP Idempotency Patterns)
Message Queue:     Producer assigns message ID, consumer dedup table, built-in dedup (SQS FIFO, Kafka)
Database Layer:    UPSERT, optimistic locking, ON CONFLICT (see Database-Level Idempotency)
Application Layer: State machines — only valid transitions execute, duplicates are no-ops
```

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

## 実世界の例

### Stripeの決済

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

- キーは24時間保存されます
- 同じキー + 同じパラメータ = キャッシュされたレスポンス
- 同じキー + 異なるパラメータ = エラー
- リトライは安全です

### AWS SQS FIFO

```
Message:
  MessageDeduplicationId: "unique-id-123"
  MessageGroupId: "group-1"

SQS deduplicates messages with same ID within 5-minute window
```

### Kafkaのexactly-once

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

## よくある落とし穴

### 落とし穴1：処理後にキーを保存する

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

### 落とし穴2：リクエストの検証をしない

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

### 落とし穴3：トランザクション外での副作用

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

### 落とし穴4：タイムスタンプをキーとして使用する

```
// WRONG
key = f"user:{user_id}:payment:{timestamp}"
// Clock skew, timing variance = different keys for retry

// RIGHT
key = f"user:{user_id}:payment:{client_generated_uuid}"
// Client generates consistent key
```

---

## 冪等性のテスト

### ユニットテスト

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

### 統合テスト

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

### カオステスト

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

## 冪等性キーの設計

### クライアント生成 vs サーバー生成のキー

```
Client-generated (recommended):
  Client creates UUID before sending request.
  On retry, client resends the same UUID.
  Server uses UUID to detect duplicates.

  ✓ Key survives response loss — client still has it
  ✓ Stripe, PayPal, Square all use this approach

Server-generated (problematic):
  Server creates key, returns it in response.
  If response is lost, client has no key to retry with.

  ✗ Defeats the purpose if the response never arrives
  ✗ Only works when the client can query for existing records
```

### キーフォーマットの選択肢

```
UUID v4 (random):
  "550e8400-e29b-41d4-a716-446655440000"
  ✓ No coordination needed
  ✗ Poor database index locality (random distribution)

UUID v7 (time-ordered, RFC 9562):
  "018f3e5c-7a1b-7000-8000-000000000001"
  ✓ Monotonically increasing — B-tree friendly
  ✓ Encodes creation timestamp
  ✓ Preferred for high-write idempotency tables

Composite key (domain-aware):
  key = sha256(f"{user_id}:{action}:{reference_id}")
  ✓ Deterministic — same intent always produces same key
  ✓ Natural deduplication without client tracking
  ✗ Requires careful design to avoid collisions
```

### キーの保存戦略

```
Same database as business data (transactional):
  BEGIN;
    INSERT INTO idempotency_keys (key, response) VALUES (...);
    INSERT INTO payments (id, amount) VALUES (...);
  COMMIT;
  ✓ Atomic — key and operation always consistent
  ✓ No split-brain between key store and data store

Redis (fast, ephemeral):
  SET idem:abc123 response_json EX 86400
  ✓ Sub-millisecond lookups
  ✗ Key can be lost on Redis restart (unless persistence is on)
  ✗ Not transactional with your main database

Dedicated dedup table:
  Separate table or service for idempotency keys.
  ✓ Clean separation of concerns
  ✗ Adds latency and a consistency boundary
```

### キーの有効期限

```
How long to keep processed keys:
  Stripe:        24 hours
  AWS SQS FIFO:  5 minutes
  Most REST APIs: 1–7 days

Formula:
  retention = max_retry_window × safety_factor

  Example: clients retry for up to 1 hour, safety_factor = 24
  retention = 1h × 24 = 24 hours

Trade-off:
  Short TTL → less storage, risk of duplicate processing on late retries
  Long TTL  → more storage, stronger guarantee against duplicates
```

### 同時重複リクエスト

```
Problem: Two requests with same key arrive at the same instant.
Both pass the "key not found" check before either inserts.

Solution 1 — Database unique constraint:
  INSERT INTO idempotency_keys (key, status)
  VALUES ('abc123', 'processing');
  -- Second insert fails with unique violation → return 409 or wait

Solution 2 — Distributed lock on the key:
  lock = redis.set("lock:abc123", owner, NX, EX=30)
  if not lock:
    wait_or_return_conflict()
```

---

## HTTPの冪等性パターン

### HTTPメソッドごとの本来の冪等性

```
GET    → Read-only, always idempotent by definition
PUT    → Full resource replacement — same payload = same result
DELETE → Deleting an already-deleted resource is a no-op (return 204 or 404)
POST   → Creates a new resource each time — NOT idempotent without explicit design
PATCH  → Depends on payload semantics — NOT guaranteed idempotent
```

### Idempotency-Keyヘッダー（Stripeパターン）

```http
POST /v1/payments HTTP/1.1
Idempotency-Key: 7c4a8d09-ca95-4c28-a1ad-8c3e2f5b3e72
Content-Type: application/json

{"amount": 5000, "currency": "usd"}
```

```
Server processing flow:
  1. Receive request with Idempotency-Key header
  2. Look up key in idempotency store
  3. If found and status = "completed" → return cached response
  4. If found and status = "processing" → return 409 Conflict (or wait)
  5. If not found → insert key with status "processing", execute operation
  6. On completion → update key with status "completed" and store full response
```

### 冪等リプレイのためのレスポンスキャッシング

```sql
-- Store the complete response alongside the key
UPDATE idempotency_keys
SET status = 'completed',
    response_code = 201,
    response_body = '{"id": "pay_abc", "amount": 5000}',
    completed_at = NOW()
WHERE key = '7c4a8d09-ca95-4c28-a1ad-8c3e2f5b3e72';

-- On retry, return the exact same response — status code and body
-- The client sees no difference between the original and the replay
```

### ETagによる条件付きリクエスト

```http
-- Client fetches resource with ETag
GET /users/123 HTTP/1.1
→ 200 OK
→ ETag: "v5"
→ {"name": "Alice", "email": "alice@example.com"}

-- Client updates with If-Match to prevent lost updates
PUT /users/123 HTTP/1.1
If-Match: "v5"
{"name": "Alice", "email": "alice@new.com"}

→ 200 OK (if version still v5)
→ 412 Precondition Failed (if another write changed it)
```

### PATCHを安全にする

```
PATCH is not naturally idempotent:
  PATCH /counter {"op": "increment"} → each call changes state

Make PATCH idempotent with versioning:
  PATCH /users/123
  If-Match: "v5"
  {"email": "new@example.com"}

  First call:  v5 matches → apply update, bump to v6
  Retry call:  v5 ≠ v6 → 412 Precondition Failed (client knows it already applied)
```

---

## データベースレベルの冪等性

### UPSERT / INSERT ON CONFLICT

```sql
-- Idempotent write in a single statement
INSERT INTO events (id, type, payload, created_at)
VALUES ('evt-001', 'order.created', '{"order_id": 42}', NOW())
ON CONFLICT (id) DO NOTHING;

-- Rows affected = 1 on first call, 0 on retry — no error, no duplicate

-- UPSERT variant: update if exists (useful for "last write wins")
INSERT INTO user_preferences (user_id, theme, updated_at)
VALUES (123, 'dark', NOW())
ON CONFLICT (user_id) DO UPDATE
SET theme = EXCLUDED.theme, updated_at = EXCLUDED.updated_at;
```

### アウトボックスパターンによるexactly-once処理

```sql
-- Process message and record it in the same transaction
BEGIN;
  -- Business logic
  UPDATE accounts SET balance = balance - 100 WHERE id = 'acc-123';

  -- Mark message as processed (dedup record)
  INSERT INTO processed_messages (message_id, processed_at)
  VALUES ('msg-789', NOW())
  ON CONFLICT (message_id) DO NOTHING;

  -- Queue outgoing event via outbox (see 05-messaging/07-outbox-pattern.md)
  INSERT INTO outbox (id, event_type, payload)
  VALUES ('out-456', 'balance.updated', '{"account": "acc-123"}');
COMMIT;

-- If message_id already exists, the ON CONFLICT makes the INSERT a no-op
-- The entire transaction is atomic — no partial processing
```

### 冪等なスキーマ設計：SET vs INCREMENT

```
Non-idempotent (INCREMENT semantics):
  UPDATE accounts SET balance = balance - 100 WHERE id = 'acc-123';
  -- Each execution subtracts another $100

Idempotent (SET semantics — final state):
  UPDATE accounts SET balance = 400 WHERE id = 'acc-123' AND version = 5;
  -- Repeated execution has no additional effect once version advances

Rule of thumb:
  ✓ SET balance = new_value        (idempotent)
  ✗ SET balance = balance - amount (not idempotent)

  Compute the final state in application code, then SET it.
  Pair with optimistic locking (version check) to prevent lost updates.
```

### 冪等な削除のためのトゥームストーンパターン

```sql
-- Soft-delete with timestamp — delete operation is always idempotent
UPDATE users
SET deleted_at = NOW()
WHERE id = 123 AND deleted_at IS NULL;

-- Re-deleting a deleted record: 0 rows affected, no error
-- Application treats deleted_at IS NOT NULL as "does not exist"

-- Advantage over hard DELETE:
--   Hard DELETE is idempotent too (deleting nothing is fine)
--   But tombstone preserves audit trail and enables undo
```

---

## 分散環境における冪等性の課題

### サービス間の冪等性

```
Scenario: Order service → Payment service → Notification service

  1. Payment service processes charge      ✓
  2. Notification service sends email      ✓
  3. Order service crashes before committing
  4. Retry: payment succeeds (dedup), notification sends AGAIN ✗

Problem: Each service saw a "new" request from its perspective.

Solution: Each service maintains its own idempotency/dedup table.
  Payment service:  dedup on payment_idempotency_key
  Notification svc: dedup on notification_id (derived from order_id + event_type)

  On retry, payment returns cached result AND notification skips the duplicate.
```

### 取り消せない副作用

```
Sending an email is not idempotent — you cannot "unsend" it.
Sending an SMS, calling a webhook, printing a receipt — same problem.

Approach: event-driven dedup at the side-effect layer.

  1. Business service writes an event: "send welcome email for user 123"
  2. Email service consumes the event
  3. Email service checks: have I already sent this? (dedup on event_id)
  4. If not sent → send and record event_id
  5. If already sent → ack and skip

  The dedup boundary is at the service that performs the irreversible action.
  See 05-messaging/04-delivery-guarantees.md for delivery guarantee details.
```

### クロックスキューとキーの有効期限

```
Scenario:
  Server A sets idempotency key TTL = 24 hours at T=0
  Client retries at T=23h59m
  Request routed to Server B whose clock is 5 minutes ahead
  Server B sees key as expired → processes request again → duplicate!

Mitigations:
  - Use generous expiration windows (add buffer beyond max retry window)
  - Synchronize clocks with NTP and monitor drift
  - Use logical timestamps (version numbers) instead of wall-clock TTLs where possible
  - Set expiration based on creation time from the key itself (UUID v7 encodes timestamp)
```

### 冪等性 vs exactly-onceセマンティクス

```
Idempotency:
  A property of an operation — calling it N times has the same effect as calling it once.
  It is a mechanism, a building block.

Exactly-once:
  A delivery guarantee — the message is processed exactly one time.
  It is an end-to-end guarantee, much harder to achieve.

Relationship:
  at-least-once delivery + idempotent processing = effectively exactly-once

  The network gives you at-least-once (retries ensure delivery).
  Your application adds idempotency (dedup ensures single processing).
  The combination behaves like exactly-once from the caller's perspective.

  See 05-messaging/04-delivery-guarantees.md for full treatment.
  See 02-distributed-databases/07-distributed-transactions.md for transactional guarantees.
```

---

## 重要なポイント

1. **複数回の実行を想定する** - ネットワークは信頼できず、リトライは必ず発生します
2. **冪等性キーを使用する** - クライアント生成で、論理操作ごとにユニークにします
3. **処理前に保存する** - 競合状態を防止します
4. **同一トランザクション内で処理する** - キーの保存と操作をアトミックに行います
5. **リクエストの一致を検証する** - 同じキーには同じパラメータが必要です
6. **副作用を適切に処理する** - キューイング、重複排除、または冪等にします
7. **適切なTTLを設定する** - ストレージとリトライウィンドウのバランスを取ります
8. **明示的にテストする** - 同時リクエスト、クラッシュシナリオ
