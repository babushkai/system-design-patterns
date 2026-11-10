# 配信保証

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../05-messaging/04-delivery-guarantees.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

配信保証は、メッセージが何回配信されるかを定義します。At-most-once（損失の可能性あり）、At-least-once（重複の可能性あり）、Exactly-once（理想的だが困難）の3種類があります。真のExactly-onceは極めて難しく、ほとんどのシステムはAt-least-once + 冪等なコンシューマーで実現します。メッセージングシステムの保証を理解し、コンシューマーを適切に設計してください。

---

## 3つの保証

### At-Most-Once（最大1回）

```
メッセージは0回または1回配信される

Send ──► Broker ──► Consumer
           │
     (失敗時にリトライしない)

起こり得る結果:
  ✓ 1回配信される
  ✗ 配信されない（損失）

重複することはない
損失する可能性がある
```

### At-Least-Once（最低1回）

```
メッセージは1回以上配信される

Send ──► Broker ──► Consumer
           │           │
     (ackがない場合リトライ) │
           │◄────(ack)─┘

起こり得る結果:
  ✓ 1回配信される
  ✓ 複数回配信される（リトライ）

損失しない（プロデューサーがリトライする場合）
重複する可能性がある
```

### Exactly-Once（正確に1回）

```
メッセージが正確に1回配信される

必要条件:
  - ブローカーまたはコンシューマーでの重複排除
  - トランザクショナル処理
  - または: At-least-once + 冪等性

理想的だが極めて困難
真の実現よりシミュレーションされることが多い
```

---

## 障害シナリオ

### プロデューサーの障害

```
シナリオ1: ブローカー到達前にメッセージ損失
  Producer ──X──► Broker

  At-most-once: 損失
  At-least-once: 損失（プロデューサーがリトライしない限り）

シナリオ2: Ackの損失
  Producer ──► Broker ──X──► Producer

  At-most-once: プロデューサーは失敗と判断し、リトライしない
  At-least-once: プロデューサーがリトライし、ブローカーで重複
```

### ブローカーの障害

```
シナリオ: ブローカーが受信後、永続化前にクラッシュ

  Producer ──► Broker (memory) ──X── (disk)

  At-most-once: メッセージ損失
  At-least-once: プロデューサーがリトライ（ackが受信されない場合）

解決策: ack前にディスクに同期、またはまずレプリケート
```

### コンシューマーの障害

```
シナリオ: コンシューマーが処理後、ack前にクラッシュ

  Broker ──► Consumer (processed) ──X── (ack)

  At-most-once: N/A（ackは期待されない）
  At-least-once: ブローカーが再配信し、2回処理される
```

---

## At-Most-Onceの実装

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

### ユースケース

```
✓ メトリクスとテレメトリ（損失OK）
✓ ロギング（ベストエフォート）
✓ リアルタイム表示（古いデータが許容）
✗ 金融取引
✗ 状態変更
✗ 信頼性が必要なもの
```

---

## At-Least-Onceの実装

### プロデューサーのリトライ

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

### 処理後のコンシューマーAck

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

### 重複の処理

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

## Exactly-Onceの実装

### アプローチ1: 重複排除

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

### アプローチ2: 冪等な操作

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

### アプローチ3: トランザクショナルアウトボックス

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

### アプローチ4: Kafkaトランザクション

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

## Kafkaの配信セマンティクス

### プロデューサー設定

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

### コンシューマー設定

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

## RabbitMQの配信セマンティクス

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

### コンシューマーのアクノリッジメント

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

## SQSの配信セマンティクス

### Standard Queue

```
At-least-once配信
ベストエフォートの順序保証

メッセージが複数回配信される可能性がある
順序は保証されない
高スループット
```

### FIFO Queue

```
Exactly-once処理
厳密な順序保証（メッセージグループ内）

以下による重複排除:
  - MessageDeduplicationId（5分ウィンドウ）
  - コンテンツベース（ボディのハッシュ）

低スループット（300-3000 msg/sec）
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

## 配信保証のテスト

### カオステスト

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

### 重複検出テスト

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

## 保証の選択

### 決定マトリクス

| 要件 | 保証 |
|------|------|
| 最大スループット、損失OK | At-most-once |
| メッセージ損失なし | At-least-once |
| 重複なし | Exactly-onceまたは冪等 |
| 金融取引 | Exactly-once推奨 |
| イベントロギング | At-least-once |
| メトリクス | At-most-onceでOK |

### コスト比較

| 保証 | レイテンシ | スループット | 複雑さ |
|------|---------|------------|--------|
| At-most-once | 最低 | 最高 | 最低 |
| At-least-once | 中 | 中 | 中 |
| Exactly-once | 最高 | 最低 | 最高 |

---

## Kafka冪等プロデューサーの内部構造

```
アプリケーションレベルのロジックなしでブローカーがどのように重複排除するか:

1. PIDの割り当て
   Producer.init() ──► ブローカーがPID（Producer ID）を割り当てる

2. シーケンスタグ付け
   すべてのProducerRecordが(PID, partition, sequence_number)を持つ
   シーケンスはパーティションごとに0から始まりインクリメントする

3. ブローカー側の重複排除
   ブローカーが保持: Map<(PID, partition), last_committed_sequence>

   受信メッセージのsequence ≤ last_committed → DUPLICATE、拒否
   受信メッセージのsequence  = last_committed + 1 → ACCEPT
   受信メッセージのsequence  > last_committed + 1 → OUT_OF_ORDER、エラー
```

**セッションスコープの制限**: PIDはエフェメラルで、`Producer.init()`で割り当てられます。プロデューサープロセスが再起動すると新しいPIDが取得されます。ブローカーは新しいPIDと古いPIDを関連付けられないため、重複排除は単一のプロデューサーセッション内でのみ機能します。

**`transactional.id`による再起動の存続**: `transactional.id`を設定すると、トランザクションコーディネーターが`transactional.id → (PID, epoch)`のマッピングを永続化します。再起動時にプロデューサーが`initTransactions()`を呼び出すと、コーディネーターが既存のPIDを検索（または新しいPIDを割り当ててepochをインクリメント）し、同じ`transactional.id`でまだ実行中の古いプロデューサーインスタンスをフェンスします。

```
# Config (default since Kafka 3.0)
enable.idempotence=true    # Implies acks=all, retries=MAX_INT, max.in.flight.requests.per.connection ≤ 5

# What it costs you: ~2-3% throughput reduction (extra sequence bookkeeping)
# What it gives you: no duplicates from producer retries within one session
```

**重要な注意点**: 冪等性だけでは、consume-transform-produceパイプライン全体でexactly-onceは得られません。単一プロデューサーからブローカーへの書き込みの重複排除のみです。エンドツーエンドのEOSにはKafkaトランザクションが必要です。

---

## Kafka EOSトランザクションプロトコル

```
トランザクションライフサイクル — 実際にワイヤ上で何が起こるか:

1. initTransactions()
   Producer ──► TransactionCoordinator
   コーディネーターがこのtransactional.idのepochをインクリメント
   同じtransactional.idの古いプロデューサーはフェンスされる（ゾンビフェンシング）

2. beginTransaction()
   ローカル状態変更のみ、ブローカーには何も送信されない

3. send() / AddPartitionsToTxn
   このtxn内の新しいパーティションへの最初の送信がトリガー:
   Producer ──► Coordinator: AddPartitionsToTxn(txnId, epoch, [topic-partition])
   コーディネーターがパーティションリストを__transaction_stateに永続化

4. データ書き込み
   Producer ──► パーティションリーダー: PID+epochでタグ付けされた通常のproduceリクエスト
   リーダーがメッセージをバッファするが「uncommitted」としてマーク

5. sendOffsetsToTransaction()
   Producer ──► Coordinator: AddOffsetsToTxn(txnId, consumerGroupId)
   Producer ──► GroupCoordinator: TxnOffsetCommit(offsets)

6. commitTransaction()
   Producer ──► Coordinator: EndTxn(COMMIT)
   コーディネーターがPREPARE_COMMITを__transaction_stateに書き込み
   コーディネーターがCOMMITマーカーをすべての関連パーティションに書き込み
   コーディネーターがCOMPLETE_COMMITを__transaction_stateに書き込み
```

**コンシューマーが見るもの**: `isolation.level=read_committed`では、コンシューマーのフェッチリクエストが`LastStableOffset`（LSO）を返します。オープントランザクションに属するLSOを超えるメッセージはバッファされますが、トランザクションが解決されるまでアプリケーションには配信されません。つまり、read_committedコンシューマーはエンドツーエンドのレイテンシが高くなる可能性があります。

**`__transaction_state`トピック**: 内部のコンパクトされたトピック（デフォルト50パーティション）。各`transactional.id`が1つのパーティションにハッシュされます。`(transactional.id, PID, epoch, state, involved_partitions, timeout)`を保存します。コンパクションがキーごとに最新の状態のみを保持します。

**障害回復**: プロデューサーがトランザクション中にクラッシュした場合、コーディネーターのトランザクションタイムアウト（デフォルト60秒）が期限切れになり自動アボートします。新しいプロデューサーが同じ`transactional.id`で初期化すると、コーディネーターがepochをインクリメントし、古いepochからの残留書き込みはパーティションリーダーに拒否されます。

---

## 冪等コンシューマーパターン

コンシューマーが重複配信を許容するための4つの戦略です。コストの低い順に説明します。

| 戦略 | メカニズム | コスト | 最適な用途 |
|------|----------|--------|-----------|
| 自然な冪等性 | 操作が本質的に安全に繰り返せる | 無料 | 可能なすべてのケース |
| データベース制約 | ユニークインデックスが重複挿入を拒否 | 低 | DBバックエンドのコンシューマー |
| 分散重複排除ストア | 処理前に外部ストアを確認 | 中 | ステートレスコンシューマー |
| バージョン付き状態 | 状態が既により高いバージョンなら拒否 | 中 | イベントソーシングシステム |

### 自然な冪等性

```python
# SET is idempotent; INCREMENT is not
# Instead of: UPDATE accounts SET balance = balance + 100
# Use:        UPDATE accounts SET balance = 1500 WHERE id = ? AND version = ?

# DELETE WHERE is idempotent
db.execute("DELETE FROM cart_items WHERE cart_id = ? AND product_id = ?", cart_id, product_id)
```

目安: 操作が何回実行されても同じ状態に収束するなら、自然に冪等です。他のすべての戦略より優先してください。

### データベース制約

```python
def process(message):
    try:
        db.execute(
            "INSERT INTO processed_events (event_id, result, created_at) VALUES (%s, %s, NOW())",
            message.id, compute_result(message)
        )
    except UniqueViolation:
        log.info(f"Duplicate {message.id}, skipping")
        return  # Safe to ack
```

ビジネスロジックと重複排除レコードが同じデータベースを共有する場合に有効です。原子性のために両方を1つのトランザクションにラップしてください。

### 分散重複排除ストア

```python
def process(message):
    # SETNX returns False if key already exists
    if not redis.set(f"dedup:{message.id}", "1", nx=True, ex=86400):
        log.info(f"Duplicate {message.id}, skipping")
        return

    do_work(message)
    # If do_work() fails, key is already set → message won't be reprocessed
    # Mitigation: set key AFTER do_work, accept small duplicate window
```

トレードオフ: 処理前にキーを設定すると重複を防げますが、クラッシュ時にメッセージ損失のリスクがあります。処理後に設定するとクラッシュ時に重複のリスクがあります。システムが許容できる障害モードに基づいて選択してください。

### バージョン付き状態

```python
def process(event):
    current = db.execute("SELECT version FROM entities WHERE id = %s", event.entity_id)
    if current.version >= event.version:
        log.info(f"Stale event v{event.version}, current v{current.version}")
        return  # Already at this version or newer

    db.execute(
        "UPDATE entities SET data = %s, version = %s WHERE id = %s AND version < %s",
        event.data, event.version, event.entity_id, event.version
    )
```

`WHERE version < ?`句によりUPDATE自体が冪等になります。同じイベントバージョンの再適用はノーオペレーションです。

---

## 重複排除ストレージのサイジング

外部の重複排除ストアを使用する場合、ストレージはメッセージレートに比例して増加します。サイジングを誤ると、メモリ不足になるか、IDの早期エビクション（再配信時に偽の「新規メッセージ」を引き起こす）が発生します。

### 保持ウィンドウ

メッセージIDを**最大再配信ウィンドウの2倍**の期間保持してください。システムが初回配信後最大4時間再配信可能な場合、IDを8時間保持します。これはコンシューマーラグ、リトライストーム、クロックスキューを考慮しています。

### ストレージ計算式

```
memory = message_rate × retention_window × id_size × overhead_factor

例:
  10,000 msg/s × 604,800 s（7日）× 36 bytes（UUID）× 1.5（ハッシュテーブルオーバーヘッド）
  = 10,000 × 604,800 × 36 × 1.5
  ≈ 327 GB

  インメモリには大きすぎます。保持期間を短くするか確率的構造を使用してください。
```

### 代替手段

**ブルームフィルタ**: 1000万エントリで1%の偽陽性率 ≈ 12 MB。メモリ効率が非常に高いですが、偽陽性は*ドロップされたメッセージ*を意味します（フィルタが見ていないのに「既に見た」と言う）。メッセージの偶発的な損失が許容される場合にのみ適用可能ですが、At-least-onceの目的に反します。

**TTLベースのクリーンアップ付きRedis sorted set**:
```
ZADD dedup <unix_timestamp> <message_id>     # O(log N) insert
ZSCORE dedup <message_id>                     # O(1) existence check
ZRANGEBYSCORE dedup -inf <cutoff_ts>          # Periodic cleanup of expired IDs
ZREMRANGEBYSCORE dedup -inf <cutoff_ts>       # Remove expired entries
```

これにより自動有効期限付きの正確な重複排除が得られます。10K msg/sで1時間の保持の場合、~3600万エントリ × 各~80バイト ≈ Redisで2.9 GB。管理可能です。

---

## パフォーマンスオーバーヘッド

Kafkaの配信保証モードには測定可能なスループットとレイテンシのコストがあります。以下の比率は例示的なもので、実際の数値はバッチサイズ、パーティション数、レプリケーションファクター、ハードウェアによって異なります。

| モード | スループット（相対） | レイテンシP99 | 主要設定 |
|--------|-------------------|-------------|---------|
| At-most-once | 1.0× ベースライン | ~2 ms | `acks=0` |
| At-least-once | ~0.85× | ~10 ms | `acks=all`, `retries=Integer.MAX_VALUE` |
| 冪等 | ~0.82× | ~12 ms | `enable.idempotence=true` |
| トランザクショナル | ~0.65-0.75× | ~25-50 ms | `transactional.id`設定、`isolation.level=read_committed` |

**トランザクショナルモードが遅い理由**:
- 各トランザクションはコーディネーターへの少なくとも2回の追加RPC（`AddPartitionsToTxn`、`EndTxn`）が必要
- コミットマーカーがトランザクションに関わるすべてのパーティションに書き込まれる必要がある
- `read_committed`コンシューマーはLSOの前進を待つ必要があり、テールレイテンシが追加される
- 小さいバッチはこのオーバーヘッドを増幅する — トランザクション使用時は積極的にバッチ化する

**チューニングレバー**:
- `linger.ms`と`batch.size`: 大きいバッチがメッセージごとのオーバーヘッドを償却する
- `transaction.timeout.ms`: 短いタイムアウト = ゾンビ検出が速いが、遅い正当なプロデューサーのアボートリスク
- パーティション数: パーティションが多い = トランザクションごとのコミットマーカーが多いが、並列性も向上

**目安**: プロデューサーあたり>100K msg/sかつP99 < 10msが必要な場合、冪等モードが実用的な上限です。トランザクショナルモードは~50-80K msg/sでP99 ~30msで実用可能です。

---

## 実際の障害シナリオ

理論は障害境界で破綻します。以下の3つのシナリオは本番で繰り返し発生します。

### シナリオ1: 重複排除ストアのダウン

```
コンシューマーがメッセージM1を受信
コンシューマーがRedis SETNXで重複チェック → Redisタイムアウト / 接続拒否
どうする？

オプションA — フェイルオープン（とにかく処理）:
  リスク: M1が既に見られていた場合、重複処理
  利点: データ損失なし

オプションB — フェイルクローズ（拒否 / nack）:
  リスク: メッセージがキューに戻り、期限切れまたはDLQに到達する可能性
  利点: 重複処理なし
```

**ほとんどのシステムはフェイルオープンを選択します。** 理由: 重複は通常、下流で処理する方が安価（冪等なDB書き込み、リコンシリエーションジョブ）であり、メッセージ損失より安全です。重複排除ストアのSLAがメッセージブローカーのSLAより低い場合、ローカルのインプロセスフォールバックキャッシュを検討してください。

### シナリオ2: オフセットコミット後のコンシューマークラッシュ

```
タイムライン:
  t1: コンシューマーが100メッセージのバッチをポーリング
  t2: コンシューマーがオフセットをコミット（非同期またはread_committed）
  t3: コンシューマーがメッセージ51のビジネスロジックを開始
  t4: コンシューマープロセスがクラッシュ（OOM、セグフォルト、kill -9）

結果: メッセージ51-100はオフセットがコミットされたが、ビジネスロジックは完了していない。
これらのメッセージは永久にスキップされる — 新しいコンシューマーインスタンスはオフセット101から開始。
```

**修正**: ビジネスロジックが成功するまでオフセットをコミットしないでください。手動コミット（`enable.auto.commit=false`）を使用し、各バッチの処理*後*にコミットしてください。Kafkaトランザクションでは、`sendOffsetsToTransaction()`を使用してオフセットコミットをproduceと原子的にしてください。トランザクションがアボートされるとオフセットもロールバックされます。

### シナリオ3: バッチ中のブローカー再起動

```
プロデューサーが50メッセージのバッチをパーティションリーダーに送信
ブローカーがメッセージ1-30をackし、31-50をackする前にクラッシュ

冪等性なしの場合:
  プロデューサーが50メッセージすべてをリトライ（どれが永続化されたか不明）
  メッセージ1-30がブローカーで重複
  コンシューマーが50ではなく80メッセージを受信

冪等性ありの場合:
  プロデューサーが同じ(PID, partition, sequence)タプルで50メッセージすべてをリトライ
  ブローカーのシーケンストラッカーがメッセージ1-30を拒否（sequence ≤ last committed）
  ブローカーがメッセージ31-50を受け入れ
  コンシューマーが正確に50メッセージを受信
```

これは冪等プロデューサーを有効にする最も強力な根拠です。ほぼコストなし（~3%のスループット）で、Kafkaにおける最も一般的な重複原因を排除します。

相互参照: データベースとメッセージキューの両方に原子的に書き込む方法（これらのシナリオの多くを引き起こすデュアルライト問題の回避）については、`07-outbox-pattern.md`を参照してください。

---

## 重要なポイント

1. **At-most-onceが最速** - ただしメッセージを失う可能性がある
2. **At-least-onceが最も一般的** - 冪等なコンシューマーが必要
3. **Exactly-onceは困難** - 通常は重複排除でシミュレート
4. **処理後にackする** - 処理前ではない
5. **冪等性が味方** - 重複を無害にする
6. **障害シナリオをテストする** - コンシューマーをクラッシュさせ、ackをドロップする
7. **システムの保証を理解する** - Kafka vs SQS vs RabbitMQは異なる
8. **重複を前提に設計する** - 必ず発生する
