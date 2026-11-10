# デッドレターキュー

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../05-messaging/08-dead-letter-queues.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

デッドレターキュー（DLQ）は、正常に処理できないメッセージをキャプチャします。失敗したメッセージを失ったりキューをブロックしたりする代わりに、調査のために別のキューに移動します。デバッグ、コンプライアンス、データ損失防止に不可欠です。リトライ制限を設定し、DLQの深度を監視し、デッドレター処理の手順を確立してください。

---

## なぜデッドレターキューが必要か？

### 問題

```
メッセージ到着 → 処理失敗 → どうする？

DLQなしの選択肢:
  1. 永遠にリトライ（キューをブロック）
  2. 破棄（データ損失）
  3. コンシューマーをクラッシュ（サービス中断）

どれも良くない！
```

### DLQソリューション

```
メッセージ到着 → 処理失敗 → N回リトライ → DLQに移動

Main Queue ──► Consumer ──► 成功
                  │
              失敗（リトライ後）
                  │
                  ▼
              Dead Letter Queue
                  │
                  ▼
          手動調査
```

---

## DLQの仕組み

### 基本フロー

```
1. コンシューマーがメッセージを受信する
2. 処理が失敗する
3. メッセージがキューに返される（nack）
4. リトライカウンターがインクリメントされる
5. N回リトライ後、DLQに移動する
6. 元のキューは処理を継続する
7. DLQが監視・調査される
```

### メッセージメタデータ

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

## 設定

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

## リトライ戦略

### 即時リトライ

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

### 指数バックオフ

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

### 遅延リトライキュー

```
即時リトライの代わりに遅延キューを使用

Main Queue → 失敗 → Delay Queue (5分) → Main Queue

Delay Queueの実装:
  - メッセージTTL + メインキューへのDLQルーティング
  - または: スケジュールされた再配信
```

---

## デッドレターの処理

### 調査ワークフロー

```
1. DLQメッセージでアラート
2. メッセージ内容と失敗理由を確認
3. 根本原因を特定
   - コンシューマーのバグ？
   - 無効なメッセージフォーマット？
   - 外部依存関係の障害？
4. 根本原因を修正
5. メッセージをリプレイまたは破棄
```

### メッセージの検査

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

### メッセージのリプレイ

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

### 選択的リプレイ

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

## エラータイプごとのDLQ

### 分離されたDLQ

```
orders-dlq-validation  → 無効なメッセージフォーマット
orders-dlq-external    → 外部サービス障害
orders-dlq-unknown     → 不明なエラー

利点:
  - タイプごとに異なる処理
  - より簡単な調査
  - 異なる保持ポリシー
```

### ルーティング実装

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

## モニタリング

### 主要メトリクス

```
DLQ深度:
  DLQ内のメッセージ数
  通常はゼロ近くであるべき

DLQ到着レート:
  1分あたりの到着メッセージ数
  スパイクは処理問題を示す

DLQ経過時間:
  最も古いメッセージの経過時間
  古いメッセージは放置を示す

障害カテゴリ:
  エラータイプ別の内訳
  システム的な問題を特定
```

### アラート

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

### ダッシュボード

```
DLQダッシュボード:
  - 現在の深度（ゲージ）
  - 到着レート（時系列）
  - 上位の失敗理由（パイチャート）
  - 経過時間分布（ヒストグラム）
  - 最近のメッセージ（テーブル）
```

---

## 保持とクリーンアップ

### 保持ポリシー

```
考慮事項:
  - コンプライアンス要件（N日間保持が必要）
  - 調査時間（デバッグのための時間を確保）
  - ストレージコスト（永遠に保持しない）

一般的: 7-30日
```

### 自動クリーンアップ

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

### 削除前のアーカイブ

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

## 一般的なパターン

### ポイズンメッセージの検出

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

### DLQコンシューマー

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

### サーキットブレーカー連携

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

## DLQ処理戦略

### 手動レビュー

```
担当者:   ダッシュボードまたはCLI経由の人間のオペレーター
いつ:     低ボリュームDLQ、コンプライアンスに敏感なデータ、不明な障害タイプ
方法:     オペレーターがメッセージボディ + 失敗理由を検査 → リプレイまたは破棄を決定

ワークフロー:
  1. DLQ深度でアラート発報
  2. オペレーターがDLQダッシュボードを開く
  3. 失敗理由 + スタックトレースを読む
  4. 根本原因を特定
  5. コンシューマーまたは上流データを修正
  6. メッセージをリプレイまたはアーカイブ

トレードオフ: 遅い、スケールしない。ただし重要な金融データやPIIには最も安全。
```

### 自動リトライ

```
スケジューラーが定期的にDLQを読み取り、元のキューにメッセージを再パブリッシュします。

スケジュール: 15分ごとに最大50メッセージを取得し、指数バックオフで再パブリッシュ。

バックオフ計算式:
  delay = min(base_delay * 2^retry_count, max_delay)
  例: 1s → 2s → 4s → 8s → ... → 5分で上限

リスク: 無限リトライループ。
  メッセージが永続的に無効な場合（不正なスキーマ、必須フィールドの欠落）、
  メインキューとDLQの間を永遠にバウンスします。

緩和策:
  - 最大ライフタイムを設定（例: 最初の失敗から24時間）。それ以降 → アーカイブ。
  - 再パブリッシュ前にリトライ可能 vs リトライ不可のエラーを区別する。
```

### 条件付きリプレイ

```
各DLQメッセージを検査し、変換またはデータ修正を適用してからリプレイします。

例:
  元のメッセージに { "price": -5 } → バリデーション失敗
  修正: priceを0に設定、またはソースシステムから正しい価格を取得
  修正されたメッセージをメインキューにリプレイ

使用する場合:
  - 上流プロデューサーが不正データを送信したが意図は回復可能
  - スキーマが進化し古いメッセージにフィールドのバックフィルが必要
  - 外部参照データが一時的に不正だった（例: 為替レート）

注意: 変換は冪等でなければなりません。リプレイされたメッセージは
      より新しいメッセージと並行して処理される可能性があります — 重複副作用がないことを確認してください。
```

### DLQコンシューマーサービス

```
DLQをプライマリ入力として消費する専用マイクロサービスです。

責任:
  - 障害タイプの分類（バリデーション、タイムアウト、認証、不明）
  - 障害タイプごとのプログラマティックな修正の適用
  - 修正されたメッセージを元のキューに再パブリッシュ
  - 修正不能なメッセージのエスカレーション（チケット作成、Slackアラート送信）
  - 修復メトリクスの追跡（自動修正率、エスカレーション率）

アーキテクチャ:
  Main Queue ──► Consumer ──► DLQ ──► DLQ Consumer Service
                                          │
                              ┌────────────┼────────────┐
                              ▼            ▼            ▼
                         自動修正      チケット作成   アーカイブ
                         & リプレイ
```

---

## DLQスキーマとメタデータ

### なぜメタデータが重要か

```
障害コンテキストがなければ、DLQはブラックホールです。

DLQにメッセージがある。回答が必要な質問:
  - どのキューから来たか？
  - なぜ失敗したか？
  - 何回リトライされたか？
  - いつ最初に失敗した？いつ最後に失敗した？
  - スタックトレースは何と言っているか？

このメタデータがなければ、トリアージは推測になります。エンジニアが
スタックトレースが数秒で説明できたはずの障害を再現するのに何時間も無駄にします。
```

### 必須メタデータスキーマ

```json
{
  "dlq_envelope": {
    "message_id": "msg-a1b2c3d4",
    "original_topic": "orders.placed",
    "original_queue": "order-processing",
    "original_partition": 3,
    "original_offset": 884201,
    "failure_reason": "ValidationError: field 'quantity' must be > 0",
    "failure_category": "VALIDATION",
    "stack_trace": "Traceback (most recent call last):\n  File \"consumer.py\", line 42 ...",
    "retry_count": 3,
    "first_failure_at": "2025-11-10T08:15:22Z",
    "last_failure_at": "2025-11-10T08:17:44Z",
    "consumer_instance": "order-consumer-pod-7b4d9",
    "consumer_version": "2.4.1"
  },
  "original_headers": {
    "correlation_id": "corr-x9y8z7",
    "content_type": "application/json"
  },
  "original_body": {
    "order_id": "ORD-12345",
    "quantity": -1,
    "product_id": "SKU-999"
  }
}
```

### メタデータガイドライン

```
- 常にfailure_reasonをキャプチャする: クラス名だけでなく例外メッセージを。
- 常にstack_traceをキャプチャする: ストレージのために必要なら最後の20フレームに切り詰める。
- 最初と最後の失敗タイムスタンプを追跡する: メッセージがどのくらいバウンスしているか示す。
- consumer_versionを含める: 特定のデプロイで導入された問題のデバッグに重要。
- 元のヘッダーをそのまま保持する: コリレーションIDがエンドツーエンドのトレーシングを可能にする。
```

---

## DLQアンチパターン

### DLQの無視

```
症状:  DLQに50,000メッセージ。誰も気づかない。
原因:  モニタリングなし、アラートなし、オーナーシップなし。
修正:  DLQ深度 > 0（警告）、> 100（重大）でアラート。
      オンコールローテーションの一部としてDLQトリアージを担当するチームを割り当てる。
```

### 修正せずにリプレイ

```
症状:  メッセージ失敗 → DLQ → リプレイ → 再び失敗 → DLQ → リプレイ → ...
原因:  根本原因分析なしの盲目的なリプレイスクリプト。
修正:  失敗理由を理解せずにリプレイしない。
      チェックでリプレイをゲートする: コンシューマーのバグは修正された？
      無効なデータは修正された？
      リプレイ回数を追跡 — メッセージが3回以上リプレイされた場合、エスカレートする。
```

### DLQメッセージにTTLなし

```
症状:  DLQに2年前のメッセージが含まれている。誰もそれが何か知らない。
原因:  保持ポリシーなし、クリーンアップジョブなし。
修正:  コンプライアンスニーズに応じて7-30日の保持を設定する。
      監査証跡が必要なら削除前にコールドストレージ（S3、GCS）にアーカイブする。
      保持期間を超えたメッセージはアクション不可能 — 削除またはアーカイブする。
```

### DLQを機能として使用

```
症状:  プロデューサーが意図的にメッセージを失敗させ、
      DLQで「後で処理」するためにキューに送信する。
原因:  DLQの目的の誤解。遅延キューとして扱っている。
修正:  代わりに専用の遅延キューまたはスケジュールキューを使用する。
      DLQは予期しない障害用であり、意図的なルーティング用ではない。
      遅延メカニズム: RabbitMQメッセージTTL + 処理キューへのデッドレタールーティング、
      SQS遅延キュー、タイムスタンプベースのコンシューマー一時停止付きKafkaトピック。
```

---

## 実システムでのDLQ

### AWS SQS

```
設定:
  メインキューにRedrivePolicy:
    { "maxReceiveCount": 5, "deadLetterTargetArn": "arn:aws:sqs:...:orders-dlq" }

動作:
  - 5回の失敗した受信+処理サイクル（削除なし）後、メッセージがDLQに移動。
  - SQSが受信カウントを自動追跡 — アプリケーションコード不要。
  - DLQでRedriveAllowPolicyを使用してターゲットにできるキューを制限。
  - ソースへの再駆動: SQSコンソールがメッセージを元のキューに戻す機能をサポート。

注意点: maxReceiveCountにはVisibility Timeoutの期限切れが含まれる。コンシューマーが
        遅くてVisibility Timeoutが切れた場合、それは受信としてカウントされる。
```

### Apache Kafka

```
KafkaにはネイティブのDLQメカニズムがありません。自分で実装します。

一般的なパターン:
  - 失敗したメッセージを別のトピックにプロデュース: orders.dlq
  - コンシューマーが例外をキャッチ → 失敗ヘッダー付きでDLQトピックに書き込み
  - DLQコンシューマーサービスがorders.dlqを読んでトリアージ

Spring Kafka連携:
  - DeadLetterPublishingRecoverer: リトライ後に<topic>.DLTに自動パブリッシュ
  - DefaultErrorHandler with BackOff: 設定可能なリトライ + DLTルーティング
  - 元のヘッダーを保持 + 例外ヘッダーを自動追加

命名規則: <original-topic>.dlq または <original-topic>.DLT（dead letter topic）
```

### RabbitMQ

```
エクスチェンジルーティングによるネイティブDLQサポート:

キュー引数:
  x-dead-letter-exchange: "dlx-exchange"
  x-dead-letter-routing-key: "orders.dlq"

メッセージがデッドレターになる条件:
  - コンシューマーがnack（basic.reject / basic.nack）でrequeue=false
  - メッセージTTLが期限切れ
  - キューのmax-lengthを超過

デッドレターエクスチェンジがルーティングキーに基づいてメッセージをDLQにルーティング。
元のdeathメタデータがx-deathヘッダー配列に追加される（キュー、理由、回数、時間）。
```

### GCP Pub/Sub

```
設定:
  サブスクリプションにdeadLetterPolicy:
    { "deadLetterTopic": "projects/.../topics/orders-dlq",
      "maxDeliveryAttempts": 5 }

動作:
  - 5回の失敗した配信試行（nackまたはackデッドライン期限切れ）後、メッセージが
    デッドレタートピックに転送される。
  - Pub/SubがCloudPubSubDeadLetterSourceDeliveryCount属性を自動追加。
  - DLQトピックにはコンシューマーが読むための別のサブスクリプションが必要。

注意点: サービスアカウントにDLQトピックへのpubsub.publisherロールと
        ソースサブスクリプションへのpubsub.subscriberロールが必要。
```

---

## 重要なポイント

1. **DLQはメッセージ損失を防ぐ** - 失敗したメッセージが保持される
2. **リトライ制限を設定する** - 永遠にリトライしない
3. **障害メタデータを含める** - 理由、タイムスタンプ、リトライ回数
4. **DLQ深度を監視する** - 蓄積時にアラート
5. **処理手順を確立する** - 調査、リプレイ、アーカイブ
6. **エラーごとに異なるDLQ** - より簡単な分類
7. **保持ポリシーが重要** - コンプライアンスとストレージコスト
8. **可能な限り自動化する** - リプレイ、クリーンアップ、アラート
