# Outboxパターン

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../05-messaging/07-outbox-pattern.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

Outboxパターンは、メッセージをビジネスデータと同じトランザクションでデータベーステーブル（outbox）に書き込むことで、信頼性のあるメッセージパブリッシングを保証します。別のプロセスがoutboxから読み取り、メッセージブローカーにパブリッシュします。これにより、データベース書き込みとメッセージパブリッシングの間の原子性が保証され、デュアルライト問題を解決します。

---

## デュアルライト問題

### 素朴なアプローチ

```python
def create_order(order):
    # Step 1: Save to database
    db.save(order)

    # Step 2: Publish event
    message_queue.publish(OrderCreated(order))
```

### 障害シナリオ

```
シナリオ1: DBは成功、パブリッシュが失敗
  db.save(order)     ✓ (committed)
  mq.publish(event)  ✗ (failed)

  結果: 注文は存在するが、イベントなし
  下流システムは決して知らない

シナリオ2: パブリッシュは成功、DBが失敗
  db.save(order)     (pending)
  mq.publish(event)  ✓ (published)
  db.commit()        ✗ (rolled back)

  結果: イベントは存在するが、注文なし
  下流システムが幻の注文を処理する
```

### なぜ分散トランザクションが役立たないか

```
XA/2PC:
  - ほとんどのメッセージブローカーでサポートされていない
  - 遅い（コーディネーターでブロック）
  - 複雑な障害処理

よりシンプルで信頼性の高いアプローチが必要
```

---

## Outboxソリューション

### アーキテクチャ

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

### 仕組み

```
1. アプリケーションがビジネスデータとoutboxレコードを
   同じトランザクションで書き込む

2. トランザクションが原子的にコミット
   注文とoutboxレコードの両方が存在するか、どちらも存在しない

3. バックグラウンドプロセスがoutboxをポーリング
   未パブリッシュのメッセージを読み取る

4. パブリッシャーがメッセージブローカーに送信
   メッセージがキュー/トピックに配信される

5. パブリッシャーがoutboxレコードをパブリッシュ済みとしてマーク
   重複パブリッシングを防止する
```

---

## 実装

### Outboxテーブルスキーマ

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

### Outboxへの書き込み

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

### Outboxパブリッシャー（ポーリング）

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

## CDCベースのOutbox

### Change Data Captureの使用

```
ポーリングの代わりにデータベースログを使用

Database ──► CDC (Debezium) ──► Kafka

Outboxテーブルの変更がbinlog/WALからキャプチャされる
ポーリングより低レイテンシ
別のパブリッシャープロセスが不要
```

### Debezium設定

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

### CDCの利点

```
+ 低レイテンシ（ほぼリアルタイム）
+ データベースへのポーリング負荷なし
+ 保証された順序（ログから）
+ メッセージの見落としなし

- より多くのインフラ（Debezium、Kafka Connect）
- CDCセットアップの複雑さ
- データベースがログアクセスをサポートする必要がある
```

---

## 重複の処理

### なぜ重複が発生するか

```
シナリオ:
  1. パブリッシャーがoutboxからメッセージを読み取る
  2. パブリッシャーがブローカーに送信 ✓
  3. パブリッシャーがパブリッシュ済みマーク前にクラッシュ
  4. 新しいパブリッシャーインスタンスが起動
  5. 同じメッセージが再びパブリッシュされる

コンシューマーが重複メッセージを受信する
```

### 冪等なコンシューマー

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

### トランザクショナル重複排除

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

## 順序保証

### アグリゲートごとの順序

```sql
-- Outbox entries ordered by aggregate
SELECT * FROM outbox
WHERE published_at IS NULL
ORDER BY aggregate_id, created_at
FOR UPDATE SKIP LOCKED
```

### アグリゲートによるパーティション

```python
def publish(entry):
    broker.publish(
        topic="order-events",
        key=entry.aggregate_id,  # Same aggregate → same partition
        value=entry.payload
    )
```

### 順序外の処理

```
厳密な順序が必要な場合:
  1. アグリゲートタイプごとに単一パブリッシャー
  2. または: メッセージにシーケンス番号
  3. または: コンシューマーの並べ替えバッファ
```

---

## クリーンアップ戦略

### パブリッシュ後に削除

```python
# Immediately delete after successful publish
db.execute("DELETE FROM outbox WHERE id = %s", entry.id)
```

### 論理削除とクリーンアップ

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

### 削除前のアーカイブ

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

## モニタリング

### 主要メトリクス

```
Outboxラグ:
  未パブリッシュメッセージの数
  低く保つべき

パブリッシュレイテンシ:
  created_atからpublished_atまでの時間
  処理速度を示す

パブリッシュ失敗:
  失敗したパブリッシュ試行の割合
  ブローカーの問題を示す

Outboxサイズ:
  テーブルの総サイズ
  制限されるべき
```

### アラート

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

### ヘルスチェック

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

## バリエーション

### Inboxパターン（冪等コンシューマー）

```
コンシューマー側のOutboxのミラー

メッセージ到着 → Inboxに書き込み → 処理 → 処理済みマーク

Inboxテーブル:
  id, message_id, payload, processed_at

コンシューマーでの冪等性を保証
```

### トランザクショナルInbox

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

## Debezium CDC実装

### アーキテクチャの詳細

DebeziumはKafka Connect上に構築された変更データキャプチャ用分散プラットフォームです。Outboxパターンでは、Debeziumがデータベースのトランザクションログを直接読み取ります — ポーリングクエリもアプリケーションレベルのフックも不要 — 行レベルの変更をKafkaトピックにパブリッシュします。

```
Database Transaction Log ──► Debezium Connector ──► Kafka Connect ──► Kafka Topic
     (WAL / binlog)            (source connector)    (worker cluster)    (outbox.events.*)
```

### PostgreSQL: 論理レプリケーション

PostgreSQLはクラッシュリカバリのためにWrite-Ahead Logging（WAL）を使用します。Debeziumは`pgoutput`プラグイン（PostgreSQL 10以降に組み込み）を使用して**論理レプリケーションスロット**を作成し、変更をストリーミングします。

- レプリケーションスロットは、DebeziumがWALセグメントを消費する前にリサイクルされないことを保証します
- `pgoutput`はWALエントリを論理変更イベント（INSERT、UPDATE、DELETE）にデコードします
- Outboxテーブルのポーリングなし — 変更がコミット時にWALからプッシュされます
- `postgresql.conf`で`wal_level = logical`が必要

### MySQL: Binlog消費

MySQLのバイナリログはすべてのデータ変更を記録します。Debeziumはレプリカとして接続します:

- Binlogイベントを読み取り、`table.include.list`でOutboxテーブルをフィルタリング
- 行ベースと混合のBinlogフォーマットの両方をサポート（完全な変更キャプチャには行ベースが必要）
- コネクターは再起動後の再開のためにBinlogファイル名 + 位置を追跡

### コネクター設定

```json
{
  "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
  "database.hostname": "db-primary",
  "database.port": "5432",
  "database.user": "debezium_replication",
  "database.dbname": "app",
  "slot.name": "outbox_slot",
  "plugin.name": "pgoutput",
  "table.include.list": "public.outbox_events",
  "transforms": "outbox",
  "transforms.outbox.type": "io.debezium.transforms.outbox.EventRouter",
  "transforms.outbox.table.field.event.key": "aggregate_id",
  "transforms.outbox.table.field.event.type": "event_type",
  "transforms.outbox.table.field.event.payload": "payload",
  "transforms.outbox.route.topic.replacement": "outbox.events.${routedByValue}"
}
```

### EventRouterトランスフォーム

`EventRouter` Single Message Transform（SMT）はDebeziumをOutbox対応にする重要なピースです:

- Outbox行の`payload`カラムからイベントペイロードを抽出 — エンベロープラッピングなし
- `aggregate_type`に基づいて正しいKafkaトピックにルーティング（例: `outbox.events.Order`）
- Kafkaメッセージキーを`aggregate_id`に設定 — エンティティごとのパーティションレベルの順序保証を確保
- オプションでパブリッシュ後にOutbox行を削除（`route.tombstone.on.empty.payload`経由）

### 順序保証

イベントは単一のKafkaパーティション内でWALコミット順にパブリッシュされます。`aggregate_id`がパーティションキーであるため、同じアグリゲートのすべてのイベントは同じパーティションに到着し、データベースにコミットされた正確な順序で届きます。パーティション間のアグリゲート間順序は保証されません — これは設計上の意図です。

---

## Outboxテーブルスキーマ設計

### 最小スキーマ

```sql
CREATE TABLE outbox_events (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_type VARCHAR(255) NOT NULL,   -- e.g., 'Order', 'Payment'
    aggregate_id  VARCHAR(255) NOT NULL,   -- entity's business ID
    event_type    VARCHAR(255) NOT NULL,   -- e.g., 'OrderCreated', 'PaymentFailed'
    payload       JSONB       NOT NULL,    -- full event body
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### なぜ`aggregate_id`が重要か

`aggregate_id`カラムは二重の目的を果たします:

1. **Kafkaパーティションキー** — DebeziumのEventRouter（またはポーリングパブリッシャー）がこれをメッセージキーとして使用します。Kafkaがパーティションにハッシュし、同じエンティティのすべてのイベントが順序通りに到着することを保証します。
2. **コンシューマーの相関** — 下流サービスが`aggregate_id`を使用してソースにクエリせずにエンティティ状態を再構築します。

### ペイロード戦略: フル vs 参照

| 戦略 | トレードオフ |
|------|------------|
| **フルペイロード**（イベントボディ全体をJSONBで） | 自己完結型イベント、行が大きい、コンシューマーは他に何も不要 |
| **参照**（イベントID + タイプ、コンシューマーが詳細を取得） | Outbox行が小さい、ただしカップリング導入 — コンシューマーがソースサービスにコールバックする必要 |

イベントサイズが定常的に1MBを超えない限り、フルペイロードを推奨します。自己完結型イベントはサービスをより効果的に分離します。

### 保持とクリーンアップ

CDC（Debezium）では、処理済み行はすぐに削除できます — Debeziumはoutboxテーブルの読み取りではなく、レプリケーションスロットを介してWAL内の位置を追跡します。テーブルを小さく保つことでVACUUMオーバーヘッドとインデックス肥大を削減します。

ポーリングベースの実装では、`published_at`が設定されるまで行を保持し、スケジュールされたクリーンアップジョブで削除します。

### インデックス

- **`id`のプライマリキー** — 重複排除とルックアップに必要
- **`published_at IS NULL`条件の`created_at`部分インデックス** — ポーリングベースパブリッシャーが未パブリッシュ行を効率的に見つけるため
- `payload`のインデックスは避ける — outboxテーブルへのJSONB GINインデックスは読み取りの利益なしに書き込みオーバーヘッドを追加

### テーブルパーティショニング

高スループットシステムでは、PostgreSQLのネイティブパーティショニングまたは`pg_partman`を使用して`created_at`でoutboxテーブルをパーティション分割します:

```sql
CREATE TABLE outbox_events (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    aggregate_type VARCHAR(255) NOT NULL,
    aggregate_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(255) NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
) PARTITION BY RANGE (created_at);
```

パーティショニングにより、行レベルの`DELETE`の代わりに`DROP`ベースのクリーンアップ（古いパーティションの削除）が可能になり、テーブル肥大と長時間のVACUUM操作を回避できます。

---

## ポーリング vs CDCのトレードオフ

### 比較マトリクス

| 側面 | ポーリング | CDC（Debezium） |
|------|----------|----------------|
| **レイテンシ** | ポーリング間隔に制約（100ms-5s が典型的） | ほぼリアルタイム（コミットから<100ms） |
| **順序** | `ORDER BY created_at`は並行書き込み下でギャップの可能性 | WAL順序は正確なコミット順 |
| **データベース負荷** | outboxテーブルへの繰り返しクエリ；`FOR UPDATE SKIP LOCKED`で緩和 | レプリケーションスロットからの読み取り — 最小限の増分負荷 |
| **運用の複雑さ** | シンプルなSQLクエリ + cronまたはループ | Debezium + Kafka Connectクラスター + モニタリング |
| **障害回復** | 最後の処理IDまたは`published_at IS NULL`から再ポーリング | DebeziumがWALオフセットから再開 |
| **インフラ** | アプリケーション + データベースのみ | Kafka、Kafka Connect、Debezium、Schema Registry |
| **スループット上限** | ポーリングクエリ速度 + バッチサイズに制約 | WALストリーミングがデータベース書き込みスループットに比例してスケール |

### ポーリングを選ぶ場合

- 小〜中規模のイベントボリューム（1,000イベント/秒未満）
- 既存のKafkaインフラがなく、採用予定もない
- チームがレイテンシより運用のシンプルさを優先
- イベントがレイテンシに敏感でない（バッチ処理、日次レポート）

### CDCを選ぶ場合

- 高スループット（1,000イベント/秒以上の持続）
- アグリゲート内の厳密な順序保証要件
- 運用専門知識を持つ既存のKafkaインフラ
- ほぼリアルタイムのイベント伝播がビジネス要件
- 複数のコンシューマーが同じイベントストリームを必要とする（Kafkaトピックファンアウト）

### ハイブリッドアプローチ

一部のシステムはポーリングから始め、スケールの要求に応じてCDCに移行します。Outboxテーブルスキーマは同一のまま — パブリッシャーメカニズムのみが変更されます。これによりポーリングは明確なアップグレードパスを持つ安全な出発点となります。

---

## Outboxパターンの障害モード

### レプリケーションスロットの肥大（CDC）

Debeziumがダウンしているか消費できない場合、PostgreSQLはレプリケーションスロットが参照するWALセグメントを保持します。チェックしないと、ディスクが満杯になりデータベースがクラッシュします。

**検出:** `pg_replication_slots`を監視 — `confirmed_flush_lsn`を`pg_current_wal_lsn()`と比較します。デルタの増大はDebeziumが遅れていることを示します。

```sql
SELECT slot_name,
       pg_wal_lsn_diff(pg_current_wal_lsn(), confirmed_flush_lsn) AS bytes_behind
FROM pg_replication_slots
WHERE slot_name = 'outbox_slot';
```

**緩和策:** `max_slot_wal_keep_size`（PostgreSQL 13+）を設定して保持WALを制限します。ラグが閾値（例: 1GB）を超えたらアラート。Debeziumが回復不能な場合、スロットをドロップして再作成します — 一部のイベントがOutboxテーブルから再パブリッシュが必要になることを受け入れます。

### コンシューマー再起動時の重複イベント

コンシューマーがイベント処理後、オフセットコミット前にクラッシュした場合、再起動時にイベントを再受信します。コンシューマーは冪等でなければなりません。パターンについては`04-delivery-guarantees.md`を参照してください。

### スキーマ進化

Outboxペイロードのフィールドの追加・削除は、固定構造を期待するコンシューマーを壊します。戦略:

- **Avro + Schema Registry:** スキーマレベルで前方・後方互換性を強制。DebeziumはConfluent Schema Registryとネイティブに統合。
- **追加変更のみのJSONB:** フィールドを削除せず、オプショナルなフィールドのみ追加。コンシューマーは未知のフィールドを無視。
- **バージョン付きイベントタイプ:** `OrderCreated.v2`を`event_type`として使用し、非互換なスキーマバージョンを区別。

### 大きなペイロード

大きなJSONBペイロード（>100KB）を持つOutbox行はWALレプリケーションを遅くし、Kafkaメッセージサイズを増加させます。選択肢:

- フルペイロードを別テーブルに保存；Outbox行は参照（イベントID + アグリゲートタイプ）を保持。コンシューマーがAPIまたはオブジェクトストアからペイロードを取得。
- Outboxカラムに書き込む前にペイロードを圧縮。
- Claim-checkパターン: ペイロードをS3/GCSに書き込み、オブジェクトキーをOutbox行に保存。

### アグリゲート間のトランザクション順序

単一のデータベーストランザクションが複数のアグリゲート（例: `Order`と`Payment`）のOutboxエントリを書き込む場合があります。これらのイベントは異なるKafkaパーティションに到着し、コンシューマーに任意の順序で届く可能性があります。コンシューマーをこれに対応するよう設計してください:

- アグリゲート間の因果順序を仮定しない
- 下流ロジックが協調処理を必要とする場合、明示的なコリレーションIDを使用
- 厳密なアグリゲート間順序が必要な場合、すべての関連イベントを同じ`aggregate_id`でルーティング — ただしパーティション並列性を制限する

---

## Outboxの代替手段

### Listen/Notify（PostgreSQL）

PostgreSQLの`NOTIFY`はビジネス書き込みと同じトランザクション内で発行できます。リスニングプロセスが通知を受信し、ブローカーにパブリッシュします。

```sql
-- Inside transaction
INSERT INTO orders (...) VALUES (...);
NOTIFY order_events, '{"order_id": "abc", "type": "OrderCreated"}';
```

**制限:** 通知は永続化されません。リスナーが切断またはクラッシュした場合、イベントは永久に失われます。リプレイ機能なし。重要でないベストエフォート通知にのみ適用。

### トランザクショナルメッセージング（XA/2PC）

XAを使用してデータベースとメッセージブローカーの両方を分散トランザクションに参加させます。両方がコミットまたは両方がロールバック。

**制限:** ほとんどのメッセージブローカー（Kafka、RabbitMQ、SQS）はXAをサポートしていません。サポートされている場合でも、2PCは遅く（コーディネーターのラウンドトリップ）、脆弱で（コーディネーター障害がすべての参加者をブロック）、運用が困難です。Outboxパターンは、XAが大規模で実用的でないために存在します。

### コミット後に発行されるドメインイベント

```python
def create_order(order_data):
    order = save_to_db(order_data)
    # DB committed, now publish
    broker.publish(OrderCreated(order))  # crash here = lost event
```

コミットとパブリッシュの間のギャップが、Outboxパターンが排除する脆弱性そのものです。そのウィンドウ内でのクラッシュ、ネットワークタイムアウト、プロセスキルは、回復パスのないイベント損失を引き起こします。

### アプリケーションポーリング付きイベントテーブル

Outboxに似ていますが、正式なOutbox構造なし — アプリケーションがイベントを汎用テーブルに書き込み、パブリッシング用にポーリングします。機能的にはOutboxパターンと同等ですが、明示的な`aggregate_id`パーティショニングと冪等性設計を欠くことが多いです。

### Outboxが過剰な場合

- 呼び出し元が障害時にリトライする内部サービス通信（リトライ付き同期HTTP）
- 偶発的な損失が許容される重要でない通知（メール、Slackアラート）
- 下流コンシューマーのない単一サービスアーキテクチャ
- 運用のシンプルさが信頼性保証を上回るプロトタイピングまたはMVP段階

---

## 重要なポイント

1. **デュアルライト問題を解決** - 原子的なデータベース + メッセージ
2. **同一トランザクションが鍵** - ビジネスデータ + Outboxを一緒に
3. **ポーリングまたはCDC** - レイテンシ要件に基づいて選択
4. **重複は発生する** - コンシューマーは冪等でなければならない
5. **アグリゲートで順序付け** - エンティティごとの順序を保持
6. **定期的にクリーンアップ** - Outboxを無制限に増大させない
7. **ラグを監視** - パブリッシング問題を早期に検出
8. **コンシューマー側にはInbox** - 受信側で同じパターン
