# イベントソーシング

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../05-messaging/05-event-sourcing.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

イベントソーシングは、アプリケーション状態へのすべての変更をイベントのシーケンスとして保存します。現在の状態を保存する代わりに、何が起こったかの履歴を保存します。現在の状態はイベントをリプレイすることで導出されます。利点として、完全な監査証跡、時間的クエリ、デバッグがあります。コストとして、複雑さ、結果整合性、ストレージの増大があります。CQRSとよく組み合わされます。

---

## 従来型 vs イベントソーシング

### 従来型（状態ベース）

```
データベースが現在の状態を保存:

Users table:  id: 123, balance: 500, updated_at: 2024-01-15

問題: 履歴が失われる
  昨日の残高はいくらだった？ どうやって500になった？ 不明。
```

### イベントソーシング

```
データベースがイベントを保存:
  AccountCreated(id=123, balance=1000)
  MoneyWithdrawn(id=123, amount=200)
  MoneyDeposited(id=123, amount=300)
  MoneyWithdrawn(id=123, amount=600)

現在の状態: リプレイ → 1000 - 200 + 300 - 600 = 500 ✓
完全な履歴が保持される
```

---

## 基本概念

### イベント

```python
@dataclass
class Event:
    event_id: str
    aggregate_id: str
    event_type: str
    timestamp: datetime
    data: dict
    version: int

# Example
AccountCreated(
    event_id="evt-001",
    aggregate_id="account-123",
    event_type="AccountCreated",
    timestamp="2024-01-15T10:00:00Z",
    data={"owner": "Alice", "initial_balance": 1000},
    version=1
)
```

### イベントストア

```
イベントの追記専用ログ

┌──────────────────────────────────────────────┐
│ Event 1 │ Event 2 │ Event 3 │ ... │ Event N │
└──────────────────────────────────────────────┘
     ↑
  追記のみ（更新なし、削除なし）
```

### アグリゲート

```
関連するイベントをグループ化するドメインエンティティ。イベントは常にアグリゲートに属します。

Accountアグリゲート:  Created, Deposited, Withdrawn, Closed
Orderアグリゲート:    Placed, Confirmed, Shipped, Delivered
```

### コマンド

```
状態変更の意図を表します。バリデーション後にイベントを生成します。

Command: Withdraw(account_id=123, amount=100)
  バリデーション: アカウントは存在する？ ✓  残高は十分？ ✓
  結果: MoneyWithdrawnイベントが生成される
```

---

## イベントストアの実装

### スキーマ

```sql
CREATE TABLE events (
    event_id UUID PRIMARY KEY,
    aggregate_id VARCHAR(255) NOT NULL,
    aggregate_type VARCHAR(255) NOT NULL,
    event_type VARCHAR(255) NOT NULL,
    event_data JSONB NOT NULL,
    metadata JSONB,
    version INT NOT NULL,
    timestamp TIMESTAMP NOT NULL,

    UNIQUE (aggregate_id, version)  -- Optimistic concurrency
);

CREATE INDEX idx_events_aggregate ON events(aggregate_id, version);
CREATE INDEX idx_events_timestamp ON events(timestamp);
```

### イベントの追加

```python
class EventStore:
    def append(self, aggregate_id, events, expected_version):
        with transaction():
            # Check optimistic concurrency
            current = self.get_latest_version(aggregate_id)
            if current != expected_version:
                raise ConcurrencyError(
                    f"Expected version {expected_version}, got {current}"
                )

            # Append events
            for i, event in enumerate(events):
                event.version = expected_version + i + 1
                self.db.insert(event)

            # Publish events
            for event in events:
                self.publish(event)
```

### アグリゲートの読み込み

```python
def load_aggregate(aggregate_id):
    # Get all events for aggregate
    events = event_store.get_events(aggregate_id)

    # Replay to rebuild state
    aggregate = Account()
    for event in events:
        aggregate.apply(event)

    return aggregate

class Account:
    def apply(self, event):
        if event.type == "AccountCreated":
            self.id = event.data["id"]
            self.balance = event.data["initial_balance"]
        elif event.type == "MoneyDeposited":
            self.balance += event.data["amount"]
        elif event.type == "MoneyWithdrawn":
            self.balance -= event.data["amount"]
```

---

## スナップショット

### 問題

```
10,000イベントを持つアカウント
読み込みのたびに: 10,000イベントをリプレイ
非常に遅い！
```

### スナップショットによる解決

```
N件のイベントごとに、現在の状態をスナップショットとして保存

イベント: 1-1000
イベント1000時点のスナップショット: {balance: 5000, ...}
イベント: 1001-2000

読み込みプロセス:
  1. スナップショットを読み込む（存在する場合）
  2. スナップショット以降のイベントのみをリプレイ

2000ではなく1000イベントのリプレイ
```

### 実装

```python
def load_aggregate_with_snapshot(aggregate_id):
    # Try to load snapshot
    snapshot = snapshot_store.get_latest(aggregate_id)

    if snapshot:
        aggregate = deserialize(snapshot.state)
        start_version = snapshot.version + 1
    else:
        aggregate = Account()
        start_version = 0

    # Replay events since snapshot
    events = event_store.get_events(
        aggregate_id,
        from_version=start_version
    )

    for event in events:
        aggregate.apply(event)

    return aggregate

def save_snapshot(aggregate_id, aggregate, version):
    snapshot_store.save(
        aggregate_id=aggregate_id,
        state=serialize(aggregate),
        version=version
    )
```

---

## プロジェクション

### 概念

```
イベント（信頼できる情報源）
    ↓ プロジェクト
リードモデル（クエリに最適化）

同じイベント → 複数のプロジェクション
各々が特定のユースケースに最適化
```

### 例

```
イベント:
  AccountCreated(id=1, owner="Alice")
  MoneyDeposited(id=1, amount=1000)
  AccountCreated(id=2, owner="Bob")
  MoneyWithdrawn(id=1, amount=500)

プロジェクション: アカウント残高
  {id: 1, balance: 500}
  {id: 2, balance: 0}

プロジェクション: アクティビティタイムライン
  [
    {time: T1, action: "アカウント1作成"},
    {time: T2, action: "アカウント1に1000入金"},
    ...
  ]

プロジェクション: オーナーディレクトリ
  {Alice: [1], Bob: [2]}
```

### プロジェクションの構築

```python
class BalanceProjection:
    def __init__(self):
        self.balances = {}

    def handle(self, event):
        if event.type == "AccountCreated":
            self.balances[event.data["id"]] = event.data.get("initial_balance", 0)
        elif event.type == "MoneyDeposited":
            self.balances[event.aggregate_id] += event.data["amount"]
        elif event.type == "MoneyWithdrawn":
            self.balances[event.aggregate_id] -= event.data["amount"]

    def rebuild_from_start(self):
        self.balances = {}
        for event in event_store.get_all_events():
            self.handle(event)
```

---

## 利点

### 完全な監査証跡

```
すべての変更が記録される
誰が何をいつ行ったか

質問: 「なぜ残高が500なのか？」
回答: イベントをリプレイして各変更を確認する
```

### 時間的クエリ

```python
def get_balance_at_time(account_id, timestamp):
    events = event_store.get_events(
        account_id,
        before=timestamp
    )

    balance = 0
    for event in events:
        if event.type == "MoneyDeposited":
            balance += event.data["amount"]
        elif event.type == "MoneyWithdrawn":
            balance -= event.data["amount"]

    return balance

# 1月1日時点の残高は？
get_balance_at_time("account-123", "2024-01-01")
```

### デバッグ

```
本番のバグ:
  1. バグを引き起こしたイベントをキャプチャ
  2. ローカルでリプレイ
  3. 完全な履歴でデバッグ
  4. 同じイベントで修正とテスト
```

### スキーマ進化

```
イベントは過去の事実
イベントを変更せず、新しいタイプを追加する

v1: UserCreated(name)
v2: UserCreated(name, email)  # 新フィールド

古いイベントは依然として有効
新しいコードが両方のバージョンを処理する
```

---

## 課題

### 結果整合性

```
イベント保存 → プロジェクション更新（非同期）

プロジェクションが古い状態のギャップ
UIが古いデータを表示する可能性がある

解決策:
  - 結果整合性を受け入れる
  - 重要な読み取りにはイベントストアから読む
  - 楽観的UIアップデート
```

### ストレージの増大

```
イベントは決して削除されない
ストレージが永遠に増大する

緩和策:
  - スナップショット（リプレイ時間の短縮）
  - アーカイブ（古いイベントをコールドストレージに移動）
  - イベントコンパクション（慎重に、特定のパターンで）
```

### イベントスキーマの変更

```
課題: 過去のイベントは不変

解決策:
  - イベントを明示的にバージョニング
  - アップキャスティング: 読み取り時に古いイベントを変換
  - 弱いスキーマ: JSONで保存し、欠落フィールドを処理
```

```python
def upcast_event(event):
    if event.type == "UserCreated" and event.version == 1:
        # Add default email for v1 events
        event.data["email"] = None
        event.version = 2
    return event
```

### 複雑なクエリ

```
イベントストアは以下に最適化:
  - 追記
  - アグリゲートによる読み取り

以下には最適化されていない:
  - アグリゲートをまたぐ複雑なクエリ
  - 集計

解決策: クエリニーズにはプロジェクションを使用
```

---

## イベントソーシングパターン

### コマンド → イベント

```python
def handle_withdraw(cmd: WithdrawCommand):
    # Load aggregate
    account = load_aggregate(cmd.account_id)

    # Validate
    if account.balance < cmd.amount:
        raise InsufficientFundsError()

    # Generate event
    event = MoneyWithdrawn(
        account_id=cmd.account_id,
        amount=cmd.amount,
        timestamp=now()
    )

    # Store event
    event_store.append(cmd.account_id, [event], account.version)

    return event
```

### サーガ/プロセスマネージャー

```
複数のアグリゲートを調整する

OrderSaga:
  On OrderPlaced:
    ReserveInventoryコマンドを送信

  On InventoryReserved:
    ChargePaymentコマンドを送信

  On PaymentCharged:
    ShipOrderコマンドを送信

  On PaymentFailed:
    ReleaseInventoryコマンドを送信
```

### マイグレーションのためのイベントリプレイ

```python
def migrate_to_new_projection():
    # Create new projection store
    new_projection = NewProjection()

    # Replay all events
    for event in event_store.get_all_events():
        new_projection.handle(event)

    # Switch over
    swap_projection(old_projection, new_projection)
```

---

## イベントソーシングを使うべき時

```
✓ 強い監査要件（金融、ヘルスケア）
✓ ビジネスルールを持つ複雑なドメイン
✓ 時間的クエリの必要性
✓ イベント駆動アーキテクチャが既に導入済み
✓ CQRSの実装
```

---

## スナップショット戦略

### なぜスナップショットを取るか

```
1,000,000イベントを持つアグリゲート → 読み込みのたびに全リプレイ？ 受け入れられない。

スナップショット = 既知のバージョンでのシリアライズされたアグリゲート状態。
スナップショットを読み込み → そのバージョン以降のイベントのみリプレイ。

スナップショットなし:  リプレイ 1..1,000,000  （~秒から分）
v999,000のスナップショットあり:  デシリアライズ + 1,000リプレイ  （~ミリ秒）
```

### いつスナップショットを取るか

```
Nイベントごと      — 100イベントごとにスナップショット。シンプルで予測可能。
時間ベース          — 最後のスナップショットがT以上古い場合。バースト書き込みに適している。
読み取り時（遅延）  — events_since_snapshot > 閾値の場合 → 読み込み後にスナップショット。
                    バックグラウンドジョブなしだが、最初の遅い読み取りがコストを負担。

トレードオフ: 頻繁すぎる → ストレージコスト / 書き込み増幅
             まれすぎる → 遅い回復 / 高いリプレイレイテンシ
```

### スナップショットストレージ

```
(aggregate_id, version)でキー付けされた別のストア:

  snapshots: aggregate_id | version | state (JSONB) | schema_version
             account-123  | 1000    | {balance:...} | 3
             account-123  | 2000    | {balance:...} | 4

schema_versionを含める — コードv3のスナップショットはv5でデシリアライズできない可能性。
schema_version < currentの場合、読み取り時にマイグレート。
```

### スナップショットマネージャー

```python
class SnapshotManager:
    def __init__(self, event_store, snapshot_store, interval=100):
        self.event_store = event_store
        self.snapshot_store = snapshot_store
        self.interval = interval

    def load(self, aggregate_id, factory):
        snapshot = self.snapshot_store.get_latest(aggregate_id)
        if snapshot:
            aggregate = deserialize(snapshot.state, snapshot.schema_version)
            from_version = snapshot.version + 1
        else:
            aggregate, from_version = factory(), 0

        events = self.event_store.get_events(aggregate_id, from_version=from_version)
        for event in events:
            aggregate.apply(event)

        if len(events) >= self.interval:  # lazy snapshot on read
            self.snapshot_store.save(
                aggregate_id=aggregate_id, version=aggregate.version,
                state=serialize(aggregate), schema_version=CURRENT_SCHEMA_VERSION)
        return aggregate
```

---

## スキーマ進化

### 問題

```
イベントは不変 — 保存されたイベントを変更できない。
しかしドメインモデルは進化する: 新しいフィールド、リネームされたフィールド、分割されたイベント。

Day 1:  OrderPlaced { order_id, total }
Day 90: OrderPlaced { order_id, total, currency, customer_tier }

古いイベントはDay 1の形状のまま。アプリケーションコードはDay 90の形状を期待。
```

### アップキャスティング

```python
# Transform old event shapes to current shape ON READ.
# Event store keeps original bytes untouched.
UPCASTERS = {
    ("OrderPlaced", 1): lambda data: {
        **data, "currency": "USD", "customer_tier": "standard",
    },
}

def upcast(event_type, version, data):
    key = (event_type, version)
    while key in UPCASTERS:
        data = UPCASTERS[key](data)
        version += 1
        key = (event_type, version)
    return data
```

### バージョン付きイベントタイプ

```
タイプ名に明示的バージョン:
  OrderPlaced_v1 { order_id, total }
  OrderPlaced_v2 { order_id, total, currency, customer_tier }

コンシューマーがmatch/switchで両方を処理。
動作するがタイプが増殖する — ほとんどの場合アップキャスティングを優先。
```

### スキーマ戦略の比較

```
弱いスキーマ（JSON、寛容なリーダー）:
  + フィールド追加が容易、レジストリ不要
  - コンパイル時安全性なし、タイプミスでサイレントエラー

強いスキーマ（Avro / Protobuf）:
  + 前方/後方互換性が強制、コンパイル時型
  - スキーマレジストリが必要、運用オーバーヘッドが大きい
```

### アンチパターン: 保存されたイベントの変更

```
ストア内のイベントを決して書き換えてはいけません。
破壊するもの: 監査証跡、決定論的リプレイ、下流コンシューマーとの因果関係。

事実を訂正するには、補償イベントを追記する:
  OrderPlaced → OrderCorrected { reason, corrected_fields }
```

---

## イベントストア技術の選択

### PostgreSQL

```
上記の「イベントストアの実装」セクションで既に示しています。
シンプルで実績があり、柔軟なイベントデータにJSONB。
(aggregate_id, version)のユニーク制約 = 楽観的並行制御。
制約違反時にアプリケーションがリトライ: 再読み込み、再バリデーション、再追記。
```

### EventStoreDB

```
専用に構築されたイベントストア（オープンソース、gRPC API）。
ネイティブなアグリゲートごとのストリーム、組み込みプロジェクション、永続サブスクリプション。
ストリームバージョンでの楽観的並行制御。リビルド用のキャッチアップサブスクリプション。
ESがアーキテクチャの中心で、チームが専用ストアを運用できる場合に選択。
```

### イベントログとしてのKafka

```
追記専用の分散ログ — イベントストアとして魅力的だが:
  - アグリゲートごとの順序保証なし（トピックパーティション ≠ アグリゲート）
  - アグリゲートごとの楽観的並行制御なし
  - 単一アグリゲートの読み取り = パーティションスキャンまたは外部インデックスの維持
  - 保持ポリシーがイベントを削除する可能性（不変性に違反）

より良い役割: イベントストアからKafkaにイベントをパブリッシュし、下流コンシューマーに配信。
イベントストア = 信頼できる情報源、Kafka = 配信レイヤー。
```

### DynamoDB

```
パーティションキー = aggregate_id、ソートキー = version。
条件付き書き込み（attribute_not_exists）= 楽観的並行制御。
サーバーレス、水平スケール、CDCのためのDynamoDB Streams。
制限: 400 KBアイテムサイズ制限、組み込みプロジェクションなし（Streams + LambdaでDIY）。
```

### 比較

```
                    PostgreSQL   EventStoreDB   Kafka      DynamoDB
楽観的並行制御       ✓ (unique)   ✓ (native)     ✗          ✓ (cond. write)
組み込みプロジェクション ✗ (DIY)   ✓              ✗          ✗ (DIY)
アグリゲートごと読取   ✓            ✓              ✗          ✓
運用の複雑さ          低           中             高         低
最適な用途           入門          ES中心        配信        サーバーレス
```

---

## イベントソーシングを使うべきでない時

```
監査不要のシンプルなCRUD
  ユーザー設定、フィーチャーフラグ、CMSコンテンツ。
  「3ヶ月前の値は何だった？」と誰も聞かない — 単純なUPDATEで十分。

意味のあるイベントがないドメイン
  設定管理、静的参照データ、ルックアップテーブル。
  まれな変更 + 興味のない履歴 = 成果のないセレモニー。

チーム経験のギャップ
  ESが要求するもの: 結果整合性、プロジェクションリビルド、冪等ハンドラー、
  スキーマ進化、アップキャスティング。急な学習曲線 → 本番でのバグ。
  完全なES採用前にイベント駆動スキルを段階的に構築する。

許容できない読み取りの古さ
  ビジネスが書き込みを即座に反映する読み取りを要求する場合、ES + CQRSの
  非同期プロジェクションラグは常に問題になる。ワークアラウンド（同期プロジェクション、
  read-your-writes）は分離の利点を損なう。

予測不能なスキーマ変更
  毎週イベント形状が変わる → アップキャスターチェーンが増大、テストマトリクスが爆発。
  まずドメインモデルを安定させ、後でESを採用する。

アンチパターン: 「すべてをイベントソーシング」
  恩恵のある境界付きコンテキストに選択的に適用:
    決済処理 → 強い監査、時間的クエリ → はい
    ユーザープロフィールCRUD → シンプルな読み書き、履歴不要 → いいえ
  同じシステムでESと非ESのコンテキストを混在させることは正常で健全。
```

---

## 重要なポイント

1. **状態ではなくイベントを保存する** - 状態は導出される
2. **イベントは不変** - 更新も削除もしない
3. **スナップショットで遅いリビルドを防ぐ** - 定期的に取得する
4. **クエリにはプロジェクション** - 同じイベントから複数のビュー
5. **結果整合性は正常** - それを前提に設計する
6. **監査証跡に最適** - 完全な履歴
7. **複雑さは現実** - シンプルなCRUDには不向き
8. **CQRSと相性が良い** - 読み取り/書き込みモデルの分離
