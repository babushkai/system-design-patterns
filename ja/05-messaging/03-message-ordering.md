# メッセージの順序保証

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../05-messaging/03-message-ordering.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

メッセージの順序保証は、メッセージが送信された順序で配信されるかどうかを決定します。選択肢は順序保証なし（最高パフォーマンス）から全体順序保証（最低パフォーマンス）まであります。ほとんどのシステムはパーティションベースの順序保証を使用します。同じキーのメッセージは順序が保証され、異なるキーは入り混じる可能性があります。ビジネス要件に基づいて選択してください。真の全体順序保証はめったに必要なく、コストが高いです。

---

## なぜ順序が重要か

### 問題

```
ユーザーアクション（この順序で送信）:
  1. アカウント作成
  2. プロフィール更新
  3. アカウント削除

順序が入れ替わって配信された場合:
  削除が最初に到着 → 「アカウントが見つかりません」エラー
  更新が到着 → 孤立したデータが作成される
  作成が到着 → アカウントが再び存在する

結果: 状態の破損
```

### 順序が重要な場合

```
重要:
  - 金融取引（入金が引き落としの前に）
  - ステートマシン遷移
  - ログ集約
  - レプリケーション

あまり重要でない:
  - アナリティクスイベント（後で並べ替え可能）
  - 通知（わずかな順序入れ替えは許容）
  - 独立した操作
```

---

## 順序保証のレベル

### 順序保証なし

```
メッセージは任意の順序で到着する可能性がある

プロデューサーが送信: A, B, C
コンシューマーが受信:  C, A, B（任意の順列）

利点:
  - 最大スループット
  - 容易なスケーリング
  - 調整が不要

使用する場合:
  - 操作が独立している
  - コンシューマーが任意の順序を処理できる
```

### プロデューサー内FIFO

```
各プロデューサーのメッセージは順序通りに到着する
異なるプロデューサーは入り混じる可能性がある

プロデューサー1: A1, B1, C1 → 順序通りに到着
プロデューサー2: A2, B2, C2 → 順序通りに到着

ただし全体として: A1, A2, B1, C2, B2, C1（入り混じり）

使用する場合:
  - 同一ソースからのイベントが順序付けされる必要がある
  - 異なるソースは独立している
```

### パーティション/キー内FIFO

```
同じキーのメッセージは順序が保証される
異なるキーは入り混じる可能性がある

Key=user1: login, update, logout → 順序保証
Key=user2: login, purchase → 順序保証

ただし: user1.login, user2.login, user1.update...（入り混じり）

最も一般的なアプローチ
Kafka、SQS FIFOがこれを使用
```

### 全体順序保証

```
すべてのメッセージが厳密なグローバル順序

送信: A, B, C, D, E
受信: A, B, C, D, E（正確に）

必要条件:
  - 単一パーティション/キュー
  - または分散合意

コストが高く、スループットを制限する
真に必要なケースはまれ
```

---

## Kafkaの順序保証

### パーティションベース

```
3パーティションのトピック:
  Partition 0: [A, D, G]
  Partition 1: [B, E, H]
  Partition 2: [C, F, I]

パーティション内: 厳密な順序保証
パーティション間: 順序保証なし

プロデューサー:
  - Key = null: パーティションへのラウンドロビン
  - Key = "user123": 一貫したパーティションへのハッシュ
```

### コンシューマーグループ

```
Consumer Group A:
  Consumer 1 ← Partition 0
  Consumer 2 ← Partition 1
  Consumer 3 ← Partition 2

各パーティションは1つのコンシューマーが処理する
パーティション内の順序が保持される

コンシューマーが失敗した場合:
  パーティションが再割り当てされる
  最後にコミットされたオフセットから継続する
```

### 順序保証

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

## SQS FIFOの順序保証

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

### 重複排除

```
FIFOキューは以下で重複排除します:
  - MessageDeduplicationId（明示的）
  - コンテンツハッシュ（コンテンツベースの重複排除が有効な場合）

ウィンドウ: 5分
ウィンドウ内の同一ID → メッセージはドロップ
```

### スループット制限

```
Standard SQS: 無制限のスループット
FIFO SQS: 300 msg/sec（バッチングで3000）

メッセージグループあたり: 最大300 msg/sec
スケールするには複数グループを使用
```

---

## 順序保証の実装

### シーケンス番号

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

### リシーケンシングバッファ

```
受信（順序外）: 3, 1, 4, 2, 5

バッファの状態:
  3を受信: buffer=[3], 1を待機
  1を受信: 1を処理, buffer=[3], 2を待機
  4を受信: buffer=[3,4], 2を待機
  2を受信: 2,3,4を処理, buffer=[], 5を待機
  5を受信: 5を処理

考慮事項:
  - バッファサイズの制限
  - 欠落シーケンスのタイムアウト
  - ギャップ検出
```

### ギャップの処理

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

## 順序保証とスケーリング

### パーティション戦略

```
エンティティIDによる:
  user-123 → partition 0
  user-456 → partition 1
  user-123のすべてのイベントが順序保証 ✓

時間バケットによる:
  イベント 00:00-00:05 → partition 0
  イベント 00:05-00:10 → partition 1
  バケット内で時間順

ハッシュによる:
  hash(key) % num_partitions
  均一分散
```

### パーティションの増加

```
初期: 4パーティション
  Key A → partition 1
  Key B → partition 3

パーティション追加後: 8パーティション
  Key A → partition 5（異なる！）
  Key B → partition 3（変わる可能性あり）

問題: キー→パーティションのマッピングが変わる

解決策:
  - 初めから多めにパーティション分割する（100以上）
  - コンシステントハッシングを使用する
  - パーティション増加をコンシューマーと調整する
```

### 並列処理の制限

```
厳密に順序付けされたキュー:
  最大並列度 = キーの数

  1000のユニークキー = 1000の並列操作

単一のキーに高ボリュームがある場合:
  そのキーがボトルネックになる
  タイムウィンドウイングまたはサブキーを検討する
```

---

## 一般的なパターン

### エンティティごとの順序

```python
# All events for an entity go to same partition
def get_partition_key(event):
    return event.entity_id

# Examples:
# Order events → key = order_id
# User events → key = user_id
# Session events → key = session_id
```

### 因果関係による順序

```
イベントBがイベントAに依存する場合:
  同じパーティションキーを使用する

ユーザーが注文を作成 → 注文イベント
  両方のキー: order_id
  作成が更新の前に保証される

ただし: ユーザープロフィール更新は注文の順序を必要としない
  異なるパーティションキーでOK
```

### ハイブリッド順序

```
クリティカルパス: FIFOキュー（順序保証あり、遅い）
ベストエフォート: Standardキュー（高速、順序保証なし）

Create/Update/Delete → FIFO（順序が重要）
アナリティクスイベント → Standard（順序は不要）
```

---

## トレードオフ

| 順序レベル | スループット | レイテンシ | 複雑さ |
|------------|------------|---------|--------|
| なし | 最高 | 最低 | 最低 |
| プロデューサーごと | 高 | 低 | 低 |
| キーごと | 中 | 中 | 中 |
| 全体 | 最低 | 最高 | 最高 |

### 決定フレームワーク

```
質問1: メッセージは共有状態に影響するか？
  いいえ → 順序保証は不要
  はい → 続ける

質問2: 状態はキーでパーティション分割されているか？
  はい → キーごとの順序保証で十分
  いいえ → 続ける

質問3: 全体順序保証が本当に必要か？
  通常はいいえ → 設計を再検討する
  はい → パフォーマンスペナルティを受け入れる
```

---

## 順序問題のデバッグ

### 順序外の検出

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

### 順序のためのロギング

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

## Kafkaパーティション順序の詳細

### 順序保証のスコープ

```
単一パーティション内:
  オフセットによる全体順序。コンシューマーは0, 1, 2, 3...と順次読み取る。

パーティション間:
  順序保証なし。Partition 0のオフセット5はPartition 1のオフセット5より
  新しいかもしれないし古いかもしれない。poll()は任意の順序でバッチを返す。
```

### パーティションキーの選択

```
キーは因果的に関連するメッセージをグループ化する:
  注文ライフサイクル → order_id（作成、支払い、発送 → 順序保証）
  ユーザーアクティビティ → user_id（ログイン、クリック、ログアウト → 順序保証）
  デバイステレメトリ → device_id（時系列の読み取り値）

アンチパターン:
  random_uuid → 負荷を分散するが順序が壊れる
  event_type  → 無関係なエンティティをグループ化する
```

### ホットパーティション問題

```
著名ユーザーのuser_id → 1つのパーティションに100倍のトラフィック、他はアイドル。

緩和策:
  1. 複合キー（user_id + session_id）— ユーザー単位からセッション単位の順序保証にトレード
  2. 不均衡を受け入れる — ホットコンシューマーを垂直にスケール、パーティションラグを監視
  3. アプリレベルシャーディング — 仮想サブユーザーに分割し、下流でマージ
```

### リバランスと順序

```
コンシューマーグループリバランス（コンシューマーの参加/離脱/クラッシュ）:

  Eagerプロトコル（2.4以前のデフォルト）:
    すべてのコンシューマーがリバランス中にフェッチを停止
    短い処理ギャップだが、順序外配信はない
    リバランス後、最後にコミットされたオフセットから再開

  Cooperativeリバランシング（Kafka 2.4+）:
    取り消されたパーティションのみ停止、他は処理を継続
    影響範囲を縮小 — 大規模コンシューマーグループに推奨
```

### インフライトリクエストとリトライ

```
プロデューサー設定: max.in.flight.requests.per.connection

  5に設定（デフォルト）:
    Batch 1が失敗、Batch 2が成功、Batch 1がリトライ
    → ブローカーが受信: Batch 2, Batch 1 → 順序外

  1に設定:
    1リクエストずつ → 正しい順序だが、スループットが低下

  冪等プロデューサー（enable.idempotence=true）:
    ブローカーがプロデューサーのシーケンスを追跡し、順序外の書き込みを拒否
    max.in.flight=5でも安全。順序重視のワークロードに推奨。
```

---

## サービス間の順序保証

### 根本的な問題

```
単一サービス: BがE2でクラッシュ → 再起動 → オフセットからリプレイ → 問題なし

サービス間: BがE1→F1、E2→F2を処理し、Cにパブリッシュ
  F2がF1よりCに早く到着（異なるトピック/パーティション）→ 順序外

独立したトピックとサービス間の順序はどのブローカーも保証しません。
```

### サービス間順序のためのシーケンス番号

```python
# Publisher embeds monotonic sequence per aggregate
def publish(self, aggregate_id, event):
    version = self.store.increment_version(aggregate_id)
    event['aggregate_version'] = version
    broker.send(key=aggregate_id, value=event)

# Consumer enforces version ordering
def on_event(self, event):
    version = event['aggregate_version']
    last = self.store.get_last_version(event['aggregate_id'])

    if version == last + 1:      # Expected → apply
        self.apply(event)
        self.store.set_last_version(event['aggregate_id'], version)
    elif version <= last:         # Duplicate → skip
        pass
    else:                         # Future → buffer
        self.buffer(event['aggregate_id'], event)
```

### ベクタークロックによる因果順序

```
イベントがエンティティ間で因果関係の依存性を持つ場合:

  Event A（ユーザー作成）  → clock {user_svc: 1}
  Event B（注文作成）      → clock {order_svc: 1, user_svc: 1}

  コンシューマーがBをAより先に受信:
    user_svc:1が欠落 → Bをバッファ → Aを受信 → A、次にBを処理

  よりシンプルな代替 — 因果トークン:
    Event AがトークンT1を生成。Event Bが依存関係を宣言: [T1]。
    コンシューマーが確認: T1を見たか？ いいえ → Bをバッファ。

パーティションキー順序が不十分な場合（エンティティ間チェーン）にのみ使用。
```

---

## 順序保証 vs パフォーマンスのトレードオフ

### 保証のスペクトラム

| 保証 | スループット | 並列度 | 使用する場合 |
|------|-----------|--------|-------------|
| 順序なし（ファンアウト） | 最大 | 無制限 | 通知、アナリティクス、ログ転送 |
| パーティション順序（キーごと） | 高 | パーティション数 | 注文ライフサイクル、ユーザーアクティビティ、デバイステレメトリ |
| 全体順序（単一パーティション） | 最低 | コンシューマー1台 | 金融台帳、分散ログ、チェンジログ |

### スループット（概算）

```
Kafka（3ブローカー、100バイトメッセージ）:
  順序なし: ~2M msg/sec | パーティション: ~1M | 全体: ~50K

SQS:
  Standard: ~120K msg/sec | FIFOグループあたり: 300（バッチ3000）

パーティション順序と全体順序のギャップは20倍以上です。
```

### 90%ルール

```
ほとんどのシステムはパーティションレベルの順序保証だけで十分です。

問いかけ: 「異なるエンティティのイベントに相対的な順序が必要か？」

  ほぼ常にいいえ:
    ユーザーA vs ユーザーB → 独立
    注文#100 vs #101 → 独立

  本当にはいの場合:
    共通のアグリゲートキーの下にエンティティをマージできるか？
    シングルライターパターンを使用できるか？
    代替手段を尽くした後にのみ: 単一パーティションペナルティを受け入れる。
```

---

## 順序回復パターン

### 順序外の検出

```
検出シグナル:
  - シーケンスジャンプ: 5を受信、3を期待（ギャップ = [3, 4]）
  - タイムスタンプの後退: event.ts < last_processed.ts
  - バージョンスキップ: aggregate_versionが2から5にジャンプ

モニタリング:
  メトリクス: ordering_gap_detected{topic, partition, consumer_group}
  ギャップ率がベースラインを超えたらアラート
```

### バッファリング戦略

```
ギャップが埋まるまで将来のイベントを保持:

  last_processed=2, buffer={5: event5}
  3を受信 → 処理、4を受信 → 処理、バッファに5 → 処理

  リスク: イベントが本当に失われた場合、バッファが無制限に増大
  緩和策: キーごとのバッファサイズを制限（例: 最大1000イベント）
```

### タイムアウトして続行

```
N秒間欠落イベントを待った後、ギャップを受け入れる:

  ギャップ [3,4] を検出 → 30秒タイマーを開始
  タイムアウト → ギャップをログ、5にスキップ、バッファされたイベントを処理

  遅延到着（タイムアウト後に3または4）:
    オプションA: 無視（通り過ぎた）
    オプションB: 冪等であれば遡及的に適用
    オプションC: 手動レビューのためデッドレターキュー
```

### ソースからの再処理

```
ソースがリプレイをサポートしている場合、欠落イベントをリクエスト:

  Kafka:  consumer.seek(partition, offset) — 前方に再消費
  SQS:    Visibility Timeout後にメッセージが戻る — 自動リトライ
  カスタム: GET /events?aggregate_id=X&after_version=2 — 直接取得

重要な原則: コンシューマーが冪等である場合にのみリプレイは安全です。
冪等性パターンについては 04-delivery-guarantees.md を参照してください。
```

---

## 重要なポイント

1. **全体順序はコストが高い** - 本当に必要でない限り避ける
2. **キーごとの順序で通常は十分** - エンティティIDでパーティション分割する
3. **同じキー → 同じパーティション → 同じコンシューマー** - 順序のチェーン
4. **シーケンス番号で検証が可能** - ギャップと重複を検出する
5. **順序外のメッセージをバッファする** - ギャップにはタイムアウトを設定
6. **パーティション増 = 並列度増** - ただしパーティション内順序保証
7. **スケーリングがキーマッピングに影響** - 初めから多めにパーティション分割
8. **独立したキーを設計する** - 並列度を最大化する
