# リーダーレスレプリケーション

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../02-distributed-databases/03-leaderless-replication.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

リーダーレスレプリケーションはリーダーを完全に排除します。クライアントは複数のノードに直接書き込み、複数のノードから読み取り、クォーラムを使用して一貫性を確保します。書き込みに単一障害点がありません。トレードオフ：クォーラムの計算が必要で、結果整合性のセマンティクス、慎重な競合処理が求められます。Dynamoによって普及し、Cassandra、Riak、Voldemortで使用されています。

---

## 仕組み

### 基本コンセプト

```
No leader. All nodes are equal.

Write request → send to ALL replicas
Read request → read from MULTIPLE replicas
Use quorum to determine success/latest value
```

### 書き込みパス

```
Client writes to N replicas simultaneously:

┌────────┐
│ Client │───write(x=1)───┬─────────┬─────────┐
└────────┘                │         │         │
                          ▼         ▼         ▼
                     ┌───────┐ ┌───────┐ ┌───────┐
                     │Node A │ │Node B │ │Node C │
                     │  ✓    │ │  ✓    │ │  ✗    │
                     └───────┘ └───────┘ └───────┘
                          │         │
                          ▼         ▼
                     2 of 3 succeeded → write succeeds (if W=2)
```

### 読み取りパス

```
Client reads from R replicas, takes latest:

┌────────┐
│ Client │───read(x)───┬─────────┬─────────┐
└────────┘             │         │         │
                       ▼         ▼         ▼
                  ┌───────┐ ┌───────┐ ┌───────┐
                  │Node A │ │Node B │ │Node C │
                  │ x=1   │ │ x=1   │ │ x=0   │
                  │ v=5   │ │ v=5   │ │ v=3   │
                  └───────┘ └───────┘ └───────┘
                       │         │
                       ▼         ▼
                  Compare versions → return x=1 (v=5 is latest)
```

---

## クォーラムの計算

### パラメータ

```
N = Number of replicas
W = Write quorum (how many must acknowledge write)
R = Read quorum (how many to read from)
```

### クォーラム条件

強い一貫性のためには：**W + R > N**

```
Example: N=3, W=2, R=2

Write to any 2: {A, B} or {A, C} or {B, C}
Read from any 2: {A, B} or {A, C} or {B, C}

Overlap guaranteed:
  Write set ∩ Read set ≠ ∅
  At least one node has latest value
```

### 視覚的な証明

```
N=5, W=3, R=3

Nodes:     [A] [B] [C] [D] [E]
            │   │   │
Write to:   ✓   ✓   ✓           (any 3)
            │   │   │   │   │
Read from:          ✓   ✓   ✓   (any 3)
                    │
              Overlap at C
```

### 一般的な設定

| 設定 | N | W | R | 特性 |
|------|---|---|---|------|
| 強い一貫性 | 3 | 2 | 2 | W+R=4 > 3 |
| 読み取り重視 | 3 | 3 | 1 | 高速読み取り、低速書き込み |
| 書き込み重視 | 3 | 1 | 3 | 高速書き込み、低速読み取り |
| 結果整合性 | 3 | 1 | 1 | 最速、古いデータを読む可能性あり |

---

## 一貫性の保証

### W + R > N の場合

**強い一貫性**（線形化可能性は実現可能だが保証されない）：
- すべての読み取りが最新の書き込みを確認します
- ただし：並行操作は依然として異常を引き起こす可能性があります

```
Scenario: W=2, R=2, N=3

Write(x=1) to A, B succeeds
Read from B, C:
  B has x=1
  C has x=0 (hasn't received write yet)

Return x=1 (latest version wins)
```

### W + R <= N の場合

**結果整合性：**
- 古いデータを読む可能性があります
- 最終的に収束します

```
Scenario: W=1, R=1, N=3

Write(x=1) to A only
Read from C only:
  C has x=0

Stale read! But eventually C will get x=1
```

### スロッピークォーラム

N台のノードが利用できない場合、「代替」ノードに書き込みます。

```
Normal: Write to {A, B, C}
A and B down: Write to {C, D, E} (D, E are substitutes)

Later: "Hinted handoff" moves data back to A, B
```

トレードオフ：
- 可用性の向上
- 一貫性の低下（クォーラムが重複しない可能性）

---

## バージョン競合

### 並行書き込み

```
Client 1: write(x, "A") → nodes {1, 2}
Client 2: write(x, "B") → nodes {2, 3}  (concurrent)

State after writes:
  Node 1: x = "A"
  Node 2: x = "A" or "B" (last one wins locally)
  Node 3: x = "B"

Read from {1, 3}: get {"A", "B"} — conflict!
```

### 競合検出のためのベクタークロック

各書き込みにはベクタークロックが付随します：

```
Write 1 at node A: {A:1}
Write 2 at node B: {B:1}

Compare:
  {A:1} vs {B:1}
  Neither dominates → concurrent → conflict

Merge or use LWW
```

### シブリング

すべての競合する値をアプリケーションに返します：

```
Read(x) → {
  values: ["A", "B"],
  context: merged_vector_clock
}

Application decides how to merge
Next write includes context → system knows what was merged
```

---

## リードリペアとアンチエントロピー

### リードリペア

読み取り時に古いレプリカを修正します：

```
Read from A, B, C:
  A: x=1, version=5
  B: x=1, version=5
  C: x=0, version=3  ← stale

Return x=1 to client

Background: update C with version=5
```

**日和見的：** たまたま読み取ったノードのみを修復します。

### アンチエントロピー（バックグラウンド修復）

プロアクティブにレプリカを同期します：

```
Periodically:
  for each pair of nodes (A, B):
    compare merkle trees
    for each different key:
      sync latest version
```

**マークルツリー**は効率的な比較を可能にします：
```
         [root hash]
         /          \
    [hash L]      [hash R]
    /      \      /      \
 [h1]   [h2]   [h3]   [h4]

Compare roots: different? → compare children
O(log n) to find differences in large datasets
```

---

## ヒンテッドハンドオフ

ノードが一時的に利用不可の場合：

```
Normal write target: A, B, C
A is down

Write to: B, C, D (D is hint recipient)
D stores: {key: x, value: 1, hint_for: A}

When A recovers:
  D sends hinted data to A
  D deletes hints
```

**目的：**
- 書き込みの可用性を維持します
- 一時的な障害中の書き込み損失を防ぎます

**制限事項：**
- 永続的な障害には対応しません
- ターゲットがダウンし続けるとヒントが蓄積する可能性があります

---

## 障害の処理

### 読み取り/書き込みの回復力

```
With N=5, W=3, R=3:
  Tolerate 2 failed nodes for writes
  Tolerate 2 failed nodes for reads

With N=5, W=2, R=4:
  Tolerate 3 failed nodes for writes
  Tolerate 1 failed node for reads
```

### 古いデータの検出

```python
def read_with_quorum(key, R):
  responses = parallel_read(key, all_replicas)
  wait_for(R, responses)

  latest = max(responses, key=lambda r: r.version)

  # Trigger read repair for stale replicas
  for r in responses:
    if r.version < latest.version:
      async_repair(r.node, key, latest)

  return latest.value
```

---

## 実際のシステム

### Amazon Dynamo

オリジナルのリーダーレスシステム（2007年の論文）：

```
- Consistent hashing for partitioning
- Vector clocks for versioning
- Sloppy quorums for availability
- Merkle trees for anti-entropy
- Hinted handoff for temporary failures

Design goal: "Always writable" shopping cart
```

### Apache Cassandra

```sql
-- Write with quorum
INSERT INTO users (id, name) VALUES (1, 'Alice')
USING CONSISTENCY QUORUM;

-- Read with one replica (fast, possibly stale)
SELECT * FROM users WHERE id = 1
USING CONSISTENCY ONE;

-- Configurable per-query
```

設定：
```yaml
# cassandra.yaml
num_tokens: 256
hinted_handoff_enabled: true
max_hint_window_in_ms: 10800000  # 3 hours
```

### Riak

```erlang
%% Write with W=2
riakc_pb_socket:put(Pid, Object, [{w, 2}]).

%% Read with R=2, return siblings
riakc_pb_socket:get(Pid, <<"bucket">>, <<"key">>, [{r, 2}]).

%% Application resolves siblings
resolve_siblings(Siblings) ->
    %% Custom merge logic
    merged_value.
```

---

## チューナブル一貫性

### リクエストごとの設定

```
Request 1: Strong consistency
  W=quorum, R=quorum

Request 2: Fast write
  W=1, R=quorum

Request 3: Fast read
  W=quorum, R=1

Request 4: Fastest (eventual)
  W=1, R=1
```

### 一貫性レベル（Cassandra）

| レベル | 意味 |
|--------|------|
| ONE | 1つのレプリカ |
| TWO | 2つのレプリカ |
| THREE | 3つのレプリカ |
| QUORUM | データセンター内の過半数 |
| EACH_QUORUM | 各データセンターの過半数 |
| LOCAL_QUORUM | ローカルデータセンターの過半数 |
| ALL | すべてのレプリカ |
| ANY | 任意のノード（ヒントを含む） |

---

## エッジケースと落とし穴

### 書き込み-読み取り競合

```
Time:     T1          T2          T3
Client A: write(x=1, W=2)
Client B:             read(x, R=2)

If B's read arrives before write propagates to quorum:
  B might read stale value

Not linearizable even with W+R > N
```

### Last-Writer-Winsのデータ損失

```
Concurrent writes:
  Client A: write(x, "A") at t=100
  Client B: write(x, "B") at t=101

LWW resolves to "B"
Client A's write is lost

No error returned to Client A!
```

### 障害時のクォーラムサイズ

```
N=5, W=3, R=3 normally

2 nodes permanently fail, not replaced:
  Effective N=3
  W=3 → requires all remaining nodes (less resilient)

Solution: Replace failed nodes, or adjust quorum settings
```

---

## リーダーレスシステムのモニタリング

### 主要メトリクス

| メトリクス | 説明 | アクション |
|-----------|------|----------|
| リードリペアレート | 秒あたりの修復 | 高い = 不整合 |
| ヒントキューサイズ | 保留中のヒント | 増加中 = ノードの問題 |
| クォーラム成功率 | クォーラム達成率 | <100% = 可用性の問題 |
| 読み取りレイテンシ p99 | 遅い読み取り | ストラグラーノードの確認 |
| バージョン競合 | 作成されたシブリング | 高い = 並行書き込み |

### ヘルスチェック

```python
def check_cluster_health():
  for node in nodes:
    # Check responsiveness
    if not ping(node):
      alert(f"Node {node} unreachable")

    # Check hint queue
    hints = get_hint_count(node)
    if hints > threshold:
      alert(f"Node {node} hint queue: {hints}")

    # Check anti-entropy
    last_repair = get_last_repair_time(node)
    if now() - last_repair > max_repair_interval:
      alert(f"Node {node} repair overdue")
```

---

## リーダーレスを使用すべきケース

### 適しているケース

- 高い書き込み可用性が重要な場合
- マルチデータセンターデプロイメント
- 結果整合性を許容できる場合
- シンプルなキーバリューワークロード
- 既知の競合解決戦略がある場合

### 適していないケース

- トランザクションが必要な場合
- 強い一貫性が必要な場合
- 複雑なクエリ
- 競合を処理できないアプリケーション
- 小規模なデータセット（オーバーヘッドに見合わない）

---

## クォーラムのエッジケース

### スロッピークォーラムとヒンテッドハンドオフの落とし穴

Dynamoスタイルのスロッピークォーラムは、ネットワークパーティション中にホームノード以外への書き込みを許可します。これによりシステムは書き込み可能な状態を維持しますが、**厳密なクォーラム保証が破られます**：

```
Home nodes for key K: {A, B, C}
Partition isolates A and B

Sloppy quorum write (W=2): writes to {C, D}
Sloppy quorum read  (R=2): reads from {A, B}

Overlap = ∅ → stale read despite W + R > N
```

ヒンテッドハンドオフは最終的にデータをAとBに返しますが、それにかかる時間に**上限はありません**。そのギャップの間、クォーラムがスロッピーであったことを示す兆候なく、クライアントは古い状態を観察します。

### リードリペアの競合

複数のコーディネーターが同じキーに対して並行してリードリペアを実行すると、修復書き込みが競合する可能性があります：

```
Coordinator X reads key K → sees v5 on A, v3 on B → repairs B to v5
Coordinator Y reads key K → sees v5 on A, v4 on C → repairs C to v5
                         → also sees v3 on B (before X's repair lands)
                         → repairs B to v5 again (duplicate but harmless)

Dangerous case: if a new write v6 lands between the read and the repair,
the repair can overwrite v6 with v5 → data regression
```

Cassandraはセルレベルのタイムスタンプを使用してこれを軽減します。修復書き込みは元のタイムスタンプを保持するため、より高いタイムスタンプを持つ新しい書き込みが古い修復で上書きされることはありません。

### 書き込みタイムアウトと永続化されたデータ

書き込みはWノードで成功する可能性がありますが、コーディネーターが確認応答を受け取る前にタイムアウトすることがあります。クライアントはエラーを確認してリトライします：

```
Attempt 1: write(x=1, id=abc) → W nodes store it → coordinator timeout → client error
Attempt 2: write(x=1, id=abc) → W nodes store again

Without idempotency keys, both writes are recorded.
For counters or append operations, this causes double-counting.
```

対策：クライアント生成の冪等性トークンを使用するか、すべての書き込みを設計上冪等にします（デルタではなくフルステート置換）。

### 異種ノードのための加重クォーラム

すべてのレプリカが同じレイテンシや信頼性を持つわけではありません。クロスリージョンのレプリカは100ms離れている可能性がありますが、ローカルレプリカはサブミリ秒です。加重クォーラムは、ローカルまたは高速なノードにより高い重みを割り当てます：

```
Node A (local):        weight=2
Node B (local):        weight=2
Node C (cross-region): weight=1

Total weight = 5
W = 3 (majority of weight)

Writing to A + B satisfies W=4 ≥ 3 without waiting for cross-region C
```

これによりテイルレイテンシは低減しますが、両方のローカルノードが同時に障害を起こした場合のデータ損失リスクが増加します。

---

## アンチエントロピーとリペア

### マークルツリーの比較

各レプリカはトークン範囲にわたるマークル（ハッシュ）ツリーを保持します。乖離を検出するために、2つのレプリカがツリーをトップダウンで比較します：

```
Replica A tree          Replica B tree
     [H_root]               [H_root']        ← roots differ
     /       \               /       \
  [H_L]    [H_R]         [H_L]    [H_R']     ← right subtree differs
  /    \    /    \        /    \    /    \
[h1] [h2] [h3] [h4]    [h1] [h2] [h3] [h4'] ← only h4 segment differs

Result: only keys in segment 4 need synchronization
Comparison cost: O(log n) instead of O(n)
```

ツリーは定期的に再構築されます（Cassandraは`nodetool repair`で再構築します）。再構築の間、ツリーは古くなる可能性があるため、リペアはポイントインタイムのスナップショット調整です。

### Cassandraリペア操作

Cassandraは2つのモードで`nodetool repair`を提供します：

| モード | 動作 | ユースケース |
|--------|------|-------------|
| フルリペア | マークルツリーを使用してトークン範囲のすべてのデータを比較 | ノード交換後、スキーマ変更後 |
| インクリメンタルリペア | 最後のリペア以降に書き込まれたSSTableのみ比較 | 定期メンテナンス、I/Oコストが低い |

**`gc_grace_seconds`との重要な相互作用：**

```
gc_grace_seconds = 864000 (10 days, default)

Day 1:  DELETE key K → tombstone created on nodes A, B
Day 5:  Node C comes back online (was down since before Day 1)
Day 11: gc_grace expires → A, B discard tombstone
Day 12: Anti-entropy runs → C still has key K, A and B do not
         → C's value propagates back → zombie data resurrection

Prevention: always run repair within gc_grace_seconds window
```

### リードリペア vs アクティブアンチエントロピー

| 特性 | リードリペア（受動的） | アンチエントロピー（能動的） |
|------|----------------------|--------------------------|
| トリガー | クライアントの読み取り | バックグラウンドのスケジュールタスク |
| カバレッジ | 読み取られたキーのみ | トークン範囲のすべてのキー |
| レイテンシへの影響 | 読み取りパスに修復書き込みを追加 | クライアント向けの影響なし |
| コールドデータ | 読み取られなければ修復されない | スケジュールに従って修復 |
| リソースコスト | 読み取りトラフィックに比例 | データサイズに比例 |

本番システムは両方を使用します。リードリペアはホットパスの不整合を即座にキャッチし、スケジュールされたアンチエントロピーはコールドデータの収束を確保します。

### リペアスケジュールのベストプラクティス

- `gc_grace_seconds`内に少なくとも1回フルリペアを実行します（10日間の猶予期間の場合、通常は毎週）
- ノード間でリペアをずらしてI/Oストームを回避します
- リペア中のストリーミングスループットを`nodetool netstats`で監視します
- サブレンジリペア（`-st`、`-et`フラグ）でトークン範囲間の並列化を行います

---

## Dynamo vs Cassandraの違い

オリジナルのDynamo論文とCassandraは、並行書き込みの処理方法で大きく異なります。これらの違いを理解することで、システムを切り替える際の誤った仮定を防ぎます。

### 競合検出と解決

| 機能 | Dynamo | Cassandra | Riak | ScyllaDB |
|------|--------|-----------|------|----------|
| バージョニング | ベクタークロック | セルレベルタイムスタンプ | ベクタークロック（ドット付き） | セルレベルタイムスタンプ |
| 競合検出 | 並行書き込みを検出 | 検出なし — 常にLWW | 並行書き込みを検出 | 検出なし — 常にLWW |
| 解決戦略 | クライアント側マージ | Last-Writer-Wins（暗黙的） | シブリングマージ（クライアント） | Last-Writer-Wins（暗黙的） |
| クライアントの複雑さ | 高 — シブリングの処理が必要 | 低 — 読み取りは1つの値を返す | 高 — シブリングの処理が必要 | 低 — 読み取りは1つの値を返す |
| 競合時のデータ損失 | なし（クライアントが両方を確認） | あり（古いタイムスタンプが破棄） | なし（クライアントが両方を確認） | あり（古いタイムスタンプが破棄） |

### Dynamo：ベクタークロックとクライアントマージ

Dynamoのショッピングカートは典型的な例です。2つの並行`add-to-cart`操作により、クライアントがユニオンマージする必要があるシブリングが生成されます：

```
Cart v1: {widget}

Client A: add(gadget) → v2a: {widget, gadget}  [clock: {A:1}]
Client B: add(gizmo)  → v2b: {widget, gizmo}   [clock: {B:1}]

Neither dominates → conflict → client receives both siblings
Client merges: {widget, gadget, gizmo} → writes v3 [clock: {A:1, B:1}]
```

トレードオフ：データ損失はありませんが、すべてのクライアントがマージロジックを実装する必要があります。「削除済み」セットで追跡しないと、削除されたアイテムが再表示される可能性があります（`04-consistency-models.md`のCRDTの議論を参照）。

### Cassandra：Last-Writer-Winsのシンプルさ

Cassandraは常に最も高いタイムスタンプを選択することで、クライアント側の複雑さを回避します。これは以下の条件で安全です：

- 各キーに単一のライターがある（並行更新なし）
- 書き込みがフルステート置換である（部分更新ではない）
- ノード間のクロックスキューが小さい（NTP同期済み）

これらの仮定が破られると、データは暗黙的に失われます。並行書き込みが破棄されたことを検出するメカニズムはありません。

### RiakとScyllaDB

**Riak**はドット付きバージョンベクター（クラシックなベクタークロックの最適化）を使用し、オリジナルのDynamoモデルに最も近い本番システムです。`allow_mult=true`で設定可能なシブリング解決をサポートしています。

**ScyllaDB**はCassandraとワイヤ互換であり、同じLWWセマンティクスを継承していますが、C++実装により大幅に低いテイルレイテンシ（p99）を提供します。同じ一貫性のトレードオフが適用されます。

---

## 本番環境の設定

### Cassandraのレプリケーションと一貫性

```sql
-- SimpleStrategy: single datacenter, development
CREATE KEYSPACE myapp
  WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3};

-- NetworkTopologyStrategy: multi-datacenter production
CREATE KEYSPACE myapp
  WITH replication = {
    'class': 'NetworkTopologyStrategy',
    'us-east': 3,
    'eu-west': 3
  };

-- Per-query consistency level
SELECT * FROM users WHERE id = ? USING CONSISTENCY LOCAL_QUORUM;
INSERT INTO events (id, data) VALUES (?, ?) USING CONSISTENCY EACH_QUORUM;
```

`LOCAL_QUORUM`は最も一般的な本番環境の選択です。単一のデータセンター内でクォーラム保証を維持しながら、クロスリージョンレイテンシを回避します。

### DynamoDBの設定

| 設定 | 結果整合性のある読み取り | 強い一貫性のある読み取り |
|------|----------------------|----------------------|
| レイテンシ | 低い | 高い（リーダーからの読み取りが必要） |
| コスト | 4 KBあたり0.5 RCU | 4 KBあたり1.0 RCU |
| 古さ | 最大約1秒 | なし |
| 可用性 | 高い | パーティション中は低い |

DynamoDBはデフォルトで結果整合性のある読み取りです。強い一貫性のある読み取りには`ConsistentRead=true`が必要で、コストは2倍になります。オンデマンドキャパシティモードはプロビジョニングを不要にしますが、リクエストあたりのコストが高くなります — 予測不可能なワークロードに適しています。

### モニタリングと可観測性

大規模なリーダーレスシステムの運用における主要メトリクス：

```
Coordinator latency vs replica latency:
  coordinator_latency = max(replica_latencies) + coordination_overhead
  If p99 coordinator >> p99 replica → straggler problem

Speculative retries:
  Send read to extra replica if first R haven't responded within p50
  Reduces tail latency but increases read amplification
  Cassandra: speculative_retry = '99p'  (retry if slower than p99)

Read repair rate:
  Sustained high rate → nodes falling behind or frequent restarts
  Sudden spike → possible clock skew or network partition recovery
```

### 一貫性を緩和すべきケース

すべてのクエリにクォーラムが必要なわけではありません。特定のユースケースで一貫性を下げることで、レイテンシとコストを節約できます：

| ユースケース | 推奨CL | 根拠 |
|-------------|--------|------|
| 分析/レポートクエリ | ONE | 古さは許容可能、クラスタ負荷の軽減 |
| 検索インデックス構築 | ONE | インデックスは更新される。古いデータは一時的 |
| キャッシュウォーミング | ONE | キャッシュミスは強い読み取りにフォールバック |
| ユーザーアクティビティフィード | LOCAL_ONE | フィードは短い古さを許容 |
| 金融トランザクション | LOCAL_QUORUM / EACH_QUORUM | 正確性が必要 |
| クロスリージョン災害復旧の読み取り | LOCAL_QUORUM | クロスリージョンレイテンシの回避 |

---

## 主要なポイント

1. **単一障害点がない** - どのノードでも読み取り/書き込みが可能です
2. **クォーラムが一貫性を決定** - W + R > N で強い一貫性
3. **競合はアプリケーションの問題** - LWWまたはシブリング解決
4. **リードリペアは日和見的** - アンチエントロピーがバックグラウンド同期を提供します
5. **ヒンテッドハンドオフは可用性を助ける** - ただし一貫性には寄与しません
6. **スロッピークォーラムは一貫性をトレード** - パーティション中の可用性のため
7. **リクエストごとにチューニング** - 異なる操作には異なる保証が必要です
8. **線形化可能ではない** - 強いクォーラムでも並行操作は異常を引き起こします
