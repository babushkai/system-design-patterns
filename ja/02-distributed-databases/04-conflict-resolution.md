# コンフリクト解決

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../02-distributed-databases/04-conflict-resolution.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

コンフリクトは、レプリカ間で同時に更新が発生した場合に起こります。解決戦略はシンプルなもの（最終書き込み優先）から複雑なもの（カスタムマージ関数）まであります。データのセマンティクスに応じて選択してください：シンプルさを優先するならLWW（データ消失リスクあり）、自動収束にはCRDT、完全な制御にはアプリケーションレベルの解決が適しています。最良のコンフリクト対策は、そもそもコンフリクトを発生させないことです。

---

## コンフリクトが発生するタイミング

### 同時更新

```
Replica A                    Replica B
    │                            │
 write(x, 1)                  write(x, 2)
    │                            │
    └────────────────────────────┘
                  │
           Both succeed locally
           Which value is correct?
```

### ネットワークパーティション

```
┌─────────────────┐     ┌─────────────────┐
│   Partition A   │     │   Partition B   │
│                 │     │                 │
│   write(x, 1)   │  X  │   write(x, 2)   │
│                 │     │                 │
└─────────────────┘     └─────────────────┘

After partition heals:
  x = 1 or x = 2?
```

### オフラインクライアント

```
Client A (online):  write(x, 1) → synced
Client B (offline): write(x, 2) → queued

Client B comes online:
  Conflict: x = 1 vs x = 2
```

---

## コンフリクトの種類

### Write-Writeコンフリクト

同じフィールドが異なる値に変更されます。

```
Replica A: user.name = "Alice"
Replica B: user.name = "Bob"

Conflict: Which name?
```

### Read-Modify-Writeコンフリクト

両方のレプリカが同じ値を読み取り、変更して書き込みます。

```
Initial: counter = 10

Replica A: read 10, increment, write 11
Replica B: read 10, increment, write 11

Expected: 12
Actual: 11 (lost update)
```

### Delete-Updateコンフリクト

一方が削除し、もう一方が更新します。

```
Replica A: DELETE FROM users WHERE id = 1
Replica B: UPDATE users SET name = 'Bob' WHERE id = 1

What happens to the update?
```

### 制約違反

両方のレプリカが矛盾するエントリを作成します。

```
Replica A: INSERT (id=1, email='x@y.com')
Replica B: INSERT (id=2, email='x@y.com')

Unique constraint on email violated
```

---

## 解決戦略

### 最終書き込み優先（LWW）

最も高いタイムスタンプが勝ちます。

```
Write A: {value: "Alice", timestamp: 1000}
Write B: {value: "Bob", timestamp: 1001}

Resolution: value = "Bob" (higher timestamp)
```

**実装：**
```python
def resolve_lww(versions):
    return max(versions, key=lambda v: v.timestamp)
```

**メリット：**
- 実装がシンプル
- 決定論的
- 自動収束

**デメリット：**
- データ消失（以前の書き込みが破棄される）
- クロックスキューにより誤った勝者が選ばれる可能性
- セマンティクスの理解がない

**使用すべきケース：**
- キャッシュエントリ
- 最終更新タイムスタンプ
- 「最新」が意味を持つデータ

### 最初の書き込み優先

最も低いタイムスタンプが勝ちます（元の値を保持）。

```
Write A: {value: "Alice", timestamp: 1000}
Write B: {value: "Bob", timestamp: 1001}

Resolution: value = "Alice" (lower timestamp)
```

**使用すべきケース：**
- 不変レコード
- 監査ログ
- 「一度だけ作成」のセマンティクス

### 値のマージ

矛盾する値を結合します。

```
Cart at A: [item1, item2]
Cart at B: [item1, item3]

Merge: [item1, item2, item3]
```

**実装：**
```python
def merge_sets(versions):
    result = set()
    for v in versions:
        result = result.union(v.items)
    return result
```

**適用可能なケース：**
- 集合（和集合）
- カウンター（最大値または合計）
- 追記専用リスト

### アプリケーションレベルの解決

すべてのバージョンを保存し、アプリケーションに判断させます。

```
Read(x) → {
    versions: [
        {value: "Alice", timestamp: 1000, source: "A"},
        {value: "Bob", timestamp: 1001, source: "B"}
    ],
    conflict: true
}

Application: Present UI for user to choose
```

**実装：**
```python
def read_with_conflicts(key):
    versions = get_all_versions(key)
    if len(versions) > 1:
        return Conflict(versions)
    return versions[0]

def resolve_conflict(key, chosen_version, discarded_versions):
    write(key, chosen_version)
    for v in discarded_versions:
        mark_as_resolved(v)
```

---

## CRDT（コンフリクトフリー複製データ型）

### コンセプト

常にコンフリクトなしでマージできるように設計されたデータ構造です。

```
Property: Merge is:
  - Commutative: merge(A, B) = merge(B, A)
  - Associative: merge(merge(A, B), C) = merge(A, merge(B, C))
  - Idempotent: merge(A, A) = A

Any order of merging produces same result
```

### G-Counter（増加専用カウンター）

```
Each node tracks its own increment:
  Node A: {A: 5, B: 0, C: 0}
  Node B: {A: 3, B: 7, C: 0}

Merge: component-wise max
  Result: {A: 5, B: 7, C: 0}

Total: 5 + 7 + 0 = 12
```

**操作：**
```python
class GCounter:
    def __init__(self, node_id):
        self.node_id = node_id
        self.counts = {}

    def increment(self):
        self.counts[self.node_id] = self.counts.get(self.node_id, 0) + 1

    def value(self):
        return sum(self.counts.values())

    def merge(self, other):
        for node, count in other.counts.items():
            self.counts[node] = max(self.counts.get(node, 0), count)
```

### PN-Counter（正負カウンター）

増加と減少をサポートします：
```
P (positive): G-Counter for increments
N (negative): G-Counter for decrements

Value = P.value() - N.value()

Merge: merge P and N separately
```

### G-Set（増加専用集合）

要素は追加のみ可能で、削除はできません。

```
Set A: {1, 2, 3}
Set B: {2, 3, 4}

Merge: union = {1, 2, 3, 4}
```

### OR-Set（観測削除集合）

「追加が勝つ」セマンティクスで追加と削除をサポートします。

```
Each element has unique tags:
  add(x) → x with new tag
  remove(x) → remove all current tags

Concurrent add and remove:
  add(x) creates new tag, remove sees old tags
  Result: x exists (add wins)
```

**実装：**
```python
class ORSet:
    def __init__(self):
        self.elements = {}  # element → set of tags

    def add(self, element):
        tag = unique_tag()
        self.elements.setdefault(element, set()).add(tag)

    def remove(self, element):
        self.elements[element] = set()  # Remove all known tags

    def contains(self, element):
        return len(self.elements.get(element, set())) > 0

    def merge(self, other):
        for elem, tags in other.elements.items():
            self.elements[elem] = self.elements.get(elem, set()).union(tags)
```

### LWW-Register

CRDTとしての最終書き込み優先レジスタです。

```python
class LWWRegister:
    def __init__(self):
        self.value = None
        self.timestamp = 0

    def write(self, value, timestamp):
        if timestamp > self.timestamp:
            self.value = value
            self.timestamp = timestamp

    def merge(self, other):
        if other.timestamp > self.timestamp:
            self.value = other.value
            self.timestamp = other.timestamp
```

### LWW-Map

各キーがLWW-Registerであるマップです。

```
Node A: {"name": ("Alice", t=100), "age": (30, t=100)}
Node B: {"name": ("Bob", t=101), "city": ("NYC", t=100)}

Merge:
  "name": ("Bob", t=101)  ← higher timestamp
  "age": (30, t=100)
  "city": ("NYC", t=100)
```

---

## バージョンベクター

### 因果関係の追跡

```
Version vector: {A: 3, B: 2, C: 1}

Meaning:
  - Incorporates 3 updates from A
  - Incorporates 2 updates from B
  - Incorporates 1 update from C
```

### バージョンの比較

```python
def compare(vv1, vv2):
    less_or_equal = all(vv1.get(k, 0) <= vv2.get(k, 0) for k in set(vv1) | set(vv2))
    greater_or_equal = all(vv1.get(k, 0) >= vv2.get(k, 0) for k in set(vv1) | set(vv2))

    if less_or_equal and not greater_or_equal:
        return "vv1 < vv2"  # vv1 is ancestor
    elif greater_or_equal and not less_or_equal:
        return "vv1 > vv2"  # vv2 is ancestor
    elif less_or_equal and greater_or_equal:
        return "equal"
    else:
        return "concurrent"  # Neither is ancestor → conflict
```

### コンフリクトの検出

```
Write 1: value="A", vv={A:1, B:0}
Write 2: value="B", vv={A:0, B:1}

Compare:
  {A:1, B:0} vs {A:0, B:1}
  Neither dominates → concurrent → conflict!
```

### バージョンベクターによる解決

```python
def read_repair(versions):
    # Find all versions that are not dominated by another
    non_dominated = []
    for v in versions:
        if not any(dominates(other.vv, v.vv) for other in versions if other != v):
            non_dominated.append(v)

    if len(non_dominated) == 1:
        return non_dominated[0]  # Clear winner
    else:
        return Conflict(non_dominated)  # Need resolution
```

---

## セマンティックコンフリクト解決

### 三方向マージ

共通の祖先を使用してインテリジェントにマージします。

```
Original (base): "The quick brown fox"
Version A:       "The quick red fox"    (changed brown → red)
Version B:       "The fast brown fox"   (changed quick → fast)

Three-way merge:
  - "quick" → "fast" (B's change)
  - "brown" → "red" (A's change)
  Result: "The fast red fox"
```

**実装：**
```python
def three_way_merge(base, a, b):
    a_changes = diff(base, a)
    b_changes = diff(base, b)

    for change in a_changes + b_changes:
        if conflicts_with(change, a_changes + b_changes):
            return Conflict(a, b)

    return apply_changes(base, a_changes + b_changes)
```

### 操作変換（OT）

操作を変換して任意の順序で適用できるようにします。

```
Base: "abc"
Op A: insert('X', position=1) → "aXbc"
Op B: insert('Y', position=2) → "abYc"

If A applied first:
  B needs transformation: position 2 → position 3
  "aXbc" + insert('Y', 3) = "aXbYc"

If B applied first:
  A remains: position 1
  "abYc" + insert('X', 1) = "aXbYc"

Same result regardless of order
```

---

## 削除の処理

### トゥームストーン

削除されたアイテムを削除せずにマークします。

```
Before: {id: 1, name: "Alice", deleted: false}
Delete: {id: 1, name: "Alice", deleted: true, deleted_at: 1000}

Why tombstones:
  - Replicas need to know item was deleted
  - Otherwise they might resurrect it from their version
```

### トゥームストーンのクリーンアップ

```
Problem: Tombstones accumulate forever

Solutions:
1. Time-based garbage collection
   Delete tombstones older than X days
   Risk: old replica comes back, resurrects data

2. Version vector garbage collection
   Delete when all replicas have seen tombstone
   Requires coordination

3. Compaction
   Periodically merge and remove tombstones
```

### ソフトデリート vs ハードデリート

```
Soft delete: Keep record, mark as deleted
  + Can undelete
  + Preserves audit trail
  - Storage overhead

Hard delete: Remove record
  + No storage overhead
  - Can cause resurrection
  - Loses history
```

---

## 実世界のアプローチ

### Amazon Dynamo（Riak）

```
Strategy: Return all conflicting versions to client

Read → [{value: "A", clock: {A:1}}, {value: "B", clock: {B:1}}]

Client merges, writes back with merged clock
  Write(merged, clock: {A:1, B:1, Client:1})
```

### CouchDB

```
Store all revisions in conflict
  _id: "doc1"
  _conflicts: ["2-abc123", "2-def456"]

Application picks winner or merges
Losing revisions become historical
```

### Cassandra

```
LWW by default per column

CREATE TABLE users (
  id uuid,
  name text,
  email text,
  PRIMARY KEY (id)
);

-- Each column resolved independently
-- Highest timestamp wins per column
```

### Git

```
Three-way merge for file contents
Conflict markers for unresolvable conflicts

<<<<<<< HEAD
my changes
=======
their changes
>>>>>>> branch

User manually resolves
```

---

## 戦略の選択

### 判断マトリクス

| シナリオ | 戦略 | 理由 |
|----------|------|------|
| キャッシュ | LWW | 古いデータが許容される |
| カウンター | CRDT G-Counter | 正確な集計 |
| ショッピングカート | OR-Set CRDT | 追加が勝ち、アイテムをマージ |
| ユーザープロフィール | LWWまたはフィールドマージ | フィールドレベルの解決 |
| ドキュメント編集 | OTまたはCRDT | リアルタイムコラボレーション |
| 金融取引 | コンフリクトの防止 | シングルリーダーを使用 |
| ソーシャルフィード | LWW | 古さが許容される |

### 解決より防止

最良のコンフリクト戦略はコンフリクトを防止することです：

```
1. Single leader for conflicting data
2. Partition data so conflicts impossible
3. Use optimistic locking with retries
4. Serialize operations through queue

Prevention is simpler than resolution
```

---

## 重要なポイント

1. **コンフリクトは不可避** - マルチリーダー/リーダーレスシステムでは必ず発生します
2. **LWWはシンプルだがデータ消失がある** - 一部のデータでは許容可能です
3. **CRDTは収束を保証する** - 設計上コンフリクトなし
4. **バージョンベクターは同時実行を検出する** - コンフリクト検出に不可欠です
5. **アプリケーションが最もよく知っている** - 時にはユーザーに判断させましょう
6. **トゥームストーンは必要** - ただしクリーンアップ戦略が必要です
7. **可能な限り防止する** - 重要なデータにはシングルリーダーを使用しましょう
8. **データのセマンティクスで選択する** - カウンター、集合、レジスタにはそれぞれ異なる戦略が必要です
