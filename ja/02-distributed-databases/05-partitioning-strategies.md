# パーティショニング戦略

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../02-distributed-databases/05-partitioning-strategies.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

パーティショニング（シャーディング）は、単一マシンの限界を超えてスケールするためにデータを複数のノードに分割します。主な戦略：均等分散にはハッシュパーティショニング、効率的な範囲クエリにはレンジパーティショニングを使用します。コンシステントハッシングはリバランシングを最小化します。ホットスポットは敵です。パーティションキーを慎重に選択してください。それによって、どのクエリが効率的で、どのクエリがスキャッターギャザーを必要とするかが決まります。

---

## なぜパーティショニングするのか？

### 単一ノードの限界

```
Data volume:    > 10 TB (exceeds disk)
Query load:     > 100K QPS (CPU bound)
Write load:     > 50K WPS (disk I/O bound)
Memory:         > 1 TB (RAM limit)
```

### パーティショニングのメリット

```
Before:
  [Single Node] ← All queries, all data

After:
  [Node 1] ← data A-M, queries for A-M
  [Node 2] ← data N-Z, queries for N-Z

Capacity: 2x data, 2x queries
Add more nodes → linear scaling
```

---

## パーティショニング方法

### レンジパーティショニング

連続するキー範囲をパーティションに割り当てます。

```
Partition 1: A-F    [aardvark, apple, ... , fox]
Partition 2: G-L    [giraffe, house, ... , lion]
Partition 3: M-R    [monkey, nest, ... , rabbit]
Partition 4: S-Z    [snake, tree, ... , zebra]
```

**メリット：**
- 範囲クエリが効率的（単一パーティションをスキャン）
- キーが自然に順序付けされる

**デメリット：**
- ホットスポットが発生しやすい（共通プレフィックス）
- 手動リバランシングが必要なことが多い

**例：時系列データ**
```
Partition by date:
  2024-01: Partition 1
  2024-02: Partition 2
  2024-03: Partition 3  ← hot (current month)

Problem: Current month gets all writes
Solution: Compound key (sensor_id, timestamp)
```

### ハッシュパーティショニング

キーをハッシュし、ハッシュ値に基づいてパーティションに割り当てます。

```
partition = hash(key) mod N

Example:
  hash("user_123") = 8742
  8742 mod 4 = 2
  → Partition 2
```

**メリット：**
- 均等分散（良いハッシュ = 均一）
- 連続キーによるホットスポットなし

**デメリット：**
- 範囲クエリが非効率（全パーティションを走査）
- ノード追加時に多くのキーが再配置される

**ハッシュ関数の要件：**
```
Requirements:
  - Deterministic (same key → same hash)
  - Uniform distribution
  - Fast computation

Examples:
  - MD5 (slow, cryptographic)
  - MurmurHash (fast, good distribution)
  - xxHash (very fast)
```

### コンシステントハッシング

キーとノードの両方をリングにマッピングします。

```
        0°
        │
   ┌────┼────┐
   │    │    │
270°────┼────90°
   │    │    │
   └────┼────┘
        │
       180°

Nodes: N1 at 30°, N2 at 120°, N3 at 250°
Key K: hash(K) = 100°
  → Goes to next node clockwise: N2 (120°)
```

**ノードの追加：**
```
Before: N1(30°), N2(120°), N3(250°)
Add N4 at 200°

Only keys between 120°-200° move to N4
Other partitions unchanged
```

**仮想ノード：**
```
Each physical node → multiple positions on ring
  N1: [30°, 100°, 220°]
  N2: [50°, 130°, 280°]

Benefits:
  - More even distribution
  - Smoother rebalancing
  - Handle heterogeneous hardware
```

---

## パーティションキーの選択

### 単一キーパーティショニング

```
Users table → partition by user_id

Query: SELECT * FROM users WHERE user_id = 123
  → Goes to one partition ✓

Query: SELECT * FROM users WHERE email = 'x@y.com'
  → Scatter to all partitions ✗
```

### 複合キー

```
CREATE TABLE posts (
  user_id INT,
  post_id INT,
  content TEXT,
  PRIMARY KEY ((user_id), post_id)
);

user_id = partition key
post_id = clustering key (sort within partition)

Query: SELECT * FROM posts WHERE user_id = 123
  → One partition, sorted by post_id ✓

Query: SELECT * FROM posts WHERE user_id = 123 AND post_id > 100
  → One partition, range scan ✓
```

### キー設計ガイドライン

| アクセスパターン | 良いキー | 悪いキー |
|-----------------|---------|---------|
| ユーザーのデータ | user_id | email |
| 時系列 | (device_id, date) | timestamp |
| 注文 | (customer_id, order_id) | order_date |
| チャットメッセージ | (room_id, message_time) | sender_id |

---

## ホットスポットの対処

### 問題

```
Celebrity user: 10M followers
  - All posts by this user → one partition
  - All reads of their posts → one partition
  - That partition is overwhelmed

Sequential key: order_id auto-increment
  - All new orders → highest partition
  - Write hotspot
```

### 緩和戦略

**ランダムプレフィックスの追加：**
```
Original key: user_123
Prefixed key: {0-9}_user_123  (random prefix)

Reads: scatter to 10 partitions, aggregate
Writes: distributed across 10 partitions

Trade-off: Single-key queries become scatter-gather
```

**タイムバケッティング：**
```
Instead of: partition by user_id
Use: partition by (user_id, time_bucket)

time_bucket = hour or day

Hot user's data spread across time buckets
Recent data in few buckets (queryable)
Old data in many buckets (archived)
```

**パーティションごとのリードレプリカ：**
```
Hot partition → more read replicas
Route reads to replicas
Writes still go to primary
```

---

## リバランシング

### リバランシングのタイミング

```
Triggers:
  - Node added
  - Node removed
  - Load imbalance detected
  - Data growth uneven
```

### 固定パーティション数

```
Create more partitions than nodes:
  100 partitions for 10 nodes (10 each)

Add node:
  Move some partitions to new node
  (10 partitions → new node)

Partition boundaries never change
Simple, predictable
```

### 動的パーティショニング

```
Partition grows too large → split
Partition shrinks → merge with neighbor

Example (HBase):
  Region grows > 10 GB → split
  Parent: [A-M]
  Children: [A-G], [H-M]
```

### パーティション比例ノード

```
Each node gets fixed number of partitions
New node → steal partitions from existing nodes
More nodes → smaller partitions

Cassandra approach:
  Each node: 256 virtual nodes
  Add node: 256 new vnodes, take data from neighbors
```

### 移動の最小化

```
Goal: Move minimum data when rebalancing

Consistent hashing: O(K/N) keys move
  K = total keys
  N = number of nodes

Naive hash mod: O(K) keys move
  Almost everything moves!
```

---

## クエリルーティング

### クライアントサイドルーティング

```
Client knows partition map
Client sends request directly to correct partition

┌────────┐
│ Client │──────knows partition map
└────┬───┘
     │  partition_for(user_123) = Node 2
     ▼
┌─────────┐
│ Node 2  │
└─────────┘

Pros: No extra hop
Cons: Client must track partition changes
```

### ルーティング層

```
All requests → Router → Correct partition

┌────────┐     ┌────────┐     ┌─────────┐
│ Client │ ──► │ Router │ ──► │ Node N  │
└────────┘     └────────┘     └─────────┘

Pros: Clients are simple
Cons: Extra network hop, router can be bottleneck
```

### コーディネーターノード

```
Any node can receive request
Node forwards to correct partition (or handles locally)

┌────────┐     ┌─────────┐     ┌─────────┐
│ Client │ ──► │ Node 1  │ ──► │ Node 3  │
└────────┘     │(coordinator)  │(partition owner)
               └─────────┘     └─────────┘

Pros: Any node is entry point
Cons: Extra hop if wrong node
```

### パーティションディスカバリ

```
Approach 1: ZooKeeper / etcd
  - Nodes register partition ownership
  - Clients/routers watch for changes

Approach 2: Gossip protocol
  - Nodes share partition knowledge
  - Eventually consistent

Approach 3: Central metadata service
  - Dedicated service tracks partitions
  - Single source of truth
```

---

## クロスパーティション操作

### スキャッターギャザークエリ

```
Query: SELECT COUNT(*) FROM users WHERE age > 30

All partitions:
  [P1] → count: 1000
  [P2] → count: 1500
  [P3] → count: 800
  [P4] → count: 1200

Coordinator aggregates: 4500
```

**パフォーマンス：**
```
Latency = max(partition latencies) + aggregation
Throughput limited by slowest partition

One slow partition → slow query
```

### クロスパーティション結合

```
Orders partitioned by customer_id
Products partitioned by product_id

SELECT o.*, p.name
FROM orders o
JOIN products p ON o.product_id = p.id
WHERE o.customer_id = 123

Strategy 1: Broadcast join
  Send all products to order partition

Strategy 2: Shuffle join
  Repartition both tables by join key

Strategy 3: Denormalize
  Store product name in orders table
```

### クロスパーティショントランザクション

```
Transfer $100 from Account A (Partition 1) to Account B (Partition 2)

Requires distributed transaction:
  1. Start transaction on both partitions
  2. Debit A, credit B
  3. Two-phase commit

Expensive and complex
Consider: same-partition transfers only
```

---

## 実践でのパーティショニング

### PostgreSQL（宣言的パーティショニング）

```sql
-- Range partitioning
CREATE TABLE orders (
    id SERIAL,
    order_date DATE,
    customer_id INT
) PARTITION BY RANGE (order_date);

CREATE TABLE orders_2024_q1 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

-- Hash partitioning
CREATE TABLE users (
    id INT,
    name TEXT
) PARTITION BY HASH (id);

CREATE TABLE users_p0 PARTITION OF users
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);
```

### Cassandra

```sql
CREATE TABLE posts (
    user_id uuid,
    post_time timestamp,
    content text,
    PRIMARY KEY ((user_id), post_time)
) WITH CLUSTERING ORDER BY (post_time DESC);

-- user_id is partition key (hashed)
-- post_time is clustering key (sorted within partition)
```

### MongoDB

```javascript
// Enable sharding
sh.enableSharding("mydb")

// Hash-based sharding
sh.shardCollection("mydb.users", { _id: "hashed" })

// Range-based sharding
sh.shardCollection("mydb.logs", { timestamp: 1 })
```

---

## アンチパターン

### 不均等なパーティションサイズ

```
Problem:
  Partition A: 100 GB
  Partition B: 10 GB
  Partition C: 500 GB ← overloaded

Causes:
  - Poor key distribution
  - Natural data skew

Solutions:
  - Better key choice
  - Salting keys
  - Dynamic splitting
```

### 単調増加キー

```
Problem:
  key = timestamp or auto_increment
  All new data → last partition

Solutions:
  - Hash the key
  - Prepend random bytes
  - Use compound key with better distribution
```

### パーティション数が少なすぎる

```
Problem:
  4 partitions, want 10 nodes
  Cannot distribute evenly

Solution:
  Create many partitions upfront (e.g., 256)
  Distribute across available nodes
  Room to grow
```

### 主要パターンとしてのクロスパーティションアクセス

```
Problem:
  Most queries span all partitions
  No benefit from partitioning

Solutions:
  - Reconsider partition key
  - Denormalize data
  - Accept scatter-gather cost
```

---

## 重要なポイント

1. **分散にはハッシュ、クエリにはレンジ** - アクセスパターンに基づいて選択しましょう
2. **コンシステントハッシングは移動を削減する** - 大規模システムに不可欠です
3. **パーティションキーは重要** - クエリ効率を決定します
4. **ホットスポットはパフォーマンスを殺す** - ソルティング、タイムバケッティングを使用しましょう
5. **リバランシングはコストが高い** - パーティション数を事前に計画しましょう
6. **スキャッターギャザーにはオーバーヘッドがある** - クロスパーティションクエリを最小化する設計を心がけましょう
7. **クロスパーティショントランザクションは困難** - 可能な限り避けましょう
8. **パーティション数が多い = 柔軟性が高い** - ただし調整オーバーヘッドも増加します
