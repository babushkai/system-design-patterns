# カラム指向ストレージ

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../03-storage-engines/06-column-storage.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

カラム指向ストレージは、行単位ではなくカラム単位でデータを格納します。これにより優れた圧縮が可能になり、必要なカラムのみの読み取りが実現します。多数の行を走査するが少数のカラムのみを使用する分析ワークロード（OLAP）に最適です。行指向ストレージはトランザクションワークロード（OLTP）で行全体にアクセスする場合に優れています。ほとんどのデータウェアハウスはカラム型ストレージを使用しています。

---

## 行ストレージとカラムストレージ

### 行指向（従来型）

```
Table: Users
  id | name    | age | city
  ---+---------+-----+---------
  1  | Alice   | 30  | NYC
  2  | Bob     | 25  | LA
  3  | Charlie | 35  | Chicago

On disk:
  [1, Alice, 30, NYC][2, Bob, 25, LA][3, Charlie, 35, Chicago]

  Entire row stored contiguously
```

### カラム指向

```
Same table, stored by column:

id:    [1, 2, 3]
name:  [Alice, Bob, Charlie]
age:   [30, 25, 35]
city:  [NYC, LA, Chicago]

Each column stored separately
```

---

## なぜカラムなのか？

### クエリパターンの違い

```
OLTP (Transactional):
  SELECT * FROM users WHERE id = 123
  → Need all columns for one row
  → Row storage efficient

OLAP (Analytical):
  SELECT AVG(age) FROM users WHERE city = 'NYC'
  → Need 2 columns, millions of rows
  → Reading all columns is wasteful
```

### カラムストレージの利点

```
Query: SELECT AVG(age) FROM users WHERE city = 'NYC'

Row storage reads:
  [1, Alice, 30, NYC][2, Bob, 25, LA][3, Charlie, 35, Chicago]...
  Read everything, use only 2 columns

Column storage reads:
  age:  [30, 25, 35, ...]
  city: [NYC, LA, Chicago, ...]
  Skip id, name columns entirely

I/O reduction: 2/4 = 50% in this example
Real tables with 100+ columns: 95%+ reduction
```

---

## 圧縮のメリット

### 同一型データはより効率的に圧縮できる

```
Row storage:
  [1, Alice, 30, NYC][2, Bob, 25, LA]...
  Mixed types (int, string, int, string)
  Poor compression

Column storage:
  age: [30, 25, 35, 40, 28, 30, 35, 30, ...]
  Same type, similar values
  Excellent compression
```

### ランレングスエンコーディング（RLE）

```
city column (sorted):
  [Chicago, Chicago, Chicago, LA, LA, NYC, NYC, NYC, NYC, NYC]

RLE compressed:
  [(Chicago, 3), (LA, 2), (NYC, 5)]

10 values → 3 pairs
```

### 辞書エンコーディング

```
status column:
  [pending, active, active, pending, active, completed, ...]

Dictionary:
  0 = pending
  1 = active
  2 = completed

Encoded: [0, 1, 1, 0, 1, 2, ...]
  String → 2 bits
  Massive space savings for low-cardinality columns
```

### ビットパッキング

```
age column (0-100 range):
  Standard int: 32 bits per value
  Bit-packed: 7 bits per value (2^7 = 128)

4.5x space reduction
```

### 圧縮方式の比較

| エンコーディング | 最適な用途 | 圧縮率 |
|----------|----------|-------|
| RLE | ソート済み、繰り返しが多いデータ | 10-100x |
| 辞書 | 低カーディナリティ | 10-50x |
| ビットパッキング | 小さな整数 | 2-8x |
| デルタ | タイムスタンプ、連番 | 5-20x |
| LZ4/Zstd | 汎用 | 2-5x |

---

## カラムストアのアーキテクチャ

### 物理レイアウト

```
Table: sales
Columns: date, product_id, quantity, price, region

Files on disk:
  sales_date.col      (dates only)
  sales_product_id.col
  sales_quantity.col
  sales_price.col
  sales_region.col

Each file:
  - Sorted by some key (often date)
  - Divided into row groups
  - Each group independently compressed
```

### 行グループ

```
┌─────────────────────────────────────────────────────┐
│                    Row Group 1                      │
│  date:[...] product_id:[...] quantity:[...] ...     │
├─────────────────────────────────────────────────────┤
│                    Row Group 2                      │
│  date:[...] product_id:[...] quantity:[...] ...     │
├─────────────────────────────────────────────────────┤
│                    Row Group 3                      │
│  date:[...] product_id:[...] quantity:[...] ...     │
└─────────────────────────────────────────────────────┘

Row group size: Typically 100K - 1M rows
Enables:
  - Parallel processing
  - Predicate pushdown (skip row groups)
  - Memory efficiency
```

### 行の再構築

```
Need to join columns back together:
  Position 0: date[0], product_id[0], quantity[0], ...
  Position 1: date[1], product_id[1], quantity[1], ...

Same position across columns = same row
Called "late materialization"
```

---

## クエリ実行

### 従来型（早期マテリアライゼーション）

```
Query: SELECT product_id, quantity
       FROM sales
       WHERE region = 'US' AND quantity > 100

1. Scan region column, find matching row IDs
2. For each matching row:
   - Fetch product_id, quantity, region
   - Build full row
   - Apply predicates
   - Return results

Reconstructs rows early, even if filtered out later
```

### カラム型（遅延マテリアライゼーション）

```
Same query:

1. Scan region column → bitmap of US rows
2. Scan quantity column → bitmap of quantity > 100
3. AND bitmaps → final row IDs
4. Only for matching rows:
   - Fetch product_id, quantity
   - Return results

Only reconstruct needed rows at the end
Significant speedup for selective queries
```

### ベクトル化実行

```
Process columns in batches (vectors):

Instead of:
  for row in rows:
    result = row.quantity * row.price

Do:
  quantities = load_vector(1024 values)
  prices = load_vector(1024 values)
  results = quantities * prices  # SIMD operation

Benefits:
  - CPU cache efficiency
  - SIMD parallelism
  - Reduced interpretation overhead
```

---

## Parquet フォーマット

### ファイル構造

```
┌─────────────────────────────────────────────┐
│ Magic Number: PAR1                          │
├─────────────────────────────────────────────┤
│ Row Group 1                                 │
│   Column Chunk 1: [Pages...] + Column Meta  │
│   Column Chunk 2: [Pages...] + Column Meta  │
│   ...                                       │
├─────────────────────────────────────────────┤
│ Row Group 2                                 │
│   ...                                       │
├─────────────────────────────────────────────┤
│ Footer                                      │
│   File Metadata                             │
│   Row Group Metadata                        │
│   Column Metadata                           │
│   Schema                                    │
├─────────────────────────────────────────────┤
│ Footer Length (4 bytes)                     │
├─────────────────────────────────────────────┤
│ Magic Number: PAR1                          │
└─────────────────────────────────────────────┘
```

### ページタイプ

```
Data Page:
  - Actual column values
  - Definition levels (for nulls)
  - Repetition levels (for nested data)

Dictionary Page:
  - Dictionary for dictionary encoding
  - Stored once per column chunk

Data Page V2:
  - Improved encoding
  - Header contains statistics
```

### クエリ計画のためのメタデータ

```
Footer contains per-column stats:
  - Min/max values
  - Null count
  - Distinct count (optional)

Query: WHERE date >= '2024-01-01'
  Check row group metadata
  Skip row groups where max_date < '2024-01-01'
```

---

## ORC フォーマット

### 構造

```
Similar to Parquet, used heavily in Hive/Hadoop:

┌─────────────────────────────────────────────┐
│ Stripe 1                                    │
│   Index Data (min/max, positions)           │
│   Row Data (column streams)                 │
│   Stripe Footer                             │
├─────────────────────────────────────────────┤
│ Stripe 2                                    │
│   ...                                       │
├─────────────────────────────────────────────┤
│ File Footer                                 │
│   Type information                          │
│   Stripe information                        │
│   Column statistics                         │
├─────────────────────────────────────────────┤
│ Postscript                                  │
│   Compression type, version                 │
└─────────────────────────────────────────────┘
```

### ORC と Parquet の比較

| 観点 | Parquet | ORC |
|--------|---------|-----|
| 開発元 | Twitter/Cloudera | Facebook/Hortonworks |
| エコシステム | Spark、汎用 | Hive、Presto |
| ネストデータ | より優れている | 良好 |
| ACIDアップデート | なし | あり（Hive使用時） |
| 述語プッシュダウン | 良好 | より優れたインデックス |

---

## カラムストアのインデックス

### ゾーンマップ（Min/Max インデックス）

```
For each row group or page:
  Store min and max value

Query: WHERE price > 1000

Row Group 1: min=50, max=500   → skip
Row Group 2: min=200, max=1500 → scan
Row Group 3: min=800, max=2000 → scan
Row Group 4: min=5000, max=8000 → scan (all match)
```

### ビットマップインデックス

```
For low-cardinality columns:

region = 'US':   [1, 0, 1, 1, 0, 1, ...]
region = 'EU':   [0, 1, 0, 0, 1, 0, ...]
region = 'APAC': [0, 0, 0, 0, 0, 0, ...]

Query: WHERE region IN ('US', 'EU')
  Bitmap OR: [1, 1, 1, 1, 1, 1, ...]
  Very fast set operations
```

### カラムのブルームフィルタ

```
Store Bloom filter per column chunk

Query: WHERE product_id = 'ABC123'

Check Bloom filter:
  Definitely not in chunk → skip
  Maybe in chunk → scan

Useful for high-cardinality equality predicates
```

---

## カラムストアへの書き込み

### 課題

```
INSERT single row:
  Row store: Append to one file
  Column store: Append to N files (one per column)

Much more I/O for writes
```

### バッチ書き込み

```
Buffer writes in memory (row format)
Periodically flush as column chunks

Write pattern:
  1. Write to in-memory buffer
  2. When buffer full (e.g., 10K rows):
     - Convert to columnar
     - Compress
     - Write to disk

Batching amortizes conversion overhead
```

### デルタストア

```
MemStore (row format) + Column files (column format)

Reads: Merge MemStore + Column files
Writes: Go to MemStore only

Periodically compact MemStore into column files
Similar to LSM tree approach
```

### 更新と削除

```
Option 1: Delete bitmap
  Mark rows as deleted
  Compact to remove later

Option 2: Merge-on-read
  Store updates separately
  Merge during query

Option 3: Copy-on-write
  Rewrite affected row groups
  Expensive but simple
```

---

## カラム型ストレージを使用するシステム

### 分析データベース

```
ClickHouse:
  - Native columnar
  - MergeTree engine
  - Very fast for time-series

Snowflake:
  - Columnar on cloud storage
  - Automatic clustering
  - Micro-partitions

BigQuery:
  - Capacitor columnar format
  - Dremel-style query engine
  - Serverless

Redshift:
  - Columnar PostgreSQL variant
  - Zone maps
  - Compression encoding per column
```

### ハイブリッドストア

```
DuckDB:
  - Embedded columnar database
  - Vectorized execution
  - Great for local analytics

CockroachDB:
  - Row store primary
  - Columnar for analytics (experimental)

PostgreSQL:
  - Row store with columnar extensions
  - cstore_fdw, Citus columnar
```

---

## カラム型の適用場面

### 適している場合

```
✓ 分析/OLAPワークロード
✓ 多数の行に対する集約処理
✓ 少数のカラムのみ使用するクエリ
✓ 追記中心のデータ
✓ 圧縮が重要
✓ 幅広いテーブル（100カラム以上）
```

### 適さない場合

```
✗ トランザクション/OLTPワークロード
✗ 主キーによるポイントルックアップ
✗ 頻繁な更新・削除
✗ すべてのカラムを必要とするクエリ
✗ リアルタイム要件
✗ 狭いテーブル（少数カラム）
```

### 比較

| 観点 | 行ストア | カラムストア |
|--------|-----------|--------------|
| ポイントルックアップ | 高速 | 低速 |
| フルスキャン | 低速 | 高速 |
| 集約 | 低速 | 高速 |
| 単一行挿入 | 高速 | 低速 |
| バルクロード | 中程度 | 高速 |
| 圧縮 | 2-3x | 10-100x |
| OLTP | 優秀 | 不向き |
| OLAP | 不向き | 優秀 |

---

## ArrowとParquet: メモリ内 vs 保存時

カラムナの世界が*2つの*フォーマットに標準化したのは、最適化の対象が異なるからです:

```
Parquet (at rest): optimize for SIZE and skippability.
  Heavy encodings (dictionary, RLE, bit-packing) + block compression.
  Values are NOT randomly accessible — you decode a page to read it.

Arrow (in memory): optimize for COMPUTE.
  Fixed-width arrays, validity bitmaps, contiguous buffers — a value
  is at base + i × width, SIMD-scannable directly, no decode step.
  The same buffer layout in every language (C++, Rust, Java, Python
  via zero-copy FFI) — "serialization" between processes sharing
  Arrow is memcpy or shared memory, not encode/decode.

The pipeline every modern engine runs:
  Parquet page → decode once → Arrow batch → all further operators
  (filter, join, aggregate) work on Arrow vectors.
  Arrow Flight / ADBC move Arrow batches over the network, replacing
  row-at-a-time protocols (JDBC/ODBC) that spend more CPU converting
  than transferring.
```

実務上の帰結: 「どちらのフォーマット？」は選択ではありません — Parquet*と*Arrowであり、境界はスキャンにあります。選択なのは: その境界より下に仕事を押し込むこと（Parquetリーダーへの述語/射影プッシュダウン）で、デコードされるページ自体を減らすことです。

### ベクトル化実行を具体的に

```
SELECT SUM(amount) WHERE region_code = 7    (1M rows)

Row engine:  1M × (interpret row layout → extract field → branch → add)
             ~5-20 ns per row of interpretation overhead

Vectorized engine on Arrow batches (~1K-4K values per batch):
  region_code: compare 32 codes per AVX-512 instruction → bitmask
  amount:      masked SIMD add, 16 int32s per instruction
  ≈ 1M rows / 16 per instr ≈ 62K instructions + branch-free selection
  → 10-100× less CPU per row; memory bandwidth becomes the limit

Two details that make it work:
  - dictionary-encoded columns can evaluate predicates on the CODES
    (compare against the dictionary once, then scan small integers)
  - selection vectors/bitmaps defer materialization — operators pass
    "which rows survive" instead of copying survivors
```

---

## 本当のインデックスはソート順

カラムストアでは、物理的なソート順がOLTPにおけるB木インデックスの仕事をします — 圧縮率と、スキャンがどれだけデータをスキップできるかの両方を決めます:

```
Same 1B-row events table, two layouts:

Sorted by (event_time):
  event_time: delta encoding → ~1-2 bits/value
  zone maps on event_time: a 1-day query scans ~1/365 of row groups
  but: WHERE user_id = X must scan EVERY row group (user_id is
  scattered — its per-group min/max spans the whole domain)

Sorted by (user_id, event_time):
  user_id: RLE → almost free; queries by user prune to a few groups
  event_time: still locally sorted within a user → decent deltas
  but: pure time-range queries now scan everything

The sort key is a QUERY WORKLOAD decision, revisited as workloads
change. Multi-dimensional compromise: Z-order / Hilbert curves
interleave dimensions so BOTH user_id and event_time predicates
prune reasonably (Delta OPTIMIZE ZORDER BY, Iceberg sort orders,
ClickHouse ORDER BY tuple).
```

そしてインジェストがソート順で届くことは稀なので、クラスタリングは*減衰*します: ストリームされたデータは到着順に着地し、ゾーンマップは広がり、スキャンは遅くなります。エンジンはバックグラウンドで再ソートします（Snowflakeの自動クラスタリング、Delta/Icebergのソート付きコンパクション）— LSMコンパクションのカラムナ版であり、同じ通貨（バックグラウンドのI/Oと計算）で支払います。

---

## 書き直しなしの更新: 削除ベクトル

古典的なカラムナの更新はcopy-on-writeでした: 1行を変えるためにローグループ全体（下手をすると128MBのファイル）を書き直す。モダンなテーブルフォーマットはmerge-on-readの中間路を追加しました:

```
Deletion vector: a compressed bitmap (often roaring) per data file
marking rows as dead. DELETE/UPDATE writes:
  - a tiny deletion-vector file (positions of deleted rows)
  - (for UPDATE) the new row versions into a new file
The 128 MB data file is untouched.

Read path: scan file ⊕ apply its deletion vector — a masked scan,
nearly free in a vectorized engine (it's just another bitmap AND).

The LSM parallel is exact: deletion vectors are tombstones, readers
pay a small merge cost, and background compaction eventually rewrites
files to fold deletes in. Same debt dynamics too — millions of
accumulated deletes without compaction degrade scans, so table
maintenance (OPTIMIZE / rewrite_data_files) is an operational duty,
not an optimization.

(Iceberg v2 position/equality deletes, Delta deletion vectors,
DuckDB and Photon read them natively — see
[レイクハウステーブルフォーマット](../13-data-pipelines/05-lakehouse-table-formats.md).)
```

---

## まとめ

1. **カラムストレージは必要なカラムのみ読み取る** - 大幅なI/O削減
2. **同一型データは効率的に圧縮できる** - 10-100倍の圧縮率
3. **遅延マテリアライゼーションでパフォーマンス向上** - 行再構築を遅らせる
4. **ベクトル化実行でSIMDを活用** - CPU効率の高い処理
5. **行グループで述語プッシュダウンを実現** - 不要なデータをスキップ
6. **Parquet/ORCが標準フォーマット** - 広範なエコシステムサポート
7. **書き込みはコストが高い** - バッチ化とバッファリングが必要
8. **分析用途に使用し、トランザクションには使わない** - 適材適所
