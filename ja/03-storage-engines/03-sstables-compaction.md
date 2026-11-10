# SSTableとコンパクション

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../03-storage-engines/03-sstables-compaction.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

SSTable（Sorted String Table）は、LSM木のディスク層を構成するイミュータブルなソート済みファイルです。コンパクションはSSTableをマージして、空間を回収し、古いデータを削除し、読み取りパフォーマンスを維持します。適切なコンパクション戦略（サイズ階層型、レベル型、FIFO）の選択は、ワークロードの読み取り/書き込みバランスとレイテンシ要件に依存します。

---

## SSTableの構造

### ファイルレイアウト

```
┌─────────────────────────────────────────────────────────────┐
│                        SSTable File                        │
├─────────────────────────────────────────────────────────────┤
│  Data Block 1  │  Data Block 2  │  ...  │  Data Block N    │
├─────────────────────────────────────────────────────────────┤
│               Meta Block (Bloom Filter)                    │
├─────────────────────────────────────────────────────────────┤
│                    Index Block                              │
├─────────────────────────────────────────────────────────────┤
│                       Footer                                │
└─────────────────────────────────────────────────────────────┘
```

### データブロック

```
Block structure:
┌────────────────────────────────────────┐
│  Entry 1  │  Entry 2  │  ...  │ Entry N │
├────────────────────────────────────────┤
│           Restart Points               │
├────────────────────────────────────────┤
│  Num Restarts (4 bytes)  │  CRC (4)    │
└────────────────────────────────────────┘

Entry format:
  [shared_prefix_len][unshared_len][value_len][unshared_key][value]

Prefix compression:
  Keys often share prefixes
  Only store the difference
  "user:1000" → "user:1001" stored as "01" suffix
```

### インデックスブロック

```
Sparse index:
  Every N-th key → block offset

┌───────────────────────────────────┐
│  "aaa" → Block 0 @ offset 0       │
│  "abc" → Block 1 @ offset 4096    │
│  "def" → Block 2 @ offset 8192    │
│  ...                              │
└───────────────────────────────────┘

Lookup:
  1. Binary search index for >= key
  2. Read data block
  3. Binary search within block
```

### フッタ

```
┌─────────────────────────────────────────┐
│  Metaindex Handle (offset, size)        │
│  Index Handle (offset, size)            │
│  Padding                                │
│  Magic Number (8 bytes)                 │
└─────────────────────────────────────────┘

Fixed size, always at end of file
Contains pointers to index and meta blocks
```

---

## SSTableの操作

### SSTableの作成（フラッシュ）

```python
def create_sstable(memtable, filename):
    writer = SSTableWriter(filename)

    # Iterate memtable in sorted order
    for key, value in memtable.sorted_iterator():
        writer.add(key, value)

    # Finalize: write index, bloom filter, footer
    writer.finish()

    return SSTable(filename)

class SSTableWriter:
    def __init__(self, filename):
        self.file = open(filename, 'wb')
        self.index = []
        self.bloom_filter = BloomFilter()
        self.current_block = DataBlock()
        self.block_offset = 0

    def add(self, key, value):
        self.bloom_filter.add(key)

        if self.current_block.size() > BLOCK_SIZE:
            self.flush_block()

        self.current_block.add(key, value)

    def flush_block(self):
        # Write block
        data = self.current_block.serialize()
        self.file.write(data)

        # Record in index
        first_key = self.current_block.first_key
        self.index.append((first_key, self.block_offset))

        self.block_offset += len(data)
        self.current_block = DataBlock()
```

### SSTableからの読み取り

```python
class SSTable:
    def get(self, key):
        # Check bloom filter first
        if not self.bloom_filter.might_contain(key):
            return None  # Definitely not here

        # Binary search in index
        block_offset = self.find_block(key)

        # Read and search block
        block = self.read_block(block_offset)
        return block.search(key)

    def find_block(self, key):
        # Binary search for last index entry <= key
        lo, hi = 0, len(self.index) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.index[mid].key <= key:
                lo = mid
            else:
                hi = mid - 1
        return self.index[lo].offset
```

### 範囲スキャン

```python
def scan(self, start_key, end_key):
    # Find starting block
    block_offset = self.find_block(start_key)

    results = []
    while block_offset < self.data_size:
        block = self.read_block(block_offset)

        for key, value in block.entries():
            if key > end_key:
                return results
            if key >= start_key:
                results.append((key, value))

        block_offset = self.next_block_offset(block_offset)

    return results
```

---

## コンパクション戦略

### サイズ階層型コンパクション（STCS）

```
Group SSTables by size, merge when count exceeds threshold.

Before:
  Tier 0 (small): [1MB] [1MB] [1MB] [1MB]  ← 4 files, trigger!
  Tier 1 (medium): [4MB]
  Tier 2 (large): [16MB]

After:
  Tier 0: []
  Tier 1: [4MB] [4MB]  ← merged output
  Tier 2: [16MB]
```

**アルゴリズム：**
```python
def size_tiered_compaction():
    for tier in tiers:
        if tier.file_count >= MIN_THRESHOLD:
            # Pick similar-sized files
            files = tier.select_files(MIN_THRESHOLD)

            # Merge into new file
            merged = merge(files)

            # Add to appropriate tier
            next_tier = find_tier_for_size(merged.size)
            next_tier.add(merged)

            # Remove old files
            tier.remove(files)
```

**特性：**
| 側面 | 値 |
|------|-----|
| 書き込み増幅 | 約3-5倍 |
| 空間増幅 | 最大2倍 |
| 読み取り増幅 | O(tiers × files per tier) |
| 最適な用途 | 書き込み中心のワークロード |

### レベル型コンパクション（LCS）

```
Organize files into levels with size limits.
Each level (except L0) has non-overlapping key ranges.

Level 0: 4 files max (may overlap)
Level 1: 10 MB total
Level 2: 100 MB total
Level 3: 1000 MB total
...

Compaction:
  When level exceeds limit, pick file and merge with overlapping files in next level.
```

```
Before:
  L0: [a-d] [c-f] [e-h]  ← overlapping
  L1: [a-c] [d-f] [g-i]  ← non-overlapping

L0 exceeds limit, compact [c-f]:
  Overlaps with L1's [d-f]
  Merge them

After:
  L0: [a-d] [e-h]
  L1: [a-c] [c-f'] [g-i]  ← f' is merged result
```

**アルゴリズム：**
```python
def leveled_compaction():
    for level in range(MAX_LEVELS - 1):
        if level_size(level) > size_limit(level):
            # Pick file to compact (round-robin or by overlap)
            file = pick_file_to_compact(level)

            # Find overlapping files in next level
            overlapping = find_overlapping(level + 1, file.key_range)

            # Merge
            merged_files = merge(file, overlapping)

            # Add to next level
            add_to_level(level + 1, merged_files)

            # Remove old
            remove_files([file] + overlapping)
```

**特性：**
| 側面 | 値 |
|------|-----|
| 書き込み増幅 | 約10-30倍 |
| 空間増幅 | 約1.1倍 |
| 読み取り増幅 | O(levels) |
| 最適な用途 | 読み取り中心、空間効率重視 |

### FIFOコンパクション

```
Delete oldest files when total size exceeds limit.
No merging, just time-based deletion.

├── newest_file.sst
├── file_2.sst
├── file_3.sst
├── oldest_file.sst  ← Delete when over limit
```

**ユースケース：**
- TTL付きの時系列データ
- N日後に期限切れになるログ
- 固定保持期間のメトリクス

### タイムウィンドウコンパクション（TWCS）

```
Combine STCS within time windows, FIFO across windows.

Time windows:
  [Today]  [Yesterday]  [2 days ago]  [3 days ago] ...

Within each window: Size-tiered compaction
When window expires: Drop entirely

Good for time-series with TTL
Avoids mixing old and new data
```

---

## マージプロセス

### N方向マージ

```python
def merge(sstables):
    # Use min-heap for efficient N-way merge
    heap = MinHeap()
    iterators = [sst.iterator() for sst in sstables]

    # Initialize heap with first entry from each
    for i, it in enumerate(iterators):
        if it.valid():
            heap.push((it.key(), it.value(), i))
            it.next()

    writer = SSTableWriter()
    last_key = None

    while not heap.empty():
        key, value, sst_idx = heap.pop()

        # Keep only latest version of each key
        if key != last_key:
            if not is_tombstone(value) or not at_bottom_level:
                writer.add(key, value)
            last_key = key

        # Advance that SSTable's iterator
        it = iterators[sst_idx]
        if it.valid():
            heap.push((it.key(), it.value(), sst_idx))
            it.next()

    return writer.finish()
```

### バージョンの処理

```
Same key in multiple SSTables:
  [key: "a", value: "v1", seq: 100]  ← older
  [key: "a", value: "v2", seq: 150]  ← newer

Merge keeps seq: 150 (latest)
But if snapshot active at seq: 120, must keep both!
```

### トゥームストーンの処理

```
Tombstone (delete marker):
  [key: "a", TOMBSTONE, seq: 200]

Can only be garbage collected when:
  - At bottom level
  - All older versions are in this compaction
  - No active snapshots before tombstone seq
```

---

## コンパクションのスケジューリング

### いつコンパクションするか

```
Triggers:
  1. L0 file count exceeds threshold
  2. Level size exceeds limit
  3. Tombstone ratio too high
  4. Manual trigger

Priority:
  L0 compaction > Other levels
  (L0 blocks writes when too many files)
```

### レート制限

```
Problem: Compaction uses disk I/O
  - Competes with foreground reads/writes
  - Can cause latency spikes

Solutions:
  - Rate limit compaction I/O (bytes/sec)
  - Priority I/O scheduling (ionice)
  - Adaptive rate based on workload
```

### スレッドプール

```
Typical configuration:
  - 1-2 threads for L0→L1 (critical path)
  - Multiple threads for other levels
  - Separate thread pool for flushing
```

---

## コンパクションのトレードオフ

### 書き込み増幅の詳細

```
Leveled with ratio R=10:

Key written to MemTable: 1 write
Flushed to L0: 1 write
Compacted L0→L1: ~1 write
Compacted L1→L2: ~10 writes (merged with 10 L1 files)
Compacted L2→L3: ~10 writes
...

Total: 1 + 1 + 1 + 10 + 10 + 10 + ... = O(10 * L)

For 4 levels: ~40x write amplification
```

### 空間増幅の詳細

```
Size-tiered worst case:
  4 files in tier, about to merge
  + space for merged output
  = 2x space temporarily

Leveled:
  Non-overlapping files
  Only small compaction in progress
  ~1.1x typical
```

### 読み取り増幅の詳細

```
Size-tiered:
  Check all tiers (O(T))
  Check all files in tier (O(F))
  Total: O(T * F)

Leveled:
  L0: Check all files (small count)
  L1+: One file per level (non-overlapping)
  Total: O(L0 files + Levels)

Bloom filters reduce actual disk reads significantly
```

---

## コンパクションのモニタリング

### 主要メトリクス

```
Compaction pending bytes:
  How much work is queued
  Growing = compaction falling behind

Compaction I/O rate:
  MB/s written by compaction
  Compare to write rate

L0 file count:
  Should stay below stall threshold
  Spikes = write bursts or slow compaction

Write stalls:
  Time spent waiting for compaction
  Should be near zero normally
```

### RocksDBの統計

```
db.getProperty("rocksdb.compaction-pending")
db.getProperty("rocksdb.num-files-at-level0")
db.getProperty("rocksdb.estimate-pending-compaction-bytes")
db.getProperty("rocksdb.compaction-reason")
```

---

## 最適化

### トリビアルムーブ

```
If L(n) file doesn't overlap with L(n+1):
  Just move file pointer, don't rewrite!

Huge win for sequential insert patterns
```

### サブコンパクション

```
Split large compaction into parallel sub-ranges:

File [a-z] overlaps with 10 files in next level
Split into:
  [a-j] + overlapping → thread 1
  [k-z] + overlapping → thread 2

Parallel merge, faster completion
```

### 動的レベルターゲット

```
RocksDB's dynamic leveling:
  Adjust level targets based on actual data size
  Avoids extra compaction when data is small
  Grows naturally as data grows
```

---

## コンパクション負債: スループットの方程式

コンパクションは生産者-消費者システムであり、多くの障害が破っている容量の方程式を持っています:

```
Sustainable ingest = usable device write bandwidth / write amplification

Example (leveled, WA ≈ 20, NVMe with 2 GB/s sustained writes,
reserving half the bandwidth for reads and flushes):
  sustainable ingest ≈ (2 GB/s × 0.5) / 20 = 50 MB/s of app writes

Ingest above that rate doesn't fail — it accrues DEBT:
  pending compaction bytes grow
  → more overlapping runs per read (read amp climbs first)
  → L0 count crosses slowdown trigger (writes throttled)
  → L0 count crosses stop trigger (writes stall — outage)

The insidious part: a burst 3× over sustainable rate for one hour
creates hours of catch-up work. Debt is repaid at
(compaction bandwidth − ongoing ingest × WA), which may be nearly
zero if steady-state ingest is close to the limit.
```

ここから導かれる運用ルール: バーストの返済余力を残すため、定常インジェストは**持続可能レートの50〜70%以下**にサイジングする。*pending compaction bytes* を第一級のSLOシグナルとして扱う（これが先行指標です — 読み取りレイテンシとストールは遅行指標）。追い付き作業中は、インジェストを明示的にレート制限するか、戦略を一時的に緩める（universal/tiered）こと — ストールトリガーにスロットリングをやらせるのは、考えうる最悪のレイテンシプロファイルをもたらします。

### 時間が実際に消える場所

```
A compaction job is a pipeline: read blocks → decompress → merge-sort
→ (re)compress → write blocks → sync + manifest update.

CPU can be the bottleneck, not the disk: zstd on the bottom level
compresses at a few hundred MB/s per core — a 4-thread compaction
budget caps at ~1 GB/s of logical throughput regardless of NVMe
speed. This is why per-level compression config (LZ4 up top, zstd
at the bottom) is a throughput decision, not just a space one.
```

---

## トゥームストーンGCの危険

上のメカニクスではトゥームストーンは最下層で落ちると述べましたが、本番では2つの制約が加わり、トゥームストーンは再発性のインシデントクラスになります:

```
Resurrection window (Cassandra): a tombstone may only be GC'd after
gc_grace_seconds (default 10 days). Why: a replica that was DOWN when
the delete happened never got the tombstone. If the other replicas GC
it before that node is repaired, repair copies the node's old value
BACK — the deleted row resurrects cluster-wide.
  → gc_grace_seconds must exceed your worst repair interval
  → dropping it to reclaim space faster trades disk for resurrections

The tombstone scan problem: reads over ranges with many deletes must
iterate every tombstone to prove rows dead. A table used as a queue
(insert → consume → delete at the head) degrades until every poll
scans millions of tombstones — Cassandra aborts such reads with
TombstoneOverwhelmingException. Fixes: don't build queues on LSM
tables; use time-bucketed tables dropped whole, or FIFO/TWCS where
expiry deletes files instead of rows.

Targeted tombstone compaction: engines can compact a single SSTable
when its tombstone ratio is high (Cassandra tombstone_threshold,
default 0.2; RocksDB CompactOnDeletionCollector) — worth enabling on
delete-heavy tables so tombstones don't wait for shape-driven
compaction to find them.
```

---

## 箱の外のコンパクション: 階層化とリモート

SSTableはイミュータブルで自己完結しているため、ファイルもマージ作業も、サービングノード上に住む必要はありません:

```
Tiered storage: bottom-level SSTables (≈90% of bytes, coldest data)
move to object storage; the node keeps hot levels + indexes/filters
locally. Reads to cold data pay object-store latency (see
Object Storage chapter); economics usually win for logs/time-series.

Remote compaction: the merge itself runs on stateless workers reading
and writing SSTables in object storage (RocksDB-Cloud, and the
pattern behind several cloud-native engines). The serving node stops
paying compaction CPU/IO entirely — the amplification triangle's
write-amp corner becomes an elastic, separately-billed compute pool.
The cost: coordination (which worker owns which file set) moves into
a manifest/metadata service — the same "truth in a small pointer,
bytes in immutable objects" pattern as lakehouse table formats.
```

これがクラウドLSMの進行方向です: ディスクシークを避けるために1996年に設計されたSSTableのイミュータブル性は、ストレージ・コンピュート分離を成立させるまさにその性質だったと判明しました。

---

## 重要なポイント

1. **SSTableはイミュータブル** - 一度書き込み、何度も読み取り
2. **コンパクションは核心的なトレードオフ** - 書き込み vs 読み取り vs 空間
3. **サイズ階層型は書き込みに有利** - ただし空間を浪費します
4. **レベル型は読み取りに有利** - ただし書き込み増幅が大きくなります
5. **FIFOは時系列データ向け** - 古いデータが自然に期限切れになる場合
6. **L0は重要** - ファイルが多すぎると書き込みがストールします
7. **ブルームフィルタは不可欠** - ディスク読み取りを劇的に削減します
8. **コンパクション負債を監視する** - 遅れるとパフォーマンスに影響します
