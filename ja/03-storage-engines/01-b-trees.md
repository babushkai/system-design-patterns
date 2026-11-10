# B木

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../03-storage-engines/01-b-trees.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

B木はほぼすべてのOLTPデータベースのデフォルトのインデックス構造であり、50年以上生き延びてきた理由はただひとつ: メモリ階層の形に一致しているからです。ファンアウトが数百のページサイズのノードからなる木は、10億キーを4回のページ読み取りで到達可能に保ちます — そのうち3回は通常キャッシュ済みなので、ポイントルックアップのコストはディスクI/O 1回かゼロです。教科書的な説明が省く部分にこそ本番の挙動が住んでいます: B木は実際には*バッファプール構造*です（「ディスク読み取り」は普通、メモリ読み取り+帳簿処理です）。書き込み増幅は行サイズではなくページサイズです（100バイトの更新が8〜16KBのページをダーティにし、さらにWAL、場合によってはフルページイメージも）。並行アクセスはラッチプロトコル（crabbing、楽観的降下、B-linkの横方向ポインタ）に支配され、それがマルチコアのスケーラビリティを決めます。そしてキーの選択 — シーケンシャルかランダムか — は挿入スループットとインデックスサイズを整数倍単位で揺さぶります。本章ではコストモデルをページから積み上げ、分割/マージ、並行性、リカバリとの相互作用、PostgreSQL/InnoDBの具体論、そして実際に人を呼び出す障害モード（ブロート、UUIDキー、過剰インデックス）を扱います。

---

## コストモデル: ページ、ファンアウト、メモリ階層

ストレージはバイト単位ではなくページ単位で読まれます。ランダム読み取りのコストは、ページの8バイトを使おうが全部を使おうがほぼ同じです:

```
Access cost (order of magnitude):
  L1/L2 cache hit:      ~1-10 ns
  DRAM (buffer pool):   ~100 ns
  NVMe SSD random 4KB:  ~20-100 μs      (~1,000× DRAM)
  SATA SSD random:      ~100-200 μs
  HDD random (seek):    ~5-10 ms        (~100,000× DRAM)

Design consequence: the only number that matters for a disk-resident
index is PAGE READS PER OPERATION. Comparisons are free by comparison.
```

B木の答え: 各ノードを丸ごと1ページにすることで、ページ読み取り1回ごとに探索範囲が2分の1ではなく*ファンアウト*（ノードあたりの子の数）分の1に狭まるようにします:

```
Fanout arithmetic (8 KB page, ~16-byte keys + 8-byte child pointers):
  entries per internal page ≈ 8192 / 24 ≈ 340 → call it ~300

  Height 1:  ~300 keys
  Height 2:  ~90,000
  Height 3:  ~27,000,000
  Height 4:  ~8,100,000,000

A billion-row table is a 4-level tree. And the top of the tree is tiny:
  root:               1 page
  level 2:          ~300 pages   (~2.4 MB)
  level 3:       ~90,000 pages   (~700 MB)
→ root + level 2 always cached; level 3 mostly cached.
  A point lookup is typically 3 buffer-pool hits + at most 1 real I/O.
```

これがゲームのすべてです: **1操作あたり log_fanout(n) 回のページアクセスで、上位レベルはデータに比べて微小なのでほぼすべてキャッシュヒットになる**。二分探索木、スキップリスト、ハッシュテーブルがディスク上で負けるのは漸近計算量が悪いからではなく、1ステップあたりの絞り込み係数がページサイズのI/Oに見合わないからです。

---

## 構造: B+木 — みんなが実際に作っているもの

データベースが「B木」と呼ぶものはほぼ常に**B+木**です: 値はリーフのみに存在し、内部ノードはルーティングキーと子ポインタを持ち、リーフは兄弟ポインタで連結されています。

```
     ┌─────────────┐
     │  30  |  60   │                ← internal: routing keys only
     └──┬────┬────┬─┘
        ↓    ↓    ↓
   ┌──────┐ ┌──────┐ ┌──────┐
   │10|20 │→│30|40 │→│60|80 │       ← leaves: keys + values, sibling-linked
   └──────┘ └──────┘ └──────┘

Why values-only-in-leaves wins:
  1. Internal nodes stay small (key + pointer, no payload)
     → maximum fanout → minimum height
  2. Every lookup has identical depth → predictable latency
  3. Range scan = locate start leaf, then walk sibling pointers
     sequentially — no re-traversal
  4. Internal keys are only separators: they can be truncated to the
     shortest prefix that still routes correctly ("suffix truncation"),
     raising fanout further
```

古典的なB木（すべてのノードに値を持つ）が生き残っているのは主に教科書の中です。SQLiteはインデックス用b-treeに使っていますがテーブルはB+であり、それ以外 — PostgreSQL nbtree、InnoDB、SQL Server、Oracle、WiredTiger — はすべて改良を加えたB+です。

### ページの内側: スロット化レイアウト

```
┌──────────────────────────────────────────────────────┐
│ header │ slot array →   ...free space...   ← records │
└──────────────────────────────────────────────────────┘
  header: page LSN, record count, free-space pointers
  slots:  (offset, length) pairs, kept sorted by key
  records: variable-length, grow from the end

Binary search runs over the slot array (fixed-width, cache-friendly);
records never move on insert — only slots do. Deletion marks a slot
dead; space is reclaimed by in-page compaction when needed.
```

ヘッダのページLSNは[Write-Ahead Logging](./04-write-ahead-logging.md)へのフックです: リカバリはこれをログレコードと比較し、どの変更が既にページに反映済みかを判定します — redoを冪等にする仕組みです。

### クラスタ化 vs セカンダリ: 「値」とは何か

```
Heap tables (PostgreSQL):
  index leaf holds (key → TID), a pointer into the heap
  every index on the table is equal; row lives in the heap

Clustered index (InnoDB, SQL Server default):
  the PRIMARY KEY B+-tree's leaves ARE the rows
  secondary index leaves hold (key → primary key value)
  → secondary lookup = two B-tree descents (secondary, then PK)
  → a fat primary key silently fattens EVERY secondary index
```

最後の行は本番で繰り返される驚きです: InnoDBで36バイトのUUID文字列を主キーにすると、そのテーブルの全セカンダリインデックスの全エントリに36バイトが加算されます。

---

## 操作、そしてコストが実際にある場所

### 探索とレンジスキャン

```
Point lookup:  descend root → leaf, binary search each page
  cost = height page accesses ≈ 3-4, nearly all cached

Range scan [a, b):  descend to leaf containing a,
  then walk sibling pointers until b
  cost = height + ⌈K / entries_per_leaf⌉ sequential page reads
  — sequential after the seek, which is why B-trees serve
    ORDER BY ... LIMIT and time-range queries so well
```

### 挿入と分割のカスケード

```
insert(k, v):
  descend to leaf; if room → write into page (common case: 1 dirty page)
  if full → SPLIT:
    allocate new page, move upper half of entries there,
    insert separator key into parent
    if parent full → split parent … possibly up to the root
    root split → tree grows one level (the ONLY way height increases,
    which is why B-trees stay balanced with no rebalancing pass)
```

分割を集計上安価に保つ事実が2つあります。第一に、分割は稀です: リーフは分割の合間に `entries_per_leaf / 2` のオーダーの挿入を吸収するので、挿入あたりの償却コストはページ書き込みの何分の一かです。第二に、エンジンは最悪になりうるパターンを特別扱いします: **シーケンシャルキーに対する右端分割**です。単調増加キーの挿入は常に最右のリーフに当たります。素朴な半分割は半分空のページの行列を残しますが、エンジンは代わりに「挿入点で」分割し（PostgreSQLのfastpath、InnoDBのシーケンシャル挿入ヒューリスティック）、左側のページをほぼ満杯のまま残し、追記型キーではインデックスを90%以上の密度に詰め込みます。

```
Split economics, 8 KB leaves, ~150 entries/leaf:

  Sequential keys (timestamps, sequences):
    splits only at right edge, pages left ~100% full
    index density ~90-100%, minimal page count

  Random keys (UUIDv4):
    every leaf equally likely to split; steady-state fill ≈ 2/3 (ln 2 ≈ 69%)
    → ~1.4× more leaf pages for the same data
    → 1.4× more buffer-pool pressure, 1.4× more pages to WAL-image
    AND: every insert touches a random page → the working set is the
    ENTIRE leaf level; with a 700 MB leaf level and a smaller buffer
    pool, every insert is a read-modify-write with a real disk read.
```

2つ目のブロックが「UUID主キーは遅い」現象の定量化です。時間順序付きID（ULID、UUIDv7、Snowflake ID）は分散生成を保ったままシーケンシャルなパターンを回復します — 通常これが正しい修正です。

### 削除: 教科書より怠惰

CLRSはアンダーフローしたノードを即座にマージします。実システムはほぼやりません: PostgreSQLはインデックスタプルを死んだとマークし、VACUUMは*空になった*ページのみ回収します（部分的に埋まったページをマージすることはありません）。InnoDBはページが閾値（`MERGE_THRESHOLD`、デフォルト50%）を下回ったときだけマージします。理由: 削除の多いワークロードは同じレンジに再挿入することが多く、マージ→再分割のスラッシングは緩みを抱えるより高くつくからです。帰結: **B木インデックスは再構築でしか引き締まりません** — 大量削除してもインデックスは `REINDEX` / `OPTIMIZE TABLE` まで同じサイズのままです。

---

## 書き込み増幅: ページサイズ、プラスログ

B木のI/O単位はページなので、小さな行に対する書き込み増幅は構造的です:

```
UPDATE of a 100-byte row, 8 KB pages, worst case (PostgreSQL-flavored):
  WAL record:                        ~150 bytes
  full-page image (first touch of the page after a checkpoint):
                                     ~8 KB into the WAL
  heap page write (at checkpoint):    8 KB
  index page write (if index updated): 8 KB
  ────────────────────────────────────────────
  ~24 KB of I/O for 100 logical bytes ≈ 240×  (worst case)

Steady state is far better: pages absorb many updates between
checkpoints (one page write amortizes over all of them), and only the
first touch per checkpoint pays the full-page image. Realistic WA for
OLTP: ~2-10×. But the WORST case is what sizes your disks and your
checkpoint tuning — spiky WAL volume right after each checkpoint is
the visible symptom.
```

フルページイメージが存在するのは**torn page**（引き裂かれたページ）のためです — 8KBのページ書き込みは4KBセクタのデバイス上でアトミックではありません。防御策（FPI、InnoDBのダブルライトバッファ）は[Write-Ahead Logging](./04-write-ahead-logging.md)で扱います。

[LSM木](./02-lsm-trees.md)と対比すると: LSMはランダムなページサイズの書き込みをシーケンシャルなバッチ書き込みに変換しますが（インジェストに優れる）、コンパクションで繰り返し支払います（*論理データ*に対して10〜30倍のWA、時間に分散）。B木のWAは更新ごと・即時、LSMのWAは遅延・バックグラウンドです。どちらが安いかは更新の局所性次第です: 繰り返し更新されるホットな行はB木ではほぼ無料（同じページ、最終的に1回の書き込み）で、LSMでは高価（全バージョンが全レベルを通して書き直される）です。

### エンジンが実際に使う緩和策

```
- Buffer pool absorbs re-writes: dirty page written once per checkpoint,
  not per update — checkpoint interval is a WA knob
- Group/async commit amortize the WAL fsync (see WAL chapter)
- HOT updates (PostgreSQL): update that changes no indexed column
  rewrites only the heap page — zero index writes
- Change buffering (InnoDB): secondary-index modifications for pages
  not in memory are buffered and merged later — turns random index I/O
  into batched I/O
- B^ε-trees push this to the limit: each internal node carries a buffer
  of pending messages flushed downward in batches — write-optimized
  B-trees (TokuDB/PerconaFT lineage), trading read latency for write WA
```

---

## 並行性: ラッチ、crabbing、そして横へ行く

並行アクセス下のB木は、物理的なページの整合性（ラッチ、マイクロ秒スケール）とトランザクション分離（ロック、トランザクションスケール）を別々に守らなければなりません。興味深いエンジニアリングはラッチ側にあります — 数百コアの時代、どうラッチするかがインデックスのスケールを決めます。

```
Latch crabbing (the classical protocol):
  descend holding parent latch until child latch acquired;
  release parent as soon as child is "safe"
  (safe = can't split for insert / can't underflow for delete)

  Readers: shared latches, release immediately → cheap
  Writers: exclusive latches; the root is the choke point —
  a pessimistic writer holds it until it knows no split will cascade
```

```
Optimistic descent (what modern engines do):
  descend with SHARED (or no) latches assuming no split will happen
  latch exclusively only the leaf; if it turns out to split,
  restart the descent pessimistically
  → splits are rare, so the fast path wins almost always
  Optimistic Lock Coupling generalizes this with per-page version
  counters: readers don't latch at all, they validate versions —
  reads scale linearly with cores
```

```
B-link trees (Lehman & Yao 1981): every node gets a HIGH KEY and a
RIGHT-SIBLING pointer. A split first creates the right sibling, then
updates the parent — and a concurrent reader that lands on the old
page mid-split detects (key > high key) and simply follows the sibling
pointer sideways. Readers never block on splits; writers latch at most
2-3 pages. This is PostgreSQL's actual implementation, and the reason
its index scans don't stall behind concurrent inserts.
```

リカバリもここに絡みます: 分割の途中でクラッシュしても、到達不能なページを残してはなりません。PostgreSQLは分割を1つのアトミックなWALレコード+redo時に完了される遅延親挿入としてログし、InnoDBはミニトランザクション（アトミックな複数ページredoグループ)を使います。不変条件: **構造変更は複数ページにまたがってもログ上アトミック** — [Write-Ahead Logging](./04-write-ahead-logging.md)参照。

### Copy-on-Write B木: もうひとつの道

LMDB、BoltDB、btrfsは書き込み側のラッチングを丸ごとスキップします: ページをin-placeで変更せず、変更されたリーフとルートまでのパスの新しいコピーを書き、ルートポインタをアトミックに差し替えます。

```
+ readers need NO latches ever (any root they hold is a consistent snapshot)
+ crash recovery is free — old root is always valid, no WAL required
+ snapshots/MVCC are a pointer copy
- every logical write rewrites height pages (WA multiplied by tree height)
- single writer at a time (LMDB), space reclamation needs GC
→ superb for read-dominated embedded workloads; wrong shape for
  write-heavy multi-writer OLTP
```

---

## PostgreSQLとInnoDB: 重要なノブ

```
PostgreSQL nbtree:
  fillfactor (default 90): headroom per leaf to absorb inserts without
    splitting — drop to 70-80 for heavy random-update columns
  HOT updates: keep frequently-updated columns OUT of indexes so
    updates skip index maintenance entirely
    (check pg_stat_user_tables.n_tup_hot_upd / n_tup_upd)
  B-tree deduplication (PG 13+): duplicate keys stored once with a
    TID list — low-cardinality indexes shrink 3-10×
  Bottom-up index deletion (PG 14+): kills dead index tuples at the
    moment a page would split, preventing bloat from update churn
  REINDEX CONCURRENTLY: the only way to un-bloat; VACUUM never
    merges partially-empty index pages
  Diagnostics: pgstatindex() → avg_leaf_density (<50% = bloated),
    bt_metap() for height; pg_stat_user_indexes.idx_scan = 0 → drop it
```

```
InnoDB:
  clustered PK: keep it SHORT and MONOTONIC (bigint auto-inc or UUIDv7)
    — every secondary index carries a copy of it
  change buffer: batches secondary-index updates for cold pages
  adaptive hash index: hash shortcut over hot B-tree pages, built
    automatically (and sometimes worth disabling under contention)
  innodb_fill_factor, MERGE_THRESHOLD per index
```

```
When a B-tree is the wrong index (PostgreSQL menu):
  BRIN: physically-ordered append-only data (time series) —
    min/max per block range, ~1000× smaller than B-tree
  GIN: contains-style queries (arrays, JSONB, full text)
  Hash: equality-only, marginal wins; rarely worth it
  Partial/covering indexes: cheaper than another full B-tree —
    index only the rows (WHERE ...) or add INCLUDE payload columns
    to enable index-only scans
Multi-column indexes route by leftmost prefix: (a,b,c) serves
  a / a,b / a,b,c — never b alone. Order columns by equality-first,
  then range; a range predicate stops index use for later columns.
```

---

## 障害モード

**更新/削除の出入りによるインデックスブロート。** 死んだインデックスタプルが蓄積し、ページは半分空のまま残り、同じ論理インデックスが2〜5倍のページを消費し、キャッシュヒット率が下がり、スキャンは*徐々に*遅くなります — エラーはなく、ただ漂流するだけ。過充填されたブルームフィルタとまったく同じです。`avg_leaf_density` を監視し、書き込みの激しいインデックスには `REINDEX CONCURRENTLY` をスケジュールしてください。PG 13/14の重複排除+bottom-up deletionはこれを劇的に削減します。それ以前のバージョンではブロート管理は常設の運用業務です。

**ランダムキー挿入のワーキングセット。** UUIDv4キーはすべての挿入を一様にランダムなリーフに当てます。リーフレベルがバッファプールを超えた瞬間、挿入1回 = ランダム読み取り1回+いずれランダム書き込み1回となり、スループットは「200GBあたりでデータベースが遅くなった」ように見える崖から落ちます。直すべきはキー（UUIDv7/ULID）であってハードウェアではありません。

**ホットな右端。** 単調キーはすべての挿入を最右リーフに集中させます — 高並行下でのラッチホットスポットです（SQL Serverの「last page insert contention」、OPTIMIZE_FOR_SEQUENTIAL_KEYで緩和。PostgreSQLのfastpathが大半を解消）。皮肉にも前項と正反対の病理です: シーケンシャルキーは1つのラッチにストレスをかけ、ランダムキーはキャッシュ全体にストレスをかけます。

**過剰インデックス。** すべてのインデックスは、すべての書き込みで維持される丸ごと1本の追加B木です: セカンダリインデックス5本 ≈ 挿入1回あたり6回のページダーティ化+そのWAL。二桁のインデックス数を持つ書き込み重視テーブルは、I/Oの大半を誰もクエリしないインデックスの維持に費やします。`idx_scan` を監査して削除しましょう。

**クリーンアップを打ち破る長時間トランザクション。** HOTプルーニング、bottom-up deletion、VACUUMはすべて最古の可視スナップショットを尊重します。忘れられた `idle in transaction` セッション1つがデータベース全体のインデックスクリーンアップを止め、出入りをそのままブロートに変換します。

**太ったキー。** 幅広のテキストキーはファンアウトを縮め（ページあたりのセパレータが減る → 木が高くなる → ルックアップあたりのI/O増）、InnoDBでは全セカンダリインデックスに複製されます。長い文字列はハッシュかプレフィックスをインデックスし、主キーは8〜16バイトに保ちましょう。

---

## 意思決定フレームワーク

| 状況 | 選ぶもの |
|---|---|
| OLTPのポイントルックアップ+短いレンジスキャン、中程度の書き込み | B+木（デフォルトであるのには理由がある） |
| 書き込み重視のインジェスト、ポイント読み取り少、キー順データ | [LSM木](./02-lsm-trees.md) — シーケンシャル書き込みがページRMWに勝つ |
| 追記のみの時系列、時間によるレンジスキャン | BRIN（PostgreSQL）または時間順キーのLSM |
| 分散ID生成+B木主キー | UUIDv7/ULID/Snowflake — UUIDv4を主キーにしない |
| 読み取り中心の組み込みストア、スナップショット読み取り | COW B木（LMDB/BoltDB） |
| 更新の激しいカラム | インデックスから外す（HOTを有効化）。fillfactorを下げる |
| 低カーディナリティのインデックス（status、type） | PG 13+の重複排除B木、またはホット値ごとの部分インデックス |
| バッファプールより大きいインデックス+ランダムアクセス | I/Oバウンドな挙動を予期する。キーを縮め、未使用インデックスを削除し、さもなくばキャッシュミスの経済を受け入れる |

---

## 重要なポイント

1. **ページ読み取りを数え、そのうちどれがキャッシュ済みかを数える** — B木のルックアップは高さ分のアクセスで、上位2〜3レベルは実質無料。設計全体がファンアウト最大化のために存在する。
2. **どこもかしこもB+**: 値はリーフに、切り詰められたセパレータは内部ノードに、リーフは兄弟連結でスキャンに備える。
3. **書き込み増幅はページサイズ+ログサイズ** — 最悪ケースで数百倍、償却で2〜10倍。チェックポイント頻度とフルページイメージがスパイクを決める。
4. **キーの順序は第一級の設計判断** — シーケンシャルキーはページを詰めキャッシュに美しく乗る（が右端で競合する）。ランダムUUIDは木を約1.4倍膨らませ、リーフレベル全体をワーキングセットに変える。
5. **削除はB木を縮めない** — 縮めるのは再構築だけ。ブロートは異常ではなく、監視され管理される量である。
6. **モダンな並行性は楽観的+横方向** — バージョン検証付き降下とB-linkの兄弟ポインタであり、ルートラッチの行列ではない。
7. **クラスタ化設計では主キーはすべてのインデックスの一部** — 短く単調な主キーはスタイルの好みではなくストレージの意思決定。
8. **B木/LSMの選択は更新の局所性の問題** — ホットな行への繰り返し更新はB木に有利、大量のユニークキーインジェストはLSMに有利。

---

## 参考文献

- Bayer, R., & McCreight, E. (1972). *Organization and Maintenance of Large Ordered Indexes*. Acta Informatica.
- Comer, D. (1979). *The Ubiquitous B-Tree*. ACM Computing Surveys.
- Lehman, P., & Yao, S. B. (1981). *Efficient Locking for Concurrent Operations on B-Trees*. TODS. (B-link trees.)
- Graefe, G. (2011). *Modern B-Tree Techniques*. Foundations and Trends in Databases. (The comprehensive survey.)
- Leis, V., et al. (2019). *Optimistic Lock Coupling: A Scalable and Efficient General-Purpose Synchronization Method*. IEEE Data Eng. Bulletin.
- Brodal, G., & Fagerberg, R. (2003); Bender, M., et al. — B^ε-tree / write-optimization line of work behind TokuDB/PerconaFT.
- PostgreSQL documentation: *nbtree README*, B-tree deduplication (13), bottom-up deletion (14); `pageinspect`, `pgstattuple`.
- MySQL/InnoDB documentation: clustered indexes, change buffer, adaptive hash index.
