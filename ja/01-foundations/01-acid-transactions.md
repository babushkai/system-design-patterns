# ACIDトランザクション

> この記事は英語版から翻訳されました。最新版は[英語版](/01-foundations/01-acid-transactions)をご覧ください。

## TL;DR

ACIDはデータベーストランザクションの信頼性を保証するプロパティの集合です。しかし「ACID」はマーケティング用語であり、実際の保証はデータベースによって大きく異なります。各文字の背後には実際のエンジニアリング上のトレードオフが隠れています。undo logとredo log、fsyncのレイテンシと耐久性、分離コストとスループット。各文字の仕組みを理解することが、クラッシュに耐えるシステムと静かにデータを破壊するシステムの違いを生みます。

---

## ACIDが解決する問題

銀行振込を考えましょう。口座Aから口座BへUSD 100を移動します。

```text
1. Read balance of A: $500
2. Subtract $100 from A: $400
3. Write new balance to A
4. Read balance of B: $200
5. Add $100 to B: $300
6. Write new balance to B
```

トランザクション保証がなければ何が起こりうるでしょうか？

**クラッシュ障害:**
- ステップ3の後にクラッシュ → AはUSD 100を失い、Bは何も得ていません。お金がシステムから消えました。
- ステップ6の最中にクラッシュ → ディスクに部分的な書き込みが残ります。Bの残高はUSD 200でもUSD 300でもなく、壊れたバイト列になります。

**並行性障害:**
- Aからの2つの送金が同時に実行されます。どちらもUSD 500を読み取り、どちらもUSD 100を引き、どちらもUSD 400を書き込みます。AはUSD 300であるべきなのにUSD 400になっています。銀行は無からUSD 100を生み出しました。
- レポートクエリがステップ3と6の間に実行されます。Aの引き落としは見えますが、Bへの入金はまだです。帳簿が合いません。

**耐久性障害:**
- データベースはCOMMIT成功と返答しました。電源が落ちます。カーネルはページキャッシュに書き込みを保持していましたが、fsyncを呼びませんでした。再起動すると書き込みは消えています。
- ディスクファームウェアは書き込みを確認しましたが、データはディスクの揮発性書き込みバッファにありました。電源喪失により「確認済み」の書き込みはプラッタに到達しませんでした。

これらは理論上の話ではありません。すべての本番データベースチームは各カテゴリの実体験を持っています。ACIDは、正しく実装・設定されれば、これらすべてを防ぐ保証の集合です。

---

## 原子性 — 詳細解説

### 実際の意味

原子性は「すべての操作が瞬時に起こる」という意味ではありません。それは分離性に近い概念です。

**原子性とは、全か無かの実行を意味します。** トランザクションがコミットされれば、そのすべての書き込みが適用されます。アボートされた場合（またはコミット前にシステムがクラッシュした場合）、その書き込みは一切見えません。

### なぜ重要か

原子性がなければ、すべての複数ステートメント操作がデータ破損の潜在的な原因になります。トランザクション途中のクラッシュ、ネットワークタイムアウト、制約違反は、データベースを不整合な中間状態に置きます。代替手段 — アプリケーションコードで手動のクリーンアップとロールバックロジックを書くこと — は、エラーが起きやすく現実的ではありません。

### Undo LogとRedo Log

データベースは原子性と耐久性のために、根本的に異なる2つのロギング戦略を使用します。ほとんどの本番システムは一方または両方を使用します。

**Undo log（ロールバックログ）:**
- ページを変更する前に、*古い値*をundo logに書き込みます
- ROLLBACKまたはクラッシュリカバリ時：undo logを再生して元の値を復元します
- InnoDB（MySQL）が原子性の主要メカニズムとして使用します
- InnoDBはundo logをシステムテーブルスペースまたは専用のundoテーブルスペースに保存します

**Redo log（先行書き込みログ / WAL）:**
- ページを変更する前に、*新しい値*をredo logに書き込みます
- クラッシュリカバリ時：redo logを再生してコミット済みの変更を再適用します
- PostgreSQLが主要メカニズムとして使用します（pg_walディレクトリ）
- PostgreSQLのWALは追記専用のシーケンシャルI/Oであり、ランダムなページ書き込みよりもはるかに高速です

**InnoDBは両方を同時に使用します（PostgreSQL 16）:**

```text
InnoDB transaction lifecycle:
1. BEGIN
2. Write old values to undo log (in buffer pool)
3. Write new values to redo log (ib_logfile0/ib_logfile1)
4. Modify buffer pool pages in memory (dirty pages)
5. On COMMIT: fsync redo log → return success to client
6. Checkpoint: flush dirty pages to tablespace files (async)
7. Purge: clean up undo log entries after no transaction needs them
```

```text
PostgreSQL transaction lifecycle (v16):
1. BEGIN
2. Write WAL records (new values) to WAL buffer
3. Modify pages in shared buffer pool (with before-images kept via MVCC)
4. On COMMIT: flush WAL buffer to pg_wal segment file → fsync → return success
5. Checkpoint: flush dirty buffers to data files (async, configurable interval)
6. Old row versions cleaned up by autovacuum (async)
```

重要な違い：InnoDBはページをインプレースで更新するため、ロールバックにundo logが必要です。PostgreSQLはMVCCを使用しており、古い行バージョンはVACUUMされるまでヒープに残るため、原子性のための独立したundo logは必要ありません。

### ROLLBACKの仕組み：LSNトラバーサルとUndoチェーン

すべてのログレコードには**Log Sequence Number（LSN）**があります。これは単調増加する識別子です。

**InnoDBのロールバック（MySQL 8.0）:**

```text
Transaction T1 modifies rows R1, R2, R3:
  LSN 1001: undo record for R1 (old value), prev_undo_ptr → NULL
  LSN 1002: undo record for R2 (old value), prev_undo_ptr → 1001
  LSN 1003: undo record for R3 (old value), prev_undo_ptr → 1002

ROLLBACK T1:
  1. Find T1's last undo record (LSN 1003)
  2. Restore R3 to old value
  3. Follow prev_undo_ptr to LSN 1002
  4. Restore R2 to old value
  5. Follow prev_undo_ptr to LSN 1001
  6. Restore R1 to old value
  7. Follow prev_undo_ptr to NULL → done
```

各トランザクションはundo recordの連結リストを保持します。ロールバックはこのチェーンを逆順にたどります。数百万行を変更したトランザクションのロールバックがトランザクション自体と同じくらい時間がかかるのはこのためです。各変更を個別に元に戻す必要があるからです。

**PostgreSQLのロールバック**はより低コストです。コミットログ（pg_xact）でトランザクションをアボート済みとマークするだけです。残された不要なタプルは可視性ルールにより後続のトランザクションからは見えず、autovacuumによって後で除去されます。

### セーブポイントと部分的ロールバック

セーブポイントを使うと、トランザクション全体をアボートせずに一部をロールバックできます。条件分岐のある複雑なビジネスロジックには不可欠です。

```sql
-- PostgreSQL 16
BEGIN;

INSERT INTO orders (id, customer_id, total) VALUES (1001, 42, 299.99);

SAVEPOINT before_inventory;

UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 7;
-- Suppose this violates a CHECK constraint (quantity >= 0)

ROLLBACK TO SAVEPOINT before_inventory;
-- The order INSERT is still intact
-- Only the inventory UPDATE was undone

-- Try alternative fulfillment
UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 7 AND warehouse = 'secondary';

COMMIT;
```

**実装の詳細：** セーブポイントはサブトランザクション（PostgreSQLではsubtransaction）を作成します。各サブトランザクションは独自のトランザクションIDを取得します。PostgreSQLはサブトランザクションの状態をpg_subtransで追跡します。InnoDBは各セーブポイントに対して新しいundo logセグメントを作成します。

**注意：** 深くネストしたセーブポイントにはオーバーヘッドがあります。PostgreSQLのpg_subtransは、数千のサブトランザクションでボトルネックになりえます。ループ内でセーブポイントが必要な場合は、トランザクション設計を見直してください。

### 分散原子性：二相コミット（2PC）

トランザクションが複数のデータベースノードにまたがる場合、ローカルのundo logでは不十分です。古典的な解決策は**二相コミットプロトコル**です。

```text
Coordinator (transaction manager)
├── Participant A (shard holding Account A)
└── Participant B (shard holding Account B)

Phase 1 — Prepare (vote):
  Coordinator → A: "PREPARE transaction T1"
  Coordinator → B: "PREPARE transaction T1"
  A: writes all changes to durable log, acquires locks, responds YES
  B: writes all changes to durable log, acquires locks, responds YES

Phase 2 — Commit (decision):
  Coordinator: all voted YES → writes COMMIT decision to its own durable log
  Coordinator → A: "COMMIT T1"
  Coordinator → B: "COMMIT T1"
  A: commits, releases locks
  B: commits, releases locks
```

```sql
-- PostgreSQL 16 native 2PC (used by connection poolers, ORMs, distributed systems)
-- On participant:
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 'A';
PREPARE TRANSACTION 'transfer_1001_partA';

-- Later, coordinator decides:
COMMIT PREPARED 'transfer_1001_partA';
-- or
ROLLBACK PREPARED 'transfer_1001_partA';
```

**コーディネータ障害問題：**

2PCの致命的な脆弱性は、フェーズ間でのコーディネータ障害です。コーディネータがすべてのYES投票を受け取った後、COMMITの決定をブロードキャストする前にクラッシュした場合：

```text
Timeline:
  t0: Coordinator sends PREPARE to A and B
  t1: A votes YES, B votes YES (both holding locks, changes durable)
  t2: Coordinator writes COMMIT to its log
  t3: *** Coordinator crashes ***

  A and B are now "in-doubt" — they cannot safely commit or abort:
  - Committing risks inconsistency if coordinator actually decided ABORT
  - Aborting risks inconsistency if coordinator actually decided COMMIT
  - Locks are held indefinitely until coordinator recovers
```

**不確定トランザクション**は運用上危険です。ロックを保持し、他のトランザクションをブロックし、コーディネータが復旧できなければ手動介入が必要になります。

```sql
-- PostgreSQL: find in-doubt transactions
SELECT gid, prepared, owner, database
FROM pg_prepared_xacts;

-- Manual resolution (ONLY when you've confirmed the correct outcome):
COMMIT PREPARED 'transfer_1001_partA';
```

**コーディネータ障害の緩和策：**
- コーディネータはフェーズ2の前に決定をレプリケートされた永続ログに書き込みます
- 参加者はタイムアウト後、コーディネータ（またはそのレプリカ）に決定を問い合わせます
- 三相コミット（3PC）はプリコミットフェーズを追加しますが、複雑さとネットワーク分断の脆弱性のため実際にはほとんど使われません
- 最新の分散データベース（CockroachDB、YugabyteDB）はコミットの決定にRaft/Paxosを使用し、単一コーディネータの障害モードを回避しています

**2PCのパフォーマンスコスト：** すべての分散トランザクションは最低でも2回の追加ネットワークラウンドトリップと3回の強制ログフラッシュ（各参加者のprepareで1回、コーディネータの決定で1回）が必要です。これは通常、ローカルトランザクションと比較して5～20msのレイテンシを追加します。

---

## 整合性 — 最も弱い文字

### データベースが強制すること vs 強制できないこと

**整合性とは、トランザクションがデータベースをある有効な状態から別の有効な状態に遷移させることを意味します。** ただし「有効」は宣言した制約によってのみ定義されます。

データベースが強制するもの：
- NOT NULL、CHECK制約
- UNIQUEおよびPRIMARY KEY
- FOREIGN KEY参照整合性
- EXCLUDE制約（PostgreSQL）
- トリガーベースの不変条件

データベースが強制できないもの：
- 「口座残高はすべての取引エントリの合計と一致すべき」（トリガーを書かない限り）
- 「すべての注文は少なくとも1つの明細行を持つべき」（テーブル横断の不変条件）
- 「すべての口座の合計は一定でなければならない」（グローバルな不変条件）
- アプリケーションコードにのみ存在するビジネスルール

**したがって整合性はACID保証の中で最も弱いものです** — これは主にアプリケーションレベルの責任です。データベースはツール（制約、トリガー）を提供しますが、正しさは開発者がそれらを使うかどうかに依存します。

### 「C」の多義性問題

文字Cは文脈によってまったく異なる意味を持ちます：

| 文脈 | 「整合性」の意味 | 保証する仕組み |
|------|-----------------|---------------|
| ACID | データが宣言された制約を満たすこと | データベース制約 |
| CAP定理 | すべてのノードが同時に同じデータを見ること（線形化可能性） | 合意プロトコル |
| レプリカ | レプリカが同じ状態に収束すること | レプリケーションプロトコル |

> [整合性モデル](04-consistency-models.md)で線形化可能性、因果整合性、結果整合性について参照してください。

これらは同じ単語を共有する、根本的に異なる3つの概念です。誰かが「このシステムは整合性がある」と言ったら、どの定義を意味しているか必ず確認してください。

### 遅延制約

一部の制約は行ごとにチェックできません。相互外部キーを考えましょう：

```sql
-- PostgreSQL 16
-- departments references employees.head, employees references departments
-- Inserting either first violates the FK of the other

-- Solution: deferred constraints
ALTER TABLE employees
  ADD CONSTRAINT fk_department
  FOREIGN KEY (department_id) REFERENCES departments(id)
  DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE departments
  ADD CONSTRAINT fk_head
  FOREIGN KEY (head_employee_id) REFERENCES employees(id)
  DEFERRABLE INITIALLY DEFERRED;

BEGIN;
INSERT INTO departments (id, name, head_employee_id) VALUES (1, 'Engineering', 100);
INSERT INTO employees (id, name, department_id) VALUES (100, 'Alice', 1);
-- Constraints checked HERE, at COMMIT time, not at each INSERT
COMMIT;
```

トランザクションごとに制約を遅延させることもできます：

```sql
BEGIN;
SET CONSTRAINTS fk_department DEFERRED;
-- ... operations that temporarily violate the constraint ...
COMMIT;  -- constraint checked here
```

**遅延制約のユースケース：**
- 循環外部キー（上記の例）
- 中間状態がユニーク制約に違反する一括データロード
- 親子の自己参照を持つグラフ構造
- データを並べ替えるスキーママイグレーション

**注意：** PostgreSQLの遅延ユニーク制約は異なるインデックスメカニズムを使用し、大きなテーブルではパフォーマンスに影響を及ぼす可能性があります。本番規模のデータでテストしてください。

### シャード間の外部キー

データベースをシャーディングすると、シャード間の外部キーに対するACID整合性はデータベース層では事実上不可能になります。

```text
Shard A (users 1-1000):
  users table, orders table for these users

Shard B (users 1001-2000):
  users table, orders table for these users

Problem: order on Shard A references a product catalog on Shard B.
  - No cross-shard FK enforcement
  - Shard B could delete the product while Shard A's order references it
  - No transaction spans both shards without 2PC (which is slow)
```

**実用的なアプローチ：**
- **非正規化：** 参照データをローカルシャードにコピーする（結果的な古さを許容する）
- **アプリケーションレベルの検証：** 書き込み前にチェックし、競合状態を許容する
- **イベント駆動のクリーンアップ：** 壊れた参照を非同期で検出・修復する
- **シャード間参照を避ける：** 関連データを同じシャードに配置する

これは分散システムが整合性を緩和することが多い根本的な理由です。 → [CAP定理](03-cap-theorem.md)を参照

### なぜ分散システムはCを捨てたか

単一ノードのデータベースでは、整合性はトランザクションレベルのプロパティです。すべての制約が宣言されていれば、DBがそれを強制します。分散データベースでは、トランザクション内でシャード間の制約を検証するコストは法外です：

- シャード間の制約チェックごとにネットワークラウンドトリップが追加される
- 分散デッドロック検出は高コスト
- グローバルな制約検証はスケールしない

これがGoogle Spanner、CockroachDB、YugabyteDBがACIDトランザクションをサポートしつつも、単一ノードのPostgreSQLと同じ保証でシャード間外部キーをサポートしない理由です。彼らのACIDにおけるCは「単一シャード上でローカルにチェックできる制約」を意味します。

---

## 分離性 — 高コストの文字

### 核心的な課題

分離性は「並行トランザクションは何が見えるか？」という問いに答えます。理想（直列化可能性）はトランザクションが1つずつ実行されたかのように振る舞うことを意味します。現実は、完全な分離は高コストなため、データベースはより弱いレベルを提供します。

### 分離レベルの概要

| レベル | ダーティリード | 非反復読み取り | ファントムリード | 書き込みスキュー |
|--------|--------------|---------------|----------------|-----------------|
| Read Uncommitted | あり | あり | あり | あり |
| Read Committed | なし | あり | あり | あり |
| Repeatable Read | なし | なし | あり（InnoDB: なし） | あり |
| Serializable | なし | なし | なし | なし |

**実装アプローチ：**
1. **ロック（2PL）：** トランザクションがロックを取得し、互いをブロックします。SQL ServerのSerializableで使用されています。
2. **MVCC：** 複数の行バージョンを保持し、読み取りが書き込みをブロックしません。PostgreSQL、InnoDBのほとんどのレベルで使用されています。
3. **OCC（楽観的並行制御）：** 競合がないと仮定し、コミット時に検証します。一部のインメモリデータベースで使用されています。
4. **SSI（直列化可能スナップショット分離）：** MVCC + 依存関係追跡。PostgreSQLの9.1以降のSerializable実装です。

> [分離レベル](/ja/01-foundations/02-isolation-levels)でMVCCの内部構造、ロッキングプロトコル、SSIの実装詳細、アノマリの詳細を参照してください。

### コネクションプールの落とし穴：接続ごとのSET TRANSACTION

コネクションプール（PgBouncer、HikariCP）使用時によくある本番バグ：

```sql
-- Developer intends Serializable for this one critical transaction:
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
BEGIN;
SELECT balance FROM accounts WHERE id = 42;
UPDATE accounts SET balance = balance - 100 WHERE id = 42;
COMMIT;
```

**問題点：** コネクションプールがこの接続をプールに返却し、別のリクエストに渡すと、分離レベルの設定が残る場合があります（プールモードとデータベースによります）。PgBouncerのトランザクションモードプーリングでは、`SET`コマンドがセッション間で漏洩します。

**正しいアプローチ：**

```sql
-- Use BEGIN with isolation level (scoped to the transaction)
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT balance FROM accounts WHERE id = 42;
UPDATE accounts SET balance = balance - 100 WHERE id = 42;
COMMIT;
-- Isolation level automatically resets after COMMIT/ROLLBACK
```

コネクションプール使用時は、分離レベルを常にBEGINの一部として設定し、別のSETコマンドとして実行しないでください。

---

## 耐久性 — レイテンシの文字

### なぜ重要か

耐久性はユーザーがデータベースを信頼し続けるための約束です。COMMITが成功を返したとき、データはプロセスクラッシュ、OSクラッシュ、電源障害を生き延びなければなりません。この約束を破ることはサイレントなデータ喪失を意味します。これは最悪の種類のバグです。データが必要になるまで誰も気づかないからです。

### fsyncの詳細

`fsync()`は耐久性を現実にするシステムコールです。何をするか（そして何をしないか）を理解することが重要です。

```text
Application writes data:
  1. write() → data goes to kernel page cache (RAM) → returns immediately
  2. fsync() → kernel flushes page cache to disk controller → waits for ack
  3. Disk controller writes to persistent media (platter or NAND cells)

What fsync actually forces:
  - Flush kernel page cache dirty pages for this file to the disk controller
  - Flush the disk's volatile write buffer to persistent storage
  - Wait for the disk to confirm the write is on stable media
```

**fsyncが嘘をつく場合：**

```text
Failure point 1: Disk write buffer
  - Some disks report fsync complete before data leaves volatile write buffer
  - Enterprise SSDs have capacitor-backed write buffers (safe)
  - Consumer SSDs may not (unsafe for databases)
  - Check: hdparm -W /dev/sda (Linux), 0 = write cache disabled

Failure point 2: RAID controller cache
  - Battery-backed (BBU) or flash-backed: safe
  - No battery: fsync lies, data in volatile controller RAM

Failure point 3: Filesystem behavior
  - ext4 with data=ordered (default): metadata journaled, data flushed before metadata
  - XFS: metadata journaled, data may have holes after crash on older kernels
  - ZFS: copy-on-write, checksums — most reliable for databases
```

**PostgreSQLとfsync — 2018年のインシデント：**

v12以前のPostgreSQLには致命的なバグがありました。fsync()が失敗した場合、PostgreSQLはダーティページがまだカーネルページキャッシュにあると仮定してfsyncをリトライしていました。しかし一部のLinuxカーネル（5.2以前）はfsync失敗時にページキャッシュからダーティページを削除していました。リトライのfsyncはクリーンなページをfsyncし、何も書き込まずに成功していました。つまりPostgreSQLはデータが永続化されたと思っていましたが、実際にはされていませんでした。

PostgreSQL 12以降はfsync失敗に対してPANIC（クラッシュリカバリ）を実行します。カーネルの状態が信頼できないためです。

### WALの仕組み

Write-Ahead LogはPostgreSQLにおける耐久性の礎石です（InnoDBではredo logが同じ役割を果たします）。

**WALセグメントファイル（PostgreSQL 16）：**

```text
$PGDATA/pg_wal/
├── 000000010000000000000001   (16 MB segment, default)
├── 000000010000000000000002
├── 000000010000000000000003
└── archive_status/

Segment naming: TimelineID + (LSN >> 24)
Default segment size: 16 MB (configurable at initdb with --wal-segsize)
```

**WALレコードの構造：**

```text
Each WAL record contains:
  - LSN (Log Sequence Number): unique, monotonically increasing position
  - Transaction ID
  - Resource manager ID (heap, btree, hash, etc.)
  - Record type (insert, update, delete, commit, etc.)
  - Before/after images of modified data (depending on full_page_writes setting)
  - CRC checksum
```

**チェックポイント頻度とクラッシュリカバリ：**

チェックポイントはshared_buffersのすべてのダーティバッファをデータファイルに書き込み、チェックポイントLSNを記録します。クラッシュリカバリ時、PostgreSQLは最後のチェックポイント以降のWALだけを再生すれば済みます。

```text
Crash recovery time ≈ (WAL generated since last checkpoint) / (sequential read throughput)

Example:
  checkpoint_timeout = 5 min (default)
  WAL generation rate = 50 MB/s (busy OLTP)
  Max WAL since checkpoint = 50 MB/s × 300s = 15 GB
  SSD sequential read = 500 MB/s
  Recovery time ≈ 15 GB / 500 MB/s = 30 seconds

Tuning tradeoffs:
  - Shorter checkpoint interval → faster recovery, more I/O during normal operation
  - Longer checkpoint interval → slower recovery, less I/O overhead
  - max_wal_size controls when a checkpoint is forced (default: 1 GB)
```

**InnoDBのredo log（MySQL 8.0）：**

```text
InnoDB uses a circular redo log (ib_logfile0, ib_logfile1 in older versions;
  #ib_redo directory with multiple files in MySQL 8.0.30+):

  - Fixed total size: innodb_redo_log_capacity (default 100 MB in 8.0.30+)
  - Circular buffer: head advances as new records are written
  - Tail advances as checkpoints flush dirty pages
  - If head catches tail: all transactions stall until checkpoint completes

  Sizing rule of thumb:
  - Redo log should hold ~1 hour of writes for busy systems
  - Too small: frequent checkpoint stalls, spiky latency
  - Too large: longer crash recovery time
```

### グループコミット：WALフラッシュのバッチ化

すべてのCOMMITにはWALのfsyncが必要です。fsyncはデータサイズに関係なく固定のオーバーヘッドがあるため、10トランザクション分を1回でfsyncするのは1トランザクション分を1回fsyncするのとほぼ同じ速さです。

**グループコミット**は複数の同時コミットを単一のWALフラッシュにバッチ化します。

```text
Without group commit:
  T1: write WAL → fsync (2ms) → return
  T2: write WAL → fsync (2ms) → return
  T3: write WAL → fsync (2ms) → return
  Total: 6ms, max throughput ≈ 500 commits/sec per disk

With group commit:
  T1: write WAL → wait
  T2: write WAL → wait
  T3: write WAL → wait
  Leader: fsync all three → return to T1, T2, T3
  Total: 2ms for all three, throughput scales with concurrency
```

**PostgreSQLのグループコミットチューニング（v16）：**

```text
# postgresql.conf

# How long to delay before flushing WAL, hoping more commits arrive
commit_delay = 10          # microseconds (default: 0 = disabled)

# Only delay if at least this many transactions are active
commit_siblings = 5        # (default: 5)

# Effect: if ≥ 5 concurrent transactions, wait 10μs before fsync
# This batches more commits into each fsync, improving throughput
# at the cost of 10μs additional latency per commit
```

**グループコミットをチューニングするタイミング：**
- 高いコミットレート（1000コミット/秒超）でfsyncがコミットレイテンシを支配している場合
- fsyncレイテンシが高いストレージ（ネットワーク接続型、クラウドボリューム）
- 小さなトランザクションが多いワークロード

**InnoDBのグループコミット（MySQL 8.0）：**

```text
InnoDB implements group commit in three stages:
1. FLUSH stage: write redo log to OS buffer
2. SYNC stage: fsync the redo log (where batching happens)
3. COMMIT stage: update transaction status

# my.cnf
innodb_flush_log_at_trx_commit = 1  # 1 = fsync every commit (default, safest)
                                     # 2 = write to OS buffer every commit, fsync once/sec
                                     # 0 = write+fsync once/sec (data loss on crash)
binlog_group_commit_sync_delay = 0   # microseconds to wait for more transactions
binlog_group_commit_sync_no_delay_count = 0  # commit immediately if this many waiting
```

### synchronous_commit = off：許容される場合

PostgreSQLの`synchronous_commit`は、COMMITがWALのfsyncを待つかどうかを制御します。

```sql
-- Per-transaction override (PostgreSQL 16)
SET LOCAL synchronous_commit = off;
-- Subsequent COMMIT returns immediately, WAL fsynced asynchronously
-- Risk window: ~10ms of data loss (3 × wal_writer_delay)
```

**失うもの：** PostgreSQLがコミットから約10ms以内にクラッシュした場合、そのトランザクションの変更は失われる可能性があります。データベースの整合性は保たれます（破損なし）が、コミット済みのトランザクションが消える可能性があります。

**許容される場合：**
- 数秒分のデータ喪失が許容されるログ/アナリティクスの挿入
- 再構築可能なセッション状態やキャッシュの書き込み
- リプレイを処理するダウンストリームコンシューマーがある高スループットのイベント取り込み

**許容されない場合：**
- 金融トランザクション
- アプリケーションがユーザーに対して成功を既に通知した書き込み
- 不可逆な副作用（送信済みメール、API呼び出し）をトリガーする書き込み

```text
Performance impact (typical SSD):
  synchronous_commit = on:  ~3,000 commits/sec
  synchronous_commit = off: ~30,000 commits/sec (10x improvement)

The gap widens on high-latency storage (cloud EBS, network-attached).
```

### クラウドの落とし穴：すべてのfsyncが同等ではない

クラウドブロックストレージは耐久性保証を変える抽象化レイヤーを導入します。

**AWS EBSボリューム：**

```text
Volume Type    | IOPS (baseline) | fsync latency  | Durability notes
---------------|-----------------|----------------|------------------
gp3            | 3,000           | 0.5–2ms        | Replicated within AZ
io2 Block Expr | up to 256,000   | 0.2–0.5ms      | 99.999% durability SLA
io1            | up to 64,000    | 0.3–1ms        | 99.8–99.9% durability
st1 (HDD)     | 500 (throughput) | 5–20ms         | Not suitable for WAL

Key insight: EBS replicates within a single AZ. An AZ outage can
lose EBS volumes. Cross-AZ replication (RDS Multi-AZ, streaming
replication) is your second tier of durability.
```

**GCP Persistent Disk：**
- pd-ssd：EBS gp3と同等のパフォーマンス
- Local SSD：最も低いレイテンシだが**エフェメラル** — VM停止/移行でデータ喪失。レプリケーションなしでWALに使用しないでください。

**クラウドストレージの一般的なルール：** クラウドプロバイダーのfsyncが正しいと仮定しつつ、`diskchecker.pl`や`fsync=1`オプション付きの`fio`で検証してください。一部のVMタイプやハイパーバイザー設定はfsyncを正しく処理しない場合があります。

### 耐久性の第二層としてのレプリケーション

単一のディスク（または単一のEBSボリューム）は本番環境の耐久性としては不十分です。ディスクは故障し、AZはオフラインになり、リージョン全体が障害を起こすこともあります。

```text
Durability tiers (PostgreSQL):

Tier 0: synchronous_commit = off
  - WAL in memory only, fsynced asynchronously
  - Risk: lose ~10ms of commits on crash
  - Use: ephemeral data

Tier 1: synchronous_commit = on (default)
  - WAL fsynced to local disk before COMMIT returns
  - Risk: disk failure loses data; AZ failure loses data
  - Use: single-node development, small deployments

Tier 2: Synchronous streaming replication
  - WAL shipped to standby AND fsynced on standby before COMMIT returns
  - synchronous_standby_names = 'standby1'
  - Risk: simultaneous failure of primary + standby
  - Cost: commit latency includes network RTT to standby (~1ms same AZ)
  - Use: production databases requiring durability

Tier 3: Synchronous replication to multiple standbys across AZs
  - synchronous_standby_names = 'FIRST 2 (standby1, standby2, standby3)'
  - Risk: simultaneous AZ failure (extremely rare)
  - Cost: commit latency = max(RTT to required standbys) (~2-5ms cross-AZ)
  - Use: critical financial/healthcare systems
```

---

## 本番環境での障害モード（トランザクション固有）

これらはトランザクションの誤用に固有の障害パターンです。 → 一般的な分類については[障害モード](06-failure-modes.md)を参照してください。

### 適切な分離性なしでの更新喪失

本番環境で最もよくあるトランザクションバグです。

**パターン（読み取り後書き込み）：**

```python
# DANGEROUS: Python with psycopg2 (PostgreSQL 16)
# Two concurrent requests both try to increment a counter

# Request 1                          # Request 2
cur.execute("SELECT count            cur.execute("SELECT count
  FROM counters WHERE id=1")           FROM counters WHERE id=1")
count = cur.fetchone()[0]  # 10      count = cur.fetchone()[0]  # 10
count += 1                            count += 1
cur.execute("UPDATE counters          cur.execute("UPDATE counters
  SET count=%s WHERE id=1",             SET count=%s WHERE id=1",
  (count,))                             (count,))
# Final value: 11 (should be 12)
```

**修正 — アトミックUPDATE：**

```sql
-- Correct: single atomic statement, no read-then-write race
UPDATE counters SET count = count + 1 WHERE id = 1;
```

**読み取り後書き込みが必要な場合（複雑なロジック）：**

```sql
-- Use SELECT FOR UPDATE to acquire a row lock
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
SELECT balance FROM accounts WHERE id = 42 FOR UPDATE;
-- Row is now locked; concurrent transactions block here
-- ... compute new balance in application ...
UPDATE accounts SET balance = 350.00 WHERE id = 42;
COMMIT;
```

### Read Committedでの部分的コミット可視性

Read CommittedはPostgreSQLのデフォルトです。トランザクション内の各**ステートメント**は新しいスナップショットを見ます。長いトランザクションで微妙なバグを引き起こします。

```sql
-- Session 1 (reporting query)
BEGIN;
SELECT sum(balance) FROM accounts WHERE region = 'US';
-- Returns $1,000,000

-- Meanwhile, Session 2 commits: moves $50,000 from US to EU account

SELECT sum(balance) FROM accounts WHERE region = 'EU';
-- This SELECT sees Session 2's commit! Different snapshot than the first SELECT.
-- The report shows $50,000 appearing from nowhere.
COMMIT;
```

**修正：** 一貫したスナップショットを必要とするレポートクエリにはRepeatable Readを使用してください。

```sql
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT sum(balance) FROM accounts WHERE region = 'US';
-- ... even if other transactions commit here ...
SELECT sum(balance) FROM accounts WHERE region = 'EU';
-- Both SELECTs see the same snapshot
COMMIT;
```

### リソースを保持する長時間トランザクション

```text
Symptoms:
  - Lock wait timeouts on unrelated queries
  - Bloated table sizes (PostgreSQL: dead tuples not vacuumed)
  - Replication lag (slot can't advance past long tx)
  - "too many clients already" connection exhaustion

Root causes:
  - BEGIN with no matching COMMIT (idle in transaction)
  - Application exception skipping COMMIT/ROLLBACK
  - Batch jobs running in a single transaction

Monitoring (PostgreSQL 16):
```

```sql
-- Find long-running transactions
SELECT pid, now() - xact_start AS duration, state, query
FROM pg_stat_activity
WHERE state = 'idle in transaction'
  AND now() - xact_start > interval '5 minutes'
ORDER BY duration DESC;

-- Nuclear option: terminate the session
SELECT pg_terminate_backend(pid);
```

**予防策：**

```text
# postgresql.conf
idle_in_transaction_session_timeout = '30s'   # kill idle-in-transaction after 30s
statement_timeout = '60s'                      # kill any statement after 60s
lock_timeout = '5s'                            # fail fast on lock waits
```

### オートコミットの誤用

ほとんどのデータベースドライバはデフォルトでautocommit=onとなっており、各ステートメントを独自のトランザクションでラップします。単純なクエリでは正しいですが、開発者が複数ステートメントのロジックに明示的なトランザクションが必要であることに気づかない場合、問題を引き起こします。

```python
# DANGEROUS: each statement is a separate transaction
conn.autocommit = True
cur.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
# ← crash here means money vanished
cur.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2")
```

```python
# CORRECT: explicit transaction
conn.autocommit = False
try:
    cur.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
    cur.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2")
    conn.commit()
except Exception:
    conn.rollback()
    raise
```

**SQLAlchemy コンテキストマネージャパターン（推奨）：**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

engine = create_engine("postgresql+psycopg2://localhost/mydb")

with Session(engine) as session, session.begin():
    # All operations in this block are a single transaction
    session.execute(text("UPDATE accounts SET balance = balance - 100 WHERE id = 1"))
    session.execute(text("UPDATE accounts SET balance = balance + 100 WHERE id = 2"))
# Automatic COMMIT on clean exit, ROLLBACK on exception
```

---

## 判断フレームワーク

### 分離レベルの選択

| ユースケース | 推奨レベル | 理由 | パフォーマンスコスト |
|-------------|-----------|------|-------------------|
| シンプルなCRUD Webアプリ | Read Committed | 重複しない書き込みには十分 | 基準値（1x） |
| 金融送金 | Serializable | 書き込みスキュー、ファントムリードを防止 | 競合時に2〜5倍遅い |
| レポート/アナリティクス | Repeatable Read | ステートメント間の一貫したスナップショット | 約1x（MVCCスナップショットは低コスト） |
| 在庫管理（在庫数） | Read Committed + SELECT FOR UPDATE | 特定行の行レベルロック | 1x + ロック待ち時間 |
| カウンターの増分 | Read Committed + アトミックUPDATE | 単一ステートメント、競合の余地なし | 1x |
| 当直の台帳照合 | Serializable | すべてのアノマリを防止する必要がある | 2〜5x、シリアライゼーション失敗時にリトライ |

### 2PC vs Saga vs Outboxの使い分け

| パターン | 保証 | レイテンシ | 複雑さ | 使用するタイミング |
|---------|------|----------|--------|------------------|
| **2PC** | 参加者間の原子性 | 参加者あたり+5〜20ms | 中 | PREPARE TRANSACTIONをサポートするデータベース。参加者が少数（5未満） |
| **Saga** | 補償アクションによる結果整合性 | 低（非同期ステップ） | 高（補償の正しい実装は難しい） | マイクロサービス、長期ワークフロー、サードパーティAPI呼び出し |
| **Outbox** | at-least-once配信、ローカル原子性 | 低（ポーリング/CDC遅延） | 中 | 単一DB → メッセージブローカー、イベント駆動アーキテクチャ |

**判断のヒューリスティック：**
- すべての参加者が同じデータベースに収まるか？ → ローカルトランザクションを使用。2PCは不要。
- すべての参加者が自分で管理するデータベースか？ → レイテンシが許容範囲なら2PCは実行可能。
- ワークフローに外部サービス（決済、メール、API）が含まれるか？ → Saga。
- データベース書き込みとアトミックにイベントを発行する必要があるか？ → Outboxパターン。

---

## コード例

### PostgreSQL：分離性を示す2つのセッション

同じPostgreSQL 16データベースに接続した2つの`psql`セッションを開きます。

**セットアップ：**

```sql
CREATE TABLE accounts (id INT PRIMARY KEY, balance NUMERIC NOT NULL);
INSERT INTO accounts VALUES (1, 500), (2, 200);
```

**デモ：Read Committedはダーティリードを防ぎますが、非反復読み取りを許容します：**

```text
Session A (default Read Committed):      Session B:
─────────────────────────────────────     ──────────────────────────────
BEGIN;                                    BEGIN;
UPDATE accounts SET balance = 400
  WHERE id = 1;
                                          SELECT balance FROM accounts
                                            WHERE id = 1;
                                          -- Returns 500 (not 400!)
                                          -- Dirty read prevented ✓
COMMIT;
                                          SELECT balance FROM accounts
                                            WHERE id = 1;
                                          -- Returns 400
                                          -- Non-repeatable read! The
                                          -- value changed within the
                                          -- same transaction.
                                          COMMIT;
```

**デモ：Repeatable Readはスナップショット整合性を提供します：**

```text
Session A:                                Session B (Repeatable Read):
─────────────────────────────────────     ──────────────────────────────
                                          BEGIN TRANSACTION ISOLATION
                                            LEVEL REPEATABLE READ;
                                          SELECT balance FROM accounts
                                            WHERE id = 1;
                                          -- Returns 500
BEGIN;
UPDATE accounts SET balance = 400
  WHERE id = 1;
COMMIT;
                                          SELECT balance FROM accounts
                                            WHERE id = 1;
                                          -- Still returns 500!
                                          -- Snapshot is frozen at BEGIN
                                          COMMIT;
```

### Python SQLAlchemy：SELECT FOR UPDATEパターン

```python
# Python 3.11+ / SQLAlchemy 2.0 / PostgreSQL 16
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

engine = create_engine(
    "postgresql+psycopg2://user:pass@localhost:5432/mydb",
    pool_size=10,
    pool_pre_ping=True,  # detect stale connections
)

def transfer(from_id: int, to_id: int, amount: float) -> None:
    """Transfer funds between accounts with proper locking.

    Acquires row locks in consistent order (lower ID first) to prevent deadlocks.
    """
    # Lock ordering: always lock lower ID first to prevent deadlock
    first_id, second_id = sorted([from_id, to_id])

    with Session(engine) as session, session.begin():
        # Acquire row locks in deterministic order
        rows = session.execute(
            text("""
                SELECT id, balance FROM accounts
                WHERE id IN (:id1, :id2)
                ORDER BY id
                FOR UPDATE
            """),
            {"id1": first_id, "id2": second_id},
        ).fetchall()

        balances = {row.id: row.balance for row in rows}

        if balances[from_id] < amount:
            raise ValueError(f"Insufficient funds: {balances[from_id]} < {amount}")

        session.execute(
            text("UPDATE accounts SET balance = balance - :amt WHERE id = :id"),
            {"amt": amount, "id": from_id},
        )
        session.execute(
            text("UPDATE accounts SET balance = balance + :amt WHERE id = :id"),
            {"amt": amount, "id": to_id},
        )
    # COMMIT happens here; ROLLBACK on exception
```

この例の重要なポイント：
- **ロック順序**（`sorted([from_id, to_id])`）は、2つの同時送金が逆方向に実行された場合のデッドロックを防ぎます
- **`FOR UPDATE`**は行レベルの排他ロックを取得し、並行する変更をブロックします
- **`session.begin()`コンテキストマネージャ**は例外時にROLLBACKを保証します
- **`pool_pre_ping=True`**はPgBouncerやネットワークタイムアウトによって切断された接続を処理します

---

## 実践でのACID

| データベース | バージョン | デフォルト分離レベル | 耐久性メカニズム | WAL/Redoサイズのデフォルト | 注意点 |
|-------------|-----------|-------------------|-----------------|--------------------------|--------|
| PostgreSQL | 16 | Read Committed | WAL + fsync | max_wal_size=1GB | `synchronous_commit=on`がデフォルト。`idle_in_transaction_session_timeout`はデフォルトで無効 |
| MySQL InnoDB | 8.0 | Repeatable Read | Redo log + doublewrite buffer | innodb_redo_log_capacity=100MB | `innodb_flush_log_at_trx_commit=1`は安全なデフォルトだがプロビジョニング後に確認が必要 |
| MongoDB | 7.0 | Read Committed（レプリカセットではsnapshot） | Journal（WiredTiger WAL） | 100MB journal | デフォルトのwrite concern `w:1`はレプリケーション待機なし。耐久性には`w:majority`を使用 |
| SQLite | 3.44 | Serializable | WALモードまたはロールバックジャーナル | N/A | WALモードは共有メモリが必要。ネットワークファイルシステムでは動作しない |
| CockroachDB | 23.2 | Serializable（唯一のレベル） | Raft合意 + RocksDB WAL | N/A | より弱い分離レベルは利用不可。アプリケーションでのシリアライゼーションリトライが必要 |
| SQL Server | 2022 | Read Committed | トランザクションログ | 自動拡張 | `READ_COMMITTED_SNAPSHOT`はデフォルトで無効（MVCCではなくロックを使用） |

### 警告：デフォルト設定を確認してください

本番データベースは単一ノードでの安全性に最適化されたデフォルトで出荷されます。しかしマネージドサービス、コンテナ、プロビジョニングスクリプトはしばしばそれらを上書きします。デプロイごとに確認してください：

```sql
-- PostgreSQL: verify critical durability settings
SHOW synchronous_commit;         -- should be 'on' for critical data
SHOW fsync;                       -- should be 'on' (NEVER disable in production)
SHOW full_page_writes;            -- should be 'on' (prevents torn pages)
SHOW wal_level;                   -- 'replica' or 'logical' for replication

-- MySQL: verify InnoDB settings
SHOW VARIABLES LIKE 'innodb_flush_log_at_trx_commit';  -- should be 1
SHOW VARIABLES LIKE 'innodb_doublewrite';               -- should be ON
SHOW VARIABLES LIKE 'sync_binlog';                      -- should be 1 for durability
```

---

## 重要なポイント

1. **原子性**はundo log（InnoDB）またはWAL + MVCC（PostgreSQL）で実装されます。ロールバックコストはInnoDBではトランザクションサイズに比例し、PostgreSQLではほぼ無料です。
2. **整合性**は最も弱い文字です。宣言した制約のみを強制します。シャード間外部キーは事実上不可能です。分散システムはこの保証を暗黙的に弱めます。
3. **分離性**にはレベルがあり、それぞれ実際のパフォーマンスコストがあります。デフォルトのRead CommittedはOLTPではほぼ間違いありませんが、レポートクエリにはRepeatable Readが必要です。分離レベルは常にBEGIN内で設定し、SETでは設定しないでください。
4. **耐久性**はスタックです：WAL → fsync → ディスクファームウェア → レプリケーション。各レイヤーが嘘をつきえます。fsyncの動作を検証し、重要なデータには同期レプリケーションを使用し、テストなしにクラウドストレージを信用しないでください。
5. **2PC**は分散原子性を可能にしますが、コーディネータが単一障害点になります。データベース間トランザクションに使用し、外部サービスを含むものにはSagaを使用してください。
6. **グループコミット**は無料のスループットです。高スループットシステムでは`commit_delay`と`commit_siblings`を調整してください。
7. **最もよくある本番バグ**はロックなしの読み取り後書き込みです。アトミックUPDATE文またはSELECT FOR UPDATEを決定的なロック順序で使用してください。
