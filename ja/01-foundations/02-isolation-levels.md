# 分離レベル

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../01-foundations/02-isolation-levels.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

分離レベルは、並行トランザクションが何を参照できるかを定義します。分離レベルが高いほどアノマリーは減りますが、パフォーマンスは低下します。ほとんどのOLTPアプリケーションではRead Committed（PostgreSQLのデフォルト）またはRepeatable Read（MySQLのデフォルト）を使用します。Serializableはすべてのアノマリーを防ぐ唯一のレベルですが、競合下ではスループットが20〜40%低下します。アノマリーを理解し、データベースの実際の実装を把握し、アプリケーションレベルのパターン（SELECT FOR UPDATE、楽観的ロック）を使用してギャップを安価に埋めましょう。

---

## 分離レベルが存在する理由

SQL標準が4つの分離レベルを定義しているのは、完全な直列化可能性はコストが高いためです。そのコストは2つの根本的な緊張関係から生じます。

**リーダーとライターの競合。** 完全に直列化されたシステムでは、ライターがロックを保持している間リーダーをブロックするか、読み取りが競合した場合にライターをアボートする必要があります。典型的なOLTPワークロード（95%読み取り、5%書き込み）では、すべての書き込みに対してリーダーをブロックするとスループットが激減します。

**スループットと正確性のトレードオフ。** 10,000 TPSを処理する決済システムを考えてみましょう。厳密な二相ロック（2PL）によるSerializableでは、ホットな行（口座残高、在庫数）へのロック競合がコンボイ効果を引き起こし、トランザクションが互いの後ろに並び、P99レイテンシが5msから500msに跳ね上がります。弱い分離レベルでは、特定のアノマリーを許容する代わりに並行アクセスを可能にします。

**エンジニアリング上のトレードオフは明確です。** SQL標準は各レベルで許容されるアノマリーを正確に定義しているため、エンジニアはアプリケーションロジックが許容できる最もコストの低いレベルを選択できます。

ほとんどのアプリケーションが完全な直列化可能性を必要としない理由：
- 多くの読み取りは情報提供目的（ダッシュボード、一覧表示）であり、古いデータでも許容される
- ビジネスロジックには自然な冪等性や補償トランザクションがある場合が多い
- クリティカルセクション（在庫の減算、残高の転送）では、全体にコストをかけずにターゲットを絞ったロックで対応できる

> 参照: WAL、fsync、undo/redoログの仕組みについては[ACIDトランザクション](01-acid-transactions.md)を参照してください。

---

## アノマリー

### ダーティリード

他のトランザクションのコミットされていないデータを読み取ることです。

```
T1: BEGIN
T1: UPDATE accounts SET balance = 0 WHERE id = 1
                                            T2: SELECT balance FROM accounts WHERE id = 1
                                            T2: Returns 0 (uncommitted!)
T1: ROLLBACK
```

T2はコミットされた状態として存在しないデータを参照しました。T2がその残高に基づいて判断した場合（例：ローンの拒否）、その判断は幻の状態に基づいたものです。

**防止されるレベル: Read Committed以上**

---

### 反復不能読み取り（リードスキュー）

同じ行を2回読み取ると異なる値が返されます。

```
T1: BEGIN
T1: SELECT balance FROM accounts WHERE id = 1  -- Returns 100
                                            T2: UPDATE accounts SET balance = 50 WHERE id = 1
                                            T2: COMMIT
T1: SELECT balance FROM accounts WHERE id = 1  -- Returns 50!
T1: COMMIT
```

T1のトランザクション中にデータの見え方が変わりました。これはバックアップ操作（不整合なスナップショット）、レポート生成（合計が合わない）、整合性チェック（外部キー参照のずれ）を破壊します。

**防止されるレベル: Repeatable Read以上**

---

### ファントムリード

クエリを2回実行すると異なる行が返されます。

```
T1: BEGIN
T1: SELECT COUNT(*) FROM accounts WHERE balance > 100  -- Returns 3
                                            T2: INSERT INTO accounts VALUES (4, 200)
                                            T2: COMMIT
T1: SELECT COUNT(*) FROM accounts WHERE balance > 100  -- Returns 4!
T1: COMMIT
```

トランザクション中に新しい行が「出現」しました。これは反復不能読み取りとは異なり、値ではなく*行の集合*が変化した点が特徴です。ファントムは範囲ベースの不変条件（例：「入金合計は出金合計に等しくなければならない」）を破壊します。

**防止されるレベル: Serializable**（MySQL RRはギャップロックにより一部のファントムを防ぎますが、すべてではありません）

---

### 書き込みスキュー

2つのトランザクションが重複するデータを読み取り、判断を行い、重複しないデータに書き込みます。

```
Constraint: At least one doctor must be on call

T1: SELECT COUNT(*) FROM doctors WHERE on_call = true  -- Returns 2
T1: I can go off-call, there's another doctor
                                            T2: SELECT COUNT(*) FROM doctors WHERE on_call = true  -- Returns 2
                                            T2: I can go off-call, there's another doctor
T1: UPDATE doctors SET on_call = false WHERE id = 1
T1: COMMIT
                                            T2: UPDATE doctors SET on_call = false WHERE id = 2
                                            T2: COMMIT
```

結果：オンコールの医師がゼロ人。制約違反です。

書き込みスキューは最も厄介なアノマリーです。各トランザクションのロジックは個別には正しいからです。競合はトランザクション間の読み取り・書き込み依存関係を追跡しなければ見えません。

**防止されるレベル: Serializableのみ**

---

### 更新の消失

2つのトランザクションが同じ値を読み取り、新しい値を計算し、書き戻します。一方の更新が暗黙的に上書きされます。

```
-- Account balance starts at 100

T1: BEGIN
T1: SELECT balance FROM accounts WHERE id = 1  -- Returns 100
                                            T2: BEGIN
                                            T2: SELECT balance FROM accounts WHERE id = 1  -- Returns 100
T1: UPDATE accounts SET balance = 100 + 50 WHERE id = 1  -- Deposit 50
T1: COMMIT
                                            T2: UPDATE accounts SET balance = 100 - 30 WHERE id = 1  -- Withdraw 30
                                            T2: COMMIT

-- Final balance: 70 (should be 120)
-- T1's deposit of 50 was lost
```

これは典型的な読み取り・変更・書き込み競合です。PostgreSQL RRはこれを検出してT2をアボートします。MySQL RRでは検出されず、T1の更新が暗黙的に失われます。Read Committedでは、両方のデータベースで更新が失われます。

**防止されるレベル:**
- PostgreSQL の Repeatable Read（先行更新者優先）
- MySQL の Serializable
- 任意の分離レベルでの `SELECT FOR UPDATE`
- アトミック操作: `UPDATE accounts SET balance = balance + 50`（読み取り・変更・書き込みを回避）

---

## MVCCの内部構造

Multi-Version Concurrency Control（MVCC）は、PostgreSQL、MySQL/InnoDB、Oracleが読み取りロックなしで分離を実装する方法です。書き込みごとに行の新しいバージョンが作成され、リーダーはスナップショットに適したバージョンを参照します。

### PostgreSQL: ヒープタプルヘッダ（v16）

PostgreSQLのすべての行は、ヒープ内にバージョンメタデータを直接保持しています。

```sql
-- Observe MVCC metadata directly
SELECT ctid, xmin, xmax, * FROM accounts;

--  ctid  | xmin | xmax | id | balance
-- -------+------+------+----+---------
--  (0,1) |  100 |    0 |  1 |    500
--  (0,2) |  100 |  105 |  2 |    300
--  (0,3) |  110 |    0 |  3 |    750
```

**フィールドの意味:**

| フィールド | 用途 |
|-----------|------|
| `xmin` | このタプルバージョンを挿入したトランザクションID |
| `xmax` | このタプルを削除/更新したトランザクションID（0 = 生存中） |
| `ctid` | ヒープファイル内の物理位置 `(ページ, オフセット)` |
| `t_infomask` | ビットマスクフラグ: `HEAP_XMIN_COMMITTED`、`HEAP_XMAX_INVALID` 等 |
| `t_infomask2` | 属性数、HOT更新フラグ |

行がUPDATEされると、PostgreSQLはインプレース変更を行いません。代わりに以下のようになります。

```
Before UPDATE:
  (0,1): xmin=100, xmax=0, balance=500        -- live tuple

After UPDATE (by xid 120):
  (0,1): xmin=100, xmax=120, balance=500      -- dead tuple (old version)
  (0,4): xmin=120, xmax=0, balance=600        -- new live tuple
```

古いタプルの `xmax` は更新トランザクションのIDに設定されます。古いタプルの `ctid` は新しいタプルの位置を指すように更新され、バージョンチェーンを形成します。

### スナップショットの構築

トランザクション開始時（RR/Serializable）に、PostgreSQLはスナップショットを取得します。

```
Snapshot = {
  xmin: 100,            -- oldest active transaction ID
  xmax: 125,            -- first unassigned transaction ID
  xip:  [105, 110, 118] -- transaction IDs that were in-progress at snapshot time
}
```

**タプルの可視性ルール:**

1. `tuple.xmin` が `xip` に含まれる（スナップショット時点で処理中）→ 不可視
2. `tuple.xmin >= snapshot.xmax`（スナップショット後に開始）→ 不可視
3. `tuple.xmin` がコミット済みかつ `tuple.xmin < snapshot.xmax` かつ `tuple.xmin` が `xip` に含まれない → 可視（`xmax` で隠されない場合）
4. `tuple.xmax` がコミット済みかつ同じルールで可視 → タプルは無効、不可視

Read Committedでは*各ステートメントごと*に新しいスナップショットが取得されるため、トランザクション中に新しくコミットされたデータが見えます。

### デッドタプルの蓄積とVACUUM

更新によって新しいタプルバージョンが作成されるため、古いバージョンが「デッドタプル」として蓄積されます。

```sql
-- Monitor dead tuple ratio (PostgreSQL 16)
SELECT relname,
       n_live_tup,
       n_dead_tup,
       round(n_dead_tup::numeric / greatest(n_live_tup, 1) * 100, 2) AS dead_pct,
       last_vacuum,
       last_autovacuum
FROM pg_stat_user_tables
WHERE n_dead_tup > 0
ORDER BY n_dead_tup DESC;
```

**VACUUMが存在する理由:** デッドタプルはディスク容量を浪費し、シーケンシャルスキャンを遅延させます（ヒープはデッドタプルをスキップする必要があります）。VACUUMはデッドタプルの領域を再利用可能としてマークします。VACUUM FULLはテーブル全体を書き換えてディスク容量を回収します（ACCESS EXCLUSIVEロックが必要）。

**Autovacuumのトリガー条件**（PostgreSQL 16のデフォルト設定）:

```
autovacuum_vacuum_threshold = 50          -- minimum dead tuples before vacuum
autovacuum_vacuum_scale_factor = 0.2      -- fraction of table size
-- Trigger: dead_tuples > threshold + scale_factor * n_live_tup
-- For a 1M row table: vacuum triggers after 200,050 dead tuples
```

高頻度更新テーブルにはスケールファクターを下げます。

```sql
ALTER TABLE hot_table SET (autovacuum_vacuum_scale_factor = 0.01);
-- Now triggers at 10,050 dead tuples for a 1M row table
```

### InnoDBとの比較（MySQL 8.0）

InnoDBはバージョニングに異なるアプローチを取ります。

| 側面 | PostgreSQL | InnoDB |
|------|-----------|--------|
| 古いバージョンの格納場所 | ヒープ（インライン） | Undoログセグメント（別テーブルスペース） |
| クリーンアップ機構 | VACUUM（外部プロセス） | パージスレッド（バックグラウンド） |
| バージョンチェーンの方向 | 前方向（古いctid → 新しいctid） | 後方向（現在の行 → Undoログ） |
| ブロート時の読み取りオーバーヘッド | ヒープスキャンが遅延 | Undoログの走査で長時間読み取りが遅延 |

InnoDBはクラスタインデックスに現在のバージョンを格納します。トランザクションが古いバージョンを必要とする場合、Undoログレコードを逆順に適用して再構築します。つまり、現在のデータの読み取りは高速ですが、古いスナップショットの読み取り（長時間実行トランザクションによるもの）はUndoチェーンを走査する必要があります。

```sql
-- Monitor InnoDB undo log usage (MySQL 8.0)
SELECT count AS undo_log_entries
FROM information_schema.innodb_metrics
WHERE name = 'trx_rseg_history_len';

-- High values (>1M) indicate long-running transactions preventing purge
```

---

## ロックの内部構造

### ロック階層

データベースはロック粒度の階層を使用します。粒度が細かいほど並行性は高まりますが、オーバーヘッドも増加します。

**PostgreSQLのロック階層:**

```
ACCESS SHARE          -- SELECT (blocks nothing except ACCESS EXCLUSIVE)
ROW SHARE             -- SELECT FOR UPDATE/SHARE
ROW EXCLUSIVE         -- INSERT/UPDATE/DELETE
SHARE UPDATE EXCLUSIVE -- VACUUM, CREATE INDEX CONCURRENTLY
SHARE                 -- CREATE INDEX (non-concurrent)
SHARE ROW EXCLUSIVE   -- triggers, some ALTER TABLE
EXCLUSIVE             -- blocks ROW SHARE and above
ACCESS EXCLUSIVE      -- ALTER TABLE, DROP, VACUUM FULL
```

実際の行レベルロックは、これらのテーブルレベルモードとは別のものです。PostgreSQLの行ロックはタプルヘッダ（`xmax` + `t_infomask` ビット）に格納され、共有ロックテーブルには格納されないため、数百万の行ロックでもメモリオーバーヘッドはほぼゼロです。

```sql
-- View current locks (PostgreSQL 16)
SELECT l.locktype, l.relation::regclass, l.mode, l.granted, l.pid,
       a.query
FROM pg_locks l
JOIN pg_stat_activity a ON l.pid = a.pid
WHERE l.relation IS NOT NULL
ORDER BY l.relation;
```

### InnoDBギャップロック

MySQL InnoDBのRepeatable Readでは、範囲クエリがファントムを防ぐために**ギャップロック**を取得します。

```sql
-- Session 1 (MySQL 8.0, default RR)
BEGIN;
SELECT * FROM orders WHERE id > 10 AND id < 20 FOR UPDATE;

-- This locks:
--   Record locks on existing rows where 10 < id < 20
--   Gap locks on the gaps between existing keys
--   Next-key lock on the supremum pseudo-record
```

```sql
-- Session 2 (blocked!)
INSERT INTO orders (id, amount) VALUES (15, 100);
-- Waits... blocked by gap lock even though id=15 doesn't exist yet
```

ギャップロックはファントム挿入を防ぎますが、予期しないブロッキングを引き起こします。セカンダリインデックスでの `SELECT ... WHERE status = 'pending'` は、無関係な挿入をブロックするギャップをロックする可能性があります。

```sql
-- Diagnose InnoDB locks (MySQL 8.0)
SELECT * FROM performance_schema.data_locks
WHERE lock_type = 'RECORD'
ORDER BY lock_data;

-- Shows lock_mode: X,GAP  or  X,REC_NOT_GAP  or  X (next-key lock)
```

### PostgreSQL述語ロック（SSI）

Serializable分離レベルでは、PostgreSQLは**SIReadLock**エントリを使用して各トランザクションが読み取った内容を追跡します。

```sql
-- Session 1 (Serializable)
BEGIN ISOLATION LEVEL SERIALIZABLE;
SELECT * FROM doctors WHERE on_call = true;

-- PostgreSQL creates SIReadLock entries for the rows and the index range
```

これらはブロッキングの意味でのロックではなく、マーカーです。SSIはブロックせず、依存関係を追跡し、サイクルを検出した場合にアボートします。

```sql
-- View predicate locks
SELECT locktype, relation::regclass, page, tuple
FROM pg_locks
WHERE mode = 'SIReadLock';

-- locktype | relation | page | tuple
-- ---------+----------+------+-------
-- tuple    | doctors  |    0 |     1
-- tuple    | doctors  |    0 |     3
-- page     | doctors  |    0 |
```

### デッドロック検出

2つのトランザクションがそれぞれ相手が必要とするロックを保持している場合、デッドロックが発生します。

```
T1: UPDATE accounts SET balance = balance - 100 WHERE id = 1;  -- holds lock on id=1
T2: UPDATE accounts SET balance = balance - 50  WHERE id = 2;  -- holds lock on id=2
T1: UPDATE accounts SET balance = balance + 100 WHERE id = 2;  -- waits for T2
T2: UPDATE accounts SET balance = balance + 50  WHERE id = 1;  -- waits for T1 -> DEADLOCK
```

**PostgreSQL**は**待機グラフ（waits-for graph）**を構築してサイクルを検出します。チェックは `deadlock_timeout`（デフォルト1秒）後に実行されます。1つのトランザクションが `ERROR: deadlock detected` でアボートされます。

```sql
-- PostgreSQL: tune deadlock detection
SET deadlock_timeout = '500ms';  -- check sooner (more CPU) or later (longer waits)

-- Log deadlocks for analysis
ALTER SYSTEM SET log_lock_waits = on;       -- log waits exceeding deadlock_timeout
ALTER SYSTEM SET deadlock_timeout = '1s';
SELECT pg_reload_conf();
```

**MySQL InnoDB**はロック待機のたびにデッドロックをチェックします（タイムアウト遅延なし）。待機グラフを使用し、Undoログレコードが最も少ないトランザクション（ロールバックのコストが最も低い）をロールバックします。

```sql
-- MySQL: view last deadlock
SHOW ENGINE INNODB STATUS\G
-- Look for "LATEST DETECTED DEADLOCK" section
```

---

## Serializable Snapshot Isolation（SSI）の詳細

SSI（PostgreSQL 9.1以降およびCockroachDBで使用）は楽観的同時実行制御メカニズムです。トランザクションをスナップショットに対して実行し（Repeatable Readと同様）、コミット時に直列化の競合を検出して違反者をアボートします。

### rw反依存関係

SSIの中核概念は**rw反依存関係**（rw-conflictとも呼ばれます）です。

> トランザクションT1がデータ項目のあるバージョンを読み取り、T2が後でそのデータ項目の新しいバージョンを書き込んだ場合、T1はT2に対してrw反依存関係を持ちます。

```
T1: reads row X (version v1)
T2: writes row X (version v2)
-- T1 has an rw-anti-dependency on T2: T1 read old data that T2 changed
```

単一のrw反依存関係は問題ありません。危険なのは、2つ以上の連続するエッジを含む**rw反依存関係のサイクル**です。

```
"Dangerous structure":
T1 --rw--> T2 --rw--> T3  (where T3 committed before T1)

If this pattern forms, one transaction must be aborted to maintain serializability.
```

### SSIのアボートと2PLのブロッキングの比較

| 状況 | 2PLの動作 | SSIの動作 |
|------|----------|----------|
| 読み書き競合 | ライターがコミットするまでリーダーがブロック | 両者とも進行。コミット時にサイクルが検出されればアボート |
| 書き込み同士の競合 | 2番目のライターがブロック | 2番目のライターがブロック（2PLと同じ） |
| 実際の競合なし | それでもブロック（悲観的） | オーバーヘッドなし（楽観的） |
| デッドロックの可能性 | あり（検出/タイムアウトが必要） | 読み取りではデッドロックなし（書き込み同士のみブロック可能） |

SSIの利点：読み取り中心のワークロードではオーバーヘッドがほぼゼロです。読み取りがブロックされないためです。コストは、リトライが必要な時折のアボートです。

### 直列化失敗のリトライパターン（PostgreSQL 16）

SSIが競合を検出すると、`ERROR 40001 (serialization_failure)` が発生します。アプリケーションは**必ずリトライ**する必要があります。

```python
import psycopg2
from psycopg2 import extensions
import time

def execute_with_retry(conn_params, operation, max_retries=5):
    """Execute a serializable transaction with exponential backoff retry.

    Args:
        conn_params: dict of psycopg2 connection parameters
        operation: callable(cursor) -> result, the transaction body
        max_retries: maximum number of retry attempts

    Returns:
        Result of the operation callable

    Raises:
        psycopg2.Error: if max retries exceeded or non-retryable error
    """
    for attempt in range(max_retries):
        conn = psycopg2.connect(**conn_params)
        conn.set_isolation_level(extensions.ISOLATION_LEVEL_SERIALIZABLE)
        try:
            with conn.cursor() as cur:
                result = operation(cur)
                conn.commit()
                return result
        except psycopg2.errors.SerializationFailure:
            conn.rollback()
            if attempt == max_retries - 1:
                raise
            # Exponential backoff with jitter
            delay = (2 ** attempt) * 0.01 * (0.5 + random.random())
            time.sleep(delay)
        except Exception:
            conn.rollback()
            raise  # Non-retryable errors propagate immediately
        finally:
            conn.close()


# Usage
def transfer_funds(cur):
    cur.execute("SELECT balance FROM accounts WHERE id = 1 FOR UPDATE")
    balance = cur.fetchone()[0]
    if balance < 100:
        raise ValueError("Insufficient funds")
    cur.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
    cur.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2")

execute_with_retry({"dbname": "myapp"}, transfer_funds)
```

重要なポイント：
- 失敗したステートメントだけでなく、*トランザクション全体*をリトライする必要があります
- `40001` は直列化失敗のSQLSTATEです。エラーメッセージではなくこのコードを確認してください
- 指数バックオフとジッタにより、競合下でのリトライストームを防ぎます

---

## 比較表

| レベル | ダーティリード | 反復不能読み取り | ファントムリード | 書き込みスキュー | 更新の消失 | パフォーマンス |
|--------|--------------|----------------|----------------|----------------|-----------|-------------|
| Read Uncommitted | あり | あり | あり | あり | あり | 最良 |
| Read Committed | なし | あり | あり | あり | あり | 良好 |
| Repeatable Read | なし | なし | PG: なし、MySQL: 部分的 | PG: あり、MySQL: あり | PG: なし、MySQL: あり | 中程度 |
| Serializable | なし | なし | なし | なし | なし | 最低 |

注記：
- PostgreSQL RRはファントムと更新の消失を先行更新者優先で防ぎますが、書き込みスキューは防ぎません
- MySQL RRは*一貫性読み取り*（MVCCスナップショット）ではファントムを防ぎますが、*ロック読み取り*やDMLでは防ぎません
- MySQL RRは読み取り・変更・書き込みパターンでの更新の消失を検出しません

---

## パフォーマンスへの影響

### ベンチマーク比率

並行ワークロード下での相対スループット（Read Committed = 1.0に正規化）。測定パターンは典型的なOLTP：80%ポイント読み取り、15%更新、5%範囲クエリです。公開されたベンチマークと一般的な業界観測に基づいています。

| ワークロード | RC | RR | Serializable (2PL) | Serializable (SSI) |
|-------------|----|----|--------------------|--------------------|
| 低競合（1%ホット行） | 1.0 | 0.95 | 0.85 | 0.92 |
| 中競合（10%ホット行） | 1.0 | 0.90 | 0.60 | 0.82 |
| 高競合（50%ホット行） | 1.0 | 0.85 | 0.30 | 0.65 |
| 読み取り専用 | 1.0 | 0.99 | 0.95 | 0.98 |

主要な観察：
- SSI（PostgreSQL）は競合下で2PL（MySQL Serializable）を大幅に上回ります。読み取りがブロックされないためです
- RRは読み取り専用ワークロードではほぼコストゼロです
- 高競合下では、Serializable（2PL）はロックコンボイ効果によりRCスループットの30%まで低下する可能性があります

### ロック待機がP99レイテンシに与える影響

```
Isolation Level     P50 Latency    P99 Latency    P99/P50 Ratio
-----------------------------------------------------------------
Read Committed      2ms            12ms           6x
Repeatable Read     2ms            18ms           9x
Serializable (2PL)  3ms            150ms          50x
Serializable (SSI)  2ms            25ms           12.5x  (includes retry cost)
```

SSIでは、P99に時折のアボート + リトライのコストが含まれます。2PLでは、P99はロック待機のキューイングを反映しています。

### 異なるブロートレベルでのMVCC読み取りオーバーヘッド

デッドタプルはシーケンシャルスキャンを遅延させます。PostgreSQLはすべてのタプル（生存・デッド問わず）の可視性をチェックする必要があるためです。

```
Dead Tuple Ratio    Seq Scan Overhead    Index Scan Overhead
------------------------------------------------------------
0% (freshly vacuumed)   1.0x            1.0x
20%                     1.15x           1.02x
50%                     1.45x           1.05x
80%                     2.5x            1.10x
```

インデックススキャンはインデックス経由で直接生存タプルにアクセスするため、影響は比較的小さいです。シーケンシャルスキャンはデッドタプルを含むヒープ全体を走査する必要があります。そのため、シーケンシャルスキャンでアクセスされるテーブルではVACUUMの頻度がより重要になります。

---

## アプリケーションパターン

### SELECT FOR UPDATE SKIP LOCKED: キュー・ワーカーパターン

外部メッセージブローカーなしでデータベースバックのジョブキューを実装するために使用します。

```sql
-- Worker picks up the next unprocessed job (PostgreSQL 16 / MySQL 8.0)
BEGIN;

SELECT id, payload
FROM job_queue
WHERE status = 'pending'
ORDER BY created_at
LIMIT 1
FOR UPDATE SKIP LOCKED;  -- Skip rows locked by other workers

-- Returns a row that no other worker is processing
-- If all pending rows are locked, returns empty set (worker sleeps and retries)

-- Process the job...

UPDATE job_queue SET status = 'completed', completed_at = now() WHERE id = $1;
COMMIT;
```

**これが機能する理由:** `SKIP LOCKED` は他のトランザクションによってロックされている行をスキップし、各ワーカーにブロッキングなしで一意のジョブを割り当てます。これは Read Committed を含む*任意の*分離レベルで動作します。

**ポーリングに対する利点:** 行の競合なし、デッドロックなし、重複処理なし。複数のワーカーが安全にキューを並行処理できます。

### バージョン列による楽観的ロック

リトライロジックを含む完全なPython実装：

```python
import psycopg2
from psycopg2.extras import RealDictCursor

class OptimisticLockError(Exception):
    pass

def update_product_price(conn, product_id: int, new_price: float, max_retries: int = 3):
    """Update product price with optimistic concurrency control.

    Works correctly under Read Committed -- no elevated isolation needed.
    """
    for attempt in range(max_retries):
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Step 1: Read current state including version
            cur.execute(
                "SELECT id, price, version FROM products WHERE id = %s",
                (product_id,)
            )
            product = cur.fetchone()
            if not product:
                raise ValueError(f"Product {product_id} not found")

            current_version = product["version"]

            # Step 2: Business logic (could be complex computation)
            validated_price = validate_pricing_rules(new_price, product)

            # Step 3: Conditional update -- only succeeds if version unchanged
            cur.execute(
                """UPDATE products
                   SET price = %s, version = version + 1, updated_at = now()
                   WHERE id = %s AND version = %s""",
                (validated_price, product_id, current_version)
            )

            if cur.rowcount == 1:
                conn.commit()
                return  # Success
            else:
                conn.rollback()  # Version changed, retry
                continue

    raise OptimisticLockError(
        f"Failed to update product {product_id} after {max_retries} retries"
    )
```

このパターンは `WHERE version = %s` 句がcompare-and-swapとして機能するため、Read Committedで動作します。分離レベルの引き上げは不要です。

### アドバイザリロック

PostgreSQLのアドバイザリロックはアプリケーションレベルの協調ロックです。データベースはテーブルに対してこれを強制しませんが、高速でデッドロック検出可能な排他制御プリミティブを提供します。

```sql
-- Transaction-scoped advisory lock (released at COMMIT/ROLLBACK)
SELECT pg_try_advisory_xact_lock(12345);
-- Returns true if acquired, false if already held by another session

-- Use case: prevent duplicate processing of an event
BEGIN;
SELECT pg_try_advisory_xact_lock(hashtext('order:' || order_id::text));
-- If false, another worker is already processing this order -- skip
-- If true, process the order
COMMIT;  -- Lock automatically released
```

```sql
-- Session-scoped advisory lock (persists until explicit release or disconnect)
SELECT pg_advisory_lock(hash_key);      -- blocks until acquired
SELECT pg_advisory_unlock(hash_key);    -- explicit release required

-- Useful for: singleton cron jobs, schema migrations, cache warming
```

アドバイザリロックは通常のロックと同じ待機グラフでチェックされるため、アドバイザリロックと行ロック間のデッドロックも検出されます。

### アンチパターン: Read Committedでの読み取り・変更・書き込み

```sql
-- WRONG: This loses updates under Read Committed
-- Two concurrent sessions can read the same balance, compute independently, overwrite

-- Session 1                                    -- Session 2
BEGIN;                                          BEGIN;
SELECT balance FROM accounts WHERE id = 1;      SELECT balance FROM accounts WHERE id = 1;
-- Returns 100                                  -- Returns 100
UPDATE accounts SET balance = 150 WHERE id = 1;
COMMIT;                                         UPDATE accounts SET balance = 70 WHERE id = 1;
                                                COMMIT;
-- Final: 70 (lost Session 1's +50 deposit)
```

```sql
-- FIX Option 1: Atomic expression (no read-modify-write)
UPDATE accounts SET balance = balance + 50 WHERE id = 1;

-- FIX Option 2: SELECT FOR UPDATE (pessimistic)
BEGIN;
SELECT balance FROM accounts WHERE id = 1 FOR UPDATE;  -- acquires row lock
-- Other sessions block on their SELECT FOR UPDATE until this commits
UPDATE accounts SET balance = balance + 50 WHERE id = 1;
COMMIT;

-- FIX Option 3: Optimistic locking (see version column pattern above)
```

可能な限りアトミックな式を使用してください。よりシンプルで高速であり、任意の分離レベルで正しく動作します。

---

## データベース固有の注意事項

### PostgreSQL（v16）

**分離の実装:**
- Read Committed: 各*ステートメント*ごとに新しいスナップショットを取得
- Repeatable Read: トランザクション全体で1つのスナップショット、書き込み競合には先行更新者優先
- Serializable: SSIベース、rw反依存関係のサイクルを検出

**デッドタプルとXIDの健全性の監視:**

```sql
-- Dead tuple monitoring
SELECT schemaname, relname, n_live_tup, n_dead_tup,
       round(n_dead_tup::numeric / greatest(n_live_tup, 1) * 100, 1) AS bloat_pct,
       last_autovacuum
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000
ORDER BY n_dead_tup DESC;
```

**XIDラップアラウンドの防止:**

PostgreSQLのトランザクションIDは32ビット（40億値）です。XIDカウンターがラップアラウンドに近づくと、PostgreSQLは積極的なバキュームを強制し、最終的にはデータ破損を防ぐためにシャットダウンします。

```sql
-- Check XID age (how close to wraparound)
SELECT datname,
       age(datfrozenxid) AS xid_age,
       round(age(datfrozenxid)::numeric / 2147483647 * 100, 2) AS pct_to_wraparound
FROM pg_database
ORDER BY xid_age DESC;

-- Danger zone: xid_age > 1 billion (autovacuum_freeze_max_age default: 200 million)
-- Emergency zone: xid_age > 2 billion (database shuts down to prevent wraparound)
```

長時間実行トランザクションはVACUUMによる凍結XID水平線の前進を阻止します。単一のidle-in-transactionセッションがクラスタ全体を遅延させる可能性があります。

### MySQL InnoDB（v8.0）

**デフォルト分離レベル:** Repeatable Read（PostgreSQLのRead Committedデフォルトとは異なります）

**主要な診断クエリ:**

```sql
-- Active transactions
SELECT trx_id, trx_state, trx_started,
       timestampdiff(SECOND, trx_started, now()) AS age_seconds,
       trx_rows_locked, trx_rows_modified, trx_isolation_level
FROM information_schema.innodb_trx
ORDER BY trx_started;

-- Lock waits
SELECT r.trx_id AS waiting_trx,
       r.trx_query AS waiting_query,
       b.trx_id AS blocking_trx,
       b.trx_query AS blocking_query
FROM information_schema.innodb_lock_waits w
JOIN information_schema.innodb_trx b ON b.trx_id = w.blocking_trx_id
JOIN information_schema.innodb_trx r ON r.trx_id = w.requesting_trx_id;

-- Detailed lock information (MySQL 8.0+)
SELECT engine_lock_id, lock_type, lock_mode, lock_status,
       lock_data, object_name
FROM performance_schema.data_locks
WHERE lock_status = 'WAITING';
```

**MySQL RRの特異な挙動:**
- 一貫性読み取り（通常のSELECT）はMVCCスナップショットを使用 -- ファントムなし
- ロック読み取り（SELECT FOR UPDATE、SELECT FOR SHARE）はスナップショットではなく*最新のコミット済みバージョン*を読み取る
- この不整合により、ロック読み取りか非ロック読み取りかによって同じWHERE句が異なる行にマッチする可能性がある

### Oracle（21c）

- Read CommittedとSerializableのみサポート
- Read UncommittedとRepeatable Readはなし
- **「Serializable」は実際にはSnapshot Isolation** -- 書き込みスキューを防ぎません
- Oracleは書き込み同士の競合（ORA-08177: can't serialize access）を検出しますが、読み書き競合は検出しません
- 真の直列化可能性にはアプリケーションレベルの `SELECT FOR UPDATE` が必要

```sql
-- Oracle: set serializable (actually SI)
ALTER SESSION SET ISOLATION_LEVEL = SERIALIZABLE;

-- Write skew IS possible here. Oracle will not detect the doctor on-call anomaly.
```

### CockroachDB（v23.x）

- **デフォルトでSerializable** -- 利用可能な唯一の分離レベル（v23.2でRead Committedがオプトインとして追加されるまで）
- ノード間の分散SSI実装を使用
- クロスノードトランザクションは調整オーバーヘッドが発生（関連Range1つあたり約2〜5ms）
- ゲートウェイノードでの自動トランザクションリトライ（クライアントに透過的）
- ホット行への競合はPostgreSQL SSIと同様に「transaction retry」エラーを引き起こす

```sql
-- CockroachDB: check contention
SELECT * FROM crdb_internal.cluster_contended_tables;
SELECT * FROM crdb_internal.cluster_contended_indexes;
```

---

## よくある間違い

### 1. コネクションプールでの分離レベルの漏洩

コネクションに分離レベルを設定し、リセットせずにプールに返すと、次の利用者がその設定を引き継ぎます。

```python
# BUG: isolation level leaks through the pool
conn = pool.getconn()
conn.set_isolation_level(ISOLATION_LEVEL_SERIALIZABLE)
# ... do work ...
pool.putconn(conn)
# Next pool.getconn() may return this connection -- still at SERIALIZABLE!
```

**修正:** プールに返す前に必ず分離レベルをリセットするか、トランザクションレベルの分離を使用します。

```sql
-- Per-transaction isolation (doesn't affect connection default)
BEGIN ISOLATION LEVEL SERIALIZABLE;
-- ... work ...
COMMIT;
-- Connection returns to its default level
```

### 2. MySQL RRは書き込みスキューを防がない

よくある誤解：「Repeatable Readはファントム以外のすべてのアノマリーを防ぐ」。これは書き込みスキューを無視した場合のみ正しく、元のSQL標準ではそうしていました。

```sql
-- MySQL 8.0, Repeatable Read: write skew succeeds (BUG if you need invariant)
-- The doctor on-call example runs without error on MySQL RR.
-- Both transactions commit successfully. Zero doctors on call.

-- Fix: Use SELECT ... FOR UPDATE to escalate to locking reads
BEGIN;
SELECT * FROM doctors WHERE on_call = true FOR UPDATE;  -- Now this blocks
```

### 3. 長時間トランザクションがVACUUMをブロック

本番環境で最もよく見られるPostgreSQLのパフォーマンス障害です。

```sql
-- This idle transaction prevents VACUUM from cleaning ANY dead tuples
-- created after its snapshot
BEGIN;  -- snapshot taken
SELECT * FROM tiny_table;  -- harmless-looking query
-- Developer forgets to COMMIT, goes to lunch
-- Meanwhile, heavy UPDATE traffic on big_table creates millions of dead tuples
-- Autovacuum runs but cannot remove tuples newer than this snapshot
-- Table bloats, sequential scans slow down, disk fills up
```

**防止策:**

```sql
-- PostgreSQL: kill idle-in-transaction sessions automatically
ALTER SYSTEM SET idle_in_transaction_session_timeout = '5min';
SELECT pg_reload_conf();

-- Monitor for long-running transactions
SELECT pid, now() - xact_start AS duration, query, state
FROM pg_stat_activity
WHERE state = 'idle in transaction'
  AND now() - xact_start > interval '1 minute'
ORDER BY duration DESC;
```

### 4. すべてのデータベースが同じレベルを同一に実装していると仮定する

SQL標準は分離レベルを防止するアノマリーによって定義していますが、実装は大きく異なります。

| 動作 | PostgreSQL RR | MySQL RR | Oracle「Serializable」 |
|------|--------------|----------|----------------------|
| ファントム防止 | あり（MVCC） | 部分的（ギャップロック） | あり（MVCC） |
| 更新の消失防止 | あり（先行更新者優先） | なし | あり（ORA-08177） |
| 書き込みスキュー防止 | なし | なし | なし |
| 真の直列化可能性 | なし（Serializableが必要） | なし（Serializableが必要） | なし（アプリロジックが必要） |

---

## 重要なポイント

1. **分離レベルが高いほどバグは減るが、スループットは低下します。** コストは現実的です：Serializable（2PL）は競合下でRead Committedスループットの30%まで低下する可能性があります
2. **通常はRead Committedで十分です。** クリティカルセクションにはアトミックSQL式とSELECT FOR UPDATEを組み合わせてください
3. **Repeatable Readはデータベースによって異なります。** PostgreSQL RRは更新の消失を防ぎますが、MySQL RRは防ぎません
4. **Serializable（SSI）は実用的です。** PostgreSQLのSSIはMySQLのロックベースSerializableよりはるかに安価です。正確性が重要なワークロードでは検討してください
5. **アプリケーションレベルのパターンは分離のギャップを安価に埋めます。** SELECT FOR UPDATE、バージョン列による楽観的ロック、アドバイザリロックにより、グローバルなSerializableのオーバーヘッドを回避できます
6. **MVCCは無料ではありません。** デッドタプルが蓄積し、VACUUMが追いつく必要があり、長時間トランザクションがクリーンアップをブロックします。`n_dead_tup` と `idle_in_transaction_session_timeout` を監視してください
7. **データベースの「分離レベル」はSQL標準と一致しません。** Oracleの「Serializable」はSIです。MySQLの「Repeatable Read」にはロック読み取りの不整合があります。常にデータベースの実際の動作をテストしてください
8. **直列化失敗（SQLSTATE 40001）では必ずリトライしてください。** SSIのアボートは例外的ではなく想定内です。データアクセスレイヤーにリトライロジックを組み込んでください

> 参照: 線形化可能性のスペクトラムと単一ノード分離を超えた分散一貫性保証については[一貫性モデル](04-consistency-models.md)を参照してください。
