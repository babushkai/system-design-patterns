# Write-Ahead Logging（WAL）

> **翻訳の状態:** 現行の[英語版](../../03-storage-engines/04-write-ahead-logging.md)からの翻訳です。

## TL;DR

Write-ahead loggingは、コミットごとにランダムI/Oを支払うことなくデータベースが永続性を約束できるようにするトリックです: 変更を追記専用ログに記述し、それをfsyncし、クライアントに応答する — 実際のデータページは今はメモリ上で更新し、ディスクへは都合のよいときに書けばよい。コミットパス上では1回のシーケンシャル追記が散在するページ書き込みを置き換え、クラッシュ後はログをリプレイして応答済みのすべてを再構築します。このひとつのアイデアは多くの機構を伴います: ログシーケンス番号（LSN）がリプレイを冪等にし、ARIESがリカバリをanalysis/redo/undoに構造化し、チェックポイントがリプレイすべきログ量を有界にし、グループコミットがfsyncを償却してスループットがディスクフラッシュ1回あたり1コミットで頭打ちにならないようにします。教科書が省く鋭い刃も伴います: ページ書き込みはアトミックではなく（torn pageにはフルページイメージかダブルライトバッファが必要）、fsync自体が嘘をつくことがあり（揮発性キャッシュ、fsyncgateのエラーセマンティクスのバグ）、同じログがレプリケーションフィードを兼ねます — 放置されたレプリケーションスロットがディスクを埋めてデータベースを落とすのはこのためです。本章ではコミットパスからプロトコルを組み立て、リカバリ、パフォーマンスエンジニアリング、障害モードを扱います。

---

## 問題: ランダムI/Oなしの永続性

コミットされたトランザクションはクラッシュを生き延びなければなりません。それを保証する素朴な方法は、コミット応答の前に変更されたすべてのデータページをディスクに書くことです — しかし1つのトランザクションが数GBのファイル中に散らばるページをダーティにでき、その一つひとつがランダム書き込みです。コミットはHDDならミリ秒単位のコストになり、NVMeでさえ小さな散在書き込みで痛めつけます。もう一方の素朴な選択肢 — ページはメモリ上で更新し後でフラッシュ — は高速ですが、マシンが悪いタイミングで死ぬたびに応答済みデータを失います。

WALは、変更の*記述*と*適用*を分離することでこのジレンマを解きます:

```
Commit path with WAL:
  1. append a log record describing the change   (memory)
  2. fsync the log                               (ONE sequential write)
  3. modify the data page in the buffer pool     (memory)
  4. acknowledge the client

The data page reaches disk later — at a checkpoint, or when the
buffer pool evicts it. If the machine dies first, recovery replays
the log record and rebuilds the page.

The invariant that names the technique: a data page may not be
written to disk before the log records describing its changes are.
Log first, data second — always.
```

これを利益にするのはI/Oの形です。ログは単一の追記専用ストリームです: すべてのコミットが同じ場所にシーケンシャルに書きます。これはあらゆるストレージデバイスが最も得意とするパターンです。ランダム性のすべて — どのページが変わったか、どこにあるか — は、バッチ化・ソート・スケジュールが可能なバックグラウンド書き込みへとクリティカルパスの外に繰り延べられます。同じ洞察が[LSM木](./02-lsm-trees.md)を駆動しています。実際、LSMはおおよそ「ログがデータベースそのものだったら？」であり、WALで守られた[B木](./01-b-trees.md)はupdate-in-place構造を保ちつつログを保険としてのみ使います。

代償は、すべての変更が2回書かれることです — 一度はログレコードとして、一度は最終的にページとして。これがWALがB木の章で論じた書き込み増幅の主要な寄与者である理由であり、データベースがログレコードを小さく保とうと戦う理由です。

---

## LSN: リプレイを冪等にする

リカバリは、ディスク上の状態が不明なページ — クラッシュ前にフラッシュされたものもあれば、されなかったものもある — に対してログをリプレイします。変更を2回適用すればスキップと同じくらい確実にデータが壊れます。リプレイを安全にする機構が**ログシーケンス番号（LSN）**です: すべてのレコードは単調増加のLSNを持ち、すべてのページヘッダはそのページに最後に適用されたレコードのLSNを記録します。

```
Log:                                 Page 5 on disk:
  LSN 100: update page 5              page_lsn = 101
  LSN 101: update page 5
  LSN 102: update page 8              Redo walks the log:
  LSN 103: commit T1                    LSN 100 vs page_lsn 101 → skip
                                        LSN 101 vs page_lsn 101 → skip
                                        LSN 102 vs page 8's LSN → apply if newer
```

`record_lsn > page_lsn` という比較がリプレイを冪等な操作に変えます: リカバリを1回走らせても、2回走らせても、リカバリの途中でクラッシュしてまた走らせても — ページは同じ状態に収束します。LSNは本章の他のすべての座標系でもあります: チェックポイントは「リカバリはLSN Xから始めてよい」を記録し、レプリカは「LSN Yまで適用済み」を報告し、ログ切り詰めは「誰かがまだ必要としている最小のLSNは？」を問います。

---

## レコードに何を入れるか: 物理 vs 論理

ログレコードが語りうる内容にはスペクトラムがあり、選択はログ量とリプレイの複雑さを交換します:

**物理ロギング**はバイトを記録します: 「ページ5、オフセット42、旧値 `A`、新値 `B`」。リプレイは自明で高速 — バイトをコピーするだけ — ですが、多くのバイトに触れる変更（数百キーを再配置するB木のページ分割）は巨大なレコードを生みます。

**論理ロギング**は操作を記録します: 「`UPDATE accounts SET balance = balance - 100 WHERE id = 5` を実行せよ」。レコードは極小ですが、リプレイは操作を決定的に — 同じ結果、同じ順序で — 再実行しなければならず、並行性や非決定性（`now()`、乱数）、バージョン間のコード変更の前では脆弱です。

**生理的（physiological）ロギング** — ページ*に対して*は物理的、ページ*の中*では論理的: 「ページ5で、キー `abc` をスロット3に挿入」。実際のエンジンが使うのはこれです。レコードはページを名指しし（リプレイにクエリプランニングは不要で、ページ単位で並列化できる）、変更はそのページの内部構造に対する操作としてコンパクトに記述されます。その唯一の仮定 — レコードのリプレイ時にページの事前状態が無傷であること — こそtorn pageが破る仮定であり、本章の後半にtorn page防御が存在する理由です。

---

## ARIES: 3パスのリカバリ

ほぼすべての本格的なデータベースは**ARIES**（Mohan et al., 1992）の変種でリカバリします。その中心的な設計判断は、理由を見るまでは奇妙に聞こえます: クラッシュ後、まず*歴史のすべてを繰り返す* — 最終的にロールバックされるトランザクションの変更も含めて — そしてその後で敗者をundoするのです。

```
1. ANALYSIS  — scan from the last checkpoint:
     which transactions were in flight at the crash?
     which pages might have unflushed changes (dirty page table)?

2. REDO      — scan forward, reapply every change whose LSN is newer
     than its page (committed or not). The database is now in the
     exact state of the crash instant.

3. UNDO      — for each transaction alive at the crash, walk its
     records backward and reverse them, logging a Compensation Log
     Record (CLR) for every reversal.
```

まず歴史を繰り返すことで、redoには判断が不要になります — 愚直で高速な、ページ順のリプレイです — そしてundoはクラッシュ状態の一貫したスナップショットの上で、通常のロールバックと同じロジックで動作します。**CLR**は再帰的な問題を解きます: undoの*最中*にクラッシュしたら？ 各CLRは「この巻き戻しは実行済み」と語り、次にundoすべきレコードを指すので、2度目のリカバリは完了済みの巻き戻しを再度巻き戻す代わりにスキップします。undoもredoと同様に冪等になり、リカバリは何度クラッシュしても収束します。

```
  100: T1 updates P1        Analysis: T2 was alive at crash
  101: T2 updates P3        Redo:     replay 100-103 (yes, including T2)
  102: T1 commit            Undo:     reverse 103, then 101,
  103: T2 updates P4                  writing a CLR for each
  --- CRASH ---             Result:   T1's work stands, T2 vanished
```

---

## チェックポイント: リプレイを有界にする

チェックポイントがなければ、リカバリは時間の始まりからログをリプレイします。**チェックポイント**は定期的に「ここが安全な開始点」を記録します: アクティブなトランザクションの集合、ダーティページテーブル、そして — フラッシュによって暗黙に — あるLSNより古いページはディスク上にあるという保証です。

モダンなエンジンは**ファジーチェックポイント**を使います: 世界を止めてすべてのダーティページをフラッシュする（レイテンシの大惨事）代わりに、チェックポイントはダーティページ*テーブル*を記録し、バックグラウンドライターに徐々にフラッシュさせます。redoのLSN比較が不正確さを許容します。ただしチェックポイントのコストは消えるのではなく、広がります。PostgreSQLの `checkpoint_completion_target` はI/Oスパイクを避けるためにフラッシュをチェックポイント間隔全体に明示的にペース配分し、間隔そのものが根本的なリカバリ時間のノブです:

```
Checkpoint interval trade-off:
  frequent  → little log to replay (fast recovery)
              but constant page flushing (foreground I/O impact)
              and in PostgreSQL: more full-page images (see below)
  rare      → cheap steady-state
              but crash recovery replays a huge log (minutes-hours)

Recovery time ≈ log volume since last checkpoint / replay speed.
If you have a recovery-time objective, this is the knob that meets it.
```

切り詰めはチェックポイントから従います: `min(最古のアクティブトランザクションの最初のLSN, 最古のダーティページのLSN, レプリカがまだ必要とする最古のLSN)` より古いログはリサイクルできます。この `min` のすべての項は、何かが停滞したときにログが無制限に成長する経路です — 忘れられた開きっぱなしのトランザクション、詰まったバックグラウンドライター、そして（最も多いのは、後述の）死んだレプリケーションコンシューマ。

---

## グループコミット: スループットのエンジニアリング

コミットパス上のfsyncがレイテンシとスループットの物語のすべてです。コミットごとに1 fsyncでは、スループットはデバイスのフラッシュレートで頭打ちになります:

```
fsync cost:      HDD ~10 ms   SATA SSD ~1 ms   NVMe ~20-100 μs
naive ceiling:   100/s        ~1,000/s          ~10,000-50,000/s

Group commit: while one fsync is in flight, arriving commits queue.
When it returns, ALL queued records flush in the next single fsync.
  20 concurrent committers on a 1 ms device ≈ 20,000 commits/s —
  the batch size self-tunes to concurrency, and each transaction's
  added latency is at most one flush interval.
```

本格的なエンジンはすべてこれを行います（PostgreSQLのWALライター、InnoDBのredoグループコミット、RocksDBのwrite groupリーダー）。残る選択は「durable（永続）」が何を意味すべきかで、エンジンはそれをスペクトラムとして公開しています:

```
synchronous_commit = on      fsync before ack        lose nothing
synchronous_commit = off     ack, fsync within ~ms   lose last few ms
                             (PostgreSQL: data stays CONSISTENT —
                              you lose recent commits, not integrity)
innodb_flush_log_at_trx_commit = 1 / 2 / 0   — same ladder for MySQL
```

緩和された永続性は、派生データや再生可能なデータ（上流にキューがあるイベントインジェスト、キャッシュ的なテーブル）には正当なエンジニアリングであり、お金には弁護の余地がありません。この判断はワークロード単位であるべきで — PostgreSQLは*トランザクション*単位で設定できます — 誰かがベンチマークのために選んだサーバー全体のデフォルトであるべきではありません。

コミットパスにはあと2つのレバーがあります。**専用ログデバイス**は、ログのシーケンシャルストリームがランダムなデータI/Oと交互に混ざるのを防ぎます（混ざれば両方がランダムI/Oになります）。そしてログバッファサイズは、バースト下でグループコミットがどれだけバッチできるかを制限します — 16〜64MBが典型で、大きくして主に効くのはバルクロードです。

---

## Torn Page: WALの下にあるアトミシティのギャップ

WALプロトコルは、データページの書き込みは「起こるか起こらないか」のどちらかだと仮定します。ハードウェアはその仮定を壊します:

```
Database page: 8 KB (PostgreSQL) / 16 KB (InnoDB)
Device atomic write unit: 4 KB sector (often 512B logically)

Crash mid-page-write → a TORN page: first 4 KB new, last 4 KB old.
Page checksum detects it — but redo may not be able to FIX it:
physiological log records ("insert key at slot 3") assume the page's
prior state is intact. A torn page has no valid prior state.
```

本番での防御は2つ:

```
PostgreSQL — full-page writes (full_page_writes = on):
  the FIRST modification to a page after each checkpoint logs the
  ENTIRE page image into the WAL. Redo restores the image, then
  applies records on top — no dependence on the on-disk page state.
  Cost: WAL volume spikes right after every checkpoint (the FPI
  burst); this is the hidden coupling between checkpoint_timeout
  and WAL bandwidth.

InnoDB — doublewrite buffer:
  pages are first written sequentially to a doublewrite area, synced,
  then written to their final locations. Torn final write → recover
  the page from the doublewrite copy. Cost: ~2× page write volume
  (mitigated by batching; can be disabled ONLY on filesystems/devices
  with guaranteed atomic writes — e.g., ZFS, or NVMe devices exposing
  atomic write units ≥ page size).
```

ページサイズのアトミック書き込みを本当に保証するストレージ上で動いているなら、どちらの防御も純粋なオーバーヘッドです — 「doublewrite/FPWを切れるか？」が現実のチューニング会話である理由であり、その答えが楽観ではなくストレージスタックのドキュメントから来なければならない理由です。

---

## fsyncは本当にsyncしているのか？

この設計全体の永続性は、ひとつのシステムコールが真実を語ることの上に載っています。それはしばしば、3つの層で真実を語りません:

```
1. Volatile drive caches: consumer SSDs/HDDs ack writes into DRAM
   cache. A power cut loses "durable" data unless the OS issues cache
   flush / FUA commands — which filesystem barriers do, but
   misconfigured stacks (some virtualized disks, RAID controllers
   without BBU set to write-back) silently don't.

2. fsync error semantics (fsyncgate, 2018): on Linux, if a background
   writeback fails, fsync() returns EIO ONCE — and marks the pages
   CLEAN. A process that retries fsync gets SUCCESS while the data
   never reached disk. PostgreSQL had assumed retry-until-success was
   safe for ~20 years; the fix (PG 11+) is to PANIC on fsync failure
   and recover from WAL, never retry.

3. fdatasync vs fsync vs directory sync: creating a new WAL segment
   requires fsyncing the DIRECTORY too, or the file itself may vanish
   after crash. Metadata (size changes) needs fsync; fdatasync
   suffices for in-place data and is cheaper.

Verification, not vibes: pull the power plug under load
(diskchecker.pl-style tests) or use dm-flakey/dm-log-writes to
simulate. Storage stacks that "lose" acked fsyncs are common enough
that serious databases treat this as a qualification test.
```

---

## レプリケーション基盤としてのWAL

クラッシュリカバリを提供するのと同じバイトストリームが、自然なレプリケーションフィードでもあります — レプリカとは、形式的には、決して終わらないリカバリプロセスです。この二重用途がWALの運用問題の大半を生みます:

```
Physical replication (PostgreSQL streaming, InnoDB redo shipping):
  replica applies page-level records → byte-identical standby.
  Fast, simple; replica must run the same major version and
  architecture.

Logical replication / CDC: decode WAL back into row-level changes
  (pgoutput, Debezium). Enables cross-version, selective, and
  cross-system replication ([CDCパイプライン](../13-data-pipelines/04-change-data-capture.md)) —
  at the cost of decoding CPU and ordering complexity.

The operational trap — replication slots pin WAL:
  a slot guarantees the WAL a consumer hasn't read yet is retained.
  A dead/abandoned consumer (a decommissioned replica, a stalled
  Debezium connector) pins WAL forever → disk fills → database down.
  This is among the most common self-inflicted PostgreSQL outages.
  Defense: monitor slot lag bytes; set max_slot_wal_keep_size (PG 13+)
  to cap retention and sacrifice the slot instead of the database.

Synchronous replication couples commit latency to the network:
  synchronous_commit = on → wait for local flush
                       remote_write / remote_apply → wait for standby
  Group commit still applies — batches of transactions share both the
  local fsync AND the replication round trip ([合意アルゴリズム](../02-distributed-databases/08-consensus-algorithms.md)
  makes the same amortization under quorum acks).
```

ログのアーカイブは、同じストリームを空間ではなく時間の方向に延長します: 閉じたWALセグメントを[オブジェクトストレージ](./08-object-storage.md)に送れば、任意のベースバックアップ+アーカイブ済みログで**任意の時点**にリプレイできます — リカバリ機構が `pg_restore --target-time` を兼ね、「14:32に間違ったテーブルをDROPした」に対する標準的な防御になります。

---

## 各エンジンにおけるWAL

**PostgreSQL**は `pg_wal/` 配下に16MBのWALセグメントを書きます。重要なノブは `synchronous_commit`（永続性の梯子。トランザクション単位で設定可能）、`max_wal_size`/`checkpoint_timeout`（リカバリ時間 vs FPI量）、`full_page_writes`（torn page）、`max_slot_wal_keep_size`（スロット保護）です。`pg_stat_wal` と `pg_stat_replication` が量とラグを公開します。

**MySQL/InnoDB**は循環redoログ（`innodb_redo_log_capacity`）を使います。`innodb_flush_log_at_trx_commit` が永続性の梯子、ダブルライトバッファがtorn pageをカバーし、— 構造的な違いとして — InnoDBはMVCC構造の第一級市民としてundoログ*も*保持します。一方PostgreSQLは古い行バージョンをヒープに保持し、undoログを必要としません。

**RocksDB**のWALはmemtableを守ります（[LSM木](./02-lsm-trees.md)）。WALセグメントはそのmemtableがSSTableにフラッシュされた時点で削除されます。ページ指向のredoはありません — リカバリは単に「ログからmemtableを再ロードする」だけです。これは、ARIESのどれだけ多くが*update-in-place*ストレージのためだけに存在するかを示しています。`manual_wal_flush` と書き込み単位の `disableWAL` が、同じ永続性の梯子を組み込み形式で公開しています。

---

## 障害モード

**WAL成長によるディスクフル。** ログは、それを留めている*すべて* — チェックポイント、アーカイブ、そしてすべてのレプリケーションスロット — が前進するまで成長します。古典的なインシデントは、放置されたスロットが数週間分のWALを留め、ボリュームが埋まり、データベースが書き込みを受け付けなくなるというものです。保持WALバイト数とスロットラグを監視し、`max_slot_wal_keep_size` で上限を。

**リカバリに数時間かかる。** 過大な `max_wal_size` にはクラッシュまで誰も気づきません。リカバリ時間はチェックポイント以降のログ量に比例します。RTOがあるなら、それをチェックポイント間隔に翻訳し、実際にレプリカをクラッシュさせて*テスト*してください — リプレイ速度（古いPostgreSQLではシングルスレッド）は思われているより遅いことが多いのです。

**FPIバースト。** PostgreSQLの各チェックポイント直後、触れられたすべてのページがフルイメージをログします: WAL量がしばらく5〜10倍に跳ね、レプリケーションリンクとアーカイブを飽和させえます。チェックポイントの分散（`checkpoint_completion_target`）、`wal_compression = on`、チェックポイントをバッチジョブと重ねないこと、が緩和策です。

**無音の永続性ダウングレード。** 新ハードウェアへの移行、VMプラットフォームの変更、善意の「パフォーマンス修正」（`synchronous_commit = off`、バリアの無効化、NFSマウント）が、応答済みコミットの意味を静かに変えます。永続性の設定はスキーマの一部として扱ってください — レビューされ、バージョン管理され、ストレージスタックが変わったら再検証（電源引き抜きテスト）されるものとして。

**誤ったスタック上でのtorn page防御の無効化。** `full_page_writes = off` や `skip-innodb_doublewrite` が安全なのは、ストレージが本当にページをアトミックに書く場合だけです。障害はクラッシュがページの途中に着地するまで不可視で — そのとき、リカバリ自体が壊れたページの上で失敗します。

---

## 意思決定フレームワーク

| 状況 | やること |
|---|---|
| デフォルトのOLTP永続性 | `synchronous_commit = on` / `innodb_flush_log_at_trx_commit = 1`。スループットはグループコミットが稼ぐ |
| 再生可能/派生データ、インジェスト律速 | テーブル/トランザクション単位で緩和（`synchronous_commit = off`、`= 2`）。反射でサーバー全体に適用しない |
| リカバリ時間目標がある | そこからチェックポイント間隔を導出。レプリカをクラッシュテストして実リプレイ速度を計測 |
| チェックポイント後にコミットレイテンシが跳ねる | FPIバースト — チェックポイントを分散、`wal_compression` を有効化、WAL帯域を確認 |
| 論理レプリケーション/CDCを使用 | スロットラグバイトでアラート。`max_slot_wal_keep_size` を設定。死ぬべきはスロットでありデータベースではない |
| ゼロデータロスのフェイルオーバーが必須 | 同期レプリケーション（`remote_apply`）と、全コミットへのRTT混入の受容 |
| ポイントインタイムリカバリが必須 | オブジェクトストレージへの継続的WALアーカイブ+定期ベースバックアップ。リストアのリハーサルを |
| 新しいストレージスタック（クラウドディスク、ZFS、新NVMe） | 信頼したり防御を切ったりする前に、fsyncの誠実さとページ書き込みのアトミシティを再検証 |

---

## 重要なポイント

1. **ログが先、データが後** — 1回のシーケンシャルfsyncが、任意に散らばるページ変更の永続性を買う。ランダムI/Oはコミットパスの外へ移る。
2. **LSNがリカバリを冪等にする** — `record_lsn > page_lsn` という比較が、リプレイ（そしてCLR経由のundo）をクラッシュ・再実行に対して安全にする。
3. **ARIES = 歴史を繰り返し、それからundo** — redoはクラッシュ瞬間までの愚直で高速なページ順リプレイ。undoはその後、補償レコードを書きながら敗者を巻き戻す。
4. **チェックポイント間隔がリカバリ時間のダイヤル** — PostgreSQLではフルページイメージ量のダイヤルでもあり、2つのコストは互いに交換される。
5. **グループコミットはfsyncの天井を並行性のゲームに変える** — バッチは負荷に自己調整する。その下の永続性の梯子はワークロード単位の判断であり、サーバーのデフォルトではない。
6. **ページ書き込みはアトミックではない** — full-page writesとダブルライトバッファはtorn pageのために存在する。無効化は文書化されたアトミック書き込み保証がある場合のみ。
7. **fsyncは嘘をつきうる** — 揮発性キャッシュ、fsyncgateのエラーセマンティクス、忘れられたディレクトリsync。ストレージはデータシートではなく電源を抜いて資格審査する。
8. **WALはレプリケーションとPITRの基盤でもある** — レプリカは終わらないリカバリ、アーカイブは時間方向のリカバリ。そしてログを留めるもの（スロット！）はディスクを埋めうる。

---

## 参考文献

- [Mohan et al., *ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging* (ACM TODS, 1992)](https://doi.org/10.1145/128765.128770): ARIES方式の原論文。ログ先行書き込みによるリカバリの基準点
- Gray and Reuter, *Transaction Processing: Concepts and Techniques* (Morgan Kaufmann, 1992)
- [Hellerstein, Stonebraker, and Hamilton, *Architecture of a Database System* (Foundations and Trends in Databases, 2007)](https://dsf.berkeley.edu/papers/fntdb07-architecture.pdf)
- [PostgreSQL documentation, *WAL Configuration*](https://www.postgresql.org/docs/current/wal-configuration.html)
- [PostgreSQL documentation, *Reliability*](https://www.postgresql.org/docs/current/wal-reliability.html)
- [PostgreSQL documentation, *Logical Decoding*](https://www.postgresql.org/docs/current/logicaldecoding.html)
- [MySQL documentation, *Redo Log*](https://dev.mysql.com/doc/refman/8.4/en/innodb-redo-log.html)
- [MySQL documentation, *Doublewrite Buffer*](https://dev.mysql.com/doc/refman/8.4/en/innodb-doublewrite-buffer.html)
- [Rebello et al., *Can Applications Recover from fsync Failures?* (USENIX ATC 2020)](https://www.usenix.org/conference/atc20/presentation/rebello)
- [LWN, *PostgreSQL's fsync() surprise* (2018)](https://lwn.net/Articles/752063/)
