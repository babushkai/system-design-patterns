# Amazon Aurora: Design Considerations for High Throughput Cloud-Native Relational Databases

> **注:** この記事は英語の原文を日本語に翻訳したものです。コードブロック、Mermaidダイアグラム、論文タイトル、システム名、技術用語は原文のまま保持しています。

## TL;DR

Auroraは以下を提供するクラウドネイティブリレーショナルデータベースです：
- コンピュートとストレージを分離する**ログがデータベース**アーキテクチャ
- 3つのアベイラビリティゾーンにまたがる**6方向レプリケーション**
- コンセンサスオーバーヘッドなしの耐久性のための**クォーラムベースI/O**
- 並列オンデマンドRedoによる**ほぼ瞬時のクラッシュリカバリ**

## 論文概要

- **タイトル**: Amazon Aurora: Design Considerations for High Throughput Cloud-Native Relational Databases
- **著者**: Alexandre Verbitski et al. (Amazon Web Services)
- **発表**: SIGMOD 2017
- **背景**: AWSは高可用性のクラウドネイティブリレーショナルデータベースを必要としていました

## 課題

### クラウドにおける従来のデータベースの限界

```mermaid
graph TD
    subgraph Primary["プライマリインスタンス"]
        BP[Buffer Pool] --> LB[Log Buffer]
        BP --> DP[("Data Pages<br/>EBS")]
        LB --> RL[("Redo Log<br/>EBS")]
    end

    Primary -->|同期<br/>レプリケーション| Standby["スタンバイインスタンス<br/>（フルコピー）"]
```

> **問題点:** 1) ネットワークI/Oの増幅（ミラーEBSで4倍）。2) 同期レプリケーションによるレイテンシ増加。3) クラッシュリカバリでRedoログ全体を再生。4) フェイルオーバーに数分かかる。

### Auroraの洞察

```mermaid
graph LR
    subgraph Compute["データベースインスタンス"]
        BP2[Buffer Pool]
    end

    subgraph Storage["ストレージサービス"]
        SN[("ストレージノード<br/>ログを適用してページを生成")]
    end

    BP2 -->|redo logs| SN
    SN -->|pages| BP2
```

> **「ログがデータベースである。」** 従来: ページ書き込み + ログ書き込み（2倍の書き込み）。Aurora: ログのみ書き込み（ストレージが適用）。
> **利点:** ネットワークトラフィックがRedoログのみに削減。ストレージが耐久性とレプリケーションを処理。クラッシュリカバリはストレージの再構築のみ。

## アーキテクチャ

### 全体システム設計

```mermaid
graph TD
    subgraph ComputeLayer["コンピュート層"]
        W["Writerインスタンス<br/>クエリ、バッファ、トランザクション管理"]
        R1["Readerインスタンス<br/>クエリ、バッファ、キャッシュ"]
        R2["Readerインスタンス<br/>クエリ、バッファ、キャッシュ"]
    end

    subgraph StorageLayer["ストレージ層 — 3 AZにわたる6コピー"]
        subgraph AZA["AZ-A"]
            N1[("Node 1")] & N2[("Node 2")]
        end
        subgraph AZB["AZ-B"]
            N3[("Node 3")] & N4[("Node 4")]
        end
        subgraph AZC["AZ-C"]
            N5[("Node 5")] & N6[("Node 6")]
        end
    end

    W -->|redo logs| StorageLayer
    R1 -->|page reads| StorageLayer
    R2 -->|page reads| StorageLayer
```

### ストレージセグメンテーション

```
┌─────────────────────────────────────────────────────────────────┐
│                   ストレージセグメンテーション                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  データベースボリューム（最大128 TB）                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                           │   │
│  │  Segment 1 (10GB)    Segment 2 (10GB)    Segment N       │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────┐  │   │
│  │  │                │  │                │  │            │  │   │
│  │  │  6レプリカ     │  │  6レプリカ     │  │ 6レプリカ  │  │   │
│  │  │  3 AZにわたる  │  │  3 AZにわたる  │  │            │  │   │
│  │  │                │  │                │  │            │  │   │
│  │  └────────────────┘  └────────────────┘  └────────────┘  │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Protection Groups (PGs):                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  各セグメントがProtection Groupを形成                    │   │
│  │  - PGあたり6つのストレージノード                         │   │
│  │  - アベイラビリティゾーンあたり2ノード                   │   │
│  │  - 独立した障害ドメイン                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  利点:                                                          │
│  - 並列修復（10GBは約10秒で修復）                               │
│  - 影響範囲が10GBセグメントに限定                               │
│  - バックグラウンド修復がフォアグラウンド操作に影響しない       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## クォーラムベースI/O

### 書き込みクォーラムと読み取りクォーラム

```python
class AuroraQuorum:
    """Aurora's quorum-based replication."""

    def __init__(self):
        self.replicas = 6     # Total copies
        self.write_quorum = 4 # Vw
        self.read_quorum = 3  # Vr

        # Vw + Vr > V (4 + 3 > 6) ensures overlap
        # Vw > V/2 (4 > 3) ensures no conflicting writes

    def write(self, log_record) -> bool:
        """
        Write log record to storage.

        Must reach write quorum (4/6) to acknowledge.
        """
        acks = 0
        futures = []

        for storage_node in self.get_nodes_for_segment(log_record.segment):
            future = storage_node.write_async(log_record)
            futures.append(future)

        # Wait for write quorum
        for future in futures:
            try:
                future.wait(timeout=50)  # 50ms
                acks += 1
                if acks >= self.write_quorum:
                    return True
            except Timeout:
                continue

        return acks >= self.write_quorum

    def read(self, page_id) -> Page:
        """
        Read page from storage.

        Only need read quorum (3/6) - but actually
        Aurora optimizes to read from single node!
        """
        # In practice, Aurora tracks which nodes are current
        # and reads from a single up-to-date node
        node = self.get_current_node(page_id)
        return node.read_page(page_id)

class QuorumProperties:
    """
    Aurora's quorum guarantees.

    With V=6, Vw=4, Vr=3:
    - Survives loss of entire AZ (2 nodes) + 1 additional node
    - Writes complete with any 4 nodes available
    - Reads complete with any 3 nodes available
    """

    def can_write_with_az_failure(self) -> bool:
        """
        AZ failure = 2 nodes down
        Remaining = 4 nodes
        Write quorum = 4
        Can still write!
        """
        return True

    def can_read_with_az_plus_one_failure(self) -> bool:
        """
        AZ + 1 failure = 3 nodes down
        Remaining = 3 nodes
        Read quorum = 3
        Can still read!
        """
        return True

    def write_read_overlap(self) -> bool:
        """
        Any read quorum overlaps with any write quorum.

        Vw + Vr = 4 + 3 = 7 > 6
        Guarantees at least 1 node has latest write.
        """
        return True
```

### 耐久性モデル

```
┌─────────────────────────────────────────────────────────────────┐
│                  Aurora耐久性モデル                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  障害シナリオ:                                                  │
│                                                                  │
│  シナリオ1: 単一ノード障害                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  AZ-A        AZ-B        AZ-C                           │    │
│  │  [1] [X]     [3] [4]     [5] [6]                        │    │
│  │                                                          │    │
│  │  5ノード稼働、読み取り(3)も書き込み(4)も可能 ✓          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  シナリオ2: AZ障害                                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  AZ-A        AZ-B        AZ-C                           │    │
│  │  [X] [X]     [3] [4]     [5] [6]                        │    │
│  │                                                          │    │
│  │  4ノード稼働、読み取り(3)も書き込み(4)も可能 ✓          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  シナリオ3: AZ + 1障害                                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  AZ-A        AZ-B        AZ-C                           │    │
│  │  [X] [X]     [X] [4]     [5] [6]                        │    │
│  │                                                          │    │
│  │  3ノード稼働、読み取り(3)は可能、書き込み不可 ✗         │    │
│  │  （修復まで読み取り専用モード）                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  修復:                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  - 10GBセグメントは約10秒で修復                         │    │
│  │  - バックグラウンドのゴシップベース修復                  │    │
│  │  - 二重障害のMTTF ≈ 極めて低い                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## ログ処理

### ログシッピングアーキテクチャ

```python
class AuroraLogShipping:
    """Aurora's log-based replication."""

    def __init__(self):
        self.current_lsn = 0  # Log Sequence Number
        self.commit_lsn = 0   # Last committed
        self.durable_lsn = 0  # Durable in storage

    def process_transaction(self, transaction):
        """
        Process transaction and ship logs.

        Only redo logs are shipped - not pages!
        """
        log_records = []

        for operation in transaction.operations:
            # Generate redo log record
            log_record = LogRecord(
                lsn=self._next_lsn(),
                transaction_id=transaction.id,
                page_id=operation.page_id,
                redo_data=operation.redo_data
            )
            log_records.append(log_record)

        # Ship to storage (in parallel across segments)
        futures = {}
        for record in log_records:
            segment = self._get_segment(record.page_id)
            if segment not in futures:
                futures[segment] = []
            futures[segment].append(
                self._ship_to_segment(segment, record)
            )

        # Wait for all segments to acknowledge
        for segment, segment_futures in futures.items():
            for future in segment_futures:
                future.wait()

        # Transaction is durable when all logs acknowledged
        self.durable_lsn = max(r.lsn for r in log_records)

        return True

    def _ship_to_segment(self, segment, log_record):
        """
        Ship log record to storage segment.

        Storage will:
        1. Persist log record
        2. Add to pending queue
        3. Eventually apply to generate page
        """
        return segment.write_log_async(log_record)

class StorageNode:
    """Aurora storage node operations."""

    def __init__(self):
        self.log_records = []
        self.pages = {}
        self.pending_queue = []

    def write_log(self, log_record) -> bool:
        """
        Receive and persist log record.

        This is the ONLY write from compute!
        """
        # Persist to local storage (SSD)
        self._persist_log(log_record)

        # Add to pending queue for page materialization
        self.pending_queue.append(log_record)

        # Acknowledge immediately - no blocking
        return True

    def read_page(self, page_id) -> Page:
        """
        Read page, applying pending logs if needed.

        Redo application happens on READ, not WRITE.
        """
        # Get base page
        page = self.pages.get(page_id)
        if page is None:
            page = Page.empty(page_id)

        # Apply any pending log records for this page
        pending_for_page = [
            r for r in self.pending_queue
            if r.page_id == page_id
        ]

        for record in sorted(pending_for_page, key=lambda r: r.lsn):
            page = self._apply_redo(page, record)

        return page

    def background_coalesce(self):
        """
        Background process to apply logs to pages.

        Reduces work on read path.
        """
        while True:
            # Group pending records by page
            by_page = defaultdict(list)
            for record in self.pending_queue:
                by_page[record.page_id].append(record)

            # Apply and persist pages
            for page_id, records in by_page.items():
                page = self.pages.get(page_id, Page.empty(page_id))
                for record in sorted(records, key=lambda r: r.lsn):
                    page = self._apply_redo(page, record)

                self.pages[page_id] = page

                # Remove applied records
                max_lsn = max(r.lsn for r in records)
                self.pending_queue = [
                    r for r in self.pending_queue
                    if r.page_id != page_id or r.lsn > max_lsn
                ]

            time.sleep(1)  # Run every second
```

### ネットワークI/O削減

```
┌─────────────────────────────────────────────────────────────────┐
│              ネットワークI/O比較                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  従来のMySQL（EBSミラーリング）:                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                                                          │    │
│  │  トランザクションあたりの書き込み:                       │    │
│  │  1. Redoログ（プライマリEBS）     → 1ネットワークI/O    │    │
│  │  2. Redoログ（ミラーEBS）         → 1ネットワークI/O    │    │
│  │  3. Binlog（プライマリEBS）       → 1ネットワークI/O    │    │
│  │  4. Binlog（ミラーEBS）           → 1ネットワークI/O    │    │
│  │  5. データページ（プライマリEBS） → 1ネットワークI/O    │    │
│  │  6. データページ（ミラーEBS）     → 1ネットワークI/O    │    │
│  │  7. Double-writeバッファ          → 1ネットワークI/O    │    │
│  │  8. FRMファイル                   → 1ネットワークI/O    │    │
│  │                                                          │    │
│  │  合計: 約8ネットワークラウンドトリップ（同期）           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Aurora:                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                                                          │    │
│  │  トランザクションあたりの書き込み:                       │    │
│  │  1. Redoログをストレージノードへ  → 1ネットワークI/O    │    │
│  │     （6ノードに並列送信）                                │    │
│  │                                                          │    │
│  │  合計: 1ネットワークラウンドトリップ（6ノードに並列）    │    │
│  │                                                          │    │
│  │  データページなし、Binlogなし、Double-writeバッファなし！│    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  結果: トランザクションあたりI/Oが35倍削減                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## リカバリとフェイルオーバー

### クラッシュリカバリ

```python
class AuroraRecovery:
    """
    Aurora crash recovery - near instant.

    Key insight: Recovery is just establishing
    consistency point, not replaying logs.
    """

    def recover_after_crash(self):
        """
        Crash recovery process.

        Traditional: Replay entire redo log (minutes to hours)
        Aurora: Find highest durable LSN (seconds)
        """
        # Step 1: Find Volume Durable LSN (VDL)
        vdl = self._find_volume_durable_lsn()

        # Step 2: Truncate any logs beyond VDL
        self._truncate_incomplete_logs(vdl)

        # Step 3: Ready to serve!
        # Pages are reconstructed on-demand during reads
        return vdl

    def _find_volume_durable_lsn(self) -> int:
        """
        Find highest LSN durable across all segments.

        Query each segment for its highest complete LSN.
        VDL = min of all segment's highest complete LSN.
        """
        segment_lsns = []

        for segment in self.segments:
            # Each segment knows its highest complete LSN
            # (based on quorum writes)
            highest = segment.get_highest_complete_lsn()
            segment_lsns.append(highest)

        # VDL is the min - guarantees all prior logs are durable
        return min(segment_lsns)

    def _reconstruct_page_on_demand(self, page_id) -> Page:
        """
        Reconstruct page when first accessed.

        Storage has all the logs needed.
        """
        segment = self._get_segment(page_id)
        return segment.read_page(page_id)  # Applies pending logs

class FastFailover:
    """Aurora fast failover mechanism."""

    def __init__(self):
        self.writer = None
        self.readers = []
        self.failover_time = 0

    def perform_failover(self, new_writer):
        """
        Failover to new writer.

        Steps:
        1. Detect failure (typically via health checks)
        2. Promote reader to writer
        3. Update DNS

        Total time: ~30 seconds
        """
        start = time.time()

        # Step 1: Detect failure
        if not self._is_writer_healthy():
            # Step 2: Promote reader
            new_writer = self._select_best_reader()

            # Reader has most of buffer pool already!
            # Just needs to:
            # - Establish write capability
            # - Catch up any missing logs
            new_writer.become_writer()

            # Step 3: Update DNS
            self._update_dns(new_writer)

        self.failover_time = time.time() - start
        # Typically < 30 seconds

    def _select_best_reader(self):
        """
        Select reader with most up-to-date buffer pool.

        Reader replicas continuously apply redo logs,
        so they're nearly current with writer.
        """
        best = None
        highest_lsn = 0

        for reader in self.readers:
            if reader.current_lsn > highest_lsn:
                highest_lsn = reader.current_lsn
                best = reader

        return best
```

## リードレプリカ

### レプリカアーキテクチャ

```mermaid
graph TD
    subgraph Writer
        WBP[Buffer Pool + Pages]
    end

    subgraph Reader["Reader(s)"]
        RBP[Buffer Pool + Pages]
    end

    SS[("共有ストレージ<br/>全レプリカが同じボリュームを共有")]

    WBP -.->|redo logs<br/>async| RBP
    WBP -->|read pages| SS
    RBP -->|read pages| SS
```

> **レプリカラグ:** 通常20ms未満（ログシッピングレイテンシ）。WriterとReader間のデータコピーなし。

### レプリカログ適用

```python
class AuroraReplica:
    """Aurora read replica implementation."""

    def __init__(self, storage):
        self.storage = storage
        self.buffer_pool = BufferPool()
        self.current_lsn = 0
        self.log_queue = []

    def receive_log_record(self, log_record):
        """
        Receive redo log from writer.

        Sent asynchronously for low overhead.
        """
        self.log_queue.append(log_record)

        # Apply to buffer pool if page is cached
        if log_record.page_id in self.buffer_pool:
            self._apply_to_buffer_pool(log_record)

    def _apply_to_buffer_pool(self, log_record):
        """
        Apply log record to cached page.

        Keeps buffer pool consistent with writer.
        """
        page = self.buffer_pool.get(log_record.page_id)

        # Check if this log is newer than page
        if log_record.lsn > page.lsn:
            # Apply redo to page
            new_page = self._apply_redo(page, log_record)
            self.buffer_pool.put(log_record.page_id, new_page)

        self.current_lsn = max(self.current_lsn, log_record.lsn)

    def read_page(self, page_id) -> Page:
        """
        Read page for query.

        Check buffer pool first, then storage.
        """
        if page_id in self.buffer_pool:
            return self.buffer_pool.get(page_id)

        # Read from shared storage
        # Storage applies any pending logs automatically
        page = self.storage.read_page(page_id)

        # Apply any pending logs in our queue
        for record in self.log_queue:
            if record.page_id == page_id and record.lsn > page.lsn:
                page = self._apply_redo(page, record)

        self.buffer_pool.put(page_id, page)
        return page

    def get_replica_lag(self) -> float:
        """
        Get replica lag in seconds.

        Typically < 20ms due to async log shipping.
        """
        if not self.log_queue:
            return 0

        oldest = min(r.timestamp for r in self.log_queue)
        return time.time() - oldest
```

## ストレージゴシップと修復

### ゴシッププロトコル

```python
class StorageGossip:
    """
    Aurora storage gossip for repair.

    Storage nodes constantly gossip to detect
    and repair missing data.
    """

    def __init__(self, node_id: int, peers: list):
        self.node_id = node_id
        self.peers = peers
        self.log_records = {}  # lsn -> LogRecord
        self.gaps = []

    def gossip_round(self):
        """
        One round of gossip with peers.

        Exchange information about what logs we have.
        """
        for peer in self.peers:
            # Send our highest LSN
            their_info = peer.exchange_info(
                my_highest_lsn=self.get_highest_lsn(),
                my_gaps=self.gaps
            )

            # Fill gaps from peer
            for gap in self.gaps:
                if peer.has_logs(gap.start, gap.end):
                    missing = peer.get_logs(gap.start, gap.end)
                    self._fill_gap(missing)

            # Provide logs to peer if they're missing
            for gap in their_info.gaps:
                if self.has_logs(gap.start, gap.end):
                    logs = self.get_logs(gap.start, gap.end)
                    peer.receive_repair_logs(logs)

    def detect_gaps(self):
        """
        Detect gaps in log sequence.

        Gaps occur when some writes didn't reach us.
        """
        self.gaps = []
        lsns = sorted(self.log_records.keys())

        for i in range(len(lsns) - 1):
            if lsns[i+1] - lsns[i] > 1:
                self.gaps.append(Gap(
                    start=lsns[i] + 1,
                    end=lsns[i+1] - 1
                ))

    def _fill_gap(self, logs: list):
        """Fill gap with received logs."""
        for log in logs:
            self.log_records[log.lsn] = log

        # Re-detect gaps
        self.detect_gaps()

class SegmentRepair:
    """
    Fast segment repair after node failure.

    10GB segment can be repaired in ~10 seconds.
    """

    def repair_segment(self, failed_node, segment_id):
        """
        Repair segment by copying from healthy nodes.

        Parallel copy from multiple sources.
        """
        # Find healthy nodes with this segment
        healthy_nodes = self.get_healthy_nodes(segment_id)

        # Divide segment into chunks
        chunks = self.divide_into_chunks(segment_id, num_chunks=100)

        # Parallel copy from different nodes
        futures = []
        for i, chunk in enumerate(chunks):
            source = healthy_nodes[i % len(healthy_nodes)]
            future = self.copy_chunk_async(source, chunk)
            futures.append(future)

        # Wait for all chunks
        for future in futures:
            future.wait()

        # Verify integrity
        self.verify_segment(segment_id)
```

## パフォーマンス

### 主要メトリクス

```
┌─────────────────────────────────────────────────────────────────┐
│                   Auroraパフォーマンス                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  スループット:                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  同じハードウェアでMySQLの5倍のスループット             │    │
│  │  r4.16xlargeで最大200K writes/sec                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  レイテンシ:                                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  コミットレイテンシ: 4-6ms（従来の20msに対して）         │    │
│  │  レプリカラグ: 通常20ms未満                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  リカバリ:                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  クラッシュリカバリ: 秒単位（従来の分〜時間に対して）    │    │
│  │  フェイルオーバー: 約30秒                                │    │
│  │  セグメント修復: 10GBあたり約10秒                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ストレージ:                                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  最大サイズ: 128 TB                                      │    │
│  │  自動スケーリング（10GB単位）                            │    │
│  │  6方向レプリケーション込み                               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  コスト:                                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  従来のエンタープライズDBの1/10のコスト                  │    │
│  │  プロビジョニングではなく使用量に応じた課金              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 影響とレガシー

### クラウドデータベースへの影響

```
┌──────────────────────────────────────────────────────────────┐
│                    Auroraの影響                               │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  イノベーション:                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  - ログがデータベースアーキテクチャ                 │     │
│  │  - コンピュートとストレージの分離                   │     │
│  │  - Redo適用をストレージにプッシュ                   │     │
│  │  - クラウドネイティブ耐久性モデル                   │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
│  インスパイアしたプロジェクト:                                │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  - Azure SQL Hyperscale                             │     │
│  │  - Google AlloyDB                                   │     │
│  │  - Snowflake（類似のコンピュート/ストレージ分離）   │     │
│  │  - PolarDB (Alibaba)                                │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
│  重要な教訓:                                                 │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  従来のデータベースアーキテクチャはクラウドに        │     │
│  │  適合しません。ストレージ層を再設計することで、      │     │
│  │  耐久性、パフォーマンス、コストの劇的な改善が        │     │
│  │  可能になります。                                    │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## 重要なポイント

1. **ログがデータベース**: ページではなくログを送信 — I/Oが35倍削減されます
2. **コンピュートとストレージの分離**: 独立したスケーリングと障害ドメインです
3. **クォーラム書き込み、単一読み取り**: 4/6の書き込みクォーラム、最適化された読み取りパスです
4. **修復性のためのセグメント化**: 10GBセグメントが秒単位で修復されます
5. **ストレージに作業をプッシュ**: Redo適用は書き込み時ではなく読み取り時に発生します
6. **ほぼ瞬時のリカバリ**: 再生ではなく整合性ポイントの特定のみです
7. **レプリカ用の共有ストレージ**: データコピーなし、20ms未満のラグです
