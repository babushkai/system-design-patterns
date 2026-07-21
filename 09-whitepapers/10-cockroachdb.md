# CockroachDB (SIGMOD 2020): Evidence-First Paper Analysis

CockroachDB's 2020 design assembles range-level Raft, MVCC timestamps, transaction records, leases, and a distribution-aware optimizer. The challenge is preserving serializable SQL semantics while every table and index is encoded across independently replicated key ranges that can move and fail.

## Publication identity and frozen version

- **Paper:** *CockroachDB: The Resilient Geo-Distributed SQL Database*
- **Authors:** Rebecca Taft, Irfan Sharif, Andrei Matei, Nathan VanBenschoten, Jordan Lewis, Tobias Grieger, Kai Niemi, Andy Woods, Anne Birzin, Raphael Poss, Paul Bardea, Amruta Ranade, Ben Darnell, Bram Gruneir, Justin Jaffray, Lucy Zhang, and Peter Mattis
- **Venue and version:** ACM SIGMOD International Conference on Management of Data, 2020, pages 1493–1509
- **Implementation evaluated:** CockroachDB v19.2.2 unless noted; the large TPC-C comparison explicitly used v19.2.0

Scope: the paper's 2020 names and guarantees. Transaction locking, storage engines, multi-region SQL, changefeeds, serverless operation, and optimizer features evolved later.

See [Distributed Transactions](../02-distributed-databases/07-distributed-transactions.md), [Consensus](../02-distributed-databases/08-consensus-algorithms.md), and [Partitioning](../02-distributed-databases/05-partitioning-strategies.md) for the isolated concepts.

## Problem and target contract

The system aims to offer familiar SQL and ACID transactions while scaling horizontally across commodity machines and regions. Any node accepts a client connection. Tables, indexes, and system metadata share one ordered key/value address space; the database automatically splits, replicates, moves, and rejoins its ranges.

The published contract includes:

- atomic transactions with serializable isolation;
- SQL schemas, secondary indexes, foreign keys, joins, and online schema changes;
- synchronous replication of every key range through Raft;
- automatic failover and rebalancing;
- configurable replica and leaseholder placement;
- hybrid logical clocks (HLCs) rather than specialized time hardware;
- local stale reads from followers through closed timestamps.

The contract does not make geography free. A write reaches the leaseholder and a Raft quorum for every touched range; multi-range transactions add coordination; globally distributed joins move rows.

## Layering and data placement

The paper's Figure 1 separates SQL, transactions, distribution, replication, and local storage.

SQL tables and indexes encode into sorted key/value pairs. Adjacent keys form **Ranges**, about 64 MiB in the paper. Ranges split by size or load and merge when small. A two-level structure of system ranges locates them, with aggressive client/node caching.

Each range normally has three replicas on different nodes. Its replicas form a Raft group. Raft commits ordered low-level storage commands; each replica applies those commands to its local RocksDB engine. A range's **leaseholder**, usually its Raft leader, is the only replica that serves authoritative current reads or proposes writes. The lease is itself acquired through Raft, preserving one-owner-at-a-time semantics.

Separating Raft leadership from leaseholdership is useful analytically: Raft orders replicated commands; the lease establishes which replica may evaluate current reads and coordinate writes. Co-location avoids an extra hop but the roles prove different properties.

Placement constraints attach locality or hardware attributes to schema partitions. Automatic rules spread replicas across failure domains and balance disk/load. Performance depends heavily on placing the leaseholder near writers and a quorum along the desired latency path.

## Timestamp and serialization model

CockroachDB uses multi-version concurrency control. A transaction reads and writes at an HLC timestamp containing a physical component and a logical counter. HLCs preserve causality when messages carry a higher timestamp while remaining close to wall time.

Unlike Spanner's TrueTime interval, CockroachDB assumes nodes keep physical clocks within a configured maximum offset. Under that assumption the paper claims single-key linearizability, but explicitly not strict serializability for transactions over disjoint key sets: their timestamp order need not match real-time order. The protocol treats recently observed timestamps inside an uncertainty interval conservatively, restarting a transaction rather than risk reading a value that is physically “from its future.”

Section 4.3 separates two consequences of excessive skew. Raft ordering plus lease-sequence checks still preserve serializable isolation, even if two replicas temporarily believe they hold a lease. Single-key linearizability between causally dependent transactions entering through different gateways can fail once clocks exceed the configured bound. Nodes therefore compare offsets and self-terminate when their offset exceeds the configured maximum by more than 80% relative to a majority of peers.

Serializable execution is built from several conflict rules:

- a lower-timestamp read encountering a higher-timestamp intent may ignore it;
- a read encountering an earlier uncommitted intent waits or helps resolve it;
- a write whose timestamp is below a recorded later read must move forward;
- when a transaction's timestamp moves, it attempts **read refresh** to prove its previous reads return the same values at the new timestamp; otherwise it retries.

The timestamp cache summarizes recent reads so a later write cannot commit beneath them and create an anti-dependency cycle. Latches provide short-lived mutual exclusion while evaluating overlapping commands; replicated intents protect transactional writes across requests and failures.

## Atomicity through intents and transaction records

Every provisional write is an **intent**: an MVCC value plus metadata pointing to one transaction record. The record, stored in the range of the transaction's first write, has state `PENDING`, `STAGING`, `COMMITTED`, or `ABORTED`.

Readers encountering an intent consult the record:

- committed: treat the intent as visible and clean up metadata;
- aborted: ignore and remove it;
- pending: wait, push, or eventually abort an expired transaction;
- staging: determine whether every declared write actually replicated.

This indirection gives one logical status authority for writes spread across many ranges. The transaction coordinator heartbeats long-running pending records. If it dies, contenders can detect expiration and resolve abandoned intents; coordinator memory is not the only record of the outcome.

## Write pipelining and parallel commits

A naive multi-range commit waits for every intent to replicate, then performs another consensus round to commit the transaction record. Section 3.1 removes much of that serial latency.

**Write pipelining** sends independent operations without waiting for earlier ones to finish replication. The coordinator records in-flight writes and forces dependencies to wait when key spans overlap.

**Parallel Commits** writes a `STAGING` transaction record containing the expected write set while outstanding intents replicate concurrently. If every write is durable, the staging state is implicitly committed and the SQL layer may receive success; an explicit `COMMITTED` update happens asynchronously. If the coordinator dies, another actor checks the declared writes: all present means committed, any prevented/missing write means abort.

The proof obligation is conditional atomicity: nobody may observe only part of a staging transaction. The authors modeled the protocol in TLA+ and checked that staging transactions resolve to one durable outcome without telling clients the opposite.

## Read paths and closed timestamps

Current consistent reads normally go to the leaseholder, which has authority over the range's write serialization. Historical reads can use followers.

Leaseholders periodically emit a **closed timestamp** below which they promise not to accept new writes. Replicas exchange that timestamp with corresponding Raft log indexes. A follower may serve a read at timestamp `t` only if `t` is closed and it has applied the needed log position. In the paper, closed timestamps typically trailed real time by about two seconds.

This is not a current follower read. It is a provably complete historical read whose staleness buys locality. The 2022 multi-region paper later changes this design space; those newer semantics should not be back-projected.

## SQL planning and schema state

Any node can be a SQL gateway. It parses and optimizes a query, determines the ranges involved, and constructs a distributed physical plan. Operators run near data where possible and stream intermediate rows between nodes. The optimizer extends a Cascades-style search with distribution properties and network-aware alternatives.

Secondary-index updates are transactional writes to other key ranges, so a single SQL row can become a multi-range transaction. Foreign keys and joins similarly cross placement boundaries. Logical convenience maps directly to coordination cost.

Online schema changes use versioned descriptors and background backfills. The paper maintains at most two successive schema versions in use, allowing old and new nodes/transactions to coexist while an index is populated before becoming readable. Schema change is a distributed state machine, not an instantaneous metadata edit.

## Failure and recovery behavior

- **Replica or node loss:** a range continues with a Raft majority. A returning replica catches up by log entries or a full snapshot, depending on divergence.
- **Long failure:** the allocator creates replacement replicas from surviving copies and rebalances placement.
- **Raft leader loss:** the group elects another leader; lease acquisition is replicated before authoritative service resumes.
- **Coordinator loss:** heartbeat expiry and the transaction record let other transactions abort pending work or prove a staging transaction committed.
- **Region/AZ loss:** only ranges with surviving majorities and usable lease placement continue; schema constraints determine which failure domains each range spans.
- **Clock anomaly:** the maximum-offset assumption protects timestamp reasoning by refusing unsafe participation rather than silently accepting arbitrary skew.

Automatic recovery does not guarantee continuous progress for every topology. Three replicas with two in a failed region lose quorum; a surviving quorum across continents may remain correct but slow.

## Evaluation evidence with experiment boundaries

### Parallel Commits microbenchmark

Figure 2 used three servers in three regions, writing one row with ten columns and varying secondary-index count. Compared with two-phase commit, Parallel Commits improved throughput by up to 72% and reduced median latency by up to 47% once one or more indexes forced cross-range work. “Up to” refers to points in this workload, not every transaction.

### Horizontal and vertical scale

Figure 4 ran SysBench reads/writes on three-node AWS clusters from c5d.large through c5d.9xlarge, then on 3–48 c5d.9xlarge nodes across three `us-east-1` AZs. Each point averaged three runs; the 48-node case held about 38 GB across four one-million-row tables per node. Throughput per vCPU stayed nearly constant for these embarrassingly parallel workloads. This does not measure cross-range contention.

### Coordination-cost experiment

Figure 5 varied cluster size, replica factor, and the percentage of remote TPC-C NewOrder transactions on 4-vCPU GCP `n1-standard-4` machines. Each point averaged three runs and found maximum tpmC sustained for at least ten minutes; the largest case used 10,000 warehouses and 800 GB.

Replication reduced throughput by up to 48% at factor three and 57% at factor five; distributed transactions imposed up to another 46% reduction. Throughput still scaled linearly with nodes in the graphed range. The important evidence is the measured tax of guarantees—not only the scale-out line.

### Large TPC-C run

Table 1 used CockroachDB v19.2.0 and spec-compliant wait times and foreign keys:

| Warehouses | Nodes / AWS type | Maximum tpmC | Efficiency | NewOrder p90 |
|---:|---|---:|---:|---:|
| 1,000 | 3 × c5d.4xlarge | 12,474 | 97.0% | 39.8 ms |
| 10,000 | 15 × c5d.4xlarge | 124,036 | 96.5% | 436.2 ms |
| 100,000 | 81 × c5d.9xlarge | 1,245,462 | 98.8% | 486.5 ms |

The largest case represented about 50 billion rows and 8 TB. The high efficiency shows throughput tracking the configured warehouse count; p90 latency also rises sharply from the small case. The paper's Aurora comparison reuses published Aurora numbers whose latency and machine details were incomplete, so it is not a controlled head-to-head benchmark.

### Multi-region failure injection

Figure 6 used nine 4-vCPU GCP nodes across three US regions plus regional load generators, running TPC-C 1,000 while injecting an AZ failure and a region failure. All four placement policies tolerated the AZ failure in the experiment. Only the geo-partitioned-leaseholder policy sustained the region-wide failure, at the price of higher p90 latency during normal operation and recovery. This demonstrates policy trade-offs for that replica layout, not automatic region survival for every table.

## Limits and assumptions

- Every range needs a Raft majority; cross-range availability is the intersection of all touched ranges.
- Maximum clock offset is an operational correctness dependency.
- Hot keys or ranges can bottleneck before splitting or if one key is indivisible.
- Secondary indexes and foreign keys silently turn local SQL writes into distributed transactions.
- Intents, transaction records, retries, and cleanup add storage and tail-latency work under contention or failure.
- Follower reads in the paper are roughly two seconds stale, not current linearizable reads.
- Results are vendor-authored, version-specific, and use several clouds and instance shapes; reused competitor numbers are not equivalent experiments.

## Later multi-region evolution

The SIGMOD 2022 paper adds declarative table-locality/availability syntax and a new transaction-management protocol for local, current reads from any replica. It changes the 2020 trade-off in which follower locality required a closed, stale timestamp. That is a substantive protocol evolution, while range-level replication, placement-aware planning, and serializable transactions remain the architectural spine.

## Design review questions

1. Which SQL operations touch more than one range after index and constraint expansion?
2. Where are leaseholders and quorum replicas relative to writers and readers?
3. What enforces the physical-clock offset assumption, and how does a node fail closed?
4. How are staging transactions resolved after coordinator loss without contradicting acknowledged success?
5. Can closed-timestamp staleness satisfy the read contract?
6. Which placement policy survives the named AZ or region failure while meeting latency goals?
7. Do benchmarks expose transaction locality, replication factor, percentiles, database version, and sustained-run duration?

## Primary sources

- [Taft et al., *CockroachDB: The Resilient Geo-Distributed SQL Database* (SIGMOD 2020), official Cockroach Labs PDF](https://www.cockroachlabs.com/pdf/cockroachdb-the-resilient-geo-distributed-sql-database-sigmod-2020.pdf)
- [Cockroach Labs research publication page for the SIGMOD 2020 paper](https://www.cockroachlabs.com/guides/cockroachdb-the-resilient-geo-distributed-sql-database-sigmod-2020)
- [Shraer et al., *Enabling the Next Generation of Multi-Region Applications with CockroachDB* (SIGMOD 2022)](https://www.cockroachlabs.com/guides/cockroachdb-the-resilient-geo-distributed-sql-database-sigmod-2022)
