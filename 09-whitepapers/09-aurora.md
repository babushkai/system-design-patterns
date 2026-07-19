# Amazon Aurora (SIGMOD 2017): Evidence-First Paper Analysis

Aurora's key idea is to treat network I/O—not disks—as the scarce database resource. The database engine sends **redo records rather than full pages** to a six-way, multi-AZ storage service. Storage nodes continuously materialize pages, repair replicas, back up data, and perform much of recovery. This removes duplicated database-to-storage traffic and turns crash recovery from a serial startup phase into ongoing distributed work.

## Publication identity and boundary

- **Paper:** *Amazon Aurora: Design Considerations for High Throughput Cloud-Native Relational Databases*
- **Authors:** Alexandre Verbitski, Anurag Gupta, Debanjan Saha, Murali Brahmadesam, Kamal Gupta, Raman Mittal, Sailesh Krishnamurthy, Sandor Maurice, Tengiz Kharatishvili, and Xiaofeng Bao
- **Venue and version:** ACM SIGMOD International Conference on Management of Data, 2017, pages 1041–1052
- **System described:** the MySQL-compatible Aurora service that had been generally available since July 2015, as implemented and measured for the paper

The paper does not describe every present Aurora engine, feature, or topology. It presents one writer with shared storage and up to 15 read replicas. Later multi-writer, serverless, PostgreSQL-compatible, cross-region, or disaggregated-service features must be evaluated from their own sources.

For conventional mechanisms, read [Write-Ahead Logging](../03-storage-engines/04-write-ahead-logging.md), [Single-Leader Replication](../02-distributed-databases/01-single-leader-replication.md), and [Failure Modes](../01-foundations/06-failure-modes.md). This chapter focuses on Aurora's changed storage contract.

## Problem and workload assumptions

A traditional primary database may send a write-ahead log, dirty pages, replicated log traffic, backups, and cache-management I/O across the network. In a multi-AZ cloud design, those flows multiply. Checkpointing, page flushing, replica apply, backup, and crash recovery also create foreground jitter.

Aurora targets MySQL/InnoDB OLTP workloads whose transaction and query processing can remain on one database instance while durable storage scales and repairs independently. Its design goals are:

- preserve familiar SQL, transactions, and MySQL wire compatibility;
- increase write throughput by shrinking database-node network traffic;
- tolerate storage-node and Availability Zone failures;
- make replicas share one durable volume rather than each copying pages independently;
- recover database compute quickly without replaying a long log before reopening.

This is **storage disaggregation**, not a shared-nothing distributed SQL engine. The writer still performs SQL execution, locking, buffer-pool management, and log-sequence allocation.

## Protection groups and quorum invariants

A logical volume is divided into 10 GB segments. Each segment is replicated six ways as a **protection group**: two copies in each of three Availability Zones. Volumes allocate more protection groups as they grow; the paper reports support up to 64 TB of unreplicated logical data.

The paper's quorum model uses:

- six replicas per segment;
- write quorum `Vw = 4`;
- read quorum `Vr = 3` during recovery or when discovering durable state.

Because `Vw + Vr > 6`, a read quorum intersects every completed write quorum. Four write acknowledgements permit continued writes after any two copy failures, including loss of one two-copy AZ. Three readable copies permit recovery even after three copy failures, although new writes then lack four acknowledgements.

Quorum counts are not the whole failure model. Placement across three independent AZs addresses correlation; rapid segment repair reduces exposure time. The paper states that a 10 GB segment can be repaired in about 10 seconds over a 10 Gbit/s link in its environment. That observed repair speed is part of the durability argument, not a universal bound.

## Redo-only storage protocol

The database assigns each redo record a monotonically increasing log sequence number (LSN) and sends only records relevant to each protection group. Storage nodes queue, persist, acknowledge, gossip missing records to peers, coalesce redo into pages, upload backups, scrub checksums, and garbage-collect old versions.

Each redo record links to the previous record for that protection group. A segment derives a **Segment Complete LSN (SCL)**: the greatest point below which it has every required record. Peer gossip uses gaps in that sequence to repair missing data.

The database tracks acknowledgements from segment replicas. Across protection groups it computes a **Volume Complete LSN (VCL)**—the highest contiguous volume point known complete. Completeness alone is not enough: InnoDB mini-transactions contain several contiguous redo records that must become visible atomically. The final record of each mini-transaction is marked a **Consistency Point LSN (CPL)**. The **Volume Durable LSN (VDL)** is the highest CPL no greater than the VCL.

This distinction is the core invariant:

- records through VCL are physically complete;
- only a CPL is a valid database consistency boundary;
- commits may be acknowledged only when VDL reaches their commit LSN;
- recovery truncates anything above the reconstructed VDL.

The database also limits LSN allocation to at most 10 million beyond the current VDL in the paper. That backpressure prevents compute from creating an unbounded recovery/truncation window when storage or network falls behind. It is an implementation value, not a tuning recommendation.

## Foreground reads, writes, and commits

### Write path

The writer batches redo records and sends each protection group's subset to all six segment copies in parallel. It advances VDL as quorums acknowledge complete mini-transactions. Unlike traditional InnoDB, it does not send dirty pages to storage on eviction or checkpoint.

### Commit path

When a transaction requests commit, its worker records the commit LSN on a waiting list and processes other work. A dedicated thread acknowledges transactions when `VDL >= commit LSN`. This asynchronous handoff avoids tying worker scheduling to individual storage jitter while retaining the WAL rule.

### Read path

The buffer cache serves present pages. On a miss, the database chooses a storage replica whose SCL covers the read point and requests the page at that VDL. No normal read quorum is necessary because the writer tracks completeness. A dirty page may be evicted only after all of its changes are durable through VDL; storage can reconstruct it later from materialized state plus redo.

The writer and read replicas share the same storage volume. Redo also streams to replica compute nodes so they can update cached pages; if a page is absent, a replica discards that redo and later reads a current page from storage. Replicas apply only records no greater than VDL and apply each mini-transaction atomically in their cache. Publication-era replicas were asynchronous from the writer's commit path.

## Failure, repair, and recovery

- **Storage-copy loss:** remaining segment replicas serve requests; peers reconstruct the missing 10 GB segment and restore the six-copy placement.
- **Slow storage node:** parallel requests and a four-of-six write quorum prevent one straggler from determining every write latency.
- **Database-process crash:** a replacement queries a read quorum for every protection group, reconstructs VDL, and sends versioned truncation ranges above it. It rebuilds volatile transaction state while storage already holds materialized pages and redo.
- **Uncommitted transactions:** undo still exists. The paper allows undo recovery after the database comes online, using the in-flight transaction list rebuilt from undo segments.
- **Interrupted recovery:** epoch-versioned truncation records make repeated recovery distinguish the newest boundary.
- **Read-replica failure:** no storage copy is lost because replicas mount the shared volume; a replacement warms its own buffer cache.

The paper says database recovery was generally under 10 seconds even after processing more than 100,000 write statements/s. This is an operational observation from Aurora, not a bound under every volume, cache, or control-plane failure.

## Evaluation evidence and methodology

The authors are the system's designers and operator. Most comparisons used r3.8xlarge EC2 instances with 32 vCPUs, 244 GB RAM, Intel Xeon E5-2670 v2 processors, and a 170 GB buffer cache. MySQL used EBS with 30,000 provisioned IOPS unless stated otherwise. Results measure particular configurations, not architecture alone.

### Network-I/O experiment

Table 1 ran a 30-minute SysBench write-only workload over 100 GB on r3.8xlarge instances. It compared synchronous mirrored MySQL across AZs with Aurora and its multi-AZ replicas.

- mirrored MySQL completed 780,000 transactions at 7.4 database-node I/Os per transaction;
- Aurora completed 27,378,000 transactions at 0.95 I/Os per transaction;
- the paper summarizes this as 35 times the transactions and 7.7 times fewer database-node I/Os per transaction.

The result supports the redo-only network thesis in that setup. It is not a controlled decomposition of every difference between the MySQL and Aurora stacks, and it does not report a cost-normalized or tail-latency comparison.

### SysBench scale experiments

For 1 GB datasets over the r3 instance family, the paper reports near-linear Aurora scaling with instance size, reaching about 121,000 writes/s and 600,000 reads/s on r3.8xlarge. Table 2's write-only runs report:

| Database size | Aurora writes/s | MySQL writes/s |
|---:|---:|---:|
| 1 GB | 107,000 | 8,400 |
| 10 GB | 107,000 | 2,400 |
| 100 GB | 101,000 | 1,500 |
| 1 TB | 41,000 | 1,200 |

The authors call the 100 GB ratio 67 times and the 1 TB ratio 34 times. The working set, EBS configuration, MySQL versions, and buffer sizes are inseparable from those ratios.

### Replica lag and contention

Table 4 defines replica lag as time from writer commit until visibility on a replica. As write rate increased from 1,000 to 10,000 writes/s, Aurora lag rose from 2.62 to 5.38 ms; the MySQL configuration rose from under 1,000 ms to 300,000 ms. The comparison demonstrates shared-storage/log-stream behavior in the test, not all possible MySQL replication designs.

The Percona TPC-C variant used 100 or 1,000 warehouses and 500 or 5,000 connections. Aurora's reported throughput ranged from 30,221 to 73,955 tpmC across those four cases, 2.3–16.3 times MySQL 5.7. It was a TPC-C-derived workload, not an audited TPC-C result.

## Limits and assumptions

- One writer is the serialization and SQL-compute bottleneck in the published architecture.
- Six-way storage replication spends capacity to reduce correlated-failure risk and tail latency.
- Correctness depends on LSN ordering, CPL placement, quorum intersection, and storage-node redo determinism.
- Shared storage removes replica storage copies but does not make asynchronous readers instantly current.
- The design assumes AWS's regional network, AZ separation, fleet repair, and managed control plane.
- Benchmarks are vendor-run and compare stacks with different implementations and, in some cases, feature paths.
- Compatibility was based on a MySQL 5.6 fork at publication; it was not identical to upstream MySQL behavior or ecosystem surface.

## What the follow-up clarified

The 2018 SIGMOD paper, *Amazon Aurora: On Avoiding Distributed Consensus for I/Os, Commits, and Membership Changes*, makes explicit how the single-writer epoch, quorum state, and locally reconstructed metadata avoid running a general consensus round for most storage I/O. It retains the 2017 architecture while deepening the invariant argument. That clarification should not be simplified to “Aurora has no consensus”: the system still establishes unique writer epochs and durable quorum state; it avoids consensus on particular steady-state paths under stated invariants.

## Architecture review questions

1. Which bytes cross the compute/storage boundary: logical operations, redo, or full pages?
2. What contiguous and transaction-consistent frontier makes an acknowledgement safe?
3. How do read and write quorums intersect for every allowed failure pattern?
4. Can storage repair keep the vulnerable window below the correlated-failure budget?
5. What state must compute rebuild before serving, and what can recover lazily?
6. Does one writer have enough CPU, lock, log-allocation, and network capacity?
7. Are comparisons matched for durability, replicas, cache size, instance, and workload compliance?

## Primary sources

- [Verbitski et al., *Amazon Aurora: Design Considerations for High Throughput Cloud-Native Relational Databases* (SIGMOD 2017), official Amazon Science PDF](https://cdn.amazon.science/dc/2b/4ef2b89649f9a393d37d3e042f4e/amazon-aurora-design-considerations-for-high-throughput-cloud-native-relational-databases.pdf)
- [Amazon Science publication record for the 2017 paper](https://www.amazon.science/publications/amazon-aurora-design-considerations-for-high-throughput-cloud-native-relational-databases)
- [Verbitski et al., *Amazon Aurora: On Avoiding Distributed Consensus for I/Os, Commits, and Membership Changes* (SIGMOD 2018)](https://www.amazon.science/publications/amazon-aurora-on-avoiding-distributed-consensus-for-i-os-commits-and-membership-changes)
