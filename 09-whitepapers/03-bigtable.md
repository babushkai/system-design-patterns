# Bigtable (OSDI 2006): Evidence-First Paper Analysis

Bigtable's central move is to expose **physical locality as part of the data model**. Rows are globally ordered by an application-chosen key; contiguous ranges become independently assignable tablets; column families determine storage and access-control boundaries. That bargain enables enormous sparse tables, but makes schema design a placement decision rather than a purely logical one.

## Publication identity and boundary

- **Paper:** *Bigtable: A Distributed Storage System for Structured Data*
- **Authors:** Fay Chang, Jeffrey Dean, Sanjay Ghemawat, Wilson C. Hsieh, Deborah A. Wallach, Mike Burrows, Tushar Chandra, Andrew Fikes, and Robert E. Gruber
- **Venue and version:** 7th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2006, pages 205–218
- **System described:** Google's production Bigtable implementation built on GFS, Chubby, SSTables, and Google's cluster-management infrastructure

The paper is a 2006 system snapshot. It is not documentation for the later Google Cloud Bigtable product, Apache HBase, or Cassandra. Those systems share ideas but differ in replication, consensus, APIs, compaction, and operations.

Scope here: how the paper composes [Data Modeling](../02-distributed-databases/10-data-modeling.md), [LSM Trees](../03-storage-engines/02-lsm-trees.md), and [SSTables and Compaction](../03-storage-engines/03-sstables-compaction.md).

## Problem and workload shape

Google needed one storage substrate for very different applications: web indexing, satellite imagery, per-user histories, analytics, and batch pipelines. A relational schema and distributed joins were not the goal. The desired workload combined:

- petabyte-scale sparse records;
- high-throughput sequential scans and random keyed access;
- dynamic columns and multiple timestamped versions;
- application control over co-location and memory residency;
- thousands of storage machines, with failures treated as normal;
- both latency-sensitive serving and throughput-oriented MapReduce jobs.

The paper deliberately excludes a full relational model, SQL query planning, arbitrary secondary indexes, and transactions across rows. Clients serialize their own structured values and choose row keys that encode the access pattern.

## Logical state and invariants

Section 2 defines Bigtable as a sparse, distributed, persistent multidimensional sorted map:

`(row string, column string, timestamp int64) -> uninterpreted byte string`

Each dimension carries a system consequence.

### Rows define order and atomicity

Rows are lexicographically ordered. Every read or write of data under one row key is atomic, even across column families. A range of adjacent rows is the unit from which a **tablet** is formed. Therefore a key such as reversed host name can place pages from one domain near each other, while a timestamp-first key can create a hot append edge. Key design is load distribution, scan efficiency, and failure-domain sizing at once.

The atomicity boundary is one row. If an invariant spans two row keys, the published interface does not supply the transaction needed to protect it.

### Column families are declared storage boundaries

A column name has `family:qualifier` form. Families are created administratively and expected to remain few; qualifiers can be dynamic. Access control, compression choices, memory placement, and locality groups operate at family granularity. This is why “schemaless” is misleading: value structure is opaque, but families and their storage policies form a real schema.

### Timestamps create cell versions

Each cell can contain multiple versions ordered by 64-bit timestamp. The client may supply the timestamp or use real time. Per-family garbage-collection policy can retain the most recent `n` versions or versions newer than a duration. A timestamp orders versions of one cell; the paper does not turn wall-clock timestamps into a global transaction order.

## System decomposition

The architecture in Sections 4 and 5 separates metadata control from data serving.

- **Client library:** locates tablets and talks directly to tablet servers.
- **Master:** assigns tablets, detects tablet-server changes, balances load, garbage-collects GFS files, and coordinates schema operations.
- **Tablet server:** owns roughly tens to thousands of tablets, serves their reads/writes, and splits tablets that grow.
- **GFS:** durably stores commit logs and immutable SSTables.
- **Chubby:** provides master election, tablet-server liveness locks, schema files, access-control data, and the root of tablet-location metadata.

User data never passes through the master. Clients cache tablet locations, so most requests avoid it. The master is a control-plane authority, not a data-plane proxy.

### Tablet location hierarchy

Section 5.1 uses three levels. A Chubby file points to the root tablet; the root tablet points to tablets in the `METADATA` table; those metadata tablets contain user-tablet locations. The root tablet is never split. The client caches resolved locations and walks the hierarchy only after a cache miss or stale entry.

The hierarchy avoids one giant central map while retaining an authoritative root. It also establishes a dependency chain: Chubby and metadata availability are required to discover uncached locations, though cached clients can continue direct communication for known tablets.

## Write, read, and compaction paths

A valid mutation is appended to the tablet server's GFS commit log, then inserted into an in-memory sorted **memtable**. Group commit amortizes the cost of small writes. A read merges the memtable with the tablet's immutable SSTables. Because each source is sorted, range merging is efficient.

The paper describes three compaction scales:

1. **Minor compaction:** freeze a memtable, start a new one, and write the frozen state as an SSTable. This bounds memory and recovery-log replay.
2. **Merging compaction:** combine several SSTables and the memtable to bound read amplification.
3. **Major compaction:** rewrite all SSTables for a tablet, removing deleted entries rather than carrying tombstones forward.

These processes run beside foreground traffic. They trade write and background I/O for fast sequential writes and immutable files. See [Bloom Filters](../03-storage-engines/05-bloom-filters.md) for the probabilistic filter Bigtable optionally builds per locality group to avoid disk reads for absent row/column pairs.

Locality groups place selected column families in separate SSTables, so a query need not read unrelated families. Groups can be compressed or pinned in memory. Two caches serve different reuse patterns: a scan cache stores returned key/value results; a block cache stores SSTable blocks fetched from GFS.

## Assignment, failure, and recovery

Each tablet is assigned to exactly one tablet server at a time. On startup, a server creates a uniquely named Chubby file and holds its exclusive lock. If it loses the Chubby session, it stops serving. The master confirms loss by acquiring that lock, deletes the server file so the old process cannot return as an owner, and reassigns its tablets. This is fencing through an external lease service, not failure detection alone.

A new master acquires a Chubby master lock, finds live servers, asks them which tablets they own, scans metadata, and assigns anything missing. A master restart does not itself revoke tablet-server assignments.

Persistent tablet state consists of SSTable references plus redo points. A replacement server reads SSTable indexes and rebuilds the memtable by replaying committed log entries after those points. Normal operation uses one interleaved commit log per tablet server to improve group commit, which makes recovery more difficult: many replacement servers could reread the same large log. Section 6 solves that by sorting log entries by `(table, row, sequence)` in parallel so each tablet's mutations become contiguous.

For planned tablet movement, the source performs a minor compaction, stops serving, then performs a final small compaction. The destination can load SSTables without replaying the old server's log. Tablet splitting also exploits immutability: children initially share the parent's SSTables rather than rewriting them.

The paper's lessons section is unusually candid. Real failures included memory and network corruption, clock skew, hung machines, extended partitions, and bugs in both the system and dependencies—not only clean crash-stop failures. The authors emphasize checksums and avoiding assumptions about rarely exercised recovery paths.

## Quantitative evidence in context

Section 7 used clusters of 1, 50, 250, and 500 tablet servers plus an equal number of clients. Each benchmark read or wrote about 1 GB of 1,000-byte values per tablet server; the in-memory random-read test used 100 MB per server. Other jobs shared the machine pool, so results include some interference.

- On one tablet server, random disk reads achieved 1,212 values/s. Each 1,000-byte lookup fetched a 64 KB SSTable block from GFS, producing about 75 MB/s of underlying traffic and exposing block/read amplification.
- Figure 6 reports per-server rates at 1 and 500 servers: random reads fell from 1,212 to 241 values/s; in-memory random reads from 10,811 to 6,250; random writes from 8,850 to 2,000; scans from 15,385 to 7,843. Aggregate throughput still rose by more than 100 times across the 500-fold scale-out, but not linearly.
- The authors attribute the per-server decline to shared 1 Gbit links, load imbalance, and competing processes. Random disk reads scaled worst: about 100 times in aggregate for 500 times the servers.

Production observations complement the microbenchmarks. As of August 2006, the paper reports 388 non-test clusters and about 24,500 tablet servers. Fourteen busy clusters totaling 8,069 servers handled more than 1.2 million requests/s, about 741 MB/s incoming RPC traffic, and 16 GB/s outgoing. Table 2 lists, among other examples, a roughly 200 TB Google Analytics raw-click table compressed to 14% of original size and a roughly 70 TB uncompressed Google Earth imagery table. These are point-in-time deployment facts, not capacity guarantees.

## Assumptions and limitations

- Clients must know their access paths in advance and encode locality into row keys.
- One-row atomicity cannot protect cross-row invariants.
- A poor row-key distribution can create hot tablets; splitting cannot divide one indivisible hot row.
- Reads may merge several SSTables, while compaction consumes substantial background I/O.
- GFS and Chubby supply durability, leases, and critical metadata; Bigtable's properties depend on theirs.
- No SQL optimizer, relational constraints, or automatic general-purpose secondary indexes are described.
- The benchmark does not compare equivalent systems, cost, tail latency, or multi-region consistency.

## Design lineage, without conflation

Later wide-column stores retained ordered partition keys, sparse dynamic qualifiers, memtable/SSTable write paths, Bloom filters, and compaction. They changed important boundaries: some use replicated consensus per range, different compaction strategies, or different consistency APIs. A recognizable storage shape does not imply Bigtable's exact failure or transaction semantics.

Google's [Spanner paper](./04-spanner.md) addresses a different requirement: synchronous replication and transactions across rows and locations. It retains range partitioning and a structured schema but adds Paxos groups, MVCC timestamps, two-phase commit, and TrueTime. That progression shows the cost of moving from “scale keyed access” to “scale externally consistent transactions.”

## Review questions for a Bigtable-like design

1. Does the row key distribute writes while preserving the scans the application needs?
2. Which invariant fits inside one row, and what protects the rest?
3. Which column families are read together, compressed together, and authorized together?
4. How much read amplification is acceptable before compaction, and what capacity is reserved for it?
5. Can metadata and lease dependencies recover without creating two owners for a range?
6. Does a tablet split relieve the actual hot spot, or is one row/key still dominant?
7. Are benchmark results reported with block size, cache state, value size, topology, and background load?

## Primary sources

- [Chang et al., *Bigtable: A Distributed Storage System for Structured Data* (OSDI 2006), official Google Research PDF](https://storage.googleapis.com/gweb-research2023-media/pubtools/4443.pdf)
- [Google Research publication record for Bigtable](https://research.google/pubs/bigtable-a-distributed-storage-system-for-structured-data/)
- [Corbett et al., *Spanner: Google's Globally-Distributed Database* (OSDI 2012)](https://www.usenix.org/system/files/conference/osdi12/osdi12-final-16.pdf)
