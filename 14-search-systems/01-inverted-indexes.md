# Search Index Architecture and Internals

A search index is a derived, query-optimized projection of an authoritative data source. It must publish coherent generations, survive process and machine failures, bound freshness, and serve a known logical snapshot while indexing and segment maintenance continue.

The physical index lifecycle covers document ingestion, immutable segments, commit points, deletion state, merges, shard placement, replication, recovery, and distributed fan-out. See [Lexical Query Execution](02-full-text-search.md) for analyzers, postings traversal, BM25, filters, aggregations, and top-k execution; see [Ranking and Evaluation](04-ranking-algorithms.md) for learned ranking and relevance measurement.

## Workload and service contract

Start with the operations the index must support:

- `upsert(document_id, source_version, document)`: make a new source version searchable eventually;
- `delete(document_id, source_version)`: prevent older versions from reappearing;
- `search(query, read_policy)`: execute against one declared index generation or a bounded set of shard generations;
- `refresh`: make accepted writes visible without necessarily making them crash-durable;
- `commit`: publish a recoverable commit point;
- `rebuild(schema_version)`: construct a complete replacement projection from the source of truth.

The product contract must make four different clocks explicit:

1. **Acceptance latency**: when the indexing API durably accepts an operation.
2. **Visibility latency**: when a searcher can observe it.
3. **Commit latency**: when it survives loss of the active process or node.
4. **Replica convergence**: when every serving copy can observe an equivalent logical state.

Calling all four “indexing latency” hides the failures operators need to reason about. A system may acknowledge into a replicated write-ahead log in 20 ms, refresh searchers every second, commit segments every 30 seconds, and allow replicas to trail for several seconds. Those are four different guarantees.

## State and invariants

An index shard commonly contains these state classes:

| State | Typical representation | Mutability | Recovery role |
|---|---|---:|---|
| accepted operations | sequence-numbered WAL or operation log | append-only | replay after an unclean stop |
| searchable content | immutable segment files | immutable | durable query data |
| document liveness | deletion bitmap or tombstone generation | copy-on-write | masks obsolete document versions |
| published view | manifest naming segment and deletion generations | atomic replacement | recovery boundary |
| serving view | opened readers plus caches | process-local | low-latency queries |
| placement metadata | shard term, primary, replicas, allocation state | control-plane state | fencing and routing |

The design should enforce these invariants:

**One visible version per logical document.** If versions 41 and 42 both exist physically because segments are immutable, the live-document state exposes only the winner. A stale replay of version 41 must not resurrect it.

**A commit is closed under references.** A published manifest never names a segment, deletion file, dictionary, or stored-field file that is absent or only partially durable.

**Generation publication is atomic.** Readers observe the old complete generation or the new complete generation, never a mixture assembled during a merge.

**Acknowledged durability matches the API.** If an acknowledgement promises survival of one node failure, the operation must be on enough independent failure domains before acknowledgement. A page-cache write on one node is not that guarantee.

**Primary epochs fence old writers.** Every mutation and replication message carries a shard term or lease token. After failover, the new primary rejects operations from an older term, even if the old process is still alive in a partition.

**The index remains rebuildable.** Search storage is not the only copy of business data. Every indexed document is derivable from an authoritative log, table snapshot plus change stream, or equivalent replayable source.

## Data plane and control plane

Separating the planes prevents an indexing outage from turning into metadata corruption.

The **data plane** accepts source changes, analyzes documents, appends operations, constructs segments, refreshes readers, executes shard-local queries, and merges results. Its hot path should not require a globally consistent control-plane lookup per query.

The **control plane** owns index definitions, field and analyzer versions, shard maps, replica assignments, shard terms, allocation decisions, rollout state, and rebuild orchestration. Serving nodes consume versioned snapshots of this state. They should continue serving a pinned, last-known-good snapshot during a transient control-plane outage, while rejecting operations whose safety requires fresher authority.

A useful lifecycle is:

```text
source transaction
    -> ordered change record(document_id, source_version, schema_version)
    -> route to logical shard
    -> append and replicate operation
    -> analyze into per-field index structures
    -> flush immutable segment
    -> publish refresh generation
    -> commit manifest
    -> merge old segments asynchronously
```

Idempotency lives at the projection boundary. The shard records the greatest accepted source version per document, or uses a monotonic source sequence whose replay position is committed with the segment generation. Transport-level duplicate suppression alone is insufficient: a rebuild, retry, or failover can deliver the same event through a different transport session.

## Inside an immutable segment

An inverted index maps each analyzed term to a sorted postings list. A production posting can include:

- a local document ordinal;
- term frequency within a field;
- positions and offsets for phrase matching and highlighting;
- per-document norms used by scoring;
- optional skip data or block metadata for fast traversal.

The term dictionary maps term bytes to posting locations and corpus statistics. Implementations often represent it with a sorted dictionary or finite-state structure so shared prefixes are compact and lookup can seek rather than scan. Stored fields retrieve result payloads. Column-oriented per-document values support sorting, faceting, and feature access without reconstructing the original document. Points or specialized trees serve numeric and geospatial ranges.

Local document ordinals are segment-private. A stable external ID therefore needs a lookup structure or source-version map for updates. An update is usually “append the replacement and mark the prior ordinal deleted,” not an in-place rewrite of every structure.

Compression is part of the performance model, not an afterthought. Sorted document ordinals become small gaps; block encodings compress those gaps while allowing vectorized decoding and skipping. The right objective is query work per byte read, not the smallest possible file. A denser encoding that costs excessive branchy CPU can lose at high query rates.

## Flush, refresh, commit, and merge

Indexing accumulates analyzed structures in memory. A **flush** writes a new immutable segment. A **refresh** opens a new reader generation so searches can see it. A **commit** publishes a durable manifest after referenced files satisfy the durability contract. These events may coincide, but they do not mean the same thing.

Immutable segments make concurrent readers simple: a reader pins a generation while new segments are created. The cost appears later as segment proliferation. Every query may need to seek multiple dictionaries, traverse multiple postings lists, and merge multiple local top-k lists. Background merging rewrites selected segments into fewer, larger segments and drops deleted documents.

Merge policy is a feedback controller over three competing costs:

- **read amplification** rises with segment count and deletion density;
- **write amplification** rises when bytes are repeatedly rewritten through merge levels;
- **space amplification** rises while old and new segments coexist and deleted documents remain unreclaimed.

Size-tiered policies merge similarly sized segments and favor write throughput, but can retain more files. Leveled policies bound overlap/read amplification more aggressively, often at higher rewrite cost. There is no universal merge factor: measure segment-size distribution, device bandwidth, deletion rate, and query sensitivity.

Merge output is published like any other generation. The safe sequence is: write all new files under a new generation; verify lengths and checksums; make files durable as required; atomically publish a manifest; wait until no reader pins the old generation; then reclaim old files. Deleting inputs before publication turns a routine crash into shard loss.

## Durability and recovery protocol

A recoverable shard needs an ordering relation between accepted operations and committed index state. One pattern is:

1. assign a monotonically increasing shard sequence number;
2. append the operation to a checksummed log and replicate according to the acknowledgement policy;
3. apply it to the in-memory indexing buffer;
4. write a segment covering a known inclusive sequence range;
5. publish a manifest containing the highest fully represented sequence number;
6. truncate log records only when the committed generation and required replicas no longer need them.

On restart, validate the latest complete manifest, open its segments, and replay later log records. A torn trailing log record is discarded only if the format can distinguish an incomplete append from a valid record. Checksum failure in the middle of the durable prefix is corruption, not an ignorable tail.

Recovery time is bounded by both log length and file transfer. A replica that is far behind may be faster and safer to seed from a committed shard snapshot and then replay the suffix than to retain an unbounded operation history. Snapshot metadata must include index identity, schema/analyzer version, commit generation, shard term, and source checkpoint; a directory of files without this provenance is not a recovery artifact.

## Sharding, replication, and distributed search

Choose the shard key from write locality, query fan-out, tenant isolation, and resharding needs. Hashing document IDs balances bytes but causes corpus-wide fan-out for unconstrained queries. Tenant or time partitioning can route queries narrowly, but hot tenants and skewed periods require subshards. A routing key that improves one query class can make another pathological, so record the expected fan-out distribution, not only average shard size.

Each logical shard has replicas that contain equivalent index generations, although physical segment layouts may differ after independent merges. The routing layer chooses one eligible replica per shard, attaches a query deadline and generation policy, and merges shard-local candidates. Replication gives availability; it does not make a broadcast query cheap.

Distributed top-k is exact only if every relevant shard participates and returns enough candidates under a compatible scoring model. Corpus statistics can be shard-local, periodically global, or gathered in an extra round trip. Each choice changes ranking consistency and latency. Query execution details, including candidate bounds, live in the next chapter; the architecture must nevertheless expose partial-result policy. “Return partial,” “fail the query,” and “retry another replica” are product decisions, not transport defaults.

Replica selection should consider queue time and generation freshness, not just network distance. Hedging can reduce tail latency but multiplies work during overload; send a hedge only after a delay, cancel losers, and debit it against an overload budget. A retry to an already saturated replica set is amplification.

## Capacity and cost model

Use measured corpus properties. Consider an explicitly illustrative workload:

- 200 million live documents;
- average source payload 2 KiB;
- measured index footprint 0.9 KiB per live document, including postings, stored fields, and column values;
- 20% temporary headroom for deletions and merge overlap;
- 12 logical shards, two serving replicas each;
- 8,000 peak queries/s with median fan-out to all 12 shards;
- 4,000 document mutations/s.

The steady live index is `200M * 0.9 KiB`, about 168 GiB. With 20% local amplification it is about 201 GiB per complete corpus copy. Two replicas require about 402 GiB before snapshots, WAL, filesystem reserve, and rebuild capacity. If one node must tolerate a merge that temporarily rewrites its largest 40 GiB segment, reserve that peak explicitly rather than relying on the steady 20% estimate.

Fan-out produces `8,000 * 12 = 96,000` shard searches/s before retries or hedges. If a shard search consumes 1.5 ms of CPU on the measured query mix, this is 144 CPU-seconds per second at 100% utilization. At a target utilization of 55% to preserve tail headroom, the query path needs about 262 logical CPU cores, plus coordination, indexing, and merge work. This arithmetic is a planning hypothesis; replay representative queries to measure the real service-time distribution.

Indexing bandwidth must include merge amplification. At `4,000 * 0.9 KiB`, logical index production is only 3.5 MiB/s. If measured merge write amplification is 8, the device writes roughly 28 MiB/s before WAL and replication. More importantly, merges arrive in bursts. Admission control should use queued merge bytes and device saturation, not just average ingress.

## Schema and analyzer evolution

An index schema changes interpretation, so a field mapping or analyzer update is a data migration. Some additive fields can begin on new documents, but tokenization, normalization, similarity, vector dimensions, and type changes usually require reindexing.

Use versioned physical indexes behind a stable logical alias:

1. freeze a source checkpoint and create index version `v2` with immutable schema metadata;
2. backfill a snapshot while consuming changes after the checkpoint into a buffered or dual projection;
3. catch `v2` up to a declared freshness bound;
4. shadow representative queries and compare coverage, latency, and result-quality metrics;
5. atomically switch the read alias or route a canary cohort;
6. keep `v1` readable for a bounded rollback window;
7. stop its change feed and reclaim it only after rollback expires.

Blind application dual writes are dangerous because a request can update one index and fail on the other. Prefer a durable source change stream with independent checkpoints. During cutover, never compare only document counts: duplicates can hide omissions. Reconcile source versions, sampled field values, per-partition counts, deletion counts, and a deterministic digest over IDs/version tuples.

## Concrete failure trace: stale primary resurrects a document

Suppose shard term 17 runs on node A. A network partition isolates A from the control plane but not from one producer. The control plane promotes node B in term 18. A receives `delete(id=9, version=52)` and acknowledges it locally; B receives `upsert(id=9, version=51)` from a delayed replay. Without fencing and source-version checks, both operations can be accepted, and replica recovery can later make version 51 visible again.

The safe design applies two independent guards:

- term 18 replicas reject all term 17 replication and mutation messages;
- the document-version invariant rejects version 51 because version 52 is the greatest accepted source version, including deletion tombstones.

Detection requires more than a healthy-node count. Alert on rejected stale-term operations, decreasing source checkpoints, replica generation divergence, and sampled source-to-index version mismatches. Repair rebuilds the affected logical shard from an authoritative checkpoint; copying A’s latest files would preserve the ambiguity.

## Overload, operations, and observability

Search nodes share CPU, memory, filesystem cache, and device bandwidth among queries, indexing, recovery, and merges. Static thread pools do not provide isolation if every pool drives the same saturated device. Define admission budgets for foreground queries, background indexing, recovery transfer, and merge I/O. When pressure rises, slow indexing, pause optional merges, reject expensive query shapes, or shed low-priority traffic before the node begins timing out everything.

Observe distributions and backlog age, partitioned by index, shard, replica role, query class, and generation:

- accepted-to-visible and source-to-visible lag;
- WAL bytes and age not represented by a commit;
- segment count, bytes by generation, deletion density, and merge debt;
- merge read/write throughput and throttled time;
- shard queue time, service time, fan-out, retries, hedges, and partial responses;
- replica commit/generation lag and recovery ETA;
- filesystem reserve, page-cache misses, checksum failures, and corruption events;
- source/index reconciliation mismatches.

Runbooks should cover a full disk during merge, lost primary, corrupt segment, stuck allocation, runaway expensive query, lagging projection, and rebuild/cutover. Practice them with production-sized snapshots; a recovery procedure tested only on tiny indexes says little about restore time.

## Security and isolation

Indexing is a write capability over a derived data product. Authenticate producers and bind them to specific index/schema versions. Validate document size, field count, nesting, token count, and malformed Unicode before resource-intensive analysis; otherwise one document can become a CPU or memory denial of service.

Search authorization has two safe shapes: route a request only to an index or partition containing allowed data, or enforce an unbypassable authorization filter inside every retrieval path. Post-filtering the returned top-k is unsafe and incomplete: it can leak counts, facets, timing, highlights, or existence, and may return too few authorized results. Treat cached results, query logs, stored fields, snapshots, and debug traces as data copies with the same retention and tenant boundaries as the source.

Control-plane changes need authenticated authorship, review, an immutable audit trail, and staged rollout. A malicious analyzer, synonym set, or schema expansion can exfiltrate data or exhaust a fleet even without executable code.

## Verification strategy

Test invariants at several layers:

- **format tests** generate segments, kill writers at every file boundary, and prove recovery chooses a complete commit;
- **model-based tests** compare random upserts/deletes/replays against a simple map keyed by document ID and source version;
- **replication tests** partition old and new primaries and verify term fencing plus monotonic document versions;
- **merge tests** pin old readers while publishing and reclaiming generations;
- **reconciliation tests** rebuild from a checkpoint and compare ID/version digests;
- **load tests** replay the measured query and mutation mix while merges, relocation, and recovery compete for resources;
- **fault drills** corrupt a segment, fill a disk, delay one replica, and remove a failure domain.

A green query test is not enough. The hard bugs live at lifecycle boundaries: acknowledgement versus durability, refresh versus commit, failover versus fencing, and publication versus reclamation.

## Decision framework

Use a database secondary index when queries are narrow, transactionally coupled to the write, and supported by the database’s index types. Use a dedicated search projection when the workload needs linguistic analysis, broad retrieval, independent scaling, relevance ranking, or rebuildable denormalized documents.

Within a search design, decide in this order:

1. define freshness, durability, partial-result, and recovery contracts;
2. identify the authoritative replay source and version semantics;
3. measure corpus fields, term distributions, update/delete rates, and query fan-out;
4. choose shard boundaries and failure domains;
5. model steady and peak bytes, shard work, merge amplification, and rebuild time;
6. design schema-version rollout and rollback before the first breaking change;
7. verify lifecycle invariants under crash, partition, corruption, and overload.

The index data structure matters, but production correctness comes from the state machine around it.

## References

- [Apache Lucene: Index File Formats](https://lucene.apache.org/core/10_1_0/core/org/apache/lucene/codecs/lucene101/package-summary.html)
- [Apache Lucene: Near Real-Time Search](https://lucene.apache.org/core/10_1_0/core/org/apache/lucene/index/package-summary.html#package.description)
- [Apache Lucene: TieredMergePolicy](https://lucene.apache.org/core/10_1_0/core/org/apache/lucene/index/TieredMergePolicy.html)
- [Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze: Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)
- [Sergey Brin and Lawrence Page: The Anatomy of a Large-Scale Hypertextual Web Search Engine](https://research.google/pubs/the-anatomy-of-a-large-scale-hypertextual-web-search-engine/)
- [Jeffrey Dean and Sanjay Ghemawat: MapReduce: Simplified Data Processing on Large Clusters](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/)
- [Elasticsearch: Reading and Writing Documents](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-replication.html)
- [Elasticsearch: Index and Shard Recovery](https://www.elastic.co/guide/en/elasticsearch/reference/current/recovery.html)
