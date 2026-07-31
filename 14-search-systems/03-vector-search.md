# Vector Retrieval Systems

Vector retrieval searches a learned geometric representation rather than a term vocabulary. The embedding model, preprocessing, distance function, index build, metadata snapshot, deletion state, and query path jointly define the neighborhood; changing any one can change results without changing the API.

Dense retrieval covers vector generation, approximate-nearest-neighbor (ANN) structures, filtered ANN, sharding, recall/latency/cost trade-offs, and embedding/index migration. [Lexical Query Execution](02-full-text-search.md) covers lexical retrieval; [Ranking and Evaluation](04-ranking-algorithms.md) covers candidate fusion, learned reranking, and product-quality evaluation.

## Workload and service contract

Define the retrieval operation in terms a caller can verify:

```text
VectorSearchRequest {
  query_vector | query_object
  embedding_space
  required_filter
  top_k
  candidate_budget
  deadline
  snapshot_policy
}

VectorSearchResponse {
  candidates[] { document_id, distance, source_version }
  embedding_space
  index_generation
  filter_snapshot
  approximation_policy
  partial
}
```

`embedding_space` is a versioned contract containing model artifact digest, preprocessing, output dimension, normalization, and distance function. A vector without that identity is unsafe data: two arrays with the same dimension can belong to incompatible spaces.

Specify separate service objectives for query embedding latency, ANN search latency, end-to-end source-to-vector freshness, deletion visibility, candidate recall against a declared oracle, availability, and rebuild time. ANN recall is a distribution over a workload and parameter set, not a permanent property of the product name.

The caller also needs exact semantics for filters and failure. Does `top_k=20` mean 20 authorized matches if they exist, or up to 20 after an approximate search? Is a timed-out shard omitted, retried, or fatal? Does a deleted object disappear immediately from retrieval, or within a freshness bound? These are API decisions.

## State and invariants

A vector collection contains more than vectors:

| State | Purpose |
|---|---|
| canonical object and source version | authoritative identity and replay |
| embedding vector | learned representation tied to an embedding-space version |
| ANN structure | graph, coarse partitions, compressed codes, or exact matrix |
| metadata/filter index | tenant, authorization, type, time, and product filters |
| liveness state | tombstones and superseded source versions |
| build manifest | training sample, parameters, codebook/model digests, checkpoints |
| serving manifest | shards, replicas, generation, compatibility policy |

Enforce these invariants:

**Space consistency.** A search compares only vectors whose embedding-space and distance contracts are compatible. Compatibility is explicit, never inferred only from dimension.

**Monotonic object version.** A delayed embedding for source version 31 cannot overwrite version 32. A deletion tombstone participates in the same version ordering.

**Filter-before-disclosure.** Unauthorized or tenant-incompatible candidates never appear in results, counts, debug traces, or shared caches. If post-filtering is used internally, the system must keep searching until it satisfies the contract or reports incompleteness; returning forbidden candidates and trimming them at an outer layer is not acceptable.

**Generation closure.** A serving manifest references an ANN structure, metadata snapshot, vector payloads, and tombstones that describe the same logical generation or a documented consistency window.

**Reproducible build.** The manifest records enough input checkpoints, artifact digests, seeds, and parameters to explain and rebuild an index. Approximate structures may differ physically under concurrency, but their lineage must not be mysterious.

## Data plane and control plane

The **data plane** consumes versioned source objects, batches inference, stores embeddings, updates or rebuilds ANN structures, applies tombstones, executes filtered searches, and returns candidates. Online query embedding is part of this path when callers send raw objects.

The **control plane** registers embedding spaces, approves model artifacts, chooses ANN family and parameters, assigns shards, publishes manifests, coordinates backfills and dual reads, and manages tenant quotas. It should publish immutable versioned snapshots; serving a last-known-good manifest is safer than partially applying a model or codebook update.

The indexing flow is normally asynchronous:

```text
source commit
  -> durable change event(object_id, source_version, payload_ref)
  -> preprocessing and embedding inference
  -> immutable embedding record(space, object_id, source_version, vector)
  -> ANN/index mutation or batch build
  -> metadata and tombstone reconciliation
  -> atomic serving-generation publication
```

Inference and indexing have independent retry semantics. Record a deterministic work identity such as `(space, object_id, source_version)`. A retry may recompute the vector, but only the greatest valid source version becomes live.

## Exact search as oracle and fallback

For `N` vectors of dimension `D`, exact dense search computes every distance, roughly proportional to `N * D`. It is simple, supports arbitrary post-hoc filters if the filtered set is known, and provides the ground truth used to measure ANN candidate recall. SIMD, accelerators, and batching can make exact search competitive for small collections or large offline batches.

Maintain an exact evaluation path over a representative sample even when production uses ANN. Without an oracle, a latency improvement can silently be a recall regression. Exact search can also be the operational fallback for a small tenant, a newly created partition, or a heavily selective filter whose eligible set is tiny.

Distance choice is part of model training and serving. Cosine similarity on normalized vectors is equivalent in ordering to dot product; Euclidean distance is not interchangeable with either for arbitrary vectors. Normalize consistently and test it. An accidental query-only normalization changes neighborhoods while every request remains syntactically valid.

## ANN index families

### Navigable proximity graphs

HNSW constructs a multilayer graph. Upper sparse layers guide a query toward the relevant region; the dense bottom layer performs a best-first neighborhood exploration. The main controls are graph connectivity, construction search effort, and query exploration effort. Increasing them generally consumes more build time, memory, or query work in exchange for better recall, but the response depends on the corpus.

Graph indexes provide strong low-latency single-query performance when the graph and vectors fit the memory hierarchy. Costs include pointer-heavy memory, expensive construction, difficult physical compaction, and deletion/update maintenance. Inserts alter graph topology; a steady stream of tombstones can degrade routing before live-vector percentage looks alarming. Periodic rebuild is part of the lifecycle.

### Inverted file indexes

IVF trains coarse centroids, assigns each vector to one or more cells, then searches the most promising cells. Query work depends on probed cells and their population. Skew matters: a centroid containing a disproportionate share of traffic or vectors becomes a latency hot spot. Training samples must represent the serving distribution, including languages, tenants, and new catalog regions.

IVF is naturally batch-oriented and works well with compression, but new distributions can outgrow the codebook. Measure residual error and per-cell population over time. “Same number of centroids” does not mean the same quality after a corpus shift.

### Product quantization and compressed search

Product quantization divides a vector into subspaces and represents each with a learned codeword. Search uses compact codes and precomputed distance tables; optional reranking reads full-precision vectors for the best compressed candidates. Compression reduces memory and bandwidth while introducing quantization error.

The full contract includes subspace layout, codebooks, training data, distance convention, and whether original vectors remain available. Codebooks are generation-specific. Reusing codes with a different codebook is silent corruption, not a low-quality fallback.

### Storage-aware structures

Disk-oriented graph designs keep a compact navigation representation in memory and fetch candidate vector pages from local SSD. They trade memory for I/O and rely on locality, caching, and bounded random reads. This can make a larger corpus feasible, but tail latency now depends on device queues and cache temperature. Recovery traffic and compaction can contend directly with queries.

Choose an ANN family from measured batch size, update rate, filter distribution, vector count and dimension, memory budget, local storage, target recall, and rebuild window. Corpus size alone is not a decision rule.

## Filtered ANN is a query-planning problem

A metadata filter can be applied in several places:

1. **Pre-filter**: enumerate eligible IDs, then exact-search or use a filter-aware ANN structure.
2. **Traversal filter**: allow graph navigation through all nodes but admit only eligible nodes to results.
3. **Strict traversal restriction**: traverse only eligible nodes, which can disconnect the graph and destroy recall.
4. **Post-filter**: retrieve ANN candidates and discard ineligible ones, increasing the candidate budget until enough remain.
5. **Physical partition**: build separate indexes by tenant, region, category, or time.

No one strategy dominates. If 50% of vectors are eligible, post-filtering may be fine. If 0.001% are eligible, it may inspect an enormous candidate set and still return too few. Physical partitioning gives strong isolation and narrow search but creates tiny indexes, skew, and operational overhead. Filter-aware graphs add complexity and can couple index shape to changing metadata.

Estimate selectivity before choosing a path. Use filter-index cardinalities, per-partition stats, and learned execution feedback. For sparse eligible sets, exact search over enumerated IDs may beat ANN. The planner should expose whether the returned set is complete under the chosen candidate budget.

Authorization is not an ordinary optional filter. A system may traverse forbidden nodes internally if its isolation model allows it, but it must not expose their payloads, IDs, distances, counts, cache entries, or trace attributes. For strict tenant boundaries, physically separate indexes and encryption domains may be simpler to audit.

## Sharding and distributed top-k

Hash sharding balances vector counts and broadcasts each unconstrained query. Semantic partitioning routes to a subset of clusters but risks routing error: the true neighbor may live in a cell the router does not probe. Tenant partitioning narrows authorization and blast radius but inherits tenant skew. Time partitioning helps retention and freshness but makes historical queries fan out.

Each shard returns local candidates; the coordinator merges distances only when they share an embedding space, distance definition, and score transformation. If some shards run the old space and others the new one, raw distance comparison is invalid. Dual-space migration should query and rank each space separately, then compare through a controlled fusion or shadow evaluation, not mix vectors in one ANN structure.

Distributed ANN compounds approximation:

- the router may omit a relevant shard;
- the shard ANN may omit a neighbor;
- per-shard candidate truncation may omit a globally top candidate;
- filtering may discard candidates after the budget is spent;
- a downstream reranker cannot recover any omitted object.

Measure recall at each boundary. A single end-to-end metric diagnoses little.

## Embedding and index migration

An embedding-model update is a full data migration even when the output dimension stays constant. Use distinct immutable spaces, such as `catalog_title@7` and `catalog_title@8`:

1. register `v8` with artifact digest, preprocessing, dimension, normalization, distance, and access policy;
2. embed a frozen evaluation corpus and compare exact-neighbor structure plus downstream quality;
3. backfill source objects at pinned versions while streaming later changes into both spaces;
4. build and validate a complete `v8` ANN generation;
5. shadow production queries, recording candidate overlap, exact recall, latency, filter completeness, and downstream outcome metrics;
6. canary reads by stable, domain-separated assignment;
7. switch the serving policy atomically;
8. retain `v7` for a bounded rollback period, then stop dual indexing and delete it according to retention policy.

Do not update vectors in place while serving the old model. A half-migrated space has no coherent geometry. Dual writes from request handlers are also fragile; project both spaces from one durable source stream with independent checkpoints.

Deletion must cover every active and rollback space. Privacy erasure workflows need a registry of derived locations plus proof of completion; retiring `v7` later cannot be the first time its forgotten vector is discovered.

## Capacity and cost model

Use measurements from the actual vector and filter distributions. Consider an illustrative collection:

- 120 million live objects;
- 768 float32 dimensions;
- two serving replicas;
- graph/index overhead measured at 45% of raw vector bytes;
- 15,000 peak searches/s, broadcast to 12 shards;
- average measured shard CPU service time 1.1 ms at the selected recall point;
- 3,000 object updates/s;
- 25% temporary space reserve for tombstones and rebuild activity.

Raw vectors require `120M * 768 * 4` bytes, about 343 GiB. With measured 45% graph/metadata overhead, one corpus copy is about 497 GiB. With 25% lifecycle reserve, it is about 621 GiB per replica, or 1.21 TiB for two replicas before snapshots and embedding records. If original vectors are stored separately for reranking, include them rather than assuming the ANN footprint already covers both.

Broadcast traffic is `15,000 * 12 = 180,000` shard searches/s. At 1.1 ms mean measured CPU time, demand is 198 CPU-seconds/s. At 50% target utilization, plan about 396 logical cores for this component before query embedding, coordination, filtering, and failure reserve. If a selective-filter path increases exploration by eight times at p99, admission and per-class pools must reflect that distribution.

Embedding throughput must exceed updates plus backfill. If one inference worker sustains a measured 320 objects/s at the chosen batch/shape, steady updates need at least 10 fully utilized workers. At a safer 60% target utilization they need 16. A 120-million-object migration completed in seven days requires an additional average `120M / 604,800`, about 199 objects/s, plus retry and skew margin. The model shows whether migration is compute- or index-build-bound; benchmark both.

## Concrete failure trace: mixed embedding spaces

A team deploys model `v8`, which happens to emit the same 768 dimensions as `v7`. The online query service switches first, while 70% of stored vectors remain `v7`. The ANN API validates only dimension, so it accepts `v8` queries against a geometrically mixed graph. Latency and health checks remain green, but relevant-item recall collapses and differs by shard according to backfill progress.

The containment action is to route queries back to the complete `v7` manifest. Repair builds `v8` as a separate physical generation, rejects cross-space writes at the storage boundary, and activates it only after reconciliation proves source-version coverage. Telemetry must attach embedding space to query, vector record, index generation, cache key, and trace; otherwise the incident looks like an unexplained relevance shift.

## Operations, observability, and repair

Observe each embedding space and filter class separately:

- source-to-embedding and embedding-to-searchable lag;
- embedding failures, batch utilization, model digest, and input-shape distribution;
- live vectors, tombstones, stale source versions, and unindexed records;
- graph degree/connectivity, IVF cell population, codebook residual error, and rebuild age;
- candidates visited, pages read, filter selectivity, post-filter survival, and incomplete top-k rate;
- exact-versus-ANN recall on a stable and a fresh evaluation sample;
- query embedding, queue, ANN, filter, fetch, and coordinator latency;
- shard fan-out, partial results, retries, cache temperature, and device queue depth;
- migration coverage and dual-space disagreement.

Repair APIs should be first-class: re-embed one object/version, reconcile a partition, rebuild a shard from a source checkpoint, verify a manifest, drain a corrupt replica, and prove deletion across spaces. Avoid ad hoc mutation of graph internals; rebuild from immutable vector records when possible.

Autoscaling solely on QPS is weak because work varies with exploration parameters, filter selectivity, batch size, and cache state. Scale or admit from queue time, visited-node/page distributions, device and CPU saturation, and freshness backlog. Recovery and rebuild need explicit bandwidth budgets so they do not erase query headroom.

## Security and privacy

Embeddings can retain sensitive attributes and support inference or membership attacks; they are not anonymous merely because humans cannot read them directly. Apply source-equivalent access control, encryption, retention, residency, backup, and deletion policy. Minimize payloads used for embedding and record their provenance.

Authenticate model and codebook publication. Verify artifact digests and restrict which service can write each embedding space. An attacker who can insert many crafted vectors can manipulate neighborhoods or exhaust graph construction. Rate-limit mutations, validate dimensions and finite numeric values, cap object fan-out, and monitor distribution drift and anomalous density.

Cache keys include tenant, embedding-space version, normalized request/input digest, filter policy, and index generation where consistency requires it. A shared cache keyed only by query text is both semantically wrong and a data-isolation risk.

## Verification strategy

- **Contract tests** reject dimension-compatible but space-incompatible vectors and stale source versions.
- **Oracle tests** compare ANN with exact search on stratified query sets, reporting recall by locale, tenant, filter selectivity, and freshness.
- **Property tests** verify metric/normalization behavior and deterministic manifest validation.
- **Filter tests** compare every execution strategy with exact eligible-set search and probe all disclosure channels.
- **Migration tests** replay concurrent update/delete events during backfill and prove both spaces converge to the same source-version set.
- **Fault tests** kill builders before publication, corrupt codebooks, delay metadata snapshots, partition shards, and exhaust storage.
- **Load tests** combine production query batches, rare selective filters, ingestion, rebuild, and cold-cache recovery.
- **Drift tests** monitor embedding norms, nearest-neighbor distances, cell populations, and downstream judgments across model and corpus versions.

The benchmark artifact must include the dataset/checkpoint, query sample, filter distribution, hardware, concurrency, build and search parameters, and exact oracle definition. A recall/latency point without those details is not portable evidence.

## Decision framework

Use vector retrieval when semantic or multimodal similarity adds measurable candidate coverage that lexical and structured retrieval cannot provide. Do not add it solely because embeddings are available.

Choose the architecture in this order:

1. define embedding-space identity, source versioning, deletion, and authorization semantics;
2. build exact-search quality and performance baselines;
3. measure vector count, dimension, update rate, filter selectivity, query batching, and recall needs;
4. benchmark multiple ANN families on the real distribution and hardware;
5. model memory, CPU/I/O, fan-out, rebuild time, and dual-space migration reserve;
6. design filtered execution and incomplete-result semantics;
7. automate recall, drift, lineage, and source/index reconciliation;
8. prove a model migration and rollback before relying on continuous updates.

Approximation is safe when its error is measured, scoped, and reversible. Unversioned geometry is not.

## References

- [Yu. A. Malkov and D. A. Yashunin: Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://arxiv.org/abs/1603.09320)
- [Hervé Jégou, Matthijs Douze, and Cordelia Schmid: Product Quantization for Nearest Neighbor Search](https://doi.org/10.1109/TPAMI.2010.57)
- [Jeff Johnson, Matthijs Douze, and Hervé Jégou: Billion-scale Similarity Search with GPUs](https://arxiv.org/abs/1702.08734)
- [Ruiqi Guo et al.: Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://proceedings.mlr.press/v119/guo20h.html)
- [Suhas Jayaram Subramanya et al.: DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)
- [FAISS Documentation](https://faiss.ai/)
- [Apache Lucene: Vector Search](https://lucene.apache.org/core/10_1_0/core/org/apache/lucene/search/knn/package-summary.html)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
