# Column-Oriented Storage

Column-oriented storage makes one physical bet: analytical queries usually touch many rows but only a subset of their attributes. Storing values from the same column together reduces projected I/O, exposes homogeneous data to encoding, and lets execution operate on vectors rather than repeatedly interpreting row objects.

This chapter owns the **analytical physical layout and scan path**: row groups, column chunks and pages, encodings, metadata pruning, late materialization, vectorized execution, update overlays, and scan capacity. [Data Encoding](./07-data-encoding.md) owns general serialization rules. [SSTables and Compaction](./03-sstables-compaction.md) owns immutable sorted-run lifecycle, and [Lakehouse Table Formats](../13-data-pipelines/05-lakehouse-table-formats.md) owns multi-file table snapshots and metadata protocols.

## Workload and correctness contract

Columnar layout is strongest when:

- scans touch a small fraction of a wide schema;
- predicates and aggregations evaluate over many rows;
- inserts arrive in batches or can be buffered;
- updates/deletes can use deltas or background rewrite;
- physical clustering lets metadata exclude substantial data;
- CPU-efficient decode and vector execution matter as much as storage size.

It is weaker for single-row fetches that need most columns, high-rate in-place updates, and latency dominated by one random lookup. Hybrid systems commonly keep a row-oriented write path or primary store and create columnar segments for scans.

The file contract must preserve more than values:

- one logical row position aligns across every projected column;
- null and nested structure are reconstructable;
- schema evolution does not reinterpret old fields;
- statistics and filters never exclude a row that could match;
- updates/deletes are applied at the correct snapshot;
- corrupt pages are detected rather than silently decoded.

Fast wrong pruning is worse than a slow scan. Metadata participates in query correctness.

## Physical state: row groups, chunks, and pages

A columnar file is horizontally divided into **row groups**. Within one row group, each field has a **column chunk** containing pages. This creates two locality dimensions:

- row groups bound parallelism, pruning, and rewrite units;
- column chunks let projection read only required attributes.

Pages provide independently encoded/compressed units and may carry checksums, value counts, null counts, encoding metadata, and page-level statistics. A file footer records schema, row-group locations, column-chunk offsets/sizes, codec information, and optional statistics/index locations.

The values at logical position `i` across chunks belong to the same row. Columns can use different page boundaries; readers align by logical value/row counts, not by assuming page 7 of every column begins at the same byte or row.

### Null and nested data

Flat fixed-width arrays often use a validity bitmap: one bit says whether each position is present. Nested records and repeated fields need more structure. Dremel's encoding represents repetition and definition levels so a reader can distinguish an absent optional field, an empty list, and values in repeated nested records without materializing the full tree first. Parquet inherits this model.

Schema fields need stable identity and repetition/nullability semantics. Matching only by ordinal is dangerous when columns are inserted or reordered; matching only by name is dangerous when names are reused. A migration plan should define field IDs, aliases, defaults, type promotions, and whether old readers may encounter new required fields.

## Encoding is chosen per data distribution

Column locality creates opportunities, but no encoding wins for every page.

- **Dictionary encoding:** store distinct values once and encode rows as integer codes. Effective for repeated values; high cardinality can make the dictionary larger than direct encoding.
- **Run-length encoding:** store `(value, run_length)` for repeated adjacent values. Physical sort order often creates long runs.
- **Bit packing:** use only the bits needed for a bounded integer/code range.
- **Delta and frame-of-reference:** store differences from a prior value or page base for timestamps and locally clustered numbers.
- **Boolean/validity bitmaps:** compact flags and support word/SIMD operations.
- **General compression:** compress encoded page bytes with codecs such as LZ4, Snappy, or Zstandard.

Writers may sample a page and fall back when a dictionary overflows or an encoding expands data. The choice trades compressed bytes, encoder CPU, decoder CPU, random access, and predicate execution. A heavier codec can reduce object-store/network time and still make the query faster; on cached data it may make CPU the bottleneck.

Dictionary codes are normally local to a page or chunk. Code 7 in one row group need not equal code 7 in another. Join, grouping, or comparison across dictionaries must compare values, unify dictionaries, or carry dictionary identity.

## The analytical scan protocol

A well-designed scan discards work in stages:

1. **Table/partition pruning:** use the table snapshot and partition metadata to choose files.
2. **Footer discovery:** fetch file metadata and identify required column chunks.
3. **Row-group pruning:** evaluate conservative min/max, null count, Bloom filters, and dictionaries.
4. **Page pruning:** use page indexes/statistics where present.
5. **Predicate decode:** decode filter columns into vectors and build a selection bitmap/vector.
6. **Late materialization:** decode projected columns only for surviving rows when the format/reader supports efficient selective access.
7. **Vector execution:** pass batches through filters, joins, aggregates, and projections.
8. **Final materialization:** construct output rows only at the API boundary that requires them.

Projection pushdown saves bytes only when the reader does not fetch unneeded chunks. Predicate pushdown saves work only when metadata or encoded execution can reject pages/rows before full decode.

### Statistics must be conservative

For predicate `price > 100`, a row group with exact maximum 90 is safely skipped. A group with `[min=20, max=500]` must be scanned even if only one row may match.

Statistics implementations must define nulls, NaNs, signed zero, decimal scale, timestamp timezone, collation, and string truncation. A truncated upper string bound that is smaller than an omitted real value can create a false exclusion. When bounds are incomplete or untrusted, the reader must scan. Bloom filters may return false positives but must not return false negatives for the encoded values and normalization rules.

Physical clustering determines pruning power. If data is ordered by `(tenant_id, event_time)`, tenant-time queries may touch few row groups, while a global time query can overlap many tenants. No one sort order serves every predicate. Multi-dimensional clustering, duplicated projections, or secondary data structures exchange storage/rewrite cost for broader pruning.

## Vectorized execution

A row-at-a-time interpreter repeatedly resolves field offsets, branches on types/nulls, dispatches an operator, and produces one result. A vectorized engine processes a batch of values with one typed operator:

```text
input vectors -> predicate -> selection vector -> aggregate/join -> output vectors
```

This improves instruction-cache locality, amortizes dispatch, enables branch-light loops and SIMD, and keeps compact arrays in CPU caches. Operators can pass a selection vector or bitmap rather than copying every surviving row. Predicates may execute on dictionary codes after evaluating the dictionary once; aggregates may consume encoded runs without expanding every value.

Batch size is a cache and latency decision. Larger batches amortize calls but consume more working memory and delay the first result; smaller batches increase dispatch overhead. Choose from measured operator and cache behavior rather than a universal row count.

Apache Arrow standardizes an in-memory columnar representation with contiguous buffers, validity bitmaps, offsets for variable-length values, and nested-array layouts. Parquet and ORC optimize persisted size and skipping; Arrow optimizes interoperable in-memory access. A reader often decodes a storage page into Arrow-like vectors, but engines may use their own vector format or execute directly on encoded pages. “Parquet becomes Arrow” is a common architecture, not a correctness requirement.

### Early versus late materialization

Early materialization reconstructs rows near the scan and lets conventional row operators consume them. It can be efficient when most rows and columns survive. Late materialization carries positions/selections deeper and fetches attributes only when needed; it excels for selective predicates and wide schemas but introduces position mapping and potentially scattered gathers.

The correct boundary depends on selectivity, projected widths, join shape, and whether the storage format supports efficient page/position access. Optimizers should cost both rather than treating late materialization as always faster.

## Inserts, updates, and deletes

Appending one row into separate durable streams for hundreds of columns is inefficient and difficult to publish atomically. Column stores buffer rows, sort/cluster a batch, transpose it into columns, encode pages, and publish a complete segment. The buffer and spill path is therefore part of ingestion capacity.

Updates typically use one of three contracts:

- **copy-on-write:** rewrite the affected row group/file and atomically replace it in the table snapshot;
- **merge-on-read:** write new row versions or equality/position deletes separately and merge them during scans;
- **delta store:** keep recent mutable rows in a row/delta structure and periodically convert them into columnar segments.

A position delete must bind to the exact immutable data-file identity and row position. Applying it to a rewritten file with shifted positions deletes the wrong row. Equality deletes require matching schema, type, and normalization semantics. Readers choose data and delete files from one table snapshot so a concurrent rewrite cannot mix generations.

Merge-on-read reduces write amplification and increases scan merge work. Copy-on-write makes reads simpler and turns small updates into large rewrites. Background maintenance eventually folds deltas/deletes into new files; its transactional multi-file publication belongs in [Lakehouse Table Formats](../13-data-pipelines/05-lakehouse-table-formats.md), while immutable run replacement inside an LSM belongs in [SSTables and Compaction](./03-sstables-compaction.md).

## Concrete failure and correctness traces

### Positional schema evolution swaps fields

1. Old files store fields `[id, country, amount]` by ordinal.
2. A new schema inserts `currency` before `amount`.
3. An old reader or ordinal-only reader interprets currency bytes as amount.

Stable field identity and compatible readers must precede writer cutover. A renamed/reordered display schema must not rewrite physical meaning implicitly.

### Unsafe min/max causes false pruning

1. A writer truncates long string maximum `mango…` to `mang` without marking it as a lower/incomplete bound.
2. A reader evaluates `value = 'mango'` and concludes the row group maximum is below the predicate.
3. It skips a group that actually contains the value.

Statistics serialization and comparison need cross-version conformance tests. Unknown or truncated bounds must remain conservative.

### Dictionary identity is lost

1. Row group A maps code 3 to `US`; row group B maps code 3 to `DE`.
2. A grouping operator concatenates code vectors and groups by integer code alone.
3. Different countries merge into one group.

Keep dictionary identity, unify dictionaries, or decode before cross-chunk comparison.

### Delete vector follows the wrong rewrite

1. Delete file D references row position 9 in data file F.
2. Compaction rewrites F into G with a different row order.
3. A bad snapshot publishes G but retains D as if it referenced G.
4. The reader deletes an unrelated row or fails to delete the intended one.

File content identity and snapshot-level replacement must couple data and position deletes.

### Partial object publication

1. A writer uploads several columnar files and fails halfway.
2. A reader discovers files by listing a directory/prefix.
3. It observes only part of a logical batch and double-reads files left by a retry.

Immutable objects need a transactional manifest/snapshot as table authority. Object listing is not an atomic commit protocol.

### Small-file metadata dominates

1. A streaming writer emits thousands of tiny files per minute.
2. A query performs object listings, opens, footer range reads, decompressor setup, and task scheduling for each.
3. It spends more time in metadata and fixed per-file work than scanning values.

Batching and compaction must be sized as part of the ingest/query cost model.

## Capacity and cost model

For a scan, estimate bytes by the actual chunks that survive pruning:

```text
scan_bytes
  ~= footer_and_index_bytes
   + sum(compressed selected-column chunks in surviving row groups)
   + delete/delta metadata
```

If `R` rows survive file/row-group pruning, projected uncompressed width is `W_p`, compressed bytes are `B_c`, storage bandwidth is `S`, decode throughput is `D`, and operator throughput is `E`, a lower-bound pipeline time is governed by the slowest resource:

```text
time >= max(B_c / S,
            decoded_bytes / D,
            R / E,
            network_result_time)
```

Parallelism can overlap stages until storage requests, memory bandwidth, CPU, or network saturates. Compression ratio alone does not predict latency.

Vector working memory is approximately batch rows times the widths of decoded columns, plus variable-length buffers, validity, selection, hash tables, join build state, and operator outputs. Multiple pipeline stages and concurrent tasks multiply it. Enforce per-query/task memory limits with spill; an underestimated variable-length column can otherwise turn projection into an out-of-memory failure.

Row-group size trades:

- larger groups: better sequential I/O, bigger encoding samples, fewer footers/tasks;
- smaller groups: finer pruning, more parallel units, cheaper copy-on-write;
- too large: one weak min/max range forces a large scan and update rewrite;
- too small: metadata, seeks/range requests, codec setup, and scheduling dominate.

Choose with the storage request latency, target concurrency, sort distribution, update size, and expected predicate selectivity. There is no universal row count or byte target.

For object storage, fixed cost matters:

```text
metadata_time ~= file_count * (open/range-request/footer/scheduling cost)
```

Even when requests run concurrently, service limits and coordinator CPU cap useful fan-out. Track files and row groups scanned, bytes read before/after pruning, bytes decoded, rows selected, and CPU per stage.

Clustering maintenance is a write cost. If ingest arrives out of order, row-group min/max ranges widen and pruning decays. Re-sorting rewrites data and needs temporary space. Model it as a recurring background workload with an SLO, not a one-time `ORDER BY` choice.

## Production operation and migration

Observe pruning by layer (file, row group, page), projected versus fetched columns, compressed bytes read, decoded bytes, rows before/after filters, codec CPU, vector batch occupancy, materialization/gather cost, spill, footer/cache hit rate, files per query, delete/delta density, and clustering quality. A query can read few bytes yet be CPU-bound on pathological decoding or nested reconstruction.

Use representative workload replay before changing sort order, row-group size, codec, dictionary policy, or page indexes. Compare not only compression but p50/p99 scan time, ingest CPU, rewrite cost, memory, and affected query classes. Backfill into new files and atomically switch table metadata; keep old readers compatible until rollback expires.

Schema migration should maintain stable field IDs, explicitly allow type promotions, and test old/new writer-reader combinations. Validate statistics against a full scan and fuzz nulls, NaNs, extreme decimals, Unicode/collation, nested empty versus null collections, dictionary fallback, page boundaries, and corrupted checksums. Differential tests should compare vectorized/encoded execution with a simple row interpreter for randomly generated predicates.

Fault tests should interrupt multipart upload, footer write, snapshot publication, delete-file replacement, and compaction. A reader must see either the old complete snapshot or the new complete snapshot—never a directory-shaped mixture.

## Decision framework

1. What percentage of rows and columns does each dominant query actually touch after pruning?
2. Which physical sort/clustering order serves those predicates, and which workloads does it harm?
3. What row-group/page size balances pruning, request overhead, parallelism, and rewrite cost?
4. Which encodings reduce end-to-end time after decoder CPU and cache behavior?
5. Where does materialization occur, and what selectivity makes that boundary worthwhile?
6. How are updates/deletes associated with the correct file generation and snapshot?
7. Can ingestion, clustering, delete folding, and small-file compaction keep up with steady state plus bursts?
8. Are schema IDs, statistics, and vectorized results cross-version and differentially tested?

## Primary references

- [Stonebraker et al., *C-Store: A Column-oriented DBMS* (VLDB 2005)](https://www.vldb.org/conf/2005/papers/p553-stonebraker.pdf)
- [Boncz, Zukowski, and Nes, *MonetDB/X100: Hyper-Pipelining Query Execution* (CIDR 2005)](https://www.cidrdb.org/cidr2005/papers/P19.pdf)
- [Abadi, Myers, DeWitt, and Madden, *Materialization Strategies in a Column-Oriented DBMS* (ICDE 2007), author-hosted PDF](https://www.cs.umd.edu/~abadi/papers/abadiicde2007.pdf)
- [Melnik et al., *Dremel: Interactive Analysis of Web-Scale Datasets* (VLDB 2010)](https://storage.googleapis.com/gweb-research2023-media/pubtools/3293.pdf)
- [Google, *Capacitor: Columnar Storage for the Cloud* (VLDB 2016)](https://research.google/pubs/capacitor-columnar-storage-for-the-cloud/)
- [Apache Parquet, official file-format specification](https://parquet.apache.org/docs/file-format/)
- [Apache Arrow, official columnar-format specification](https://arrow.apache.org/docs/format/Columnar.html)
