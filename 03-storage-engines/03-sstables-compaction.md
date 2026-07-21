# SSTables and Compaction

An SSTable is an immutable sorted run plus the metadata needed to search and validate it. Immutability makes publication, caching, replication, and recovery tractable; it also means updates and deletes accumulate as newer records. Compaction is the background protocol that rewrites overlapping runs into a new authoritative set without losing snapshots, resurrecting deleted values, or exposing half-built files.

Scope: the **immutable sorted-run lifecycle**: file format state, publication, lookup across runs, compaction selection and commit, version/tombstone collection, and compaction capacity. [LSM Trees](./02-lsm-trees.md) owns the complete write path from WAL and memtable through the LSM architecture. [Bloom Filters](./05-bloom-filters.md) owns filter design. [Column-Oriented Storage](./06-column-storage.md) owns analytical column layout, pruning, and vectorized execution; a Parquet row group is not an SSTable level merely because both may be immutable.

## Workload and storage contract

Sorted runs fit workloads where random updates can first become sequential output and reads can tolerate consulting multiple immutable components. The file layer should promise:

- keys ordered by one stable byte comparator;
- checksummed blocks and metadata;
- an index that finds candidate blocks without scanning the file;
- optional membership filters that have no false negatives for the represented key domain;
- snapshot-aware version ordering;
- atomic publication of a new live-file set;
- recovery that distinguishes live files, obsolete files, and incomplete output.

The format is not one universal diagram. Bigtable SSTables, LevelDB tables, RocksDB block-based tables, and other engines differ in block indexes, filters, compression, range deletions, properties, and footer versions. The invariant is that readers can interpret a fully published immutable file and the manifest can name exactly which files form one database version.

## File and manifest state

A block-based sorted table commonly contains:

- sorted data blocks with prefix-compressed keys and values;
- restart points or another mechanism for local binary search;
- a block index mapping separator keys to block handles;
- filter and properties blocks;
- per-block checksums and compression identifiers;
- a meta-index and fixed footer locating top-level metadata.

File metadata held by the version/manifest normally includes file number or object ID, byte size, smallest and largest key, smallest and largest sequence number, level/run identity, checksum or unique ID, and format version.

Many LSM engines sort an **internal key** rather than only the application key:

```text
(user_key ascending, sequence descending, record_kind)
```

For one user key, newer versions appear first. A point tombstone is another record kind. A reader at snapshot sequence `S` chooses the newest version whose sequence is at most `S`; versions newer than the snapshot are invisible. Exact encoding differs by engine, but comparator stability is non-negotiable. Changing it can make already-sorted files unreadable or cause two components to disagree about overlap.

The manifest is the authority over file membership. A directory listing is not: it can contain incomplete outputs, obsolete inputs retained for readers, temporary files, logs, and files from a failed compaction.

## Flush and publication protocol

A safe memtable flush follows a state transition:

1. freeze a sorted memtable and assign its immutable sequence range;
2. stream entries into one or more temporary table files;
3. finish indexes, filters, properties, footer, and checksums;
4. make the output durable under the filesystem/object-store contract;
5. atomically append and persist a manifest/version edit that adds the files;
6. expose the new version to readers;
7. release the covered WAL only when every recovery path can reconstruct the data.

The ordering handles both crash sides. A complete file not named by a durable manifest is an orphan and can be collected. A manifest that names output whose bytes or footer were not made durable can make recovery fail or silently lose keys. On local filesystems, rename and file sync do not necessarily persist the parent directory entry; implementations must follow their actual filesystem contract. On object storage, an immutable upload plus a separately committed manifest avoids relying on directory rename semantics.

Readers pin a version of the manifest. Publishing a new version does not invalidate an in-flight reader's old version. Obsolete files are physically removed only after no pinned version, snapshot, iterator, backup, or checkpoint can reference them.

## Read protocol across runs

### Point lookup

A point read first checks mutable/immutable memtables, then candidate SSTables in recency order consistent with the engine's level rules. For each file it can:

1. reject the key outside the file's smallest/largest range;
2. ask a Bloom or partitioned filter whether the key may occur;
3. use the top-level index to locate a data block;
4. verify/decompress the block and seek the internal key;
5. stop when it finds the newest visible value or tombstone whose placement rules prove older files cannot win.

Level 0 files often overlap and may need to be searched newest first. In a leveled organization, files within levels 1 and below usually have non-overlapping user-key ranges, so at most one file per such level is a candidate. Filters reduce negative disk reads but do not remove index/filter-cache or CPU cost.

### Range scan

A range iterator performs a k-way merge over all overlapping memtables and table iterators. It orders internal keys, suppresses shadowed versions, applies point/range tombstones, and respects the snapshot. For `E` visited entries across `k` runs, a heap-based merge has roughly `O(E log k)` comparison work; blocked/loser-tree variants change constants. When many obsolete versions survive, physical entries visited can greatly exceed logical rows returned.

Range scans make compaction quality visible: filters do little when the requested range really overlaps every run. Read-ahead, block cache, compression, and the number of overlapping runs determine throughput.

## Compaction as a versioned transaction

A compaction is not “merge files and delete the old ones.” It is an atomic metadata replacement:

1. choose an input set whose overlap closure satisfies the strategy;
2. pin that input version and reserve output file IDs;
3. merge entries in comparator order;
4. apply snapshot/version/tombstone retention rules;
5. split output at target boundaries without splitting an indivisible record incorrectly;
6. finish, checksum, and durably publish every output file;
7. persist one manifest edit that adds outputs and removes inputs;
8. expose the new version;
9. delete inputs only after old references drain.

If the worker crashes before step 7, outputs are unreferenced garbage. If it crashes after the durable edit, recovery uses outputs and treats inputs as obsolete. The manifest edit is the commit point.

Compaction may also update per-file sequence bounds, range-deletion metadata, blob/value-log references, and encryption/checksum metadata. Every side structure must change consistently with the manifest or retain a recoverable reconciliation path.

## Choosing a compaction shape

### Leveled compaction

Level 0 accepts overlapping flushes. Lower levels have increasing byte targets and non-overlapping key ranges within each level. When a level exceeds its target or creates too much overlap, selected files merge with all overlapping files in the next level.

Leveled compaction bounds point-read candidates and usually keeps space amplification low. It repeatedly rewrites keys as they descend and can incur high write amplification, especially when a small incoming range overlaps a large amount of next-level data. Compaction pointers, overlap-aware selection, trivial moves, and subcompactions distribute that work but do not remove it.

### Tiered, size-tiered, and universal compaction

Tiered policies accumulate similarly sized sorted runs and merge several into a larger run. Universal compaction is a flexible run-based form used by RocksDB. Keys are typically rewritten fewer times on the write path, while more overlapping runs remain visible to reads and transient/live space can be larger.

This suits write-heavy or short-lived data when read amplification and spare space are affordable. A giant final merge can create severe temporary space and I/O demand; run count and size ratio need explicit caps.

### FIFO and time-window policies

FIFO drops old files rather than merging the full key space. Time-window policies compact data within time windows and expire whole old windows. They are powerful only when retention semantics align with file boundaries and late updates/deletes cannot require an expired file.

If a late write for an old timestamp lands in a new file, file creation time may not equal data expiry time. If a tombstone expires before the older value file, deletion can resurrect. The policy must use trustworthy per-file time bounds and a late-arrival contract, not merely file names.

### Hybrid policies

Real engines may use tiering near the write frontier and leveling below it, different compression by level, or workload-specific compaction per column family. The correct unit of choice is a workload with measured point reads, scans, overwrites, deletes, TTL, value size, and available background resources.

## Version and tombstone collection

Compaction sees several versions of one key, but it can discard one only if no supported reader or unselected file can need it.

An old value can be removed when a newer visible value shadows it for every retained snapshot. A point tombstone can be dropped only when:

1. no retained snapshot may need to observe the deletion event or an older value;
2. the compaction covers every place an older value for that key could exist, or level metadata proves none exists below;
3. replication/backup/restore policy will not reintroduce an older value outside this local file set.

“At the bottom level” is a sufficient shortcut in some designs, not the general proof. An engine can sometimes drop earlier with complete range-coverage metadata; it must sometimes retain at the bottom for snapshots or external replication semantics.

Range tombstones are harder because one deletion interval may overlap many output files and snapshots. Implementations fragment or replicate tombstone metadata at output boundaries and must preserve ordering against point keys. A boundary error can expose values in only part of a deleted range.

TTL expiry has the same shape. Treat expiration as a versioned deletion under the database's time semantics, and decide whether historical snapshots may read pre-expiry values.

## Concrete failure traces

### Manifest references non-durable output

1. Compaction writes `out-91.sst` into buffered filesystem state.
2. It durably appends a manifest edit replacing inputs with `out-91` but never syncs the file.
3. Power fails; the manifest survives, while some output blocks do not.
4. Recovery trusts the manifest and encounters missing/corrupt committed state.

Output durability must precede manifest commit.

### Inputs deleted while a reader pins the old version

1. Reader R opens manifest version 30 and begins a long scan of files A and B.
2. Compaction publishes version 31 replacing A/B with C.
3. A cleanup thread deletes A/B immediately.
4. R seeks its next block and fails even though its snapshot should remain valid.

Version/reference pinning must control physical deletion.

### Tombstone resurrection through incomplete overlap

1. Level 1 file N contains tombstone `k@200`; level 3 file O contains value `k@100`.
2. A compaction rewrites N without including or proving absence in lower overlaps and drops the tombstone.
3. A later read reaches O and returns `k@100`.

Garbage collection needs whole-tree coverage proof, not only “the tombstone is old.”

### Snapshot pins write amplification and space

1. A backup opens snapshot sequence 1,000 and runs for hours.
2. Foreground writes replace the same hot keys thousands of times.
3. Compaction must retain versions visible to the snapshot.
4. Output files grow and repeated compactions carry old values forward.

Long snapshots are a storage-capacity input. Limit, isolate, or account for them rather than blaming compaction after space is exhausted.

### Compaction debt becomes a write outage

1. Logical ingest exceeds the background rewrite capacity during a burst.
2. Overlapping L0 runs and pending bytes grow.
3. Negative reads and scans touch more files; cache churn and latency rise.
4. The engine slows or stops flushes to keep L0/recovery state bounded.

The stall is backpressure from an unsustainable file-lifecycle rate, not a random engine pause.

## Capacity and cost model

Measure the three amplifications from engine counters. For compaction-capacity work, define `F` as the initial encoded/compressed flush bytes produced per second and define file-layer write amplification against that baseline:

```text
file_write_amplification = (flush + compaction output bytes) / initial flush bytes
read_amplification  = physical_blocks_or_bytes_read / logical_result_blocks_or_bytes
space_amplification = live_physical_bytes / live_logical_bytes
```

Application bytes become `F` after record framing, key/value encoding, compression, and flush fragmentation. Also report a conventional end-to-end write amplification against application bytes, but define whether WAL, compaction input reads, replication, and compression are included before comparing systems. A leveled formula depends on size ratio, number of levels, overlap, overwrite distribution, file selection, and the chosen numerator. Fixed claims such as “leveled is 20x” are workload observations, not strategy constants.

If usable device write bandwidth after reserving foreground reads and safety margin is `C_write`, and measured file-layer write amplification is `WA_file`, sustainable flush output is bounded by:

```text
F <= C_write / WA_file
```

After the initial flush, approximate background output demand is `F * (WA_file - 1)`. If allowed background capacity is lower, pending compaction debt grows. If debt is `D` bytes of required physical work and spare catch-up capacity is `C_spare`, recovery takes at least `D/C_spare`; `C_spare <= 0` means the system never catches up without throttling ingest or adding resources.

Point-read I/O depends on candidate runs and filter false positives. For negative lookups with candidate filters of false-positive probability `p_i`, expected false-positive data-block reads are approximately `sum(p_i)` if filter and index metadata are cached. Cache misses, L0 overlap, and correlations add cost.

Peak disk space must cover current live files, new compaction outputs before commit, old inputs retained by readers/checkpoints, write bursts, and safety reserve. A job merging `I` input bytes may produce output near the surviving logical bytes, but the inputs cannot be reclaimed until publication and reference drain. Large universal/tiered merges can temporarily approach another full run's size.

Compaction can be CPU-bound: checksum, key comparison, decompression, recompression, encryption, filter construction, and range-tombstone processing all consume cores. Track logical and physical bytes per CPU second rather than assuming NVMe bandwidth is the limit.

## Production operation and migration

Monitor pending compaction bytes/work, L0 and run counts, bytes read/written by level and reason, logical ingest, write/read/space amplification, compaction CPU, stall/slowdown duration, oldest snapshot, obsolete bytes pinned by readers, tombstone/expired-version ratios, cache hit by block type, and checksum failures. Node-wide disk utilization can look healthy while one column family accumulates fatal debt.

Rate limiting should preserve foreground latency and enough compaction progress to avoid an eventual hard stall. Prioritize write-frontier work that bounds run count, but prevent bottom-level starvation. Manual full compaction is a major rewrite with transient space and cache effects; estimate inputs, outputs, duration, and rollback before invoking it.

File-format migration needs readers that understand old and new footers/encodings, new writers enabled only after compatibility deploys, and compaction or explicit rewrite to retire the old form gradually. Comparator changes generally require a full rebuild into a separate database because mixed sorted orders cannot safely share one merge.

Crash tests should stop the process after every file sync, rename/upload, manifest append, manifest sync, version install, and obsolete-file deletion. Corrupt blocks, footers, filters, and manifests independently. Property tests should compare arbitrary flush/compaction schedules against a simple MVCC model across snapshots, point/range tombstones, TTL, and comparator edge cases. Restore old checkpoints only as non-authoritative copies unless their log/file generation is proven current.

## Decision framework

1. What point-read, scan, overwrite, delete, TTL, and snapshot distributions must the run layout support?
2. Which manifest edit is the atomic publication point, and what durability ordering precedes it?
3. How many overlapping runs can a read encounter in steady state and during debt?
4. What proof permits each old version, point tombstone, or range tombstone to be dropped?
5. What measured write amplification and background bandwidth bound sustainable ingest?
6. How much transient space can the largest compaction and longest reader require?
7. Can recovery distinguish orphan outputs from live files without trusting a directory listing?
8. How will format, comparator, and encryption changes coexist during migration?

## Primary references

- [O'Neil et al., *The Log-Structured Merge-Tree (LSM-Tree)* (Acta Informatica 1996)](https://doi.org/10.1007/s002360050048)
- [Chang et al., *Bigtable: A Distributed Storage System for Structured Data* (OSDI 2006)](https://storage.googleapis.com/gweb-research2023-media/pubtools/4443.pdf)
- [Google LevelDB, official table-format specification](https://github.com/google/leveldb/blob/main/doc/table_format.md)
- [Facebook RocksDB, official leveled-compaction description](https://github.com/facebook/rocksdb/wiki/Leveled-Compaction)
- [Dayan, Athanassoulis, and Idreos, *Monkey: Optimal Navigable Key-Value Store* (SIGMOD 2017), author publication page](https://stratos.seas.harvard.edu/publications/monkey-optimal-navigable-key-value-store)
- [Dayan and Idreos, *Dostoevsky: Better Space-Time Trade-Offs for LSM-Tree Based Key-Value Stores* (SIGMOD 2018), author publication page](https://stratos.seas.harvard.edu/publications/dostoevsky-better-space-time-trade-offs-lsm-tree-based-key-value-stores)
