# B-Trees

A database B-tree is an ordered map engineered around fixed-size pages, a buffer pool, concurrent structural changes, and crash recovery. Its value is not the textbook `O(log n)` claim; many structures have that asymptotic cost. The value is high fanout: a handful of page accesses locates a key among billions, upper levels remain cached, and linked leaves turn an ordered range into mostly sequential work.

A B+-tree implementation must account for page structure, search and mutation algorithms, latching, split recovery, and the physical cost model. [Secondary Indexes](../02-distributed-databases/06-secondary-indexes.md) owns distributed placement and consistency of alternate access paths. [Write-Ahead Logging](./04-write-ahead-logging.md) owns transaction recovery theory, while [LSM Trees](./02-lsm-trees.md) covers the write-buffered alternative.

## Workload and storage contract

Characterize point lookups, bounded ranges, ordered scans, inserts, in-place updates, and deletes separately. Record key and payload width distributions, ordering and skew, buffer-pool budget, page size, concurrent readers and writers, durability mode, snapshot lifetime, and latency percentiles. A tree serving immutable 8-byte keys with 99% cache hits is a different system from one indexing variable-length URLs under random updates.

Define what the leaf value means. In a heap-organized table it is usually a stable or version-aware row locator. In a clustered table the primary tree’s leaf contains the row, while each secondary leaf often contains the primary key. A wide clustered key therefore widens every secondary index. Covering payloads save a second lookup but enlarge leaves and make more updates touch the index.

The API contract should distinguish logical locks from physical latches. A transaction lock protects a key or predicate for isolation. A latch protects page bytes for microseconds while a thread reads or modifies them. Holding a page latch while waiting for network, user input, or a transaction lock is a convoy bug.

## Page state and structural invariants

Most database “B-trees” are B+-trees: internal pages contain separator keys and child page IDs; leaves contain all logical entries and form an ordered sibling chain. A slotted page supports variable-size records:

```text
+---------------- page ----------------+
| header | slot array -> free <- records|
+---------------------------------------+

header: page type, format version, generation, page LSN,
        lower/upper free-space bounds, sibling IDs, high key, checksum
slot:   key order plus offset/length into the record area
```

The tree’s persistent state includes a root page ID and generation, allocation metadata, page contents, and recovery log coordinates. Readers may also pin a root or page generation for a snapshot. The implementation maintains these invariants:

1. Keys are ordered within a page according to one versioned comparator and collation.
2. An internal separator covers every key reachable through its child, without gaps or ambiguous ownership.
3. Every live leaf is reachable from the root and the sibling chain preserves global order.
4. Non-root pages respect implementation occupancy rules, except during logged transitions.
5. A page is not reused while a reader, WAL record, backup, or parent pointer can still identify its old generation.
6. A page whose update is visible on disk has a page LSN no greater than the durable WAL frontier required to redo it.
7. A structural operation is recoverable from every crash point; readers see the old route, the new route, or a valid sideways path.

Separator keys need not duplicate a complete user key. Prefix or suffix truncation can retain only the shortest value that distinguishes adjacent children, increasing fanout. The comparator and truncation rules are persistent-format semantics; upgrading them requires validation or rebuild.

## Cost model: fanout, height, and cache

For page size `P`, header and slot overhead `H`, average internal separator bytes `K`, and child-pointer bytes `C`, approximate fanout is:

```text
fanout F ~= floor((P - H) / (K + C + slot_overhead))
height h ~= ceil(log_F(number_of_leaf_pages)) + 1
leaf pages ~= live_entry_bytes / (P * target_leaf_fill)
```

With an 8 KiB page and roughly 24 bytes per separator-plus-pointer, fanout is on the order of hundreds. Four levels can cover billions of entries. The root and first internal levels occupy little memory, so a point lookup usually performs several buffer hits and zero or one storage reads, not four independent random I/Os.

A range returning `Krows` costs one descent plus approximately `ceil(Krows / entries_per_leaf)` leaf visits, adjusted for visibility checks and heap fetches. If the leaf contains only row locators, unordered heap access may dominate the apparently sequential index scan. A covering or clustered layout changes that cost, at the price of wider entries.

Capacity must use measured cache misses. If point-query rate is `Qp`, tree height is `h`, and per-level miss probabilities are `m_i`, expected page reads are:

```text
random page reads/s ~= Qp * sum(m_i) + heap_fetch_misses
index bytes         ~= leaf_pages * P + internal_pages * P
```

Once randomly touched leaves outgrow the buffer pool, `m_leaf` rises sharply and latency follows device IOPS. Adding CPU does not fix that phase change.

## Read and write paths

A point read pins the current root, compares the search key with separators, follows child IDs, and searches the leaf slot array. With B-link-style pages, each page carries a high key and right-sibling pointer. If a concurrent split moved the desired key to the right, the reader follows the sibling rather than restarting or trusting a temporarily stale parent.

An ordered range descends once, then walks leaves. The scan records enough page generation and key state to resume after concurrent splits. It must not hold an entire leaf chain latched. Snapshot visibility is evaluated by the transaction layer; physical key order alone does not decide which row version is visible.

### Insert and split protocol

The common insert path latches one leaf, verifies that the search route is still valid, writes the record or slot, emits WAL, and releases. If the leaf lacks space, a robust B-link split proceeds conceptually as follows:

1. allocate and initialize a new right page;
2. move the upper key range and copy the old right sibling into it;
3. set the old page’s high key and right pointer to the new page;
4. WAL-log the transition so both pages are recoverable;
5. install the new separator in the parent, splitting ancestors if needed.

Publishing the sideways link before depending on the parent prevents a concurrent reader from losing the new page. The parent update may be helped by another thread or completed during recovery. A root split creates a new root and changes the root metadata atomically under its own logged protocol.

Sequential keys repeatedly touch the rightmost leaf. They pack cache and pages efficiently but can contend on one latch. Random keys distribute latch ownership, yet touch the entire leaf working set and tend to leave more split slack. Time-ordered distributed IDs often balance locality and decentralised generation, but they still need a primary-key width and hotspot analysis.

### Update, delete, and space reuse

An update that leaves indexed columns unchanged may avoid the tree entirely or only change a clustered leaf. Changing a key is logically delete-old plus insert-new. In heap databases, a row relocation also requires stable indirection or updates to every referring index.

Production engines rarely merge every underfull page as textbooks suggest. They mark entries dead, compact within a page, unlink wholly empty pages, and merge only when benefit exceeds latch and WAL cost. This avoids merge/split oscillation but permits bloat after mass deletes. Reclamation waits for the oldest reader and recovery/backup frontier; only then may the allocator reuse a page ID with a new generation.

## Concurrency and recovery

Classical latch crabbing acquires a child before releasing its parent and retains ancestors only while a split could propagate through them. Modern fast paths descend optimistically with shared latches or version counters, acquire an exclusive leaf latch, validate that nothing relevant changed, and restart rarely. B-link high keys make structure changes observable without blocking readers on parent repair.

Logical key and predicate locks remain separate. A range scan under serializable isolation may lock gaps or use predicate-conflict tracking even though its physical latches are already released. Confusing these layers either destroys concurrency or admits phantoms.

WAL makes dirty-page writeback safe only if the log describing a page change reaches durable storage first. Page LSNs make redo idempotent. Full-page images, doublewrite buffers, checksums, or copy-on-write protect against torn pages; those mechanisms are detailed in [Write-Ahead Logging](./04-write-ahead-logging.md). Structural changes spanning pages use one atomic log record, a mini-transaction, or a recoverable sequence with explicit completion state.

Copy-on-write B-trees take another route: write a new leaf and every changed ancestor, then atomically publish a new root. Readers pin immutable roots and need almost no page latching. The cost is path-copy write amplification, delayed garbage collection, and often restricted writer concurrency. This is attractive for read-heavy embedded stores and filesystem metadata, not an automatic OLTP replacement.

## Specialized failure traces

### Crash makes a split page unreachable

The engine allocates a right page and moves half the keys, then crashes before either linking it from the old leaf or logging a parent route. Recovery sees allocated bytes but no reachable path; acknowledged keys disappear. The split protocol must WAL-log and publish a recoverable sibling relationship before the old page stops owning those keys.

### Reused page ID causes an ABA read

A long reader remembers child page 900. Vacuum unlinks 900, the allocator immediately reuses it for an unrelated leaf, and the reader follows its stale pointer into valid but wrong bytes. Pin/epoch reclamation and page generations prevent reuse until no old route can survive.

### Random-key cache cliff

An index grows beyond the buffer pool while UUID-like keys distribute inserts uniformly. Each insert now reads a cold leaf before dirtying it; device queue depth rises, checkpoint writes compete with reads, and p99 jumps although QPS changed little. The working-set model, not tree height, explains the incident.

### Right-edge latch convoy

A monotonically increasing key sends every writer to one leaf. The pages are dense and cached, yet threads wait on the same exclusive latch and throughput stops scaling with cores. Partitioning the key space, batching inserts, shortening the critical section, or using engine-specific sequential-insert mitigation addresses contention; replacing the disk does not.

### Comparator upgrade corrupts routing

Old pages were ordered with collation version X. New code compares with version Y, where two strings reverse order. A search follows the wrong separator and reports a missing row that is physically present. Persist comparator identity and rebuild into a new tree before activating new ordering semantics.

## Operations, isolation, and migration

Observe height, internal and leaf density, split rate by level, rightmost versus random splits, latch wait time by page role, buffer hits and physical reads, dirty-page age, WAL bytes per logical write, empty-page reclaim backlog, scan pages per returned row, and corruption/checksum errors. Relate them to key distribution and tenant, not only table totals. A hot tenant can monopolize a leaf range, cache, and latch even when aggregate load is normal.

Tenant prefixes improve locality and make range deletion possible, but expose tenant size and create hot contiguous ranges; hashing tenant plus key spreads load but makes tenant scans expensive. Enforce authorization above every index-only path, because avoiding a heap fetch must not bypass row security. Encrypt pages and backups, authenticate page-level repair sources, and treat index keys as sensitive: emails or document titles leak even when payload columns are encrypted.

Page-size, collation, key-encoding, and clustered-key changes generally require a side-by-side rebuild. Create a new tree at snapshot frontier `L`, capture later mutations, apply them in key/version order, validate counts and sampled ranges, then atomically swap the root or catalog entry. Keep the old tree readable through rollback and cursor/snapshot lifetime. Page headers should carry format versions so recovery and upgrade binaries can reject rather than misparse an old page.

Verification combines a structural checker with a reference ordered map. Randomized tests insert, update, delete, split, and scan variable-width keys while checking reachability, separator coverage, leaf order, and uniqueness. Concurrency tests pause threads at every latch transition. Crash tests cut power after each WAL and page-write boundary, including root splits and page reuse. Corruption tests flip sectors and checksums; upgrade tests open N-1 data and recover N-1 WAL with the new binary.

## Decision framework

Choose a B+-tree when predictable point reads, ordered ranges, and repeated updates to a cacheable working set dominate. It is the natural general-purpose OLTP index. Choose an [LSM tree](./02-lsm-trees.md) when foreground random page modification is the ingest bottleneck and background rewrite capacity is available. Choose a hash index only when equality dominates and ordered operations truly have no value; choose specialized spatial, inverted, or vector structures for their corresponding predicates.

The most consequential B-tree choices are often outside the algorithm: key width and order, clustered versus heap layout, number of maintained indexes, buffer-pool residency, and snapshot lifetime. Model those before tuning split thresholds.

## Primary references

- Bayer, R., and McCreight, E. [Organization and Maintenance of Large Ordered Indices](https://doi.org/10.1007/BF00288683). Acta Informatica, 1972.
- Lehman, P. L., and Yao, S. B. [Efficient Locking for Concurrent Operations on B-Trees](https://doi.org/10.1145/319628.319663). ACM TODS, 1981.
- Mohan, C., and Levine, F. [ARIES/IM: An Efficient and High Concurrency Index Management Method Using Write-Ahead Logging](https://doi.org/10.1145/128765.128770). SIGMOD, 1992.
- Graefe, G. [Modern B-Tree Techniques](https://doi.org/10.1561/1900000028). Foundations and Trends in Databases, 2011.
- PostgreSQL source documentation. [B-Tree implementation README](https://github.com/postgres/postgres/blob/master/src/backend/access/nbtree/README).
- SQLite. [Database File Format: The B-Tree Pages](https://www.sqlite.org/fileformat.html#b_tree_pages).
