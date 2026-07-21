# Bloom Filters

## TL;DR

A Bloom filter answers membership with one-sided error: "definitely absent" or "probably present," never a false negative. Roughly 10 bits per element yields a 1% target false-positive rate, versus an information-theoretic minimum near 6.6 bits. Production design must account for overfill, random-memory probes, nonuniform LSM-level budgets, and more compact immutable-set alternatives such as Ribbon and binary fuse filters.

---

## Why the Filter Exists: Negative Lookups Are the Expensive Ones

A B-tree answers "key not present" cheaply: one root-to-leaf descent, mostly cached, and the leaf proves absence. An LSM tree ([LSM Trees](./02-lsm-trees.md)) cannot — the key could be in *any* run, so absence can only be proven by checking every candidate SSTable.

Count the candidates in a typical leveled LSM:

```
Leveled compaction, fanout 10, 7 levels:
  L0: up to 4 overlapping files   → 4 candidate runs
  L1–L6: one sorted run each      → 6 candidate runs
  Total: ~10 runs may contain any given key

Point lookup for a MISSING key, no filters:
  10 runs × (index block + data block read) ≈ 10–20 block reads
  At ~100μs per cached-miss NVMe read: 1–2ms to learn "not found"

Same lookup with a 1% FPR filter per run:
  Expected disk reads = 10 runs × 0.01 = 0.1 reads
  → 100–200× reduction in read amplification for negative lookups
```

Negative lookups are common in deduplication, insert-if-absent and uniqueness checks, cache-fill probes, and conditional creation. A filter converts that common miss from "read every level" to "read nothing."

**A Bloom filter prepays the decision "is disk access warranted?"** Sizing, hashing, and layout determine that decision's cost and false-positive rate.

---

## The Structure and Its One-Sided Error

A Bloom filter is a bit array of `m` bits and `k` hash functions. Insert sets `k` bits; query checks `k` bits.

```
Insert "hello":                    Query "world":
  h₁("hello") = 3                    h₁("world") = 5 → bit 5 is 0 ✗
  h₂("hello") = 7                    → "definitely not present"
  h₃("hello") = 12                   (stop at first zero bit)
  set bits 3, 7, 12
                                   Query "hello":
┌─────────────────────────────┐      all of bits 3, 7, 12 are 1
│ 0 0 0 1 0 0 0 1 0 0 0 0 1 0 │      → "probably present"
└─────────────────────────────┘
        ↑       ↑         ↑
        3       7         12
```

The asymmetry of the error is structural, not incidental:

- **No false negatives**: bits are only ever set, never cleared. If an element was inserted, its bits are 1 forever, so a query for it can never see a 0. (This is also why standard Bloom filters cannot support deletion — clearing a bit might clear evidence of *another* element that hashed to the same position, manufacturing a false negative.)
- **False positives**: a never-inserted element can find all its `k` bits already set by the union of other elements' insertions. The filter says "probably present," the disk read happens, and finds nothing. This is wasted work, not wrong answers — provided the surrounding system treats "probably" as a hint, not an authority (see [Failure Modes](#failure-modes)).

---

## The Math, Derived Honestly

### False-positive probability

After inserting `n` elements with `k` hashes into `m` bits, the probability a particular bit is still 0:

```
P(bit = 0) = (1 - 1/m)^(kn) ≈ e^(-kn/m)
```

A false positive requires all `k` probed bits to be 1:

```
p ≈ (1 - e^(-kn/m))^k
```

Two honest caveats the textbooks skip. First, this treats bit occupancies as independent, which they aren't; the exact rate (Bose et al., 2008) is slightly *higher* than the formula, though the gap is negligible for filters of realistic size (m in the millions). Second, the formula assumes the k hash functions are truly uniform and independent — real filters approximate this (see [Hashing](#hashing-where-implementations-actually-go-wrong)), and a bad approximation shows up as a real FPR above the theoretical one.

### Optimal k, and what "optimal" looks like

Minimizing `p` over `k` for fixed `m/n`:

```
k* = (m/n) × ln 2  ≈ 0.693 × bits-per-key

At the optimum:
  - exactly half the bits are set (load factor 1/2 — maximum entropy per bit)
  - p = 2^(-k*) = 0.6185^(m/n)
```

Inverting: to hit a target FPR `p`, you need

```
m/n = log₂(1/p) / ln 2 = 1.44 × log₂(1/p) bits per key

p = 1%    → 1.44 × 6.64 ≈  9.6 bits/key, k = 7
p = 0.1%  → 1.44 × 9.97 ≈ 14.4 bits/key, k = 10
p = 0.01% → 1.44 × 13.3 ≈ 19.2 bits/key, k = 13
```

Each extra ~4.8 bits/key buys one decimal order of magnitude of FPR. Note what this formula does *not* contain: the size of the elements. A Bloom filter for 64-byte keys and one for 4KB documents cost the same 9.6 bits per element at 1% — the filter stores evidence of hashes, not data.

### The 44% tax, and why post-Bloom filters exist

Information theory sets a floor: any structure answering membership with false-positive rate `p` needs at least `log₂(1/p)` bits per element. Bloom's `1.44 × log₂(1/p)` is therefore **44% above optimal** — the price of a structure so simple it was designed in 1970 and needs nothing but a bit array. The modern filters covered below (cuckoo, ribbon, xor/binary-fuse) exist precisely to claw back that 44%, trading away either mutability or build speed:

```
Bits per key at ~1% FPR (log₂(1/p) = 6.64 → theoretical floor 6.64 bits):

  Standard Bloom       9.6 bits   (1.44× floor)   dynamic inserts
  Blocked Bloom       ~10.5 bits  (1.6× floor)    dynamic, 1 cache miss
  Cuckoo filter       ~10.1 bits  (1.5× floor)    dynamic + deletion
  Ribbon filter       ~7.0 bits   (1.05–1.1×)     static, slow build
  Binary fuse filter  ~7.5 bits   (1.13×)         static, fast build
```

### The overfill curve: FPR degrades brutally, not gracefully

The most operationally important consequence of the math: FPR is set by the *actual* `n`, not the planned one. Take a filter sized for 1M keys at 1% (m = 9.59M bits, k = 7) and keep inserting:

```
actual n     kn/m     fraction of bits set     actual FPR
─────────────────────────────────────────────────────────
1.0M (as planned)  0.73        52%                 1.0%
1.5M               1.09        66%                 5.8%
2.0M               1.46        77%                16%
3.0M               2.19        89%                44%
5.0M               3.65        97%                83%
```

At 2× overfill the filter is 16× worse than configured; at 5× it approves 83% of garbage — you're paying the memory *and* the disk reads. The failure is silent: nothing errors, latency just drifts up as "definitely not" quietly becomes "go check disk." Two structural defenses: size for 1.5–2× expected growth, and prefer architectures where filters are per-immutable-artifact (one filter per SSTable, sized at flush time from the exact key count — which is why LSM engines never hit this in steady state) over one long-lived global filter that grows past its design point.

---

## Hashing: Where Implementations Actually Go Wrong

### Double hashing: k hashes for the price of two

Computing 7 independent high-quality hashes per key is wasteful. The standard trick (Kirsch & Mitzenmacher, 2006) derives all `k` probes from two base hashes:

```
h_i(x) = h₁(x) + i × h₂(x)   (mod m),  for i = 0..k-1
```

and provably preserves the asymptotic FPR. Two implementation traps:

- **h₂ = 0 collapse**: if `h₂(x) ≡ 0 (mod m)`, all k probes hit the same bit and that key's effective k is 1. Force `h₂` odd (with power-of-two m) or add a quadratic term `+ i²`.
- **In practice, one 64-bit hash is enough**: split it into two 32-bit halves for h₁ and h₂. One xxHash/Murmur3 evaluation per key, total.

### Choosing the hash

```
Good:  xxHash (fastest), MurmurHash3 (ubiquitous), CityHash/FarmHash
Bad:   MD5/SHA-256 — cryptographic strength buys nothing here, costs 10–20× cycles
       Language-default hashCode() — weak avalanche, correlated outputs,
         and often deliberately randomized per-process (breaks serialized filters!)
       h(x) mod m with structured keys — clustering
```

The last parenthetical is a real bug class: serialize a filter built with a per-process-seeded hash (Python's `hash()`, Java's default `String.hashCode` is stable but many others aren't), load it in another process, and every query probes the wrong bits — the filter returns "definitely not" for keys it contains. **A persisted filter must pin the hash function, its seed, and the bit-index derivation as part of its serialization format.**

### 32-bit hashes stop working before you expect

With a 32-bit hash, bit indices repeat once `m` approaches 2³². But the damage starts earlier: RocksDB's legacy block-based filter derived probes from a single 32-bit hash, and above a few million keys per filter the collision structure put a *floor* under the FPR — adding bits per key beyond ~14 barely improved the real FPR even as the formula promised 0.1%. The modern full/blocked filter implementations moved to 64-bit hashing specifically to fix this. If a filter will ever hold >1M keys, use 64-bit hashes.

### Adversarial inputs

A Bloom filter with a public, unkeyed hash is an oracle: an attacker who can query it (or just knows the implementation) can compute keys whose probes land on already-set bits, manufacturing unlimited false positives — every one a disk read, i.e., a cheap request-amplification attack on exactly the path the filter was supposed to protect. Where inputs are attacker-controlled (URL dedup on a crawler, per-user rate-limit filters, spam-seen filters), use a keyed hash (SipHash) with a per-deployment secret, exactly as hash tables did after the 2011 hash-flooding attacks.

---

## The Memory Hierarchy: Blocked Bloom Filters

The textbook cost model says a Bloom query is O(k) "operations." The real cost model is cache misses. A standard filter's k probes are k independent random positions across a filter that is megabytes to gigabytes — each probe is a DRAM cache miss:

```
Standard bloom, k = 7, filter >> L3 cache:
  Negative query (probe until first 0): ~2 probes average → ~2 misses
  Positive/false-positive query:         all 7 probes     → ~7 misses
  At ~100ns per DRAM miss: 200–700ns per query

For comparison: the entire rest of a memtable lookup can be ~100–200ns.
The "free" in-memory check can dominate the in-memory path.
```

The **blocked Bloom filter** (Putze, Sanders & Singler, 2007) fixes this: the first hash selects one 64-byte cache-line-sized block; all k bits are set/probed *within that line*.

```
Query = exactly 1 cache miss, regardless of k.
Within the line, probing 7 bits is a handful of register ops —
and with SIMD, all probes resolve in a couple of instructions.

The price: keys are no longer spread evenly. Some blocks get
overloaded (Poisson variance across blocks), raising FPR.
Cost ≈ 0.5–1 extra bits/key to hit the same FPR as standard bloom.
```

This is not an exotic variant; it is what production engines actually deploy. RocksDB's `format_version=5` filter (available since RocksDB 6.6) is a blocked, SIMD-friendly "fast local bloom" — the flat ~duplicated-probe layout was chosen because one cache miss per query beat 30% space savings in every read-heavy benchmark. When you see "bloom filter" in a modern storage engine, assume blocked.

---

## Inside the LSM Read Path

This is where the filter earns its memory. The mechanics, using RocksDB's vocabulary (Cassandra and HBase differ in knobs, not structure):

### Where filters live and when they're checked

```
Point lookup Get(key):
  1. memtable        — optional memtable bloom (prefix-based) skips probing
  2. immutable memtables
  3. L0 files, newest first — each: check filter block → maybe read data
  4. L1..Lmax        — binary search file boundaries, one candidate file
                       per level: check filter block → maybe read data

Filter blocks are stored per-SSTable (in the table's metadata),
fetched through the block cache like any other block, and — if
cache_index_and_filter_blocks=true — subject to eviction like any
other block. For very large SSTables, partitioned filters split the
filter into a two-level structure so only the needed partition loads.
```

Two consequences worth internalizing. First, the *aggregate* false-positive cost of a lookup is the **sum of per-run FPRs** — 10 runs at 1% each means ~10% of missing-key lookups still touch disk somewhere. Second, a filter block that has been evicted from block cache must be read from disk *before* it can save you a disk read; under memory pressure filters can invert into pure overhead (see Failure Modes).

### Whole-key vs. prefix filters — and why range scans get nothing

A Bloom filter can only answer questions about exact hashed values. A range scan `[a, b)` asks about a continuum — no finite set of exact-match probes covers it, so **standard bloom filters are useless for range queries** (Cassandra lets you set `bloom_filter_fp_chance = 1.0` to disable them on scan-only tables for exactly this reason).

The partial exception: **prefix bloom filters**. Define a prefix extractor (say, the first 8 bytes = user ID), build the filter over prefixes, and iterator seeks *within one prefix* ("all events for user X") can consult the filter. RocksDB exposes this as `prefix_extractor` plus a memtable bloom (`memtable_prefix_bloom_size_ratio`); MyRocks leans on it heavily for index-prefix scans. The trade: iterators must promise not to cross prefix boundaries (`prefix_same_as_start`), or results are silently wrong — a filter skip is only sound if the query truly stays inside the filtered domain. For true range filtering, newer structures exist (SuRF, Rosetta, range-partitioned fence pointers) but none is yet a default.

### Monkey: uniform bits-per-key is the wrong allocation

Every engine's default — "10 bits/key for every SSTable" — is provably suboptimal. The insight (Dayan, Athanassoulis & Idreos, *Monkey*, SIGMOD 2017): lookup cost is the **sum of FPRs across runs**, while memory cost of achieving FPR `p` on a run of `n` keys is `∝ n·ln(1/p)`. Minimizing total expected disk reads under a fixed memory budget yields a skewed allocation:

```
With fanout 10, the last level holds ~90% of all keys.
Uniform 10 bits/key spends ~90% of filter memory on that one level.

Monkey's optimal allocation: give SMALLER (upper) levels exponentially
LOWER FPRs — they're cheap to filter aggressively because they hold
few keys — and let the largest level run at a higher FPR.

Result: same total memory, sum-of-FPRs (expected wasted reads for a
missing key) shrinks; Monkey reports the same lookup latency with
~2× less filter memory, or up to ~2× lower lookup cost at equal memory.
```

The general principle transfers beyond LSM trees: when one filter guards many tiers of different sizes and probe frequencies, allocate false-positive budget where a bit of memory buys the most avoided work — never uniformly.

---

## Beyond Bloom: The Modern Filter Zoo

### Counting Bloom filter — deletion, at 4× the price

Replace each bit with a small counter (typically 4 bits): insert increments, delete decrements, query checks all counters > 0. It works, but: 4× the memory, counters can saturate at 15 (after which a decrement would risk false negatives, so saturated counters must stay stuck — a slow FPR leak), and in practice a cuckoo filter dominates it on every axis. Counting blooms survive mostly in older network gear and papers.

### Cuckoo filter — deletion done right

The cuckoo filter (Fan, Andersen, Kaminsky & Mitzenmacher, 2014) stores small **fingerprints** (hash fragments, e.g. 8–12 bits) in a 4-way bucketed hash table using partial-key cuckoo hashing:

```
Each key has two candidate buckets:
  b₁ = hash(x)
  b₂ = b₁ XOR hash(fingerprint(x))     ← computable from b₁ + fingerprint
                                          alone, enabling eviction without x

Insert: place fingerprint in b₁ or b₂; if both full, evict a resident
        fingerprint to ITS alternate bucket, repeat (cuckoo hashing).
Query:  check both buckets for the fingerprint — exactly 2 cache misses.
Delete: remove one matching fingerprint copy. Sound, with one caveat:
        inserting the same key twice then deleting once leaves it present;
        deleting a never-inserted key can evict a colliding victim.
        Deletion is only safe if inserts/deletes are balanced by protocol.

Space: bits/key ≈ (log₂(1/p) + 3) / α,  α ≈ 0.95 at 4-way buckets
  p = 1%  → (6.64 + 3)/0.95 ≈ 10.1 bits/key   (≈ bloom)
  p = 0.1% → (9.97 + 3)/0.95 ≈ 13.6 bits/key  (beats bloom's 14.4)
```

Rule of thumb: below ~3% target FPR, cuckoo is at least as compact as Bloom *and* gives you deletion and 2-cache-miss queries. Its weaknesses: inserts can fail outright near full load (must size with headroom and handle failure), and insert cost spikes as load approaches the maximum.

### Quotient filter — merge- and resize-friendly

Stores `fingerprint = quotient‖remainder`: the quotient addresses a slot, the remainder is stored in it, collisions resolve by linear probing with three metadata bits per slot to reconstruct runs. Everything sits in contiguous memory (cache-friendly), supports deletion, can be **resized without rehashing the original keys** (fingerprints carry enough information), and two filters **merge like a sorted-list merge** — attractive for LSM compaction. Costs ~10–25% more space than Bloom and the implementation is genuinely fiddly; used more in research systems (e.g., counting quotient filters in genomics) than mainstream engines.

### Ribbon filter — RocksDB's static space-saver

For an SSTable, the key set is frozen at build time — a *static* filter can exploit that. The Ribbon filter (RocksDB ≥ 6.15, `NewRibbonFilterPolicy`) treats "key i's probes XOR to its expected value" as a system of linear equations over GF(2) and solves it at construction:

```
  ~30% less space than blocked bloom at equal FPR (~1.05–1.1× the
   entropy floor), same query speed, but 3–4× slower to BUILD.

RocksDB's guidance: use ribbon for lower levels (built rarely, by
background compaction, hold most keys → memory savings dominate) and
bloom for L0/high levels (built on every flush → build speed dominates).
That per-level policy split is Monkey-style thinking applied to filter
*type*, not just filter *budget*.
```

### Xor and binary fuse filters — the static state of the art

Same static regime as ribbon, different construction (hypergraph peeling): each key maps to 3 positions whose stored values XOR to the key's fingerprint. Xor filters (Graf & Lemire, 2020) achieve `1.23 × log₂(1/p)` bits/key; **binary fuse filters** (2022) tighten this to ~1.13× with near-linear build time — for 8-bit fingerprints, ~9 bits/key at p ≈ 0.4%, where Bloom would need ~11.5 bits — about 20% smaller. No inserts, no deletes: build once from the complete key list. Ideal for shipped artifacts — compiled block lists, CDN "does this object exist at origin" maps, malware signature sets — anywhere the set is versioned and replaced wholesale rather than mutated.

### Choosing

| Filter | Mutability | Space @ ~1% | Query cost | Use when |
|---|---|---|---|---|
| Blocked Bloom | insert-only | ~10.5 bits/key | 1 cache miss | Default for dynamic sets; L0/flush-path SSTable filters |
| Standard Bloom | insert-only | 9.6 bits/key | up to k misses | Only when simplicity beats latency (small filters that fit in cache) |
| Cuckoo | insert + delete | ~10.1 bits/key | 2 cache misses | Need deletion (caches, membership with churn); FPR ≤ 3% |
| Quotient | insert + delete + merge + resize | ~11–12 bits/key | 1 locality-friendly probe run | Need merging (compaction pipelines) or in-place growth |
| Ribbon | static | ~7 bits/key | ~1 cache miss | Immutable, build-time tolerant: deep LSM levels |
| Binary fuse / xor | static | ~7.5 bits/key | 3 misses (fuse: ~1–2) | Immutable, build-speed sensitive: shipped/versioned sets |

---

## Bloom Filters Beyond Storage Engines

The same "prepaid negative answer" pattern recurs anywhere a cheap check can veto an expensive operation:

**Distributed joins and query engines.** In a hash join where the build side is small and the probe side is a huge scan, engines build a Bloom filter over the build-side join keys and push it *down into the probe-side scan* (Spark's runtime filters, Snowflake/BigQuery semi-join reduction, Parquet row-group skipping). Probe rows that fail the filter never leave the scan operator — often eliminating >90% of the shuffled/scanned data. Same idea across the network: ship the filter, not the table, as a compact semi-join.

**CDN caching: the one-hit-wonder problem.** Akamai measured that ~74% of requested URLs are requested exactly once in a multi-day window (Maggs & Sitaraman, 2015). Caching an object on first request means most disk writes are for objects never read again. Fix: a Bloom filter of recently-seen URLs; cache only on the *second* request (filter hit). This "bloom filter as admission policy" roughly halved disk writes and improved hit rates — the filter isn't guarding reads at all, it's guarding *writes*.

**Bitcoin SPV — a cautionary tale (BIP-37 → BIP-158).** Light clients once sent a Bloom filter of their addresses to full nodes, which returned matching transactions; false positives were *supposed* to provide plausible-deniability privacy. It failed: across multiple filters and sessions, the intersection of "maybe" sets de-anonymizes the wallet almost completely, and serving the filtered scans enabled cheap DoS. The replacement (BIP-158) inverts the direction — the *server* publishes a compact per-block filter (a Golomb-coded set, ~same job as a Bloom filter but optimally compressed for one-shot transfer) and the client downloads whole blocks that match locally, revealing nothing. Lesson: a probabilistic filter leaks its contents under repeated observation; it is a performance structure, not a privacy mechanism.

**Stream dedup with rotating filters.** "Exactly-once-ish" event pipelines ([Delivery Guarantees](../05-messaging/04-delivery-guarantees.md)) often need "have I seen this event ID in the last hour?" A single filter would overfill (see the overfill curve above); the standard pattern is N time-bucketed filters (e.g., 6 × 10-minute filters), querying all, inserting into the newest, and dropping the oldest wholesale — deletion by retirement, never by mutation.

**Others in the same shape:** web-crawl frontier dedup (billions of URLs, false positive = a page unnecessarily skipped — tolerable); Squid's cache digests (peers exchange Bloom filters of their cache contents to route requests); [secondary-index scatter-gather pruning](../02-distributed-databases/06-secondary-indexes.md) (query only partitions whose filter says "maybe").

---

## Failure Modes

**Silent overfill.** Covered above, but it's the #1 real-world failure: no error, no log line, just FPR drifting from 1% toward 80% as `n` outgrows the design point. Detection requires *measuring* the effective FPR: `false positives / (false positives + true negatives)`, where a false positive = filter said maybe, disk said no. RocksDB exposes exactly this as `rocksdb.bloom.filter.full.positive` vs `.full.true.positive`; Cassandra as `BloomFilterFalseRatio`. Alert when measured FPR exceeds ~2× configured.

**Filter blocks evicted under memory pressure.** In RocksDB with `cache_index_and_filter_blocks=true`, filter blocks compete with data blocks for cache. Under pressure, the cache evicts the filter, and the next lookup must read the filter *from disk* before it can (maybe) save a data-block read — negative lookups get slower than having no filter, intermittently, in exactly the overloaded moments that matter. Mitigations: `pin_l0_filter_and_index_blocks_in_cache`, `cache_index_and_filter_blocks_with_high_priority`, or budget filters outside the block cache and watch the `filter block read` tickers.

**Merging or serialization mismatch → false negatives.** Two filters can be merged by OR-ing bit arrays *only* if `m`, `k`, hash function, and seed are all identical; a mismatch produces a structure that confidently returns wrong "definitely not" answers — the one error class Bloom filters are never supposed to make. The same failure arrives via serialization: loading a filter into a runtime whose hash seed differs (per-process randomized hashes, endianness, a library upgrade that changed bit-index derivation). Version the filter format, embed the hash ID and seed, and refuse to load on mismatch.

**Treating "maybe" as "yes."** A username-availability check backed only by a filter will reject ~p% of *available* names (false positive → "taken") — an availability bug that ships silently because tests rarely hit a colliding name. The filter must always be a fast path in front of an authoritative check, never the authority. The dual failure: using a counting/cuckoo filter's deletion against unbalanced protocol traffic (delete-without-insert), which manufactures false negatives and *does* corrupt correctness.

**Hot false positive.** One popular missing key that happens to be a false positive turns into a 100%-of-the-time disk-read hotspot (the filter re-lies on every single query — probes are deterministic). If a "missing key" is hot enough to matter, layer a small negative cache (exact, e.g. a tiny LRU of confirmed-absent keys) behind the filter.

**Weak or misused hashing.** Correlated probe bits (bad hash, `h₂=0` degeneracy, 32-bit hash on a large filter) put an invisible floor under FPR regardless of bits/key; adversarial inputs against an unkeyed hash turn the filter into an amplification vector. Both covered in the hashing section — both show up in production as "we doubled bits per key and the FPR didn't move."

---

## Decision Framework

| Situation | Do this |
|---|---|
| LSM/SSTable point-lookup filter, default | Blocked bloom ~10 bits/key (RocksDB `format_version≥5`); ribbon for bottom levels if memory-bound |
| Read path is mostly *positive* lookups | Shrink or skip the filter — it only pays on misses; measure `useful / queries` first |
| Table accessed only by range scan | Disable the filter (`bloom_filter_fp_chance = 1.0`); consider prefix bloom only if scans are prefix-bounded |
| Need deletion with churn | Cuckoo filter with ≥5% load headroom; ensure protocol never deletes non-inserted keys |
| Immutable, versioned set shipped to many nodes | Binary fuse (fast build) or ribbon (max compactness); rebuild per version |
| Streaming dedup over a time window | Rotating bucketed blooms; retire whole filters, never delete |
| Set will grow unboundedly / unknown n | Per-artifact filters sized at seal time (LSM-style), or scalable bloom (chained filters with tightening p); never one global filter |
| Attacker-influenced keys | Keyed hash (SipHash) with per-deployment secret |
| Filter memory across many tiers | Monkey-style skew: lowest FPR where runs are small and probed often, not uniform bits/key |
| Guarding writes, not reads (admission) | Bloom of recently-seen keys; act on second occurrence (one-hit-wonder pattern) |

---

## Key Takeaways

1. **The filter prepays the answer to "is disk worth it?"** — its value is proportional to your *negative*-lookup rate; on positive-heavy workloads it's pure overhead.
2. **Sizing is 1.44 × log₂(1/p) bits/key** — 9.6 bits for 1%, +4.8 bits per extra decimal digit of FPR, independent of element size.
3. **FPR degrades brutally with overfill** — 2× over design n ⇒ ~16× the configured FPR; measure effective FPR in production, don't trust the config.
4. **Count cache misses, not hash ops** — blocked bloom (1 miss) is the production default; standard bloom's k scattered probes can dominate in-memory latency.
5. **Bloom is 44% above the entropy floor** — for static sets (SSTables, shipped artifacts) ribbon and binary fuse filters buy that back.
6. **Filters can't see ranges** — range scans get nothing; prefix blooms help only for prefix-bounded iteration.
7. **Never OR-merge or deserialize across mismatched m/k/hash/seed** — that's how a filter learns to produce false negatives, the one lie it must never tell.
8. **Allocate FPR budget unevenly** (Monkey): spend bits where runs are small and probed often; let the biggest tier run looser.
9. **Probabilistic ≠ private** — repeated observation of a filter reconstructs its contents (BIP-37); it's a performance structure, not an information barrier.

---

## References

- Bloom, B. H. (1970). *Space/Time Trade-offs in Hash Coding with Allowable Errors*. CACM.
- Broder, A., & Mitzenmacher, M. (2004). *Network Applications of Bloom Filters: A Survey*. Internet Mathematics.
- Kirsch, A., & Mitzenmacher, M. (2006). *Less Hashing, Same Performance: Building a Better Bloom Filter*. ESA.
- Putze, F., Sanders, P., & Singler, J. (2007). *Cache-, Hash- and Space-Efficient Bloom Filters*. WEA.
- Bose, P., et al. (2008). *On the False-Positive Rate of Bloom Filters*. Information Processing Letters.
- Fan, B., Andersen, D., Kaminsky, M., & Mitzenmacher, M. (2014). *Cuckoo Filter: Practically Better Than Bloom*. CoNEXT.
- Maggs, B., & Sitaraman, R. (2015). *Algorithmic Nuggets in Content Delivery*. ACM SIGCOMM CCR. (One-hit wonders / cache admission.)
- Dayan, N., Athanassoulis, M., & Idreos, S. (2017). *Monkey: Optimal Navigable Key-Value Store*. SIGMOD.
- Graf, T. M., & Lemire, D. (2020). *Xor Filters: Faster and Smaller Than Bloom and Cuckoo Filters*. ACM JEA; and (2022) *Binary Fuse Filters*.
- RocksDB Wiki: *RocksDB Bloom Filter* (format_version 5 fast local bloom) and *Ribbon Filter*.
- BIP-37 (Connection Bloom Filtering) and BIP-158 (Compact Block Filters) — Bitcoin Improvement Proposals.
