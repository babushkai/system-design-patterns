# Bloom Filters

## TL;DR

A Bloom filter is a probabilistic data structure for set membership testing. It can tell you "definitely not in set" or "probably in set" using far less memory than storing actual elements. False positives are possible; false negatives are not. Critical for avoiding unnecessary disk reads in databases and caches.

---

## The Problem

### Expensive Lookups

```
Query: Does key "xyz" exist?

Without Bloom filter:
  Check each SSTable on disk
  Multiple disk reads per lookup
  Slow for non-existent keys

With Bloom filter:
  Check in-memory filter first
  "Definitely not there" → skip disk read
  "Maybe there" → check disk

90%+ of disk reads avoided for negative lookups
```

---

## How It Works

### Structure

```
Bit array of m bits, initially all 0

┌─────────────────────────────────────────┐
│ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │
└─────────────────────────────────────────┘
  0 1 2 3 4 5 6 7 8 9 ...

k hash functions: h₁, h₂, h₃, ...
Each maps element to a bit position
```

### Adding an Element

```
Insert "hello":
  h₁("hello") = 3
  h₂("hello") = 7
  h₃("hello") = 12
  
Set bits 3, 7, 12 to 1

┌─────────────────────────────────────────┐
│ 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 │
└─────────────────────────────────────────┘
        ↑       ↑         ↑
        3       7         12
```

### Testing Membership

```
Query "hello":
  h₁("hello") = 3  → bit 3 is 1 ✓
  h₂("hello") = 7  → bit 7 is 1 ✓
  h₃("hello") = 12 → bit 12 is 1 ✓
  All bits set → "probably yes"

Query "world":
  h₁("world") = 5  → bit 5 is 0 ✗
  At least one bit is 0 → "definitely no"
```

### False Positives

```
After many insertions:
┌─────────────────────────────────────────┐
│ 1 0 1 1 0 1 1 1 0 1 0 1 1 0 1 1 0 1 0 1 │
└─────────────────────────────────────────┘

Query "never_inserted":
  h₁("never_inserted") = 3  → bit 3 is 1 ✓
  h₂("never_inserted") = 6  → bit 6 is 1 ✓
  h₃("never_inserted") = 11 → bit 11 is 1 ✓
  All bits set by OTHER elements!
  → False positive: "probably yes" but actually no
```

---

## Mathematics

### False Positive Probability

```
m = number of bits
n = number of elements
k = number of hash functions

After inserting n elements:
  Probability a bit is 0: (1 - 1/m)^(kn) ≈ e^(-kn/m)

False positive probability:
  p = (1 - e^(-kn/m))^k
```

### Optimal Parameters

```
Given n elements and desired false positive rate p:

Optimal bits: m = -n × ln(p) / (ln(2))²
Optimal hash functions: k = (m/n) × ln(2)

Example: n=1 million, p=1%
  m = -1,000,000 × ln(0.01) / (ln(2))² ≈ 9.6 million bits
  k = (9.6M/1M) × ln(2) ≈ 7 hash functions
  
  ~1.2 MB for 1 million elements with 1% false positive rate
```

### Bits Per Element

```
For p = 1%:   ~10 bits per element
For p = 0.1%: ~15 bits per element
For p = 0.01%: ~20 bits per element

Each ~0.7 additional bits per element → halve false positive rate
```

---

## Implementation

### Basic Implementation

```python
import mmh3  # MurmurHash3

class BloomFilter:
    def __init__(self, expected_elements, fp_rate):
        self.size = self._optimal_size(expected_elements, fp_rate)
        self.hash_count = self._optimal_hash_count(self.size, expected_elements)
        self.bit_array = bytearray((self.size + 7) // 8)
    
    def _optimal_size(self, n, p):
        return int(-n * math.log(p) / (math.log(2) ** 2))
    
    def _optimal_hash_count(self, m, n):
        return int((m / n) * math.log(2))
    
    def _hash(self, item, seed):
        return mmh3.hash(str(item), seed) % self.size
    
    def add(self, item):
        for i in range(self.hash_count):
            idx = self._hash(item, i)
            self.bit_array[idx // 8] |= (1 << (idx % 8))
    
    def might_contain(self, item):
        for i in range(self.hash_count):
            idx = self._hash(item, i)
            if not (self.bit_array[idx // 8] & (1 << (idx % 8))):
                return False  # Definitely not present
        return True  # Probably present
```

### Using Two Hash Functions

```python
# Optimization: Use only 2 hash functions
# Combine them to create k hash functions

def hash_i(item, i, h1, h2, m):
    return (h1 + i * h2) % m

h1 = hash(item)
h2 = hash2(item)

for i in range(k):
    idx = hash_i(item, i, h1, h2, m)
    # use idx
```

---

## Variants

### Counting Bloom Filter

Replace bits with counters.

```
Standard:   [0][1][0][1][1][0]  (bits)
Counting:   [0][2][0][1][3][0]  (counters, e.g., 4-bit)

Supports deletion:
  add("x")    → increment positions
  remove("x") → decrement positions

Trade-off: 4x+ more space
```

### Scalable Bloom Filter

Handle unknown number of elements.

```
Start with small filter
When too full (high FP rate):
  Create new, larger filter
  New elements go to new filter
  Query checks all filters

Maintains target FP rate as data grows
```

### Cuckoo Filter

Alternative with better properties.

```
Supports deletion (without counting)
Better space efficiency at <3% FP rate
Similar lookup speed

Structure: Hash table with cuckoo hashing
Stores fingerprints (not bits)
```

### Quotient Filter

Cache-friendly alternative.

```
Good locality of reference
Supports deletion and merging
Slightly higher memory than Bloom

Stores quotient and remainder of hash
Uses linear probing
```

---

## Use Cases

### Database/LSM Trees

```
Each SSTable has a Bloom filter
Before reading SSTable from disk:
  if bloom_filter.might_contain(key):
      read_sstable()  # ~1% false positive
  else:
      skip()  # Definitely not there

Huge savings for point lookups
```

### Distributed Systems

```
Cache stampede prevention:
  Before expensive computation:
    if bloom_filter.might_contain(request):
        might_be_in_cache = True
    else:
        definitely_not_cached = True

Routing decisions:
  Which server has this data?
  Check each server's Bloom filter
```

### Web Crawlers

```
Have we seen this URL before?
  if bloom_filter.might_contain(url):
      # Probably seen, skip or verify in DB
  else:
      # New URL, definitely crawl

Billions of URLs with modest memory
```

### Spell Checkers

```
Is this a valid word?
  if dictionary_bloom.might_contain(word):
      # Probably valid
  else:
      # Definitely misspelled

Fast first-pass check
```

### CDN / Caching

```
Is this content in cache?
  if cache_bloom.might_contain(content_id):
      check_actual_cache()
  else:
      fetch_from_origin()

Avoid cache misses going to disk
```

---

## Operational Considerations

### Sizing Guidelines

```
Too small:
  - High false positive rate
  - Defeats the purpose

Too large:
  - Wasted memory
  - Possibly slower (cache misses)

Rule of thumb:
  10 bits per element → ~1% FP
  15 bits per element → ~0.1% FP
```

### Memory Calculation

```
For 10 million keys with 1% FP:
  Bits needed: 10M × 10 = 100M bits = 12.5 MB
  Hash functions: ~7

For 1 billion keys with 0.1% FP:
  Bits needed: 1B × 15 = 15B bits = 1.875 GB
  Hash functions: ~10
```

### Serialization

```python
# Save to disk
def save(self, filename):
    with open(filename, 'wb') as f:
        f.write(struct.pack('II', self.size, self.hash_count))
        f.write(self.bit_array)

# Load from disk
@classmethod
def load(cls, filename):
    with open(filename, 'rb') as f:
        size, hash_count = struct.unpack('II', f.read(8))
        bit_array = bytearray(f.read())
    # Reconstruct filter
```

### Merging Filters

```
Two Bloom filters with same size and hash functions:
  merged = filter1.bit_array | filter2.bit_array

Result contains union of elements
FP rate increases
```

---

## Limitations

### No Deletion (Standard)

```
Why can't we delete?
  Element A sets bits: 3, 7, 12
  Element B sets bits: 3, 8, 15
  
Delete A: Clear bits 3, 7, 12
  But bit 3 was also set by B!
  Now B shows "not present" → false negative!

Use counting Bloom filter if deletion needed
```

### No Enumeration

```
Cannot list elements in filter
Filter only answers: "Is X present?"
To enumerate, need separate storage
```

### False Positive Rate Increases

```
As filter fills up:
  More bits set
  Higher FP rate
  
If n grows beyond expected:
  FP rate degrades
  Need to resize (rebuild with larger filter)
```

---

## Bloom Filters in Practice

### RocksDB Configuration

```cpp
Options options;
options.filter_policy.reset(NewBloomFilterPolicy(
    10,    // bits per key
    false  // use full filter (not block-based)
));
```

### Cassandra Configuration

```yaml
# In table schema
bloom_filter_fp_chance: 0.01  # 1% false positive rate

# Trade-off:
# Lower FP → more memory, fewer disk reads
# Higher FP → less memory, more disk reads
```

### Redis

```
# Using RedisBloom module
BF.ADD myfilter item1
BF.EXISTS myfilter item1  # Returns 1
BF.EXISTS myfilter item2  # Returns 0 (definitely not)
```

---

## Comparison

| Structure | Space | Lookup | Insert | Delete | FP Rate |
|-----------|-------|--------|--------|--------|---------|
| HashSet | O(n) | O(1) | O(1) | O(1) | 0% |
| Bloom | O(n) bits | O(k) | O(k) | No | ~1% |
| Counting Bloom | O(n) × 4 | O(k) | O(k) | O(k) | ~1% |
| Cuckoo | O(n) bits | O(1) | O(1)* | O(1) | ~1% |

*Cuckoo insert may require rehashing

---

## Key Takeaways

1. **False positives, never false negatives** - "Maybe yes" or "definitely no"
2. **10 bits per element for 1% FP** - Simple sizing rule
3. **Critical for LSM trees** - Avoid disk reads
4. **No deletion without counters** - Standard Bloom is insert-only
5. **Space-efficient set approximation** - Bits vs full elements
6. **Merge with OR** - Union of filters is trivial
7. **Size upfront** - Can't resize without rebuilding
8. **Cuckoo for deletions** - Modern alternative
