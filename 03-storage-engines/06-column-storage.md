# Column-Oriented Storage

## TL;DR

Column-oriented storage stores data by column rather than by row. This enables excellent compression and allows reading only needed columns. Ideal for analytics workloads (OLAP) that scan many rows but few columns. Row stores excel at transactional workloads (OLTP) that access entire rows. Most data warehouses use columnar storage.

---

## Row vs Column Storage

### Row-Oriented (Traditional)

```
Table: Users
  id | name    | age | city
  ---+---------+-----+---------
  1  | Alice   | 30  | NYC
  2  | Bob     | 25  | LA
  3  | Charlie | 35  | Chicago

On disk:
  [1, Alice, 30, NYC][2, Bob, 25, LA][3, Charlie, 35, Chicago]
  
  Entire row stored contiguously
```

### Column-Oriented

```
Same table, stored by column:

id:    [1, 2, 3]
name:  [Alice, Bob, Charlie]
age:   [30, 25, 35]
city:  [NYC, LA, Chicago]

Each column stored separately
```

---

## Why Columns?

### Query Pattern Difference

```
OLTP (Transactional):
  SELECT * FROM users WHERE id = 123
  → Need all columns for one row
  → Row storage efficient
  
OLAP (Analytical):
  SELECT AVG(age) FROM users WHERE city = 'NYC'
  → Need 2 columns, millions of rows
  → Reading all columns is wasteful
```

### Column Storage Advantages

```
Query: SELECT AVG(age) FROM users WHERE city = 'NYC'

Row storage reads:
  [1, Alice, 30, NYC][2, Bob, 25, LA][3, Charlie, 35, Chicago]...
  Read everything, use only 2 columns
  
Column storage reads:
  age:  [30, 25, 35, ...]
  city: [NYC, LA, Chicago, ...]
  Skip id, name columns entirely
  
I/O reduction: 2/4 = 50% in this example
Real tables with 100+ columns: 95%+ reduction
```

---

## Compression Benefits

### Same-Type Data Compresses Better

```
Row storage:
  [1, Alice, 30, NYC][2, Bob, 25, LA]...
  Mixed types (int, string, int, string)
  Poor compression

Column storage:
  age: [30, 25, 35, 40, 28, 30, 35, 30, ...]
  Same type, similar values
  Excellent compression
```

### Run-Length Encoding (RLE)

```
city column (sorted):
  [Chicago, Chicago, Chicago, LA, LA, NYC, NYC, NYC, NYC, NYC]
  
RLE compressed:
  [(Chicago, 3), (LA, 2), (NYC, 5)]
  
10 values → 3 pairs
```

### Dictionary Encoding

```
status column:
  [pending, active, active, pending, active, completed, ...]
  
Dictionary:
  0 = pending
  1 = active
  2 = completed
  
Encoded: [0, 1, 1, 0, 1, 2, ...]
  String → 2 bits
  Massive space savings for low-cardinality columns
```

### Bit-Packing

```
age column (0-100 range):
  Standard int: 32 bits per value
  Bit-packed: 7 bits per value (2^7 = 128)
  
4.5x space reduction
```

### Compression Comparison

| Encoding | Best For | Ratio |
|----------|----------|-------|
| RLE | Sorted, repetitive | 10-100x |
| Dictionary | Low cardinality | 10-50x |
| Bit-packing | Small integers | 2-8x |
| Delta | Timestamps, sequences | 5-20x |
| LZ4/Zstd | General purpose | 2-5x |

---

## Column Store Architecture

### Physical Layout

```
Table: sales
Columns: date, product_id, quantity, price, region

Files on disk:
  sales_date.col      (dates only)
  sales_product_id.col
  sales_quantity.col
  sales_price.col
  sales_region.col

Each file:
  - Sorted by some key (often date)
  - Divided into row groups
  - Each group independently compressed
```

### Row Groups

```
┌─────────────────────────────────────────────────────┐
│                    Row Group 1                      │
│  date:[...] product_id:[...] quantity:[...] ...     │
├─────────────────────────────────────────────────────┤
│                    Row Group 2                      │
│  date:[...] product_id:[...] quantity:[...] ...     │
├─────────────────────────────────────────────────────┤
│                    Row Group 3                      │
│  date:[...] product_id:[...] quantity:[...] ...     │
└─────────────────────────────────────────────────────┘

Row group size: Typically 100K - 1M rows
Enables:
  - Parallel processing
  - Predicate pushdown (skip row groups)
  - Memory efficiency
```

### Reconstructing Rows

```
Need to join columns back together:
  Position 0: date[0], product_id[0], quantity[0], ...
  Position 1: date[1], product_id[1], quantity[1], ...
  
Same position across columns = same row
Called "late materialization"
```

---

## Query Execution

### Traditional (Early Materialization)

```
Query: SELECT product_id, quantity 
       FROM sales 
       WHERE region = 'US' AND quantity > 100

1. Scan region column, find matching row IDs
2. For each matching row:
   - Fetch product_id, quantity, region
   - Build full row
   - Apply predicates
   - Return results

Reconstructs rows early, even if filtered out later
```

### Columnar (Late Materialization)

```
Same query:

1. Scan region column → bitmap of US rows
2. Scan quantity column → bitmap of quantity > 100
3. AND bitmaps → final row IDs
4. Only for matching rows:
   - Fetch product_id, quantity
   - Return results

Only reconstruct needed rows at the end
Significant speedup for selective queries
```

### Vectorized Execution

```
Process columns in batches (vectors):

Instead of:
  for row in rows:
    result = row.quantity * row.price
    
Do:
  quantities = load_vector(1024 values)
  prices = load_vector(1024 values)
  results = quantities * prices  # SIMD operation

Benefits:
  - CPU cache efficiency
  - SIMD parallelism
  - Reduced interpretation overhead
```

---

## Parquet Format

### File Structure

```
┌─────────────────────────────────────────────┐
│ Magic Number: PAR1                          │
├─────────────────────────────────────────────┤
│ Row Group 1                                 │
│   Column Chunk 1: [Pages...] + Column Meta  │
│   Column Chunk 2: [Pages...] + Column Meta  │
│   ...                                       │
├─────────────────────────────────────────────┤
│ Row Group 2                                 │
│   ...                                       │
├─────────────────────────────────────────────┤
│ Footer                                      │
│   File Metadata                             │
│   Row Group Metadata                        │
│   Column Metadata                           │
│   Schema                                    │
├─────────────────────────────────────────────┤
│ Footer Length (4 bytes)                     │
├─────────────────────────────────────────────┤
│ Magic Number: PAR1                          │
└─────────────────────────────────────────────┘
```

### Page Types

```
Data Page:
  - Actual column values
  - Definition levels (for nulls)
  - Repetition levels (for nested data)
  
Dictionary Page:
  - Dictionary for dictionary encoding
  - Stored once per column chunk
  
Data Page V2:
  - Improved encoding
  - Header contains statistics
```

### Metadata for Query Planning

```
Footer contains per-column stats:
  - Min/max values
  - Null count
  - Distinct count (optional)

Query: WHERE date >= '2024-01-01'
  Check row group metadata
  Skip row groups where max_date < '2024-01-01'
```

---

## ORC Format

### Structure

```
Similar to Parquet, used heavily in Hive/Hadoop:

┌─────────────────────────────────────────────┐
│ Stripe 1                                    │
│   Index Data (min/max, positions)           │
│   Row Data (column streams)                 │
│   Stripe Footer                             │
├─────────────────────────────────────────────┤
│ Stripe 2                                    │
│   ...                                       │
├─────────────────────────────────────────────┤
│ File Footer                                 │
│   Type information                          │
│   Stripe information                        │
│   Column statistics                         │
├─────────────────────────────────────────────┤
│ Postscript                                  │
│   Compression type, version                 │
└─────────────────────────────────────────────┘
```

### ORC vs Parquet

| Aspect | Parquet | ORC |
|--------|---------|-----|
| Origin | Twitter/Cloudera | Facebook/Hortonworks |
| Ecosystem | Spark, general | Hive, Presto |
| Nested data | Better | Good |
| ACID updates | No | Yes (with Hive) |
| Predicate pushdown | Good | Better indexes |

---

## Indexing in Column Stores

### Zone Maps (Min/Max Index)

```
For each row group or page:
  Store min and max value

Query: WHERE price > 1000

Row Group 1: min=50, max=500   → skip
Row Group 2: min=200, max=1500 → scan
Row Group 3: min=800, max=2000 → scan
Row Group 4: min=5000, max=8000 → scan (all match)
```

### Bitmap Indexes

```
For low-cardinality columns:

region = 'US':   [1, 0, 1, 1, 0, 1, ...]
region = 'EU':   [0, 1, 0, 0, 1, 0, ...]
region = 'APAC': [0, 0, 0, 0, 0, 0, ...]

Query: WHERE region IN ('US', 'EU')
  Bitmap OR: [1, 1, 1, 1, 1, 1, ...]
  Very fast set operations
```

### Bloom Filters on Columns

```
Store Bloom filter per column chunk

Query: WHERE product_id = 'ABC123'

Check Bloom filter:
  Definitely not in chunk → skip
  Maybe in chunk → scan

Useful for high-cardinality equality predicates
```

---

## Writes in Column Stores

### The Challenge

```
INSERT single row:
  Row store: Append to one file
  Column store: Append to N files (one per column)
  
Much more I/O for writes
```

### Batch Writes

```
Buffer writes in memory (row format)
Periodically flush as column chunks

Write pattern:
  1. Write to in-memory buffer
  2. When buffer full (e.g., 10K rows):
     - Convert to columnar
     - Compress
     - Write to disk

Batching amortizes conversion overhead
```

### Delta Stores

```
MemStore (row format) + Column files (column format)

Reads: Merge MemStore + Column files
Writes: Go to MemStore only

Periodically compact MemStore into column files
Similar to LSM tree approach
```

### Updates and Deletes

```
Option 1: Delete bitmap
  Mark rows as deleted
  Compact to remove later
  
Option 2: Merge-on-read
  Store updates separately
  Merge during query
  
Option 3: Copy-on-write
  Rewrite affected row groups
  Expensive but simple
```

---

## Systems Using Columnar Storage

### Analytical Databases

```
ClickHouse:
  - Native columnar
  - MergeTree engine
  - Very fast for time-series

Snowflake:
  - Columnar on cloud storage
  - Automatic clustering
  - Micro-partitions

BigQuery:
  - Capacitor columnar format
  - Dremel-style query engine
  - Serverless

Redshift:
  - Columnar PostgreSQL variant
  - Zone maps
  - Compression encoding per column
```

### Hybrid Stores

```
DuckDB:
  - Embedded columnar database
  - Vectorized execution
  - Great for local analytics

CockroachDB:
  - Row store primary
  - Columnar for analytics (experimental)
  
PostgreSQL:
  - Row store with columnar extensions
  - cstore_fdw, Citus columnar
```

---

## When to Use Columnar

### Good Fit

```
✓ Analytics/OLAP workloads
✓ Aggregations over many rows
✓ Queries use few columns
✓ Data is append-mostly
✓ Compression important
✓ Wide tables (100+ columns)
```

### Poor Fit

```
✗ Transactional/OLTP workloads
✗ Point lookups by primary key
✗ Frequent updates/deletes
✗ Queries need all columns
✗ Real-time requirements
✗ Narrow tables (few columns)
```

### Comparison

| Aspect | Row Store | Column Store |
|--------|-----------|--------------|
| Point lookup | Fast | Slow |
| Full scan | Slow | Fast |
| Aggregation | Slow | Fast |
| Insert single row | Fast | Slow |
| Bulk load | Medium | Fast |
| Compression | 2-3x | 10-100x |
| OLTP | Excellent | Poor |
| OLAP | Poor | Excellent |

---

## Key Takeaways

1. **Column storage reads only needed columns** - Huge I/O savings
2. **Same-type data compresses well** - 10-100x compression
3. **Late materialization improves performance** - Delay row reconstruction
4. **Vectorized execution uses SIMD** - CPU-efficient processing
5. **Row groups enable predicate pushdown** - Skip irrelevant data
6. **Parquet/ORC are standard formats** - Wide ecosystem support
7. **Writes are expensive** - Batch and buffer
8. **Use for analytics, not transactions** - Right tool for the job
