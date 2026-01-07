# Isolation Levels

## TL;DR

Isolation levels define what concurrent transactions can see. Higher isolation = fewer anomalies but worse performance. Most apps use Read Committed or Repeatable Read. Serializable is rarely used due to cost.

---

## Why Isolation Levels Exist

Full serialization is expensive. It requires either:
- Heavy locking (transactions block each other)
- Tracking all reads/writes (abort on conflict)

Most applications can tolerate weaker guarantees for better performance.

---

## The Anomalies

### Dirty Read

Reading uncommitted data from another transaction.

```
T1: BEGIN
T1: UPDATE accounts SET balance = 0 WHERE id = 1
                                            T2: SELECT balance FROM accounts WHERE id = 1
                                            T2: Returns 0 (uncommitted!)
T1: ROLLBACK
```

T2 saw data that never existed in committed state.

**Prevented by: Read Committed and above**

---

### Non-Repeatable Read (Read Skew)

Reading the same row twice yields different values.

```
T1: BEGIN
T1: SELECT balance FROM accounts WHERE id = 1  -- Returns 100
                                            T2: UPDATE accounts SET balance = 50 WHERE id = 1
                                            T2: COMMIT
T1: SELECT balance FROM accounts WHERE id = 1  -- Returns 50!
T1: COMMIT
```

T1's view of the world changed mid-transaction.

**Prevented by: Repeatable Read and above**

---

### Phantom Read

A query returns different rows when executed twice.

```
T1: BEGIN
T1: SELECT COUNT(*) FROM accounts WHERE balance > 100  -- Returns 3
                                            T2: INSERT INTO accounts VALUES (4, 200)
                                            T2: COMMIT
T1: SELECT COUNT(*) FROM accounts WHERE balance > 100  -- Returns 4!
T1: COMMIT
```

New rows "appeared" mid-transaction.

**Prevented by: Serializable**

---

### Write Skew

Two transactions read overlapping data, make decisions, write non-overlapping data.

```
Constraint: At least one doctor must be on call

T1: SELECT COUNT(*) FROM doctors WHERE on_call = true  -- Returns 2
T1: I can go off-call, there's another doctor
                                            T2: SELECT COUNT(*) FROM doctors WHERE on_call = true  -- Returns 2
                                            T2: I can go off-call, there's another doctor
T1: UPDATE doctors SET on_call = false WHERE id = 1
T1: COMMIT
                                            T2: UPDATE doctors SET on_call = false WHERE id = 2
                                            T2: COMMIT
```

Result: Zero doctors on call. Constraint violated.

**Prevented by: Serializable only**

---

## Isolation Level Implementations

### Read Committed

**What you get:**
- Only see committed data
- Each statement sees latest committed data

**How it works:**
- Readers acquire row-level shared locks, release immediately
- Writers acquire exclusive locks, hold until commit
- Or: MVCC where each statement sees its own snapshot

**The catch:**
Non-repeatable reads allowed. Same query can return different results.

---

### Repeatable Read (Snapshot Isolation)

**What you get:**
- See a consistent snapshot from transaction start
- Same query always returns same results

**How it works (MVCC):**
```
Transaction starts at timestamp T=100

Row versions:
  id=1: value=A at T=50, value=B at T=150
  
Transaction sees value=A (latest version <= 100)
Even if another transaction commits value=B at T=150
```

**The catch:**
- Phantoms still possible (new rows can appear in range queries)
- Write skew possible

**MySQL's "Repeatable Read":**
Actually closer to snapshot isolation. Uses gap locks to prevent some phantoms, but not write skew.

---

### Serializable

**What you get:**
- Transactions appear to execute one at a time
- No anomalies possible

**How it works (multiple approaches):**

1. **Actual Serial Execution**
   - Single-threaded execution
   - Used by VoltDB, Redis
   - Fast if transactions are short

2. **Two-Phase Locking (2PL)**
   - Acquire locks as you go, release all at commit
   - Shared locks for reads, exclusive for writes
   - Predicate locks for range queries
   - Deadlock risk, poor concurrency

3. **Serializable Snapshot Isolation (SSI)**
   - Optimistic: assume no conflicts
   - Track reads/writes, detect conflicts at commit
   - Abort if serialization would be violated
   - Used by PostgreSQL, CockroachDB

---

## Comparison Table

| Level | Dirty Read | Non-Repeatable | Phantom | Write Skew | Performance |
|-------|------------|----------------|---------|------------|-------------|
| Read Uncommitted | Yes | Yes | Yes | Yes | Best |
| Read Committed | No | Yes | Yes | Yes | Good |
| Repeatable Read | No | No | Maybe | Yes | Medium |
| Serializable | No | No | No | No | Worst |

---

## Practical Guidance

### When to Use Read Committed

- Default choice for most OLTP applications
- Analytics queries that don't need consistency
- When stale reads are acceptable

### When to Use Repeatable Read

- Need consistent view during transaction
- Generating reports
- Read-modify-write patterns

### When to Use Serializable

- Financial transactions
- Inventory management (prevent overselling)
- Any case where write skew would be catastrophic

### Application-Level Alternatives

Often cheaper than serializable:
- **SELECT FOR UPDATE**: Acquire exclusive lock on read
- **Optimistic Locking**: Version column, check on write
- **Application Locks**: External mutex/semaphore

```sql
-- Optimistic locking example
UPDATE products 
SET stock = stock - 1, version = version + 1
WHERE id = 123 AND version = 5;  -- Fails if version changed

-- Returns 0 rows affected if concurrent modification
```

---

## Database-Specific Notes

### PostgreSQL
- Read Committed: Each statement sees new snapshot
- Repeatable Read: True snapshot isolation
- Serializable: SSI-based, very good

### MySQL InnoDB
- "Repeatable Read" is default but has gaps
- Uses gap locks to prevent some phantoms
- Write skew still possible
- Serializable uses full locking

### Oracle
- Only Read Committed and Serializable
- "Serializable" is actually snapshot isolation
- True serializability requires application logic

---

## Key Takeaways

1. Higher isolation = fewer bugs, worse performance
2. Read Committed is usually good enough
3. Repeatable Read adds consistency within transaction
4. Serializable prevents all anomalies but costs throughput
5. Application-level locking is often cheaper than serializable
6. Database "isolation levels" don't always match SQL standard
