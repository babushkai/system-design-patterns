# ACID Transactions

## TL;DR

ACID is a set of properties that guarantee database transactions are processed reliably. But "ACID" is a marketing term - the actual guarantees vary wildly between databases.

---

## The Problem ACID Solves

Consider a bank transfer: move $100 from Account A to Account B.

```
1. Read balance of A: $500
2. Subtract $100 from A: $400
3. Write new balance to A
4. Read balance of B: $200
5. Add $100 to B: $300
6. Write new balance to B
```

What can go wrong?
- Crash after step 3: A lost $100, B gained nothing
- Concurrent read at step 1: Another transaction sees stale data
- Power failure: Writes partially persisted to disk

---

## Atomicity

### What It Actually Means

Atomicity does NOT mean "all operations happen instantaneously."

**Atomicity means: all-or-nothing execution.**

If a transaction commits, all its writes are applied. If it aborts, none are applied.

### The Mechanism: Undo Logging

```
BEGIN TRANSACTION
1. Write to undo log: "A was $500"
2. Update A to $400
3. Write to undo log: "B was $200"  
4. Update B to $300
COMMIT
```

If crash before COMMIT: replay undo log, restore original values.

### Important Nuance

Atomicity is about the database, not the application. If your app sends an email then commits, the email is sent even if the transaction later fails.

---

## Consistency

### The Most Overloaded Term in Computing

ACID Consistency differs from:
- CAP theorem consistency (linearizability)
- Replica consistency (eventual consistency)

### What It Actually Means

**Consistency means: transactions preserve database invariants.**

Constraints like:
- Account balance >= 0
- Foreign keys must reference existing rows
- Unique constraints

The database enforces rules you define. It cannot enforce business logic you haven't expressed as constraints.

---

## Isolation

### The Core Challenge

Isolation answers: "What do concurrent transactions see?"

The ideal (serializability): transactions execute as if they ran one-at-a-time.

The reality: full isolation is expensive. Databases offer weaker levels.

### Isolation Levels Summary

| Level | Dirty Reads | Non-Repeatable Reads | Phantom Reads |
|-------|-------------|----------------------|---------------|
| Read Uncommitted | Yes | Yes | Yes |
| Read Committed | No | Yes | Yes |
| Repeatable Read | No | No | Yes |
| Serializable | No | No | No |

### Implementation Approaches

1. **Locking**: Transactions acquire locks, block each other
2. **MVCC**: Keep multiple versions, readers don't block writers
3. **OCC**: Assume no conflicts, validate at commit

---

## Durability

### What It Actually Means

**Durability means: committed transactions survive crashes.**

Once COMMIT returns, data is safe even if server crashes immediately.

### The Mechanism: Write-Ahead Logging (WAL)

```
1. Write changes to WAL (sequential disk write)
2. fsync() the WAL
3. Return COMMIT to client
4. Apply to data files (can be async)
```

### Durability Tradeoffs

fsync is slow (~10ms HDD, ~0.1ms SSD). Options:
- Batch commits: group transactions
- Async replication: risk data loss
- Disable fsync: dangerous

**Single disk isn't enough.** True durability requires replication.

---

## ACID in Practice

| Database | Default Isolation | Durability |
|----------|------------------|------------|
| PostgreSQL | Read Committed | fsync |
| MySQL InnoDB | Repeatable Read | fsync |
| MongoDB | Read Committed | Journaled |
| SQLite | Serializable | fsync |

### Warning: Check Your Defaults

Many databases have unsafe defaults for performance:
- MySQL: innodb_flush_log_at_trx_commit
- PostgreSQL: synchronous_commit
- MongoDB: write concern

---

## Common Misconceptions

1. **"ACID means my data is safe"** - ACID doesn't prevent app bugs, hardware failures, or operator errors
2. **"NoSQL isn't ACID"** - Many support transactions (MongoDB 4.0+, DynamoDB)
3. **"Distributed can't be ACID"** - Spanner, CockroachDB prove otherwise

---

## Key Takeaways

1. Atomicity: All-or-nothing via undo logs
2. Consistency: App-level invariants, weakest guarantee
3. Isolation: Has levels, affects concurrency
4. Durability: WAL + fsync, but replicate for safety
