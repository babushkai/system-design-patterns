# Isolation Levels

## TL;DR

An isolation level is a contract over **which transaction histories may commit**. It is not a performance preset and its SQL name is not portable across engines. Start from a business invariant, construct the smallest concurrent history that could break it, and then choose a mechanism that rejects that history: a declarative constraint, one atomic statement, an explicit lock, compare-and-swap, snapshot isolation, or serializability. Multi-version concurrency control makes readers cheap by retaining old versions; it does not by itself make a history serializable. Lock-based serializability waits, serializable snapshot isolation aborts dangerous dependency structures, and both move cost into different queues. Every application using an abort-capable level needs a bounded retry protocol, a way to resolve ambiguous commits, and a boundary that keeps external effects out of speculative transactions.

---

## Scope: Transaction Histories, Not Replica Freshness

Transaction isolation governs concurrent access to one transactional authority through snapshots, locks, serialization conflicts, and retry behavior.

- [ACID Transactions](01-acid-transactions.md) covers atomic commit, durability, WAL, and recovery.
- [Consistency Models](04-consistency-models.md) covers client-visible guarantees across replicas and services.
- [Distributed Transactions](../02-distributed-databases/07-distributed-transactions.md) covers commit across multiple resource managers.
- [Idempotency](08-idempotency.md) covers duplicate requests and durable effect deduplication.

Those boundaries matter. A serializable transaction on a lagging read replica can be internally serializable yet stale. A linearizable key-value operation can be individually current yet fail a multi-row invariant. “The database is ACID” does not specify either behavior.

---

## Model the History Before Naming the Level

Write a transaction as reads and writes over logical items. `r1(x=0)` means transaction 1 read version 0 of `x`; `w2(y=1)` means transaction 2 wrote `y`; `c1` is commit and `a1` is abort.

Two histories are **conflict-equivalent** when they preserve the order of every conflicting pair: read/write, write/read, or write/write on the same logical item. A committed history is serializable when it is equivalent to some serial execution of those transactions. Build a serialization graph with one vertex per committed transaction and an edge $T_i \rightarrow T_j$ when their conflicting operations require $T_i$ before $T_j$. A cycle proves that no serial explanation exists.

Serializability deliberately says nothing about wall-clock order. If transaction B starts after transaction A has returned, a merely serializable system may still choose a serialization order that places B before A. **Strict serializability** adds real-time order. Linearizability is the corresponding single-operation/object property; strict serializability applies the idea to transactions.

Snapshot isolation (SI) is different again:

1. a transaction reads from a stable snapshot,
2. its writes become visible atomically at commit, and
3. concurrent transactions that write the same item cannot both commit, usually through a first-committer-wins rule.

SI prevents dirty reads and many lost updates, but two transactions may read the same predicate and write different rows. Their writes do not conflict, so both commit even when the combined state violates an invariant. That is write skew, and it is the canonical proof that snapshot isolation is not serializable.

### SQL names are lossy labels

The SQL names `READ COMMITTED`, `REPEATABLE READ`, and `SERIALIZABLE` describe minimum observable behavior, not a universal implementation. Engines differ in whether a snapshot lasts a statement or transaction, whether concurrent updates wait or abort, how range predicates are protected, and whether “serializable” means strict two-phase locking, optimistic validation, or serializable snapshot isolation. Treat the engine manual and a concurrency test as part of the interface contract.

---

## Anomalies as Broken Invariants

Memorizing four ANSI phenomena is insufficient. The useful question is: *which dependency cycle can this operation create?*

| Anomaly | Minimal shape | Broken assumption | Typical defense |
|---|---|---|---|
| Dirty read | T2 reads T1's uncommitted write; T1 aborts | Decisions use committed state | Read committed or stronger |
| Non-repeatable read | T1 reads `x`; T2 commits `x`; T1 rereads `x` | One transaction has one view | Transaction snapshot or locking read |
| Read skew / fractured read | T1 sees new `x` and old related `y` | Correlated values move atomically | One consistent snapshot |
| Lost update | T1 and T2 read `x=10`; both write derived values | Both transformations survive | Atomic update, version CAS, write conflict |
| Phantom | T1 evaluates predicate P; T2 inserts a matching row | Predicate result stays stable | Range/predicate protection |
| Write skew | T1 and T2 read overlapping state, write disjoint rows | Cross-row invariant remains true | Constraint, explicit lock, serializable validation |

### Lost update: the application overwrites evidence

```
T1: read seats_remaining = 2
T2: read seats_remaining = 2
T1: write seats_remaining = 1; commit
T2: write seats_remaining = 1; commit
```

Two reservations succeeded, but only one decrement remains. Raising isolation is one option; expressing the transition as one conditional statement is often more direct:

```sql
UPDATE flights
   SET seats_remaining = seats_remaining - 1
 WHERE flight_id = :id
   AND seats_remaining > 0;
```

The invariant is now enforced at the write item. The application treats `row_count = 0` as “sold out.” No preliminary read participates in correctness.

### Write skew: disjoint writes can still conflict logically

Suppose at least one doctor must remain on call:

```
Initial: Alice=on, Bob=on

T1 reads Alice=on, Bob=on       T2 reads Alice=on, Bob=on
T1 writes Alice=off             T2 writes Bob=off
T1 commits                      T2 commits

Final: Alice=off, Bob=off
```

The rows written are disjoint, so a row-level write conflict cannot detect the broken predicate. Defenses, in order of locality, are:

1. redesign the state so one constrained row represents available coverage;
2. lock a stable parent or guard row shared by both transitions;
3. use a serializable implementation that tracks predicate dependencies; or
4. serialize this command through one ordered owner.

“Use repeatable read” is not a defense unless that engine explicitly rejects this history.

### Phantoms are about predicates, not mystical rows

If a transaction checks `SUM(amount) < credit_limit` and another inserts a matching charge, the conflict is with the *predicate range*, not an existing tuple. A B-tree next-key lock can protect an indexed range. SSI can remember that the predicate was read and abort a later dangerous structure. An unindexed predicate may require coarse protection because the engine cannot name a narrow range.

---

## MVCC: Snapshots Are Retained Versions

Multi-version concurrency control (MVCC) replaces read/write blocking with version selection. A logical row has multiple physical versions, each associated with creation and retirement transaction metadata. A snapshot contains enough information to decide which transactions were visible when it was taken.

```text
logical account A

version v17: balance=100, created by tx 17, retired by tx 24
version v24: balance= 80, created by tx 24, still current

snapshot S started before tx 24 committed -> reads v17
snapshot S2 started after tx 24 committed -> reads v24
```

The central invariant is: **a reader sees only versions committed before its snapshot, plus its own writes, and never versions from aborted or still-invisible transactions**. The exact metadata differs (transaction identifiers, commit timestamps, undo records, or immutable key suffixes), but the visibility rule is the product contract.

### Statement and transaction snapshots

At a common read-committed implementation, each statement obtains a new snapshot. Two queries in one transaction can therefore observe different committed worlds. At repeatable-read/SI, the transaction keeps one snapshot, so rereads are stable. Neither choice automatically protects a value that the application read and then later overwrites; that depends on update conflict handling.

### Version retention is a capacity obligation

Old versions cannot be reclaimed while any active snapshot might still need them. Let:

- $u$ be updated bytes per second, including index/version overhead,
- $T_{old}$ be the age of the oldest snapshot, and
- $r$ be the fraction of changed bytes that require retained history.

An initial retention estimate is:

$$
B_{retained} \approx u \times T_{old} \times r
$$

This is a planning model, not a vendor promise. It explains why a forgotten analytical transaction can create table bloat, undo-space growth, longer vacuum work, or a replication-retention emergency. “Readers do not block writers” often means “readers create deferred cleanup work.”

### Snapshot construction is shared state

At high transaction rates, allocating transaction IDs, publishing active-transaction state, and constructing visibility snapshots can become coordination points. Measure snapshot acquisition time and active-set size; do not assume MVCC read scaling is free simply because row locks are absent.

---

## Three Ways to Enforce Serial Order

### Strict two-phase locking

Two-phase locking (2PL) acquires locks while a transaction executes and releases them only after the lock-growing phase. Strict 2PL holds write locks through commit, preventing other transactions from observing uncommitted writes and making recovery tractable.

Predicate correctness needs more than row locks. An engine may lock index gaps/ranges, materialize predicate locks, or fall back to coarser locks. The cost appears as waiting and deadlocks:

```text
T1 holds A, waits for B
T2 holds B, waits for A
                 -> wait-for cycle -> abort one victim
```

Global lock ordering prevents cycles when the application can enumerate resources. A deadlock detector handles the rest, but victim aborts still require the same retry discipline as optimistic concurrency.

### Optimistic validation

Optimistic concurrency executes against a snapshot, records a read/write set, and validates at commit that serial order remains possible. It avoids waiting when conflicts are rare. Under contention, completed work is thrown away and retried, so CPU and downstream reads amplify precisely when the system is stressed.

A version-column compare-and-swap is a narrow form:

```sql
UPDATE documents
   SET body = :body, version = version + 1
 WHERE id = :id AND version = :observed_version;
```

Zero updated rows means “the premise changed,” not an infrastructure error. The command must re-read and either merge, reject, or retry from the beginning.

### Serializable snapshot isolation

Serializable snapshot isolation (SSI) preserves nonblocking snapshot reads but tracks **read-write anti-dependencies**: T1 read a version that T2 later replaced, so T1 appears before T2 in the dependency graph. A cycle containing these edges is nonserializable.

Implementations need not retain the entire graph. SSI detects a dangerous shape:

$$
T_{in} \xrightarrow{rw} T_{pivot} \xrightarrow{rw} T_{out}
$$

and aborts a participant when commit ordering makes the structure capable of closing a cycle. This may abort a history that would ultimately have been serializable (a conservative false positive), but it never permits a known serialization cycle. Read-only safe snapshots can avoid tracking when the engine proves that concurrent read/write transactions cannot create the dangerous structure.

SSI moves the cost from lock waits to dependency metadata, predicate-read footprints, and aborts. Missing indexes can make one logical predicate cover a huge part of the database, increasing memory use and false conflicts even when the query itself is fast enough.

---

## Put Invariants at the Narrowest Reliable Boundary

Choose the cheapest mechanism whose scope exactly covers the invariant.

| Invariant | Preferred expression | Why |
|---|---|---|
| Email unique per tenant | Unique constraint on `(tenant_id, normalized_email)` | Database arbitrates every writer |
| Counter never below zero | Conditional atomic update | One write item, no read/write race |
| User edit based on version seen | Version CAS | Conflict is part of product behavior |
| Sum across mutable rows under limit | Serializable transaction or locked guard row | Predicate spans multiple items |
| One command produces DB state and event | Local transaction plus outbox | External broker is outside DB isolation |
| Cross-database invariant | Redesign ownership or distributed commit/workflow | One engine cannot serialize unknown state |

Declarative constraints are usually stronger than application checks because all writers pass the same authority. A preflight `SELECT` followed by `INSERT` is a user-experience optimization; the unique constraint is the correctness mechanism.

### Read-only transactions still make promises

A report may require a mutually consistent snapshot but not the latest value. A fraud decision may require freshness relative to a just-completed command. Label those separately. Route snapshot-consistent analytics and read-your-writes traffic differently when replicas cannot provide both.

### Authorization predicates belong inside the transaction model

Tenant and policy filters are correctness predicates. If an authorization check reads membership and a later statement mutates a protected resource, define whether membership revocation may race that mutation. Row-level security, explicit tenant keys, and transaction-local identity reduce “check then use” gaps. Never let a pooled connection retain a previous request's tenant, role, or isolation setting.

---

## Retry Is Part of the Transaction Protocol

Serializable, deadlock-detected, and optimistic transactions may abort during healthy operation. Retry the **entire logical transaction**, not only the last statement, because every earlier read belonged to an invalidated premise.

```text
for attempt in 0..max_attempts:
    begin transaction
    read all decision inputs
    validate request id / command state
    apply state transition
    insert outbox record in the same transaction
    try commit
      committed        -> return durable result
      serialization    -> rollback, jittered backoff, retry whole body
      definite failure -> rollback, classify
      outcome unknown  -> resolve by command id before any retry
```

The **outcome-unknown** branch is different from a serialization abort. A connection can disappear after the commit record becomes durable but before the response arrives. Blindly repeating the business action can double-charge or double-reserve. Give each logical command a stable identifier, store it with the result under a uniqueness constraint, and query that record to resolve the outcome. See [Idempotency](08-idempotency.md).

Keep network calls, email, and irreversible device actions outside the speculative transaction. Persist intent in an outbox or workflow state, commit, then deliver the effect with its own idempotency key. Holding database locks while waiting on an external dependency couples two unrelated failure domains and makes tail latency the lock duration.

### Bound the retry amplifier

If a transaction has abort probability $p$ and attempts are independent, expected attempts per success are:

$$
E[A] = \frac{1}{1-p}
$$

At $p=0.2$, useful throughput requires 1.25 attempts per success; at $p=0.5$, it requires 2. The independence assumption usually becomes optimistic under a hot-key incident because immediate retries collide again. Use randomized backoff, a finite retry budget, admission control, and a contention-specific error to callers. Retries are load; they do not create capacity.

---

## Capacity and Contention Model

For arrival rate $\lambda$ transactions per second and mean time $W$ seconds spent inside a transaction, Little's Law estimates in-flight transactions:

$$
N = \lambda W
$$

This connects application behavior to database state. A remote call that raises transaction time from 10 ms to 500 ms multiplies concurrent snapshots and possible lock holders by 50 at the same throughput.

Build an isolation budget from four resources:

1. **lock residency:** lock count, wait duration, and hot-resource queue depth;
2. **version residency:** update bytes times oldest-snapshot age;
3. **validation work:** read/predicate footprint and conflict edges retained until commit;
4. **retry work:** attempted transactions divided by committed logical commands.

Do not publish universal “serializable costs X%” claims. Cost is a workload property: conflict topology, transaction duration, index quality, write-set overlap, and the engine's mechanism dominate the result. Benchmark with the real invariant-preserving transaction shape and a skew distribution that includes hot tenants and keys.

---

## Failure Modes

### The pooled-session leak

Request A changes a session default to serializable or installs tenant context, then returns the connection without resetting it. Request B inherits that state. The symptom may be unexpected aborts or cross-tenant data exposure. Prefer transaction-scoped settings, reset connections on checkout/check-in, and test pool reuse explicitly.

### The long snapshot retention incident

An analyst opens a repeatable-read transaction and leaves it idle. Update traffic continues. Cleanup cannot remove versions visible to that snapshot; storage and replication logs grow, compaction/vacuum falls behind, and latency degrades. A restart “fixes” the oldest snapshot while discarding the evidence. Alert on oldest transaction/snapshot age well before disk pressure, label the owning workload, and enforce idle-in-transaction limits.

### The retry storm on a hot invariant

A new campaign makes one inventory row hot. Serializable aborts rise. Every client retries immediately, so attempted TPS grows while committed TPS falls. Queueing moves from the database to CPU and connection pools. Bound attempts, add jitter, shed optional work, and consider a per-key command queue or escrow-style allocation when the product permits it.

### The replica-read premise

A command reads eligibility from a lagging replica and writes to the primary. No primary isolation level can repair the stale premise because the read was outside the transaction authority. Move the decision read to the writer, use a session/LSN fence, or explicitly accept the stale decision as product semantics.

### The missing predicate index

A serializable query scans broadly because the intended filter lacks an index. Locking engines protect a large range; SSI engines record coarse predicate evidence. Unrelated transactions block or abort. Query plans are therefore part of the concurrency contract, not only a latency concern.

---

## Observability as History Evidence

Monitor the mechanism and the user-visible outcome together:

- commits, aborts, deadlock victims, and serialization failures by transaction class;
- retry attempts per logical command and exhausted retry budgets;
- lock wait time, wait-for edges, oldest waiter, and hot locked resource;
- transaction duration and idle-in-transaction age;
- oldest snapshot, retained-version/undo bytes, cleanup debt, and WAL retention;
- predicate/read-set memory where the engine exposes it;
- ambiguous commit resolutions and duplicate-command constraint hits;
- query-plan changes for invariant-bearing predicates.

Use stable operation names rather than raw SQL as metric labels. Raw statements contain sensitive values and create unbounded cardinality. During an incident, retain sampled transaction traces with transaction ID, snapshot/commit coordinate, rows or key ranges touched, wait/abort reason, and logical command ID.

---

## Migration and Verification

### Strengthening isolation safely

1. Inventory transaction classes and state the invariant each owns.
2. Add stable command IDs and whole-transaction retry support before enabling an abort-capable level.
3. Move external effects behind outbox/workflow boundaries.
4. Add missing constraints and predicate indexes.
5. Shadow or canary the stronger level for selected operations; measure abort topology, not only average latency.
6. Increase coverage gradually and keep a per-operation rollback switch.

A rollback to weaker isolation is a semantic rollback. Document which anomaly becomes legal again; do not call it a transparent performance change.

### Deterministic concurrency tests

Unit tests that run transactions sequentially prove little. Use barriers to force the dangerous interleaving:

```text
T1 read predicate ---- barrier ---- write A ---- commit
T2 read predicate ---- barrier ---- write B ---- commit
```

Assert that at most one commits or that the invariant still holds. Add histories for lost update, phantom insertion, deadlock victim retry, process crash after commit-before-response, pool reuse, long snapshots, and replica lag. For a custom store, record invocation/completion histories and check them against the promised model; fault injection should include pauses, partitions, and clock changes even though isolation itself must not depend on wall-clock correctness.

---

## Decision Framework

1. **Can the invariant be a constraint or one atomic conditional write?** Use that first.
2. **Is conflict expected product behavior, such as concurrent editing?** Use version CAS and expose merge/reject semantics.
3. **Does the invariant span a stable, small resource set?** Lock those resources in a global order.
4. **Does it span predicates or dynamically discovered rows?** Use true serializability and design for aborts or waits.
5. **Are reads outside the writer or effects outside the database?** Isolation alone is insufficient; add freshness fences and durable effect protocols.
6. **Is contention sustained rather than accidental?** Change ownership or data shape instead of increasing retries.

The correct level is selected per transaction class. One service can reasonably use read committed for independent inserts, version CAS for user edits, and serializable transactions for a cross-row financial invariant.

---

## Key Takeaways

1. Isolation is a set of admitted histories; engine labels are only hints.
2. Snapshot isolation prevents many anomalies but permits write skew across disjoint writes.
3. Constraints and atomic state transitions are narrower and often stronger than a global level change.
4. MVCC trades blocking for retained versions, cleanup debt, and snapshot-management work.
5. Serializable implementations pay through waits, validation metadata, aborts, or some combination.
6. Whole-transaction retry, ambiguous-commit resolution, and external-effect isolation are part of correctness.
7. Test the exact bad interleaving and observe committed logical commands, not merely SQL attempts.

---

## References

- Atul Adya, [*Weak Consistency: A Generalized Theory and Optimistic Implementations for Distributed Transactions*](https://pmg.csail.mit.edu/papers/adya-phd.pdf), MIT PhD thesis, 1999.
- Hal Berenson et al., [*A Critique of ANSI SQL Isolation Levels*](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-95-51.pdf), Microsoft Research, 1995.
- Dan R. K. Ports and Kevin Grittner, [*Serializable Snapshot Isolation in PostgreSQL*](https://www.vldb.org/pvldb/vol5/p1850_danrkports_vldb2012.pdf), VLDB, 2012.
- PostgreSQL, [Transaction Isolation](https://www.postgresql.org/docs/current/transaction-iso.html), current documentation.
- MySQL, [InnoDB Transaction Model](https://dev.mysql.com/doc/refman/8.4/en/innodb-transaction-model.html), 8.4 Reference Manual.
- Philip A. Bernstein, Vassos Hadzilacos, and Nathan Goodman, [*Concurrency Control and Recovery in Database Systems*](https://www.microsoft.com/en-us/research/people/philbe/book/), 1987.
