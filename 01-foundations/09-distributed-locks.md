# Distributed Locks and Leases

A distributed lock is a protocol for serializing a named critical section across processes that can pause, disconnect, restart, and disagree about time. Mutual exclusion at the lock service is only half the safety argument. A former holder may continue after its grant expires, so correctness-critical effects need a monotonically ordered fencing token that the protected resource validates atomically with the effect.

A distributed-lock contract must specify **lock API semantics, critical-section scope, wait queues, fencing at protected resources, multi-lock deadlocks, and lock-service operation**. [Leader Election](../02-distributed-databases/09-leader-election.md) owns long-lived leadership terms, activation barriers, and failover. [Leases, Heartbeats, and Recovery](../18-workflow-job-systems/08-leases-heartbeats-recovery.md) owns task-claim renewal and abandoned-work recovery. [Consensus Algorithms](../02-distributed-databases/08-consensus-algorithms.md) owns how a replicated service agrees on lock state.

## Primary Evidence and Scope

| Primary evidence | What it establishes | Boundary |
|---|---|---|
| Gray and Cheriton, SOSP 1989 | Time-bounded leases trade renewal traffic against recovery delay and clock uncertainty | File-cache leases, not a universal lock implementation |
| Burrows, Chubby, OSDI 2006 | A consensus-backed coarse-grained lock service, sessions, sequencers, lock delay, caching, and operational lessons | Historical Google design; optimized for coarse coordination, not high-rate transactions |
| Hunt et al., ZooKeeper, ATC 2010, plus official recipes | Ordered ephemeral nodes and predecessor watches support observable queued locks without waking every waiter | ZooKeeper-specific session and ordering semantics |
| etcd API guarantees and lock API | Mutations receive increasing revisions; lock ownership is tied to a lease/key and can be composed with etcd transactions | Guarantees inside etcd do not automatically fence another resource |
| PostgreSQL 18 documentation | Transaction/session advisory locks, row/table conflicts, deadlock detection, and lock observability | Locks apply within one database cluster and remain advisory where documented |

## Lock contract before implementation

“Lock `orders`” is not a contract. Define:

| Field | Required answer |
|---|---|
| **Resource key** | Which exact entity or invariant is serialized, and how is the key canonicalized? |
| **Mode** | Exclusive, shared/read, intent, semaphore, or try-once efficiency claim? |
| **Protected operations** | Which state changes require the grant, and where is that requirement enforced? |
| **Grant lifetime** | Transaction, session, explicit release, or time-bounded lease? |
| **Acquisition semantics** | Blocking, deadline, cancellation, queue order, priority, and lost-response behavior? |
| **Owner identity** | Process incarnation plus request identity, not a reusable hostname or PID? |
| **Fencing** | Which monotonic token is returned, and which resources reject older tokens? |
| **Failure outcome** | May work run twice, be delayed, be rejected, or require compensation? |
| **Scope of correctness** | Efficiency only, or does duplicate/concurrent execution violate a business invariant? |

An **efficiency lock** avoids duplicate computation while tolerating overlap. A **correctness lock** protects an invariant whose violation loses money, corrupts state, or exposes data. The latter requires a linearizable grant authority plus enforcement at every side-effect boundary; “the client checks that it still owns the key” is insufficient.

Prefer the smallest lock scope that contains the invariant. A global lock is easy to name and turns unrelated work into one queue. A per-entity lock increases concurrency but requires callers to derive exactly the same canonical entity key.

## State, authority, and invariants

The authoritative lock record contains at least:

```text
namespace and resource key
mode and compatibility class
owner incarnation and acquisition request ID
grant/fencing generation
session or lease identity and server-side deadline
grant state: WAITING, HELD, REVOKING, RELEASED, EXPIRED
queue/order identity when fairness is promised
record revision and policy revision
```

The protected resource separately stores the greatest accepted fencing generation for the relevant scope. That value is durable correctness state; it cannot live only in the holder or lock service cache.

**Reference-design invariants:**

1. At one lock-service revision, incompatible grants for one resource are never simultaneously valid.
2. A new grant has a generation greater than every prior grant for that resource, including across service failover and restore.
3. A protected effect with generation `g` is accepted only if `g` is not older than the resource's durable accepted generation.
4. Release, cancel, and renew identify the exact owner incarnation and grant; an old process cannot release a successor's lock.
5. A lost acquire response is resolved by request identity or read-back, not by issuing an unrelated second acquisition.
6. Cancellation removes a waiter without silently transferring its queue position to another request.
7. Multi-resource acquisitions use a declared order or transactional protocol and have bounded wait.
8. Losing the lock service does not authorize holders to extend grants locally.

## Control path and protected data path

~~~mermaid
sequenceDiagram
    participant C as Contender
    participant L as Linearizable lock service
    participant R as Protected resource

    C->>L: acquire(resource, incarnation, request_id, deadline)
    L->>L: serialize grant; allocate generation 84
    L-->>C: HELD(resource, generation 84, lease evidence)
    C->>R: mutate(command, generation 84, operation_id)
    R->>R: atomically compare fence and apply
    R-->>C: committed at generation 84
    C->>L: release(resource, incarnation, generation 84)
~~~

The **control path** serializes acquisition, queues contenders, renews sessions, and allocates generations. The **data path** is the business mutation. Keeping the control path off the data path improves availability only if the data path can validate cached grant evidence or a fencing token without calling the lock service synchronously for every operation.

**Inference:** if a protected resource must make a linearizable lock-service read before every write, the lock service is effectively in the write transaction anyway. A conditional write or database transaction at the resource may be simpler and stronger.

## Why expiry does not stop a holder

A lease makes abandoned state reclaimable; it does not terminate the process that held it. The holder may be paused by garbage collection, host suspension, page fault, debugger, scheduler starvation, or a partition. It cannot run a “lease lost” callback while paused.

Detailed renewal timing and failure detection belong to [Leader Election](../02-distributed-databases/09-leader-election.md) and [workflow lease recovery](../18-workflow-job-systems/08-leases-heartbeats-recovery.md). For locks, the key consequence is simple: self-demotion narrows risk, while fencing closes the stale-holder safety gap.

**Documented:** Chubby's 2006 paper describes sequencers that protected services can validate and a lock-delay fallback for services that cannot validate them. Lock delay waits before regranting after session loss; it reduces the chance that old requests survive but is a timing mitigation, not proof that an arbitrarily delayed request is gone.

## Fencing at the effect boundary

Suppose client A holds generation 83, pauses beyond its lease, and client B receives generation 84. The resource applies:

```text
apply(resource, command, generation, operation_id):
    atomically:
        reject if generation < highest_generation
        deduplicate operation_id
        highest_generation = max(highest_generation, generation)
        apply command
```

The comparison and effect must be one transaction or one conditional operation. Checking a token in application memory and then issuing an unfenced write creates a time-of-check/time-of-use race.

Token scope matters. A generation for `tenant/42/invoice/7` does not order writes to `tenant/42/account`. If one critical section writes a database, object store, and external API, each boundary must understand the token, or the unfenced effect must use a stable [idempotency key](./08-idempotency.md), transactional outbox, or compensation.

Do not use wall-clock timestamps as fencing tokens. Clock regressions, equal timestamps, restore, and multi-writer allocation can violate monotonicity. Use a sequence allocated by the linearizable authority and preserve it through backup/restore.

## Acquisition protocols

### Linearizable compare-and-set

For an uncontended exclusive lock, atomically create the grant only when the resource key is absent or expired under server authority. Attach the grant to a session/lease and return its committed revision as generation. Retries carry one acquisition request ID so a lost reply can return the same result.

This is safe only for effects inside the same transactional store unless external resources fence the revision. etcd's documentation explicitly supports composing the returned lock key with etcd transactions; that does not make an arbitrary object store aware of etcd ownership.

### Ordered wait queue

Polling every contender at a fixed interval wastes read capacity and synchronizes retries. An ordered queue records one wait node per request. The first compatible waiter holds the lock; each exclusive waiter watches only its immediate predecessor or a compact queue revision.

**Documented:** ZooKeeper's lock recipe uses ephemeral sequential nodes and predecessor watches. Session loss removes ephemeral nodes, and sequence order exposes contention. Implementations still must handle connection loss after create: the client may not know the created node name unless the request identity is recoverable.

Fairness is a product decision. Strict FIFO bounds starvation but can cause head-of-line blocking when a slow or paused first waiter cannot proceed. Priority can protect critical work and starve normal work unless priority aging or quotas are explicit.

### Database transaction and advisory locks

If all protected state is in one strongly consistent database, use its transaction, conditional update, row lock, exclusion constraint, or advisory lock before adding a separate coordinator.

**Documented, PostgreSQL 18:** transaction-level advisory locks release at transaction end; session-level advisory locks outlive transaction rollback until explicit release or session end. Advisory locks are application-defined and share bounded lock-manager memory. PostgreSQL also detects database deadlocks and aborts a participant.

An advisory lock protects only cooperating callers. A unique constraint or conditional write is often stronger because the database enforces it for every writer.

### Best-effort cache lock

A single cache `SET if absent` with random owner value and TTL can be an efficient duplicate-work suppressor. Release must compare the owner value and delete atomically; a bare delete can remove a successor's grant.

Redis's Redlock documentation proposes quorum acquisition across independent Redis masters under timing assumptions. The published safety debate centers on pauses, clock bounds, and absent resource fencing. **Reference design:** do not make correctness depend on a timing-only cache lock. If overlap is acceptable, label it an efficiency lock; if overlap is not acceptable, use a linearizable authority and resource-side fencing.

## Shared locks, upgrades, and multiple resources

A shared/exclusive lock needs a compatibility matrix and one atomic grant decision. Writer starvation occurs when new readers continually bypass a queued writer; reader starvation occurs when writer preference is absolute. State the fairness policy and test it under cancellation and session expiry.

Lock upgrade from shared to exclusive is hazardous:

1. A and B both hold shared mode.
2. Both wait to upgrade while retaining shared mode.
3. Neither can obtain exclusive mode.

Release-and-reacquire avoids deadlock but loses atomicity between observations. A true upgrade requires one ordered upgrader or transaction support. Prefer optimistic version validation when possible.

For several lock keys, acquire in one global canonical order. If dynamic discovery makes the full set unknown, use bounded try-lock plus release/backoff, a database transaction, or redesign the invariant. Never hold lock A while making an unbounded network call to discover lock B.

## Contention and capacity model

One exclusive lock is a single-server queue. Let acquisition arrival rate be $\lambda$, mean protected hold time be $S$, and utilization be $\rho$. Under an illustrative M/M/1 assumption:

$$
\rho = \lambda S, \qquad W = \frac{S}{1-\rho}
$$

If $\lambda = 80$ acquisitions/s and $S = 8\,\mathrm{ms}$, then $\rho = 0.64$ and modeled mean acquisition-plus-hold time is about $22\,\mathrm{ms}$. At $\lambda = 120$/s, $\rho = 0.96$ and the same model gives $200\,\mathrm{ms}$. Real critical sections have heavy tails, retries, pauses, and non-Poisson arrivals, so measure the distribution; the example shows why a hot lock becomes nonlinear before the lock service's CPU saturates.

For $H$ held lease-backed locks renewed every $T$ seconds:

$$
\lambda_{\mathrm{renew}} = \frac{H}{T}
$$

Add waiter watches, acquire/release writes, audit events, and recovery bursts. A service managing 600,000 held grants at 15-second renewal intervals sees 40,000 renewals/s before retries. Shard the lock namespace only if one resource's grants remain serialized and generations cannot regress across shard movement.

Capacity planning must include the hottest resource key, not just total acquisitions. Ten million cold locks do not offset one global lock at 99% utilization.

## Specialized failure traces

### Paused holder writes after regrant

1. A receives generation 83 and pauses.
2. Its lease expires; B receives generation 84 and commits.
3. A resumes and sends its already-buffered command.

The resource rejects 83 after observing 84. If the resource cannot compare tokens, the lock did not protect correctness.

### Acquire succeeds but response is lost

1. The service commits A's grant and generation, then the reply is lost.
2. A retries with a new request identity and queues behind itself.
3. Its first grant waits for a holder that does not know it holds anything.

Acquisition must be idempotent by request ID or support read-back by owner incarnation. Timeouts are ambiguous outcomes, not proof of failure.

### Old release deletes a new grant

1. A's TTL expires; B acquires the same key.
2. A's delayed cleanup executes an unconditional delete.
3. C acquires while B is still active.

Release compares exact grant identity/generation atomically. Human “break lock” operations allocate a new generation and are audited; they never delete blind.

### Waiter herd expires healthy sessions

1. One hot lock has 20,000 waiters watching one key.
2. Release wakes all waiters; each reads and attempts compare-and-set.
3. The coordination service overload delays unrelated renewals, expiring healthy grants.

Predecessor watches, bounded queues, randomized backoff, and per-namespace admission isolate the herd. A client should not enqueue more work than can meet its deadline.

### Restore regresses fencing generation

1. The lock database restores a snapshot whose last generation is 500.
2. A protected resource has already accepted generation 740.
3. New grants 501–739 are all rejected, or accepted incorrectly by resources that reset too.

Restore must preserve an epoch greater than every pre-restore grant, or rotate to a new higher namespace epoch. Test control-plane disaster recovery with live resource fence state.

### Deadlock spans services

A holds customer lock and calls service B, which holds inventory lock and calls back for customer lock. Neither database sees the whole wait-for graph. Deadlines eventually abort both, possibly after holding scarce resources. Enforce cross-service lock ordering, avoid lock-held RPCs, or move the invariant into one transactional authority.

## Security and abuse boundaries

Lock possession is authority metadata, not user authorization. Authenticate clients and authorize exact namespaces, modes, TTL ranges, and administrative break operations. Canonicalize Unicode and path forms so two names cannot alias one resource or one tenant access another's key.

Protect the coordinator from denial of service through waiter, lock-count, watch, renewal, and key-cardinality quotas. Do not put credentials or personal data in lock keys, owner strings, metrics, or traces. Encrypt transport and at-rest state, audit grant/release/break, and separate lock-service administration from application ownership.

A malicious holder can retain the critical section until expiry or deliberately create hot-lock contention. The protected resource still enforces business authorization and rate limits on every operation.

## Operations, rollout, and migration

To introduce a correctness lock safely:

1. inventory every writer and side-effect boundary for the invariant;
2. deploy fencing-token storage and comparison at resources in observe-only mode;
3. propagate tokens through every legitimate path and measure unfenced effects;
4. enable lock acquisition while old serialization remains authoritative;
5. enforce resource rejection only after token coverage reaches the declared gate;
6. remove the old mechanism after pause, partition, and restore tests pass.

Rolling upgrades keep key encoding, owner incarnation, queue ordering, and token width compatible. Do not change coordinator, namespace hash, and resource fence representation together. Rollback must not reset the token or allow an old binary that omits it to write.

Runbooks cover hot-key overload, stuck waiter, lost session, mass expiry, coordinator quorum loss, stale-holder rejections, token exhaustion/wrap, restore, and authorized emergency break. Breaking a lock is not “making the error go away”; it deliberately creates a newer authority generation and may start duplicate work.

## Observability and verification

Measure by namespace and resource class:

- acquire attempts, success, timeout, cancellation, and ambiguous outcomes;
- wait and hold distributions, queue depth/age, utilization, and hottest keys;
- active sessions, renewal margin, expiry, reconnect, and watch backlog;
- grant generation, rejected stale effects, unfenced effects, and release mismatches;
- deadlocks, victim selection, priority starvation, and emergency breaks;
- coordinator revision lag, quorum health, restore epoch, and capacity saturation.

Verification includes linearizability/history tests for incompatible grants, deterministic key encoding, acquire-response loss, duplicate release, cancellation races, process pauses beyond expiry, asymmetric partitions, session loss, coordinator failover and restore, queue herds, cross-resource deadlocks, and generation enforcement at every side effect.

The decisive assertion is historical: after a resource accepts generation `g`, it never accepts a mutating operation from a generation lower than `g`.

## Designing the lock away

Prefer an invariant-enforcing primitive:

| Intended lock purpose | Often stronger design |
|---|---|
| Create one entity | Unique constraint or conditional create |
| Read-modify-write | Compare-and-set/version column or serializable transaction |
| Process one message | Idempotent outcome plus durable message identity |
| One writer per key | Partition ownership and fenced generation |
| Prevent duplicate API call | Idempotency key at the external API boundary |
| Coordinate a long-lived controller | Leader election with activation/failover semantics |

A lock reduces concurrency and adds another availability dependency. Use it when serialization matches the invariant and the protected boundary can enforce the grant, not because “distributed systems need locks.”

## Decision framework

1. What exact invariant and resource key require serialization?
2. Is overlap merely wasteful or correctness-breaking?
3. Can a conditional write, uniqueness rule, idempotency key, or partition owner replace the lock?
4. Which service serializes grants, and how does it preserve generations across failover and restore?
5. Where is the fencing token compared atomically with each protected effect?
6. What happens after an ambiguous acquire or release response?
7. What is the hottest lock's arrival/hold distribution and modeled queue knee?
8. Can queue fairness, cancellation, and multi-key order avoid starvation and deadlock?
9. Which effects cannot be fenced, and how are duplicates repaired?
10. Can the lock service and protected resources survive pause, partition, herd, and restore tests?

## Primary references

- [Gray and Cheriton, *Leases: An Efficient Fault-Tolerant Mechanism for Distributed File Cache Consistency* (SOSP 1989)](https://web.stanford.edu/class/cs240/readings/leases.pdf)
- [Burrows, *The Chubby Lock Service for Loosely-Coupled Distributed Systems* (OSDI 2006)](https://research.google.com/archive/chubby-osdi06.pdf)
- [Hunt et al., *ZooKeeper: Wait-free Coordination for Internet-scale Systems* (USENIX ATC 2010)](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Hunt.pdf)
- [Apache ZooKeeper, *Recipes and Solutions: Locks*](https://zookeeper.apache.org/doc/current/recipes.html#sc_recipes_Locks)
- [etcd, *API guarantees*](https://etcd.io/docs/v3.5/learning/api_guarantees/)
- [etcd, *Lock service API*](https://etcd.io/docs/v3.6/dev-guide/api_concurrency_reference_v3/)
- [PostgreSQL 18, *Explicit Locking and Advisory Locks*](https://www.postgresql.org/docs/current/explicit-locking.html)
- [Redis, *Distributed Locks with Redis*](https://redis.io/docs/latest/develop/clients/patterns/distributed-locks/)
- [Kleppmann, *How to do distributed locking* (2016)](https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html)
