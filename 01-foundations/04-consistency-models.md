# Consistency Models

A consistency model is a contract over histories: given concurrent invocations, responses, failures, and replication, which results may a client observe? “Strong,” “eventual,” and a database consistency-level name are not adequate specifications. The contract must name its scope, ordering relation, failure outcome, and whether it covers one object, a session, or a transaction.

Scope: client-observable models—linearizable, sequential, causal, PRAM/session, bounded-staleness, and eventual/convergent behavior—their composition, mechanisms, and verification. [CAP Theorem](./03-cap-theorem.md) applies linearizability to one partitioned read/write object. [ACID Transactions](./01-acid-transactions.md) owns database isolation and invariant enforcement; [Conflict Resolution](../02-distributed-databases/04-conflict-resolution.md) owns merge algebra and CRDT mechanics.

## Start with histories, not product labels

An operation has an invocation and a matching response. Its interval lies between them. A history records operations from all clients; each client’s **program order** is the order of its own calls. Operation `a` precedes `b` in real time when `a` responds before `b` is invoked. Overlapping operations are concurrent and may be ordered either way by many models.

For a register, a legal sequential history returns the value of the latest preceding write. A queue, set, compare-and-set register, and transaction have different sequential specifications. A consistency model selects which concurrent histories are equivalent to or compatible with such legal behavior.

Timeouts matter. An invocation without a successful response may have taken effect. A checker may complete a pending operation with a compatible response or omit it, according to the model; the application still needs idempotency and outcome lookup. Recording only successful calls erases the histories most likely to expose a failover bug.

Every stated model needs four coordinates:

```text
scope:       key, object, partition, table, session, transaction, or database
operations:  read/write, CAS, range, batch, transaction, side effect
ordering:    real-time, program, causal, prefix, version, or none
liveness:    when writes propagate and what happens if dependencies are unreachable
```

Consistency and liveness are separate. A service can preserve linearizability by never responding. Eventual convergence needs assumptions such as eventual delivery and no new writes. Availability behavior during communication loss is covered in [CAP Theorem](./03-cap-theorem.md).

## The models are not one total ladder

Some guarantees imply others for the same object and operation set, but the useful models live on different axes. Session guarantees constrain one client. Causal consistency constrains dependency order. Bounded staleness constrains age or version distance. Transaction isolation constrains groups of reads and writes. A simple strongest-to-weakest spectrum hides those scope differences.

### Linearizability: one legal order that respects real time

A history is linearizable if its completed operations—and a permissible completion of some pending operations—can be placed in a legal sequential order that:

1. preserves each operation’s result under the object specification; and
2. preserves real-time precedence between non-overlapping operations.

If `write(x,1)` returns before another client invokes `read(x)`, that read returns `1` or a later value. If the read overlaps the write, either old or new may be legal. Linearizability does not require synchronized wall clocks; real-time order comes from invocation and response intervals.

Linearizability is **local**: a system is linearizable if each object is linearizable, assuming operations truly act on those objects. This compositional property makes per-key checking scalable. It does not make two separate key operations an atomic transaction or preserve an application invariant spanning them.

Common mechanisms are one fenced write authority, a consensus log with leader/read barriers, or an atomic-register quorum protocol that queries and writes back ordered versions. A quorum equation alone is insufficient. A leader lease supports local reads only while its timing and fencing assumptions remain valid; otherwise the leader must confirm authority. [Consensus Algorithms](../02-distributed-databases/08-consensus-algorithms.md#linearizable-reads) covers those protocols.

### Sequential consistency: program order without real-time order

Sequential consistency requires one legal total order containing every process’s operations in that process’s program order. It does **not** require that total order to respect real-time precedence across processes.

```text
real time:  client A write(x,1) returns ----- client B read(x) begins

sequentially consistent explanation may order:
            client B read(x)->0 ; client A write(x,1)
linearizability may not, because the calls do not overlap
```

This can be useful when all participants consume one ordered log but external response timing is not part of the abstraction. Unlike linearizability, sequential consistency is not generally local: independently valid per-object sequential orders may be impossible to combine while preserving every process’s program order. A system must define the shared ordering domain.

### Causal consistency: preserve dependencies, not one total order

The happens-before relation includes a client’s program order, reads-from edges (a write precedes a read that observes it), and transitive closure. Under causal consistency, every observer sees causally related writes in that order. Concurrent writes have no causal edge and may be observed in different orders unless an additional convergence/arbitration rule is specified.

```text
w1: publish post
r1: another client reads that post
w2: publish reply after r1

w1 -> r1 -> w2, so no observer may expose w2 without the required w1.
```

Implementations carry dependency metadata, such as version vectors, dotted versions, or compact partition/session frontiers. A replica delays visibility until predecessors are applied or routes the request to a replica whose frontier dominates the token. Lamport timestamps provide an order consistent with causality but cannot by themselves distinguish concurrency; a scalar timestamp is not a complete dependency set.

“Causal+” is used for causal visibility plus convergent conflict handling, but exact definitions vary. State the concrete visibility and arbitration rules. The merge laws and metadata-reclamation boundary belong in [Conflict Resolution](../02-distributed-databases/04-conflict-resolution.md).

### PRAM and the four session guarantees

PRAM/FIFO consistency makes every process’s writes visible to others in that writer’s program order; writes from different processes may be interleaved differently. It does not automatically preserve a write that depends on something the writer read.

Session guarantees constrain a sequence of operations associated with one client context:

- **Read your writes:** a read reflects the session’s preceding writes.
- **Monotonic reads:** later reads include at least the write set reflected by earlier reads; the session does not go backward.
- **Monotonic writes:** the system orders a session’s writes after its preceding writes.
- **Writes follow reads:** a write is ordered after the writes reflected by preceding session reads.

Together, these make roaming among replicas much less surprising, but they are not global linearizability. “Sticky session” is only a fragile implementation if that replica is lost or lags after failover. A portable session token should encode the required progress/dependencies and be integrity-protected, tenant-bound, bounded in size, and available across devices if the product promises continuity there.

### Consistent prefix and bounded staleness

Consistent-prefix reads expose only a prefix of an ordered history: a reader may be behind but does not see entry 12 without required entry 11. This is useful for replicated logs and follower reads. It says nothing about how far behind the prefix is.

Bounded staleness adds a measurable limit, such as at most `K` committed versions or at most `T` time behind a declared authority. A version bound is often easier to establish from replication state than a time bound. Time-based bounds require a trustworthy relationship between commit timestamps and real time plus bounded replication measurement. On breach, the API must wait, route elsewhere, or return a typed “freshness unavailable” result; silently serving older data violates the contract.

### Eventual consistency and convergence

“Eventual consistency” is incomplete unless the liveness assumptions and convergence rule are stated. A useful decomposition is:

- **eventual delivery/visibility:** every accepted update eventually reaches each in-scope live replica after communication recovers;
- **convergence:** replicas receiving the same relevant updates eventually reach equivalent state;
- **termination:** local operations complete under the specified failure conditions.

Strong eventual consistency additionally requires replicas that have incorporated the same update set to be equivalent regardless of delivery order. CRDTs can supply this when their algebra and delivery assumptions hold. A last-writer-wins register may converge through a total tie-broken order while silently discarding concurrent intent. Eventual propagation alone does not choose either rule.

## Transactional isolation is a separate axis

Linearizability concerns individual object operations. **Serializability** asks whether committed transactions are equivalent to some serial transaction order, but that order need not respect real time. **Strict serializability** combines serializability with real-time precedence. Snapshot isolation provides a consistent snapshot and write-conflict checks but can permit write skew; read committed permits still more histories.

Atomic visibility is another obligation: a reader should not see half of a multi-key commit when the API promises an atomic transaction. Two individually linearizable keys do not provide that automatically. Conversely, a serializable database can expose a stale but serializable snapshot unless it also promises real-time recency. See [ACID Transactions](./01-acid-transactions.md) and [Distributed Transactions](../02-distributed-databases/07-distributed-transactions.md#isolation-fails-despite-atomic-commit) for transaction mechanisms and anomalies.

## Composition, scope, and mixed modes

Always attach the model to a scope. “Linearizable database” may mean point operations per key while range indexes, follower reads, and transactions have different contracts. A batch API may read keys at different frontiers. A cache can turn a linearizable source into an eventual endpoint. An external side effect is outside the database history unless it participates through fencing or idempotent workflow state.

Mixing modes creates a new contract rather than preserving the strongest component. A quorum write followed by an eventual follower read may violate read-your-writes. A causal write routed through a consumer that discards its dependency token becomes an ordinary asynchronous write. A strong primary-key lookup and stale secondary index can still produce false negatives. Each path—including retries, background jobs, and failover routing—must propagate the required frontier.

Linearizability composes across objects, but multi-object **operations** still need their own sequential specification. Sequential consistency, session guarantees, and causal consistency need shared program/dependency context to compose. When teams independently assign scopes, document where contexts join and where the guarantee ends.

## Mechanisms and their real costs

| Contract | Essential mechanism/state | Foreground consequence | Failure behavior |
|---|---|---|---|
| Linearizable register | Fenced authority or ordered quorum tags; durable commit frontier; authoritative read path | Writes coordinate; reads prove authority or query/write back | Side without authority waits or rejects |
| Sequential order | One ordering domain/log and preserved client program order | Operations enter total order; response need not reflect external real time | Failover must preserve log order |
| Causal | Dependency context plus causal delivery/visibility gate | Metadata travels; a missing predecessor can delay visibility | Independent partitions may progress on concurrent work |
| Session guarantees | Client token/frontier and wait, route, or fallback policy | Roaming read may wait or move to a fresher replica | Token loss weakens the session unless rejected |
| Bounded staleness | Measured authority and replica frontiers | Serve locally within bound; otherwise wait/route/fail | Bound is observable and may become unavailable |
| Eventual convergence | Durable asynchronous log, idempotent delivery, deterministic merge/repair | Local acceptance can avoid remote coordination | Divergence is allowed until delivery assumptions recover |

Stronger history does not have one universal latency multiplier. Placement, durability, batching, read leases, contention, and requested operation matter. Model components explicitly: for a quorum write, completion follows the required acknowledgement order statistic; for a causal read, cost may be zero when dependencies are local or an unbounded wait while one predecessor is missing.

**Illustrative calculation, not a product claim.** A replica applies 150 log positions/s while its source commits 200/s for 20 seconds. Its deficit grows by `(200 - 150) * 20 = 1,000` positions. If apply capacity later rises to 300/s while commits remain 200/s, catch-up needs at least `1,000 / (300 - 200) = 10` seconds. A session requiring the latest source position cannot receive a read-your-writes result there before catch-up unless the request routes elsewhere. Queueing, batches, and retries make real tails worse.

**Illustrative metadata bound.** A dense causal vector with one unsigned 64-bit counter for each of `A` actors needs at least `8A` bytes before actor IDs and framing. At 1,000 actors that lower bound is about 8,000 bytes per context, motivating dotted, hierarchical, partition-scoped, or server-held contexts. The calculation is a representation example, not a claim that causal consistency always has that overhead.

## Make the contract observable

An API should return enough evidence for the chosen model: commit/request ID, authority epoch, logical position or version, session/dependency token, and for stale reads an `as_of` frontier or timestamp. The request can carry a minimum frontier and a policy such as `wait`, `route`, or `fail`; it should not rely on sleeping an assumed replication delay.

Document downgrade behavior. If a strong read cannot prove authority, does it return unavailable, fall back only when the caller explicitly accepts staleness, or serve a separate cached representation? Do timeouts mean unknown commit? How long are tokens valid, and across which tenants, regions, restores, and schema versions? These are part of consistency just as much as the replication algorithm.

Authorization can itself require consistency. Credential revocation, ownership transfer, quota reservation, and policy changes may need a current/fenced read, while public content can tolerate a stale cache. Bind tokens to principal and tenant, prevent clients from forging progress, and avoid leaking internal topology in externally visible versions. A consistency downgrade must never imply an authorization downgrade.

Observe required versus served frontier, replica apply position and age, wait/route/fallback counts, session-token size and rejection, authority term and lease evidence, stale-read age distribution, causal dependency queue, incomplete-operation outcomes, conflict backlog, and contract mode by endpoint and tenant. Measure semantic outcomes rather than calling every successful HTTP response “consistent.”

## Specialized failure traces

### Acknowledgement precedes the durable linearization point

A leader returns success before the entry is committed on the required quorum, then fails. A new leader without the entry serves the old value. The completed write has no legal place before the later read. Acknowledgement must follow the protocol’s durable commit point, not local receipt.

### Paused leaseholder serves after its lease

A process pauses while holding a read lease. The cluster advances the epoch and commits a new value. The old process resumes using cached time and serves the prior value as linearizable. Lease safety needs a valid clock/expiry model and fencing; otherwise use a read barrier against current consensus state.

### Sequential consistency surprises a real-time observer

Client A’s write returns, and only afterward client B reads the old value. A legal sequential order can place B’s read before A’s write because their program orders do not conflict. If the product says “completed changes are immediately visible,” it needs linearizability, not sequential consistency.

### Reply appears before its causal parent

A user reads post `p` and writes reply `r` with dependency `{p}`. A remote indexer publishes `r` before receiving `p`, so readers see a reply to nothing. The pipeline discarded or ignored the causal context; wall-clock sorting afterward cannot repair the visibility violation.

### Roaming session loses read-your-writes

A write in region A returns position 91. The client reconnects to B but does not present its token; B is at 87 and returns the previous profile. Sticky routing happened to supply the guarantee until failover. Persist and enforce the session frontier or make the weaker cross-device contract explicit.

### Per-key linearizability exposes an impossible snapshot

A transfer changes `debit` and `credit` under separate linearizable key operations. A reader sees new debit and old credit. Each key history is legal; the multi-key observation is not atomic. Use a transaction/snapshot contract rather than assuming object composition creates transaction isolation.

### Replicas receive the same updates but disagree forever

Two eventual replicas apply concurrent values through a resolver that depends on local iteration order. Delivery completes, yet final states differ. Eventual delivery is not convergence; use a deterministic total choice or merge satisfying the required algebra.

## Verification and evolution

Capture client-side invocation and response events, values, stable operation IDs, requested model, tokens, and unknown outcomes. Do not reconstruct real-time order solely from unsynchronized server timestamps. Inject process crashes, pauses, asymmetric partitions, reordered and duplicate delivery, clock faults, disk faults, lease expiry, replica migration, and mixed software/configuration versions.

Use a checker matched to the claim:

- linearizability: search for a legal real-time-respecting history, partitioned by object only when the API is truly local;
- sequential consistency: preserve program order but deliberately omit cross-client real-time edges;
- causal: generate explicit reads-from dependencies and ensure descendants never appear without predecessors;
- session: roam clients among replicas and assert each of the four token/frontier properties independently;
- bounded staleness: compare the served frontier with the authoritative frontier under the declared metric;
- convergence: deliver the same operation set in duplicated and permuted orders and compare state;
- transactions: infer dependency cycles and isolation anomalies with an isolation checker such as Elle.

Check safety and liveness separately. A finite test finding no violation is evidence, not proof; a concise counterexample is decisive. Preserve the exact binary, configuration, topology, and checker model with every history so a claimed guarantee is reproducible.

Changing consistency is an API migration. Introduce new tokens and response metadata before requiring them; dual-run old and new read paths; compare histories; then gate writes or reads at a versioned activation frontier. Rollback must not route a token-requiring client to a server that silently ignores the token. Never change a mode name in place while retaining its old clients.

## Decision framework

Choose from the invariant outward:

| Need | Candidate minimum contract | Question that can force something stronger |
|---|---|---|
| Lock, leader record, unique reservation, revocation | Linearizable conditional object | Does the invariant span several objects or an external side effect? |
| Multi-key transaction with real-time commit order | Strict serializability plus atomic durability | Are stale snapshots or weaker isolation actually allowed? |
| Conversation, dependency graph, collaborative workflow | Causal visibility plus explicit convergence | Must all concurrent actions have one immediate winner? |
| User roaming among replicas | Required session guarantees | Is the token durable across devices and failover? |
| Follower read with freshness SLO | Consistent prefix plus version/time bound | What happens when the bound cannot be met? |
| Cache, derived view, offline mergeable state | Eventual delivery plus deterministic convergence | Are false negatives, stale positives, deletes, and conflicts acceptable? |

Do not pay for a stronger model by reflex, and do not weaken one based on generic latency folklore. State the smallest history that keeps the product invariant true, expose its evidence and failure outcome, then test that exact contract across every serving path.

## Primary references

- Herlihy, M. P., and Wing, J. M. [Linearizability: A Correctness Condition for Concurrent Objects](https://doi.org/10.1145/78969.78972). ACM TOPLAS, 1990.
- Lamport, L. [How to Make a Multiprocessor Computer That Correctly Executes Multiprocess Programs](https://doi.org/10.1109/TC.1979.1675439). IEEE Transactions on Computers, 1979.
- Ahamad, M., Neiger, G., Burns, J. E., Kohli, P., and Hutto, P. W. [Causal Memory: Definitions, Implementation, and Programming](https://doi.org/10.1007/BF01784241). Distributed Computing, 1995.
- Terry, D. B., et al. [Session Guarantees for Weakly Consistent Replicated Data](https://doi.org/10.1109/PDIS.1994.331722). PDIS, 1994.
- Attiya, H., Bar-Noy, A., and Dolev, D. [Sharing Memory Robustly in Message-Passing Systems](https://doi.org/10.1145/200836.200869). Journal of the ACM, 1995.
- Adya, A., Liskov, B., and O’Neil, P. [Generalized Isolation Level Definitions](https://doi.org/10.1109/ICDE.2000.839388). ICDE, 2000.
- Horn, A., and Kroening, D. [Faster Linearizability Checking via P-Compositionality](https://doi.org/10.1007/978-3-319-19195-9_4). FORTE, 2015.
- Kingsbury, K., and Alvaro, P. [Elle: Inferring Isolation Anomalies from Experimental Observations](https://www.vldb.org/pvldb/vol14/p268-alvaro.pdf). PVLDB, 2020.
