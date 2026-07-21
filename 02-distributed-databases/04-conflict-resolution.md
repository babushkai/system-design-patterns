# Conflict Resolution

Conflict resolution is the policy that turns **independently accepted, causally concurrent updates** into one convergent state. It cannot recover an intention the system never recorded. A merge can preserve both shopping-cart additions, for example, but it cannot decide whether two users who independently claimed the same username should share it. The data type and business invariant must define that answer.

Conflict resolution must define detection, representation, merge semantics, deletion knowledge, and repair for conflicting versions. [Multi-Leader Replication](./02-multi-leader-replication.md) and [Leaderless Replication](./03-leaderless-replication.md) own the replica protocols that create and exchange those versions. [Distributed Transactions](./07-distributed-transactions.md) owns coordination when an invariant cannot be merged safely.

## Start with the contract, not the merge function

Four properties are often collapsed into “eventual consistency,” but they are different:

- **Convergence:** replicas that receive the same updates eventually hold equivalent state.
- **Causality preservation:** a version known to include another version does not lose it during reconciliation.
- **Intention preservation:** the merged state reflects the users' operations, such as two increments rather than one overwritten number.
- **Invariant preservation:** every reachable state satisfies rules such as non-negative inventory or unique ownership.

A deterministic winner supplies convergence. It does not automatically preserve causality, intention, or invariants. A CRDT supplies convergence under a particular state and delivery contract. It does not automatically make every application invariant coordination-free.

Before choosing a strategy, write the semantic table for each operation:

| Concurrent operations | Required result | Can information be discarded? | Coordination needed? |
|---|---|---:|---:|
| profile photo A / profile photo B | one deterministic photo | perhaps | no, if arbitrary winner is acceptable |
| increment 3 / increment 5 | total increases by 8 | no | no, with a counter CRDT |
| add item / remove an observed add | item absent | no | no, with causal remove metadata |
| two claims on one username | at most one owner | losing claim must be rejected | normally yes |
| debit / concurrent debit | balance never below zero | no overspend | yes, or rights must be escrow-partitioned |

The table is the contract. “Use last writer wins” is an implementation choice only after the first row has explicitly accepted data loss.

## State required for correct reconciliation

A conflict-aware record is more than a value and a timestamp. Depending on the strategy, durable state includes:

- the payload or operation;
- a unique event identity, often `(replica_id, counter)`;
- causal context describing updates already observed;
- a deterministic order key if one winner is required;
- deletion knowledge or removal context;
- a schema and merge-policy version;
- sibling versions that are concurrent and not yet reconciled.

Replica identity and counters are correctness state. Restoring a replica from backup while reusing an old identity and counter can recreate an event ID already covered by a tombstone; a genuinely new add can then disappear. Decommissioning and restoring actors therefore require an identity-lifecycle protocol, not only a data backup.

### Version vectors detect ancestry and concurrency

A version vector maps each actor to the greatest counter observed from that actor. Vector `A` dominates vector `B` when every component of `A` is at least the corresponding component of `B` and one is greater. `B` is then an ancestor and can be discarded. If neither dominates, the versions are concurrent.

For example:

```text
v1 = {A: 4, B: 1}
v2 = {A: 3, B: 2}

v1 contains a later A event; v2 contains a later B event.
Neither dominates, so discarding either loses an update.
```

A **dotted version vector** separates the new event's dot from the causal context it observed. This represents causality more precisely when different servers coordinate writes for the same logical object. Physical timestamps cannot substitute for this relation: clock order is not evidence that one writer observed another.

Vectors have a membership cost. A dense vector is `O(r)` metadata for `r` actors, and actor churn can grow it indefinitely. Practical systems use stable actors, sparse maps, dotted contexts, interval compression, or deliberately lossy truncation. Dynamo's timestamp-based vector-clock truncation is explicit about the trade: bounded metadata can forget ancestry and create siblings that a complete clock would have eliminated.

## Resolution families

### Serialize the operation when one answer is mandatory

“First writer wins” is not choosing the lowest client timestamp after replicas diverge. It is accepting one operation at a serialization point and rejecting the rest through compare-and-set, a unique index, a consensus-backed create-if-absent operation, or a transaction.

Use this contract for one-time idempotency-key ownership, uniqueness, inventory rights that cannot be oversold, and other non-mergeable invariants. Independent replicas may still accept requests for availability, but they cannot both promise globally exclusive success unless ownership rights were partitioned in advance.

### Last-writer-wins register

An LWW register chooses the maximum element under a **total order**, commonly `(timestamp, stable_writer_id)`. The stable tie-breaker is required: if equal timestamps let replica A prefer its local value while replica B prefers its local value, anti-entropy never converges.

```text
winner = max(versions, key = (logical_or_physical_time, stable_writer_id))
```

The tie-breaker guarantees convergence, not that an unsynchronized clock identified the real last writer. A hybrid logical clock preserves causal order better than a raw wall clock, but LWW still deliberately discards concurrent information. It is suitable only when any deterministic winner is semantically acceptable: cache hints, replaceable preferences, or fields whose authoritative source will refresh them.

Deletes in an LWW register are values too. A tombstone must participate in the same order; physically removing it early allows an older live value to win again.

### Multi-value register and application merge

A multi-value register removes causally dominated versions and returns every maximal concurrent sibling. A later write that includes all sibling contexts supersedes them. This avoids silent loss and lets a domain-aware resolver decide.

The application merge must itself be retry-safe and deterministic. If two replicas resolve the same siblings differently, resolution creates another conflict. Persist the resolved value with causal context covering **all** consumed siblings; merely deleting sibling rows locally does not prevent another replica from returning them.

This model works well when conflicts are rare and a user can make a meaningful decision. It works poorly when a hot key routinely has many independent writers: sibling count, payload storage, read latency, and human repair queues all grow.

### State-based CRDTs

A state-based CRDT represents state as a join-semilattice. Local updates move state monotonically upward, and merge computes the least upper bound. The merge is commutative, associative, and idempotent, so duplicate and reordered state transfer converges.

A grow-only counter stores one monotonically increasing component per actor. Merge takes the component-wise maximum; the displayed value is the sum of components. Plain `sum` of replica totals double-counts retransmitted state, while plain `max` loses independent increments. A positive-negative counter uses two grow-only component maps, one for increments and one for decrements.

An observed-remove set gives every add a durable unique dot. Removing element `x` records exactly the add dots for `x` that the remover has observed:

```text
A observes add(x, dot=A:7), then removes x
  -> tombstone/context covers A:7; that add stays removed after merge

B concurrently adds(x, dot=B:4), unseen by A
  -> B:4 is not covered; x remains present (add-wins)
```

The simple representation retains add dots and removed dots forever. Optimized forms such as ORSWOT summarize removal knowledge with causal context. Compaction is safe only after **causal stability** establishes that no replica can later reintroduce a covered add. A wall-clock retention period alone is unsafe if an old replica or backup may rejoin after that period.

### Operation-based and delta-state CRDTs

Operation-based CRDTs disseminate operations rather than full state. Their convergence usually requires either reliable causal delivery plus duplicate suppression or operations designed to commute under a weaker delivery order. Delta-state CRDTs disseminate small joinable state fragments and retain the idempotent join model while reducing bandwidth.

Do not label arbitrary event handlers a CRDT. The proof must name the state order or operation-delivery contract and demonstrate convergence for every concurrent operation pair, including add/remove, update/delete, and duplicate delivery.

### Semantic operation logs

Some domains merge intentions more safely than values. Recording `add 5` and `subtract 2` retains more information than two replicas writing totals. Collaborative editors similarly reconcile insert/delete operations with stable element identities; operational transformation and sequence CRDTs solve different protocol problems and should not be mixed casually.

Operation logs require bounded retention, deterministic replay, idempotency, and a snapshot/compaction rule. An operation that calls an external service is not made safe merely by putting it in a mergeable log; see [Idempotency](../01-foundations/08-idempotency.md).

## Concrete failure traces

### Equal timestamps without a stable tie-breaker

1. Replica A accepts `name = Alice` at timestamp 100.
2. Replica B accepts `name = Bob` at timestamp 100.
3. Each implementation keeps its local value on equality.
4. Every exchange repeats the tie and leaves A and B different.

The fix is a total deterministic order (or retaining both versions), not hoping timestamp precision makes ties impossible.

### Delete resurrection

1. A and B contain add dot `A:9` for item `x`; C is offline with the old value.
2. A removes `x`, and A/B later physically discard removal context after an elapsed-time rule.
3. C returns and sends `A:9`.
4. No surviving state proves that `A:9` was removed, so `x` reappears.

Safe collection requires causal stability, an operational guarantee that C can never rejoin, or rebuilding C from a current snapshot before it participates.

### Retry double-counts a non-idempotent merge

1. A exports counter total 10; B exports total 7.
2. A resolver adds them and stores 17.
3. The reply is lost, so repair repeats and adds B's 7 again.
4. The total becomes 24 even though no new increment occurred.

Per-actor components and component-wise max turn retransmission into an idempotent merge.

### Locally valid values violate a global constraint

1. During a partition, A assigns username `river` to user 1.
2. B assigns `river` to user 2.
3. LWW can converge on one owner, but both users may already have received success and created dependent state.

The conflict is not merely which row remains. The API promised something it could not preserve without coordination, escrowed ownership, or a later explicit revocation workflow.

## Capacity and cost model

Conflict resolution moves cost from the write path into metadata, reads, repair, and garbage collection.

Let `r` be active causal actors, `s` unresolved siblings for a key, `v` average payload bytes, and `m` metadata bytes per version. A multi-value record occupies roughly `s * (v + m)`. Naive pairwise vector dominance costs `O(s^2 * r)` component comparisons; good implementations maintain a maximal frontier incrementally, but large sibling sets still make one hot key expensive.

For state-based anti-entropy, network demand is approximately:

```text
repair_bytes_per_second
  = changed_keys_per_second * replicas_contacted * transferred_state_per_key
```

Full-state CRDTs make `transferred_state_per_key` grow with accumulated history. Delta-state transfer reduces the common case but still needs a full-state/bootstrap path when deltas are lost or a replica is new.

Tombstones and causal context are compaction debt. Track bytes and count by age, but never turn an age metric directly into a deletion rule. The safe-removal frontier must come from replica progress or an explicit membership decision. Repair bandwidth must also be capacity-reserved; foreground success during a partition can create a backlog whose reconciliation later saturates CPU, disk, or network.

## Production operation and migration

### Make merge policy versioned data

Changing from LWW to an OR-set, changing timestamp sources, or altering add/remove precedence changes persisted semantics. Store a policy/schema version beside the record. During migration, readers must understand old and new forms; writers should dual-write or translate under a documented cutover frontier. Never allow two live versions of the resolver to interpret the same bytes differently without a compatibility proof.

### Operate actor identity explicitly

- Persist `(actor_id, counter)` across process restarts.
- Allocate a new identity after restoring an old backup unless counter continuity is proven.
- Mark retired actors in membership state.
- Rebuild long-offline replicas from a current snapshot when removal context may have been collected.
- Quarantine replicas whose clock or causal metadata moves backward.

### Observe the hidden queues

Useful signals include unresolved siblings per key and percentile, causal-context bytes, tombstone bytes, merge retries, reconciliation outcomes by policy version, anti-entropy backlog, oldest unacknowledged removal, and replicas behind the stability frontier. A converged cluster can still be wrong, so convergence alone is not an SLO.

### Test algebra and histories

Property tests should randomly permute, duplicate, batch, and replay updates and then assert merge commutativity, associativity, and idempotence where promised. Generate concurrent operation histories and verify domain invariants, not only equal final bytes. Fault tests must include partitions, process pauses, lost acknowledgements, actor restart from backup, schema-version skew, and a replica returning after removal compaction.

## Decision framework

1. What concurrent operation pairs can occur, including delete/update and retry/retry?
2. Must both intentions survive, or may one deterministic value be discarded?
3. Is the business invariant closed under the proposed merge operation?
4. What causal metadata proves ancestry, and how does actor churn affect its size?
5. What exact condition makes removal knowledge safe to collect?
6. Can the application resolve every sibling deterministically and idempotently?
7. What repair capacity drains divergence after the longest supported partition?
8. If coordination is required, should the API reject immediately rather than acknowledge a promise it may later revoke?

## Primary references

- [DeCandia et al., *Dynamo: Amazon's Highly Available Key-value Store* (SOSP 2007)](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf)
- [Terry et al., *Managing Update Conflicts in Bayou, a Weakly Connected Replicated Storage System* (SOSP 1995)](https://doi.org/10.1145/224056.224070)
- [Shapiro et al., *A Comprehensive Study of Convergent and Commutative Replicated Data Types* (INRIA Research Report 7506, 2011)](https://inria.hal.science/inria-00555588/document)
- [Almeida, Baquero, and Fonte, *Interval Tree Clocks: A Logical Clock for Dynamic Systems* (OPODIS 2008)](https://gsd.di.uminho.pt/members/cbm/ps/itc2008.pdf)
- [Almeida, Baquero, and Fonte, *Dotted Version Vectors: Logical Clocks for Optimistic Replication* (2010)](https://arxiv.org/abs/1011.5808)
- [Kleppmann and Beresford, *A Conflict-Free Replicated JSON Datatype* (IEEE TPDS 2017)](https://arxiv.org/abs/1608.03960)
