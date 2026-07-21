# Distributed Transactions

A distributed transaction makes one outcome authoritative across multiple independently durable participants. The central problem is not sending `COMMIT` twice. It is preserving atomicity when processes crash at every instruction boundary, messages are lost or duplicated, and some participants have made promises they can no longer revoke.

Scope: **atomic commit across storage participants**, its interaction with isolation, and alternatives whose contracts are explicitly weaker. [ACID Transactions](../01-foundations/01-acid-transactions.md) and [Isolation Levels](../01-foundations/02-isolation-levels.md) define local transaction semantics. [Consensus Algorithms](./08-consensus-algorithms.md) owns replication of a participant or decision record. [Outbox Pattern](../05-messaging/07-outbox-pattern.md) owns reliable database-to-message publication, and [Database Sharding](../06-scaling/03-database-sharding.md) owns operational shard migration.

## Define the outcome contract

Three different problems are commonly called a distributed transaction:

1. **Atomic commit:** every participant commits, or every participant aborts.
2. **Isolation:** concurrent transactions observe an allowed serial/snapshot order.
3. **Durable workflow:** a sequence of local transactions eventually completes or compensates.

Two-phase commit solves the first problem. It does not by itself prevent write skew, lost updates, or phantoms; participants need strict two-phase locking, serializable MVCC/OCC, or another concurrency-control protocol. A saga solves the third problem and intentionally exposes intermediate states. Calling compensation a rollback hides a materially weaker contract.

For each use case, state:

- the participants and failure domains;
- whether a client success response may precede cleanup at every participant;
- the required isolation level;
- the maximum time locks or intents may remain unresolved;
- whether an irreversible external effect is inside the atomic boundary;
- what a retry with the same transaction ID means.

A transfer inside one distributed database is different from “update a database, charge a third-party card, and publish an email.” The latter systems do not share one prepare/commit protocol, so atomic rollback is generally unavailable.

## Durable state and atomicity invariants

Let transaction `T` have coordinator `C` and participant set `P`. A recoverable protocol needs:

- a globally unique transaction ID;
- the complete participant set or a durable way to discover it;
- per-participant provisional writes, undo/redo information, and held locks/intents;
- participant state such as `ACTIVE`, `PREPARED`, `COMMITTED`, or `ABORTED`;
- one authoritative durable decision record;
- idempotent handlers for repeated prepare, commit, abort, and status queries;
- enough retention that a recovering participant can still learn the decision.

The critical invariants are:

1. A participant votes `YES` only after it can commit despite a local crash.
2. After voting `YES`, a participant does not unilaterally abort or forget its prepared state.
3. The coordinator records `COMMIT` durably only after every required participant voted `YES`.
4. `COMMIT` and `ABORT` are mutually exclusive terminal decisions for one transaction ID.
5. A client is told success only after the commit decision and all state needed to complete it are durable under the advertised failure model.

Atomicity is a recovery property. A transaction may temporarily be committed at A and still prepared at B after a lost message; the protocol remains atomic only if readers cannot expose an invalid mix under the isolation model and recovery inevitably drives B to the same durable decision.

## Two-phase commit, precisely

### Phase 1: prepare

The coordinator freezes the participant set and sends `PREPARE(T)`. Each participant:

1. finishes execution and validates local constraints/concurrency control;
2. obtains the locks or records the intents needed to prevent a conflicting outcome;
3. writes a `PREPARED(T, digest, coordinator)` record and provisional data to stable storage;
4. replies `YES` only after that state meets its crash-durability contract.

A participant may reply `NO` before a `YES` promise, causing abort. A timeout is not evidence that it voted `NO`: its `YES` response may have been lost after the durable prepare.

### Phase 2: decide and disseminate

If any participant votes `NO` or the prepare deadline expires, the coordinator chooses abort. If every participant votes `YES`, it writes the commit decision durably **before** announcing it. It then repeatedly sends the decision until every participant acknowledges.

On `COMMIT`, a participant makes provisional state visible under its concurrency-control rules, records the terminal state, and releases locks. On `ABORT`, it discards/undoes provisional state and releases locks. Repeated decisions return the same result.

The coordinator does not need to wait for every phase-two acknowledgement before returning success if its decision is durable and a recovery service will continue dissemination. That optimization trades lower client latency for a potentially large background resolution queue. The API must not imply that every replica or downstream observer is already updated.

### Why prepared participants block

Once prepared, a participant cannot safely infer the outcome from silence:

- the coordinator may have logged `COMMIT` and lost the outbound message;
- another participant may have voted `NO`, so commit is illegal;
- a replacement coordinator may be replaying the decision slowly.

Timing out and aborting can contradict an already durable commit. Timing out and committing can contradict an abort. Classic 2PC therefore blocks an uncertain prepared participant until it can learn the decision. Replicating the decision record reduces the coordinator failure window, but it does not let a participant invent an answer when the decision quorum or its own replica quorum is unavailable.

## Failure traces

### Participant crashes before voting

1. A prepares; B crashes before a durable `PREPARED` record.
2. The coordinator times out and durably decides abort.
3. A receives abort and releases locks.
4. B recovers with no prepared promise and treats `T` as aborted when it receives replay.

No participant promised commit, so abort is safe.

### Vote is lost after durable prepare

1. B writes `PREPARED(T)` and sends `YES`.
2. The response is lost; the coordinator times out and decides abort.
3. B must retain prepared state until abort arrives or it queries the decision record.

If B presumed abort merely because the vote timed out, it could race a coordinator that actually received every `YES` and committed.

### Coordinator crashes after recording commit

1. A and B vote `YES`.
2. C durably records `COMMIT` and sends it only to A.
3. A commits; C crashes; B remains prepared and holds locks/intents.
4. C recovers—or another process reads the replicated decision—and replays `COMMIT` to B.

The interval is partially applied but not an arbitrary outcome. Decision durability and idempotent replay are the recovery spine.

### Decision record disappears too early

1. Every live participant acknowledges commit, so cleanup deletes the coordinator record.
2. An old participant snapshot later rejoins with `PREPARED(T)`.
3. No authority can prove commit versus abort.

Transaction metadata garbage collection requires a recovery horizon and membership rule. A replica older than that horizon must be rebuilt from a current snapshot, not allowed to ask questions whose answers were discarded.

### Isolation fails despite atomic commit

1. `T1` reads doctors A and B as on call, then updates A off call.
2. `T2` reads the same snapshot, then updates B off call.
3. Each touches one participant and commits atomically.
4. Both doctors are off call; the cross-row invariant failed through write skew.

Atomic all-or-nothing outcome is not serializable isolation. The read/write conflict protocol must cover the invariant.

## Making the coordinator and participants fault tolerant

In a sharded database, each participant is often a replicated state machine. Preparing a shard means committing the prepare/intents through that shard's consensus group. The coordinator decision may itself be a replicated transaction record.

This composition changes availability:

- each touched shard needs its replica quorum;
- the decision-record shard needs its quorum;
- the transaction waits for the slowest required prepare path;
- coordinator process loss is survivable because authority is in replicated state.

Spanner composes Paxos-replicated participants with 2PC. CockroachDB stores intents plus a transaction record; Parallel Commits uses a `STAGING` record declaring the write set so intent replication and record staging overlap. Another actor can prove commit only if every declared write is present. These are optimizations of the durable proof, not permission to skip it.

Gray and Lamport's Paxos Commit replaces the single 2PC coordinator's volatile role with consensus on participant outcomes. It improves non-blocking progress under its quorum assumptions, while requiring more replicated protocol state and messages. Consensus and atomic commit remain distinct: consensus chooses one value within a replica group; atomic commit couples the outcomes of all resource managers.

## Latency, throughput, and contention model

Let `p` be the number of participants. Prepare and decision dissemination send `O(p)` messages, and every participant must persist at least a prepare outcome plus terminal/cleanup state according to its logging design. With replicated participants, each “persist” is itself a quorum operation.

A useful latency decomposition is:

```text
prepare_latency
  = max over participants(execution + validation + durable_prepare)

commit_latency
  = prepare_latency
  + durable_coordinator_decision
  + client_visibility_requirement
```

The last term is zero only if the client may return after a recoverable decision; it includes the slowest decision application if the contract requires every participant visible before success. Batching can amortize log sync and network packets but increases queue delay and the failure impact of one batch.

If participant availability during the commit window is `a_i` and the decision service availability is `a_c`, an independence approximation gives transaction-path availability `a_c * product(a_i)`. Correlated zones and shared dependencies make the real result worse. More participants narrow the window in which all required quorums are simultaneously available.

Lock or intent hold time begins before prepare and ends only after decision application:

```text
hold_time = execution + prepare + decision_delay + recovery_delay_if_any
```

One in-doubt transaction can therefore block unrelated work on every key it touched. Capacity models must include maximum prepared transactions, lock-table/intents bytes, decision-replay throughput, and the oldest unresolved age—not only committed transactions per second.

Tail latency also amplifies with fan-out. If each participant completion has CDF `F(t)` and were independent, all `p` complete by `t` with probability `F(t)^p`. A transaction coordinator observes the maximum, so a rare slow shard becomes common at high fan-out.

## Optimizations and alternate execution models

- **Read-only or single-participant fast paths:** skip distributed prepare only after proving the participant set cannot expand.
- **Presumed abort/commit:** reduce log records for the common outcome by defining what missing recovery state means. The presumed outcome is a protocol rule, not a timeout guess after `YES`.
- **Parallel prepare/commit work:** overlap independent replication while retaining a durable write-set proof.
- **Timestamped MVCC:** reduce read locks, but still needs validation and an atomic outcome for writes.
- **Deterministic ordering:** Calvin sequences transactions before execution and schedules known read/write sets deterministically. It exchanges per-transaction commit coordination for predeclared access sets, a replicated sequencer, and deterministic execution. Data-dependent interactive transactions do not automatically fit.

No optimization removes the need to identify which durable facts make success irrevocable after each crash point.

## Sagas are durable workflows, not atomic commit

A saga is a durable sequence of local transactions with compensating actions. Intermediate states are visible, and compensation is a new business operation—not time travel.

For a trip workflow:

```text
reserve flight -> reserve hotel -> charge payment
cancel flight <- cancel hotel <- refund payment
```

A refund can fail, arrive after a statement closes, or incur a fee. An email cannot be unsent. Inventory released by compensation may already have affected another customer. The contract is eventual business reconciliation, usually with explicit `PENDING`, `CONFIRMED`, `COMPENSATING`, `COMPENSATED`, and `MANUAL_REVIEW` states.

An orchestrated saga stores the step state, attempt ID, result, next retry, and compensation progress in a durable coordinator. Choreography distributes transitions among event consumers; it removes one code owner but makes global state reconstruction and cycle prevention harder. Neither style is automatically more available—the relevant question is whether every transition and message is durable and idempotent.

Use semantic locks, reservations, version checks, or rereads to limit the isolation anomalies caused by visible intermediate state. Put irreversible actions late when possible, and define an operator-owned terminal path when compensation cannot restore the original business state.

For a database update plus event publication, a transactional outbox places the business row and event record in one local transaction; a publisher or CDC process delivers the event with retries. Delivery is normally at least once, so consumers deduplicate by stable event/operation ID. This solves a dual-write boundary without pretending the broker joined the database transaction.

## Production operation and migration

### Recovery is a continuously running subsystem

Maintain indexed queues for prepared transactions, decisions awaiting acknowledgement, expired coordinator heartbeats, and intents requiring resolution. Recovery work needs reserved I/O and concurrency limits so an outage does not produce a repair storm that starves foreground commits.

Operators need a transaction-status query that reports participant set, authoritative decision, coordinator epoch, prepared age, held resources, and last recovery error. Manual “heuristic commit/abort” is a data-consistency decision; require audited approval and record that atomicity may have been broken.

### Garbage collection needs a proof frontier

Delete intents, undo, and decision records only after every possible recovering participant can determine the terminal outcome through newer durable state. Tie this to replica snapshots, log truncation, membership removal, and backup restore policy. A time-to-live without a stale-replica rule creates unrecoverable in-doubt transactions.

### Migrate deliberately

When introducing cross-shard atomicity, first assign stable transaction IDs and idempotent local handlers, then deploy durable participant/decision state, recovery, and observability before routing production multi-participant writes. Shadow the participant-discovery logic: an omitted index or constraint write is an atomicity bug even if 2PC works perfectly.

When removing 2PC in favor of a saga, change the API contract. Expose pending state, define compensation, update readers that previously assumed atomic visibility, and backfill a workflow state for in-flight transactions. It is not a transparent performance optimization.

### Test every durable boundary

Crash the coordinator and each participant immediately before and after prepare persistence, vote send, decision persistence, decision send, participant application, acknowledgement, and garbage collection. Duplicate and reorder every message. Partition the decision-record quorum, restore an old participant snapshot, change membership mid-transaction, and force lock conflicts. History checking must assert both atomic outcomes and the advertised isolation level.

## Decision framework

1. Is the requirement atomic visibility, serializable isolation, or eventual workflow completion?
2. Can all invariant-related data be placed in one transaction participant?
3. Which participants are discovered dynamically through indexes, constraints, or reads?
4. What durable record makes commit irrevocable, and which quorum protects it?
5. How long can prepared state block foreground work during the worst supported outage?
6. What recovery and metadata-retention rule handles a replica restored from an old snapshot?
7. Are external effects prepare/commit capable, or must they use outbox, idempotency, and compensation?
8. Does the latency and availability budget tolerate the slowest of all touched participant quorums?

## Primary references

- [Gray, *Notes on Data Base Operating Systems* (1978)](https://jimgray.azurewebsites.net/papers/dbos.pdf)
- [Gray and Lamport, *Consensus on Transaction Commit* (ACM TODS 2006), Microsoft Research publication](https://www.microsoft.com/en-us/research/publication/consensus-on-transaction-commit/)
- [Garcia-Molina and Salem, *Sagas* (SIGMOD 1987)](https://doi.org/10.1145/38713.38742)
- [Corbett et al., *Spanner: Google's Globally-Distributed Database* (OSDI 2012)](https://www.usenix.org/system/files/conference/osdi12/osdi12-final-16.pdf)
- [Peng and Dabek, *Large-scale Incremental Processing Using Distributed Transactions and Notifications* (OSDI 2010)](https://research.google/pubs/large-scale-incremental-processing-using-distributed-transactions-and-notifications/)
- [Thomson et al., *Calvin: Fast Distributed Transactions for Partitioned Database Systems* (SIGMOD 2012)](https://cs.yale.edu/homes/thomson/publications/calvin-sigmod12.pdf)
- [Taft et al., *CockroachDB: The Resilient Geo-Distributed SQL Database* (SIGMOD 2020)](https://www.cockroachlabs.com/pdf/cockroachdb-the-resilient-geo-distributed-sql-database-sigmod-2020.pdf)
- [Helland, *Life Beyond Distributed Transactions: an Apostate's Opinion* (CIDR 2007)](https://www.cidrdb.org/cidr2007/papers/cidr07p15.pdf)
