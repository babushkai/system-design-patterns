# Consensus Algorithms

Consensus is the safety core of a replicated state machine: replicas choose one ordered history even when processes crash, messages are delayed or duplicated, and the network partitions. Its value is not that every server is always current. Its value is that no two successful majorities can make incompatible decisions under the stated failure model.

Consensus design must preserve **replicated-state-machine safety, quorum intersection, log choice, recovery, membership, and explicit liveness assumptions**. [Leader Election](./09-leader-election.md) covers candidacy, authority epochs, leases, external-resource fencing, failover timing, and handoff. The [Raft paper analysis](../09-whitepapers/07-raft.md) follows one protocol; the sections below compare common proof obligations.

## The contract and failure model

For a single consensus instance, the usual properties are:

- **Agreement:** no two correct processes decide different values.
- **Validity/non-triviality:** a decided value came from an allowed proposal.
- **Integrity:** a process decides at most once.
- **Termination:** correct processes eventually decide, under explicit liveness assumptions.

A replicated log repeats consensus for slots and adds total order: replicas apply the same committed command at each index, in index order. A service normally also needs linearizable client semantics, duplicate suppression, snapshots, membership changes, and deterministic state-machine execution. Those are not automatic consequences of “using Raft” or “using Paxos.”

The failure model matters. Classic Paxos, Viewstamped Replication, and Raft tolerate non-Byzantine crash/recovery and arbitrary message loss, delay, duplication, and reordering. They assume persistent protocol state does not roll backward and nodes do not forge messages. Byzantine behavior, undetected disk corruption, and compromised identities require different protocols or additional integrity mechanisms.

### Safety is asynchronous; liveness is conditional

The FLP result says a deterministic consensus protocol cannot guarantee termination in a fully asynchronous system with even one possible crash. It does not say consensus safety is impossible. A protocol can refuse to decide forever rather than decide inconsistently.

Practical systems obtain liveness from partial synchrony, randomized timeouts, failure detectors, or a stable leader: after some unknown point, enough messages and processing complete within a useful bound. Timeouts choose when to try another leader; they must never be used as proof that an old value did not commit.

## State that carries the proof across crashes

A production consensus group persists enough information to prevent a restart from contradicting promises made before the crash. Depending on the protocol, that includes:

- the highest promised ballot/term;
- accepted or logged entries with their ballot/term and index;
- the current membership/configuration epoch;
- snapshot index, term, checksum, and state-machine image;
- client-session deduplication state if the API promises retry safety.

Volatile state includes replication progress, next indexes, timers, and cached read barriers. A server may reconstruct volatile progress conservatively. It must not reconstruct a vote, promise, or accepted entry by guessing.

The replicated state machine separately tracks `commit_index` and `applied_index`. An entry can be durably chosen but not yet applied. A read that observes application state must wait until the required committed index is applied.

## Quorum intersection is the safety mechanism

With `n = 2f + 1` voters and majority quorums of `f + 1`, any two majorities intersect. If an earlier quorum chose a value, a later protocol round contacts at least one member of that quorum. The protocol's state and selection rule force the later round to preserve the choice.

Intersection alone is insufficient:

- the intersecting node must have persisted the relevant state;
- a new proposer must actually read and honor that state;
- membership changes must preserve intersection across configurations;
- replica placement must make the desired quorum available after correlated failures.

Flexible Paxos observes that not every quorum type must intersect every other type; phase-one quorums must intersect phase-two quorums. This can trade read/recovery and write quorum sizes, but it changes the liveness and load calculation. Never change quorum numbers without restating which quorum pairs must intersect.

Replica count is not failure tolerance by itself. Five voters on one rack still share a rack failure. Place voters across the failure domains named in the availability contract, and calculate whether a quorum remains after each domain loss.

## Paxos: choosing one value

Paxos makes the intersection argument explicit through numbered ballots.

### Phase 1: prepare and promise

A proposer chooses a globally unique, monotonically ordered ballot `b` and sends `PREPARE(b)`. An acceptor replies if `b` is higher than its persisted promise, records that it will reject lower ballots, and returns its highest previously accepted `(ballot, value)`, if any.

After responses from a phase-one quorum, the proposer must select the value attached to the **highest accepted ballot** it observed. Only if none reported an accepted value may it choose a new client value.

### Phase 2: accept

The proposer sends `ACCEPT(b, value)`. An acceptor accepts if `b` is at least its current promise, persists the acceptance, and replies. The value is chosen after a phase-two quorum accepts it.

The selection rule is the bridge between quorums. Suppose `X` was already chosen. Any later phase-one quorum intersects the quorum that accepted `X`; the highest relevant accepted response constrains the new proposer to carry `X` forward. A proposer that takes a local majority vote over returned values, or always prefers its client's new value, is not Paxos.

Multiple proposers can repeatedly preempt one another with higher ballots while preserving safety and making no progress. A stable distinguished proposer supplies liveness and lets subsequent instances skip phase one in the normal case.

## From one value to an ordered replicated log

Multi-Paxos runs a consensus instance per log slot. A stable leader completes phase one for a ballot and drives phase two for many slots. Implementations must handle holes, retransmit chosen entries, prevent two commands at one slot, and apply only a contiguous committed prefix.

Viewstamped Replication and Raft organize the same replicated-state-machine problem around a strong primary/leader and explicit logs. Raft narrows the legal states through several coupled rules:

- voters grant leadership only to candidates whose logs are sufficiently up to date;
- AppendEntries verifies the previous `(index, term)`, giving the log-matching property;
- a leader repairs follower suffixes from its own log;
- a leader counts replicas to directly commit only entries from its current term; earlier entries become committed through the committed prefix.

The current-term rule is essential. An older-term entry can be present on a majority in an intermediate execution and still be legally overwritten until a current leader establishes the necessary election/log relationship. “Stored on most nodes” is not a protocol-independent definition of committed.

Paxos and Raft can implement equivalent fault-tolerant logs, but their state decompositions and proofs differ. Mixing Raft voting with an ad hoc Paxos-style log or changing one commit rule in isolation invalidates the original proof.

## Client commands and reads

### Writes and retry identity

A leader normally appends a command, replicates it to the required quorum, marks it committed under the protocol's rule, applies it, and then replies. Replying after only a local append exposes an uncommitted command that a later leader may discard.

The reply can be lost after application. If the client retries `increment`, the state machine may increment twice even though consensus delivered every log entry exactly once by index. Exactly-once-looking APIs need a stable client/session ID, monotonically increasing request number, and a replicated response cache or an idempotent business operation.

### Linearizable reads

Reading from “the leader” is not automatically linearizable. An isolated old leader may still believe it is authoritative while a majority has elected another. Safe common patterns are:

- append or wait for a committed barrier in the current leadership epoch;
- confirm current authority with a quorum, capture a read index, and wait until it is applied;
- use a rigorously bounded lease whose clock and renewal assumptions are part of the proof.

Follower reads without such a proof are stale reads and should expose a timestamp/index or staleness contract. Lease construction and external fencing belong in [Leader Election](./09-leader-election.md).

## Membership, learners, and snapshots

### Reconfiguration is consensus state

Switching directly from old voters `{A,B,C}` to new voters `{D,E,F}` permits disjoint majorities to choose incompatible logs. Safe protocols make configuration part of the replicated history and require an overlap rule. Raft's joint consensus temporarily requires majorities of both old and new configurations, then commits the final configuration. Other systems use proven one-at-a-time or epoch-based protocols with their own constraints.

A new server should normally catch up as a non-voting learner before it is counted toward quorum. Adding an empty voter can raise the quorum size before that voter can acknowledge, reducing availability at the exact moment of expansion.

### Snapshot installation

Logs cannot grow forever. A snapshot represents state after an applied log index and must include or align with:

- the last included index and term/ballot;
- current membership;
- state-machine bytes and checksum;
- client deduplication/session state;
- any application schema version required to interpret it.

Publish a snapshot atomically, retain log entries needed by active readers/followers until installation is safe, and install into a temporary location before switching the state pointer. A follower receiving a snapshot concurrently with log entries must not apply an old suffix over the new image. Snapshotting is part of the safety state machine, not a file-copy optimization.

## Concrete failure traces

### An acceptor forgets durable state

1. In a three-node group, A and B accept `X` at ballot 10; `X` is chosen.
2. B restarts from a stale disk image that forgot its promise and accepted value.
3. A is unreachable. A proposer runs phase one at ballot 11 with B and C.
4. Neither response reports `X`, so it proposes `Y`; B and C accept it.

Two values are now chosen. The majority intersection was B, but B forgot the fact that made intersection useful. Stable storage and restore procedures are part of consensus correctness.

### Reply before quorum commit

1. Leader L appends command `charge` locally and replies success.
2. L crashes before any follower accepts it.
3. A new leader is elected without the command and commits later entries.
4. The client has a success response for an operation absent from the authoritative log.

Local durability is not replicated commitment.

### Unsafe reconfiguration

1. Old configuration `{A,B,C}` allows A+B to commit `X`.
2. An external control plane activates `{D,E,F}` without a jointly committed transition.
3. D+E commit `Y` at the same log index.

No quorum intersects. Reconfiguration must be ordered by the old safety regime before the new regime acts.

### Non-deterministic application

1. Every replica commits command `allocate_random_discount` at index 80.
2. Each calls its local random generator or wall clock during apply.
3. Logs match, but materialized state diverges.

Replicate the chosen random value/time as command data or make the state-machine transition deterministic. Consensus orders inputs; it does not reconcile arbitrary execution.

### Old leader serves a stale read

1. L is partitioned from the majority but can still reach a client.
2. The majority elects L2 and commits `x = 9`.
3. L reads local `x = 7` and responds as if current.

Leader identity is historical unless refreshed by a quorum/read barrier or a valid lease.

## Capacity and cost model

For `n` full voters, a stable leader sends each log byte to roughly `n - 1` followers. At command rate `lambda` and average encoded entry size `s`:

```text
leader_replication_egress ~= lambda * s * (n - 1)
cluster_log_write_rate    ~= lambda * s * n   before compaction
```

Protocol headers, retransmission, snapshots, encryption, and checksums add overhead. A sharded system scales by using many consensus groups with leaders distributed across machines; one global group still has one ordering bottleneck.

Normal-case commit latency is governed by the local durable append and the `q-1` fastest follower acknowledgements needed with the leader to form quorum `q`:

```text
commit_path >= max(local_stable_write,
                   quorum_order_statistic(follower_network + stable_write))
```

The slowest follower can lag without delaying a majority, but when a fast quorum member degrades, the next-fastest path defines latency. Batching amortizes log sync and packet cost across commands; larger batches increase queueing delay and recovery units.

Throughput is bounded by the minimum of leader CPU/apply capacity, durable-log bandwidth and sync rate, leader replication egress, quorum follower ingest, and state-machine execution. Snapshots and catch-up consume the same disks and network. Reserve recovery capacity; a cluster sized only for foreground traffic may be unable to replace a failed voter before another failure.

Under independent per-node availability `a`, majority availability is the binomial sum for at least `q` available nodes. This is only a sanity check. Shared racks, zones, software versions, disks, and control planes dominate real outages, so evaluate named correlated failure scenarios directly.

## Production operation and verification

Track proposal rate and failure, commit latency by quorum member, committed-to-applied lag, follower log/snapshot lag, retransmission, durable-log sync tails, snapshot build/install duration, leader changes, current membership, and time without quorum. Alert on **progress**, not just process health: five live processes can be unable to commit.

Keep consensus WAL and metadata on storage whose flush semantics are understood. Test power loss, torn/corrupt writes, full disks, restored virtual-machine snapshots, and filesystem errors; crash-fault algorithms do not mask arbitrary storage lies. Backups must capture state-machine and consensus metadata at a compatible index, or restore as a fresh learner rather than a voting replica with rolled-back state.

Use model checking or specification for ballot/log/membership invariants, implementation-level fault injection at every persistence boundary, and history checking for the client contract. Test duplication, delay, reordering, asymmetric partitions, long pauses, clock jumps if leases are used, and membership changes during snapshot/catch-up. A green happy-path integration test provides little evidence for consensus safety.

## Decision framework

1. What exact operations require one linearizable order, and can they be partitioned into independent groups?
2. What crash, storage, network, and Byzantine behaviors are inside the failure model?
3. Which persisted fields preserve promises and accepted history across restart and restore?
4. Which quorum types must intersect, including during membership change?
5. What read protocol proves current authority and applied state?
6. How are duplicate client commands recognized after leader and process failure?
7. Can quorum replication, snapshots, and replacement fit the latency and recovery-capacity budget?
8. Has the implemented protocol (not only the paper algorithm) been model- and history-tested?

## Primary references

- [Fischer, Lynch, and Paterson, *Impossibility of Distributed Consensus with One Faulty Process* (JACM 1985)](https://groups.csail.mit.edu/tds/papers/Lynch/jacm85.pdf)
- [Lamport, *Paxos Made Simple* (2001)](https://lamport.azurewebsites.net/pubs/paxos-simple.pdf)
- [Oki and Liskov, *Viewstamped Replication: A New Primary Copy Method to Support Highly-Available Distributed Systems* (PODC 1988)](https://doi.org/10.1145/62546.62549)
- [Ongaro and Ousterhout, *In Search of an Understandable Consensus Algorithm* (USENIX ATC 2014)](https://www.usenix.org/system/files/conference/atc14/atc14-paper-ongaro.pdf)
- [Howard, Malkhi, and Spiegelman, *Flexible Paxos: Quorum Intersection Revisited* (2016)](https://arxiv.org/abs/1608.06696)
- [Ongaro, *Consensus: Bridging Theory and Practice* (Stanford dissertation, 2014)](https://web.stanford.edu/~ouster/cgi-bin/papers/OngaroPhD.pdf)
