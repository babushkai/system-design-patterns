# Raft (USENIX ATC 2014): Evidence-First Paper Analysis

Raft is not merely “leader plus majority”: it reduces legal states and strengthens leadership so consensus safety is easier to implement. Its election and log restrictions force a new leader to contain every committed entry.

## Publication identity and scope

- **Paper:** *In Search of an Understandable Consensus Algorithm*
- **Authors:** Diego Ongaro and John Ousterhout
- **Venue and version:** USENIX Annual Technical Conference, 2014, pages 305–319; Best Paper Award
- **Companion:** the extended technical report and Ongaro's 2014 Stanford dissertation include client interaction, log compaction, and additional proof detail omitted for conference space

The paper specifies non-Byzantine replicated-state-machine consensus. Servers may crash, restart, lose messages, delay them, duplicate them, or experience partitions; they do not forge protocol messages or arbitrarily corrupt persisted state. Raft orders deterministic commands. It does not itself provide a SQL transaction layer, a scheduler, or application-level exactly-once effects.

This chapter stays with the 2014 protocol; [Consensus Algorithms](../02-distributed-databases/08-consensus-algorithms.md) and [Leader Election](../02-distributed-databases/09-leader-election.md) provide broader comparisons.

## Problem and system contract

A replicated state machine needs every surviving replica to apply the same commands in the same order. Clients should see one reliable service even while some servers fail. The consensus module therefore must decide a single log prefix that cannot later be replaced by a different committed prefix.

Raft decomposes that task into:

1. elect one leader for a numbered term;
2. accept new log entries only through that leader;
3. replicate entries until a majority stores them;
4. commit entries under a rule that preserves leader completeness;
5. apply committed entries in order to deterministic state machines;
6. change membership through a configuration with overlapping quorums.

Safety must hold without timing assumptions. Timing affects whether elections eventually settle and requests complete.

## State and safety properties

Figure 2 is the normative condensed specification. Every server persists `currentTerm`, `votedFor`, and its log before responding in ways that depend on them. It keeps volatile `commitIndex` and `lastApplied`; a leader additionally tracks each follower's `nextIndex` and `matchIndex`.

A log entry contains the state-machine command and the leader term in which it was received. The pair `(index, term)` is the identity used to compare prefixes.

Figure 3 names five properties:

- **Election safety:** at most one leader is elected in a term.
- **Leader append-only:** a leader never overwrites or deletes its own log entries.
- **Log matching:** equal `(index, term)` entries imply identical prefixes through that index.
- **Leader completeness:** an entry committed in a term appears in every leader elected in a later term.
- **State-machine safety:** no two servers apply different commands at the same log index.

These are not five independent wishes. One vote per term plus majority intersection gives election safety; the up-to-date voting rule and current-term commit rule produce leader completeness; AppendEntries prefix checks produce log matching; applying only committed entries in order yields state-machine safety.

## Terms, roles, and election protocol

A server is follower, candidate, or leader. Terms form a monotonically increasing logical clock. Any RPC carrying a higher term makes a server update its term and step down. Stale-term requests are rejected.

Followers expect periodic AppendEntries heartbeats. If an election timeout expires, a follower:

1. becomes candidate and increments its term;
2. votes for itself and persists that vote;
3. sends RequestVote RPCs;
4. becomes leader after votes from a majority, returns to follower on a valid higher/equal-term leader, or starts another term after timeout.

Election timeouts are randomized independently to reduce repeated split votes. Randomization is a liveness device, not the safety proof.

A voter grants at most one vote per term and only if the candidate's log is at least as up to date as its own. “Up to date” compares the last-log term first, then the last-log index. This restriction is essential: it prevents a candidate missing a committed entry from collecting a majority merely because it responds quickly.

## Log replication and commitment

The leader appends each client command locally and sends AppendEntries containing `prevLogIndex` and `prevLogTerm`. A follower accepts new entries only if it has that matching predecessor. On conflict it deletes its divergent suffix and appends the leader's entries. The leader backs up `nextIndex` until it finds the common prefix, then advances the follower.

An entry is stored on a majority when the leader's `matchIndex` values prove it. But Section 5.4.2 adds a subtle restriction: the leader advances `commitIndex` by counting replicas only for an entry from its **current term**. Once a current-term entry is committed, all earlier entries in its prefix become committed indirectly.

Why not count an older-term entry directly? Figure 8 constructs an execution where an old entry exists on a majority yet a later leader can legally overwrite it. The current-term rule closes that hole by linking commitment to the election restriction. Many broken “Raft-like” implementations miss precisely this condition.

Followers learn the commit index in subsequent AppendEntries and apply entries sequentially. A leader must not answer a client before the corresponding command is committed and applied under the service's response protocol.

## Membership change

Changing directly from configuration `C_old` to `C_new` can elect two leaders if disjoint majorities operate during the transition. Section 6 uses **joint consensus**: a transitional `C_old,new` entry requires separate majorities of both old and new configurations for decisions. After that entry commits, the leader appends and commits `C_new`.

During joint consensus, log entries replicate to all servers in both sets and any leader must satisfy both quorum systems. Overlap carries the safety proof across the transition. A real implementation must also handle removed leaders, newly added catch-up replicas, and crashes during each phase; “update the member list” is not a safe reconfiguration algorithm.

## Failure and recovery reasoning

- **Follower crash:** the leader retries AppendEntries. Duplicate RPCs are harmless because prefix matching and log indexes make replication idempotent.
- **Leader crash:** followers elect a new leader. Uncommitted suffixes may be overwritten; committed entries cannot be lost if persistence and the election rules hold.
- **Network partition:** only a component containing a majority can elect or retain a productive leader. A minority leader may believe it is leader, but cannot commit new entries and steps down when it later sees a higher term.
- **Candidate split vote:** no entry is committed by the election itself; randomized timeouts trigger another term.
- **Restart:** persisted term, vote, and log prevent a server from voting twice or forgetting accepted entries. The paper's safety assumes those writes really reach stable storage before replies.

Read-only requests need care. A partitioned old leader cannot safely answer from local state merely because it once held leadership. The extended report uses a committed no-op in the leader's term plus quorum confirmation/lease reasoning. Client retries also need unique request IDs and a state-machine response cache to avoid applying a command twice. These details are not fully presented in the shortened conference paper and must not be inferred from basic log replication.

## Evidence and methodology

Raft's evaluation has three distinct parts; only one measures runtime performance.

### Understandability study

Forty-three students at Stanford and UC Berkeley watched one Raft lecture and one Paxos lecture in counterbalanced order, then took paired quizzes. Fifteen reported prior Paxos experience, and the Paxos video was 14% longer. The authors published materials and used a rubric and randomized grading to reduce bias.

Thirty-three of 43 scored higher on Raft. Mean scores were 25.7/60 for Raft and 20.8/60 for Paxos, a 4.9-point observed difference. A paired t-test gave a 95% confidence lower bound of 2.5 points on the mean difference. This supports an educational claim for these materials and participants; it does not prove every Raft implementation is easier to make correct.

### Formal analysis

The authors wrote a roughly 400-line TLA+ specification. At publication, the Log Completeness Property had a mechanically checked proof that relied on auxiliary invariants not all mechanically verified; State Machine Safety had a detailed informal proof. This is stronger evidence than testing alone but not a claim that every implementation refines the model.

### Election measurements

Figure 14 repeatedly crashed the leader of a five-server cluster whose broadcast time was about 15 ms. Each line represents 1,000 trials, except the no-randomness case with 100. The test deliberately varied log lengths and synchronized a final heartbeat to encourage split votes.

- With no timeout randomness, elections repeatedly exceeded 10 seconds.
- A 150–155 ms timeout range produced 287 ms median downtime.
- A 50 ms random spread capped the worst observed completion at 513 ms over 1,000 trials.
- With 12–24 ms timeouts, mean election time was 35 ms and the longest trial 152 ms, but smaller timeouts violated the needed timing separation and caused unnecessary elections.

The paper recommends a conservative 150–300 ms range for that environment. It is not a universal configuration value: real deployments must measure heartbeat broadcast tails, pauses, storage latency, and network paths.

## Assumptions and omissions

- A majority is required for availability; a two-of-three outage stops progress.
- One leader orders each log, so its CPU, disk, and network can bound throughput.
- Safety assumes durable persistence and non-Byzantine behavior.
- Deterministic state-machine execution is outside consensus but required for identical replicas.
- The conference paper omits complete snapshot/log-compaction and client-session protocols.
- Joint consensus is safe but operationally intricate; later implementations may choose different proven reconfiguration protocols.
- Election benchmarks are small-cluster experiments, not WAN or overload studies.

## Later practice versus the paper

Production Raft families often add pre-vote phases, leadership transfer, accelerated conflict repair, snapshot streaming, read-index protocols, leases, witness/non-voting members, and single-server-at-a-time membership changes. Some additions appear in the dissertation or later implementation work; others are separate protocols. They must be reviewed with their own invariants.

What should be retained from the paper is the review method: enumerate persistent state, term transitions, quorum intersections, the exact commit rule, and every recovery state. A product calling its metadata service “Raft-based” does not establish which of these extensions it uses.

## Implementation review questions

1. Which state is persisted before granting a vote or acknowledging an append?
2. Does RequestVote compare last term before last index?
3. Does a leader commit older-term entries only through a committed current-term entry?
4. Can an old leader answer a read without proving current leadership?
5. How are retried client commands deduplicated at the state machine?
6. What quorum intersection protects every membership transition?
7. Are election timeouts based on measured worst-case broadcast and pause tails?

## Primary sources

- [Ongaro and Ousterhout, *In Search of an Understandable Consensus Algorithm* (USENIX ATC 2014), official PDF](https://www.usenix.org/system/files/conference/atc14/atc14-paper-ongaro.pdf)
- [USENIX publication and presentation page](https://www.usenix.org/conference/atc14/technical-sessions/presentation/ongaro)
- [Ongaro and Ousterhout, extended Raft technical report](https://raft.github.io/raft.pdf)
- [Ongaro, *Consensus: Bridging Theory and Practice* (Stanford dissertation, 2014)](https://web.stanford.edu/~ouster/cgi-bin/papers/OngaroPhD.pdf)
