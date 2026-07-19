# Leader Election

Leader election assigns temporary authority over a resource. The safety problem is not preventing two processes from ever *believing* they are leader—pauses and partitions make that impossible to guarantee from local state. The safety problem is ensuring that only the current generation can make accepted changes to the protected system.

This chapter owns **terms and leadership epochs, failure detection, leases, activation barriers, fencing tokens, failover, and operational handoff**. [Consensus Algorithms](./08-consensus-algorithms.md) owns replicated-log agreement and quorum safety. [Distributed Locks](../01-foundations/09-distributed-locks.md) owns the broader lock API and critical-section semantics. A consensus protocol may contain an election, but external work still needs the authority and fencing reasoning developed here.

## Define the leadership contract

Leadership is always scoped:

- leader for one consensus log;
- primary for one database range;
- scheduler for one tenant or queue partition;
- controller for one reconciliation key;
- writer for one external storage resource.

A global singleton often creates an unnecessary bottleneck and failure domain. Prefer many independently elected leaders when operations commute across resources.

For each resource, specify:

- what actions only the leader may perform;
- where authority is validated;
- whether followers may serve stale or read-only work;
- how long failover may take;
- what state a new leader must catch up before activation;
- what happens to commands issued by an old generation but delivered late;
- whether the protected resource can enforce a fencing token.

“Only one pod runs the cron job” is not a safety contract. If the old pod pauses after charging a card and the new pod retries, the external effect needs its own idempotency or fencing boundary.

## State and invariants

A robust election carries these durable or replicated fields:

- **resource ID:** the exact scope of authority;
- **epoch/term/generation:** a monotonically increasing leadership number;
- **candidate/leader identity:** unique across process reincarnations;
- **membership and votes or lease grant:** evidence authorizing the generation;
- **log/checkpoint position:** state the candidate has incorporated;
- **lease deadline and renewal evidence**, if time bounds authority;
- **fencing token:** the generation understood by the protected resource;
- **activation state:** elected, catching up, active, draining, or revoked.

The key invariants are:

1. The election authority does not grant two leaders for the same resource and generation.
2. A newer generation supersedes every older generation, even if an old process is still running.
3. A protected resource rejects state-changing operations from an older generation.
4. A leader does not serve authoritative work until its state satisfies the activation barrier.
5. Losing renewal or quorum causes prompt self-demotion, but safety does not rely only on that self-demotion.

An epoch orders leadership, not application operations by itself. The storage engine, queue, or downstream service must compare it on the operation that matters.

## Election mechanisms

### Consensus-integrated election

Raft candidates increase a term, persist their vote, and need a majority. Voters also require an up-to-date log, so a candidate missing committed entries cannot become leader. A server seeing a higher term steps down. Multi-Paxos and Viewstamped Replication use different mechanics but similarly stabilize one proposer/primary to drive progress.

Majority intersection prevents two leaders in the same term under the protocol. It does not prevent an isolated term-4 leader from running concurrently in physical time with a term-5 leader. The consensus log rejects old-term replication, but a separate object store or payment API knows nothing about those terms unless the application propagates and validates them.

Election timing and the detailed log-safety proof belong in [Consensus Algorithms](./08-consensus-algorithms.md) and the [Raft paper analysis](../09-whitepapers/07-raft.md). Operationally, the leader must still establish current-term state before serving linearizable reads or external work.

### Coordination-service election

A strongly consistent service such as Chubby, ZooKeeper, or etcd can serialize contenders. A common ZooKeeper recipe creates an ephemeral sequential node under an election path; the lowest sequence is leader, and each contender watches its predecessor rather than all contenders watching one node. Session expiry removes abandoned candidacies.

An etcd-style recipe uses a transaction to create an election key only if the prior version is absent and attaches it to a lease. The acquisition transaction's monotonic revision can identify the generation. A random lease ID or wall-clock expiry is not a fencing token.

The coordination service chooses the winner; it does not fence another database automatically. The elected process must carry the sequence/revision to the protected resource, which must remember the greatest accepted generation.

### Lease-based authority

A lease is a grant that remains exclusive for a bounded interval. It improves failover over an unbounded lock because authority eventually expires when a holder disappears. It also introduces time into the proof.

The granting service measures expiry on its own monotonic time. A client receiving a lease cannot simply store `wall_clock_now + TTL`: its clock may differ, and the grant was already aging in transit. Safe clients use a conservative local deadline derived from monotonic elapsed time, subtract communication and clock uncertainty, renew well before that deadline, and enter a non-serving “jeopardy” state when renewal evidence is missing.

A longer lease reduces renewal load and false failover but lengthens the worst crash-detection window. A shorter lease improves potential failover and increases sensitivity to scheduler pauses, overloaded coordination services, and network tails. Renewal cadence must come from measured delay and pause distributions plus the service's clock model; `TTL/3` is an implementation convention, not a proof.

Chubby combines sessions, leases, and **sequencers** so a protected service can reject a former lock holder. When sequencer validation is impossible, its lock-delay fallback merely waits before regranting; timing reduces risk but is weaker than fencing.

### Static-rank and ring elections

Bully and ring algorithms can choose a deterministic candidate in a reliable, fully connected environment. Without quorum or an external fencing authority, two network components can each choose a winner. They are membership/discovery algorithms, not a safe basis for correctness-critical leadership under partitions.

## Fencing the old leader

Consider a lease grant returning generation 100 to A and the next grant returning 101 to B. Every state-changing request includes the generation:

```text
resource.write(command, generation)

if generation < greatest_generation_seen:
    reject STALE_LEADER
else:
    durably advance greatest_generation_seen as required
    apply command
```

The validation and update must be atomic with respect to the protected operation. A check in application memory followed by an unfenced write has a time-of-check/time-of-use race.

Tokens are scoped. If a job writes database X and object store Y, both must reject old generations; seeing token 101 at X does not teach Y about it. A newly elected leader may need to establish a fence at each resource before publishing itself active.

If a third-party API cannot validate generations, alternatives are weaker or more expensive:

- make each effect idempotent with a stable operation ID;
- route all effects through a fenced proxy or transactional outbox;
- physically isolate/terminate the old process before activating the new one (STONITH);
- redesign the operation so duplicate execution is harmless.

Killing the old process is an operational fence only if the kill authority and network isolation are themselves reliable. “The old leader should notice” is never a fence.

## Activation, service, and handoff protocol

Election victory should begin a transition, not immediately enable writes.

### Activation barrier

1. **Acquire generation:** obtain majority votes or a linearizable lease/election record.
2. **Recover state:** replay the log, load checkpoints, and reconcile work left by the prior leader.
3. **Establish current authority:** for a replicated log, commit or confirm a barrier in the new term and apply through it.
4. **Fence dependencies:** make external resources reject older generations.
5. **Publish readiness:** advertise the endpoint and accept leader-only work.

The order prevents a log-stale winner or an elected-but-unfenced process from acting. A readiness probe should represent completion of this barrier, not merely process health.

### Losing authority

On failed renewal, observed higher term, quorum loss, or explicit transfer, stop admitting new leader work, cancel or drain in-flight work according to its idempotency contract, and withdraw readiness. Some requests may already be buffered in the network; protected resources must still fence them.

### Graceful transfer

For maintenance, catch the target up first, stop or bound new work, transfer/elect a higher generation, establish its barrier, and only then demote the old leader. A transfer protocol can reduce downtime, but “zero gap” and “zero overlap” are different goals. Fencing makes a brief physical overlap safe; clients retry through a brief service gap.

## Concrete failure traces

### Process pause outlives its lease

1. A obtains lease generation 40 and starts work.
2. A pauses for garbage collection or host suspension.
3. The lease expires; B obtains generation 41 and writes successfully.
4. A resumes with stale local state and sends a delayed write.

Self-demotion could not run during the pause. The resource must reject generation 40, or the delayed write can corrupt state.

### Isolated old consensus leader reaches an external API

1. Term-8 leader A loses contact with the majority but still reaches a payment service.
2. The majority elects B in term 9.
3. A cannot commit to the consensus log, but it can still call the external API.

Consensus protects the replicated log, not an unrelated effect. Put an outbox command in the log and execute it idempotently/fenced, or propagate term 9 authority to a validating proxy.

### Candidate is elected before catch-up

1. A coordination service grants leadership to C based only on liveness.
2. C restored an hour-old checkpoint and immediately schedules jobs it believes incomplete.
3. The prior leader had completed those jobs after the checkpoint.

Election chose one process but did not choose correct state. Activation must include a durable progress barrier, and job execution must deduplicate by operation ID.

### Aggressive timeout creates leadership churn

1. Election timeout is below the high-percentile durable-log or scheduler pause.
2. A healthy leader occasionally misses the window.
3. Followers start elections; requests pause, caches cold-start, and leadership moves load to another node.
4. The load spike makes that node miss its window too.

Failure detection should be tuned from end-to-end heartbeat processing tails, not median network RTT. Pre-vote-style checks can prevent an isolated node from inflating terms, but cannot fix an overloaded quorum.

### Watch herd overloads the election service

1. Thousands of contenders watch one leader key.
2. It disappears; every contender wakes, reads metadata, and attempts compare-and-set.
3. The coordination service overload delays lease renewals for unrelated leaders.

Predecessor watches, randomized campaign backoff, sharded election scopes, and admission control bound this control-plane burst.

## Failover and capacity model

Failover time has separable components:

```text
failover_time
  = failure_detection_or_remaining_lease
  + election
  + state_catch_up
  + activation/fencing
  + client_reroute
```

Reporting only “election completed in 300 ms” ignores a 20-second lease remainder or minutes of state catch-up. Track and budget each term.

For `M` independently leased resources renewed every `h` seconds, baseline keepalive demand is roughly `M/h` renewals per second, before retries and watches. A single global leader minimizes renewals but concentrates work; per-partition leaders distribute data-plane load while multiplying control-plane state. Proxying/aggregating sessions can reduce connections, as Chubby's production experience showed.

If a successor lags by `L` log bytes and effective catch-up bandwidth is `B`, activation needs at least `L/B`, plus snapshot creation/install and replay. Reserve bandwidth for failover; a saturated leader cannot both serve peak traffic and seed a replacement quickly.

Election timeouts must exceed normal end-to-end heartbeat delay, including leader scheduling, durable-log stalls if they share the event loop, follower scheduling, and network tails. Randomization should be wide enough to separate contenders under the measured distribution. Larger values improve stability and worsen detection latency; there is no topology-independent constant.

## Production operation and migration

Observe current leader and generation per resource, leadership duration, renewal margin, campaign attempts, vote rejection reasons, changes per hour, activation-stage duration, log/checkpoint lag at election, stale-token rejections, unfenced external calls, watch backlog, and client reroute time. Frequent successful elections are still an availability problem.

During rolling upgrades, keep election-record and token formats backward compatible. Drain leadership before stopping a node, but test abrupt termination too. Do not combine a membership change, election-protocol change, token-width change, and storage migration in one step; each changes a proof boundary.

To add fencing to an existing service, first deploy resource-side token storage and validation in observe-only mode, propagate tokens on every caller path, verify no unfenced writes remain, then enforce rejection. Starting token enforcement before all legitimate writers carry a generation creates an outage; generating tokens before resources validate them creates a false sense of safety.

Test process pauses longer than leases, clock steps and monotonic-clock behavior, asymmetric partitions, lost renew replies, duplicate grants after restore, old packets released after new activation, coordination-service overload, successor catch-up failure, and fencing at every external resource. Assert history properties: no accepted lower-generation operation after a higher generation at that resource.

## Decision framework

1. What exact resource does one election govern, and can leadership be partitioned more finely?
2. Which service serializes generations, and what failure model protects it?
3. What state/log position must a winner reach before it becomes active?
4. Where is the fencing token validated atomically with the protected operation?
5. Which external effects cannot be fenced, and how are duplicates made safe?
6. What are the measured detection, election, catch-up, fencing, and reroute components of failover?
7. Can renewal/watch load survive a leader-loss herd without expiring healthy leases?
8. How does restore or rolling upgrade preserve monotonic generations and identity?

## Primary references

- [Gray and Cheriton, *Leases: an Efficient Fault-Tolerant Mechanism for Distributed File Cache Consistency* (SOSP 1989)](https://web.stanford.edu/class/cs240/readings/leases.pdf)
- [Chandra and Toueg, *Unreliable Failure Detectors for Reliable Distributed Systems* (JACM 1996)](https://www.cs.cornell.edu/courses/cs734/2000FA/cached%20papers/ct96.pdf)
- [Burrows, *The Chubby Lock Service for Loosely-Coupled Distributed Systems* (OSDI 2006)](https://research.google.com/archive/chubby-osdi06.pdf)
- [Hunt et al., *ZooKeeper: Wait-free Coordination for Internet-scale Systems* (USENIX ATC 2010)](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Hunt.pdf)
- [Ongaro and Ousterhout, *In Search of an Understandable Consensus Algorithm* (USENIX ATC 2014)](https://www.usenix.org/system/files/conference/atc14/atc14-paper-ongaro.pdf)
