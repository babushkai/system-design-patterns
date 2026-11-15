# Failure Semantics, Detection, and Recovery Boundaries

## TL;DR

A failure model is part of a system's correctness contract. It states which components may stop, restart, delay, omit, corrupt, or equivocate; which failures may be correlated; what evidence other components can observe; and which safety properties must survive when progress is impossible.

Production failures are rarely clean. A process may answer health checks while its disk stalls, one network direction may fail, an acknowledged write may have committed before the client timed out, a restarted worker may still hold stale authority, and a failover can overload the remaining capacity. The useful design questions are therefore:

- What state is durable, volatile, or externally visible at every interruption point?
- Which outcomes are distinguishable from the caller's evidence?
- How is stale authority fenced after failover or restart?
- Which failure domains are independent, and which share power, network, identity, configuration, or load?
- What work is admitted during degradation, and how is backlog drained after recovery?
- How will the invariants be tested under crash, delay, loss, reordering, duplication, partition, corruption, and operator error?

Safety and liveness must be separated. During uncertainty, a system may have to stop making progress to preserve an invariant. Redundancy improves availability only when replicas do not share the same fault and recovery does not create a second failure.

---

## Specify the Failure Contract Before the Happy Path

A program running on one machine either works or stops. A system made of many machines has a third condition, which is by far the commonest: part of it has failed and the rest carries on. A request sent to another machine may be answered, or refused, or it may simply receive no reply, and in the last case the sender cannot tell whether the request was lost on the way, whether it arrived and the machine crashed before acting, whether it was carried out and the reply was lost, or whether the reply is merely late. These possibilities look identical from where the sender stands, and they call for different responses.

This is the central difficulty of distributed systems, and it cannot be engineered away. What can be done is to design each operation so that its possible outcomes are few, are distinguishable where it matters, and each have a safe next step. That design has to come first. A system built for the case where everything works, with failure handling added afterward, tends to treat the ambiguous outcome as either success or failure, and each of those guesses is wrong some of the time.

We begin by naming, for each operation, the outcomes that mean something to the caller:

| Outcome | What the caller knows | Safe next action |
|---|---|---|
| Committed success | operation effect and identity are durable | continue |
| Definitive rejection | no effect occurred under this operation identity | correct request or stop |
| Ambiguous | request or response may have been lost; effect may exist | retry with same operation identity or query status |
| Superseded/fenced | caller no longer has authority for this epoch | reacquire authority; never retry blindly |
| Indeterminate integrity | stored or replicated state may be corrupt | isolate, verify, repair from trusted evidence |

An API that returns only `200` or `500` hides the most important state: whether a timed-out mutation committed. A timeout proves only that the caller did not receive a response; it does not prove rollback. Retrying under a new identity can duplicate a payment or reservation, while refusing to retry can lose an operation that never committed. The safe protocol reuses a stable idempotency key and queries the recorded outcome until the caller can distinguish committed, rejected, or still unknown. [Idempotency and Operation Identity](./08-idempotency.md) defines that protocol; this chapter defines the failure semantics that make it necessary.

### Safety, liveness, and durability

Three properties are in play whenever we reason about failure, and they are often confused. *Safety* says that something forbidden never happens; for example, two owners never both commit writes for the same lease epoch. *Liveness* says that a valid operation eventually completes under stated timing and fault assumptions. *Durability* says that once an operation reaches its documented commit point, its effect survives the documented failures.

These are not interchangeable, and a design frequently has to give up one to keep another. A minority replica may reject writes to preserve safety, sacrificing liveness until communication recovers. A queue may remain available by accepting work to local disk, but whether that work is durable depends on whether the disk lies within the promised failure domain. When a design document says a system is "reliable," we ask which of the three it means, and under which faults.

### An explicit model

A failure model is only useful if it is written down precisely enough for a reviewer to disagree with it. A reviewable contract might say:

```text
Components:
  5 replicated state-machine nodes across 3 zones

Assumptions:
  at most 2 crash-stop or crash-recovery faults
  messages may delay, drop, duplicate, reorder, or partition
  stable storage can lose only writes not acknowledged after fsync
  nodes are authenticated but not Byzantine
  clocks have bounded drift only while synchronized

Required safety:
  one committed log entry per index
  only the current fenced leader may mutate the external resource

Required liveness:
  progress resumes after a majority can communicate for one election interval
```

Every line can be checked against the implementation or challenged against reality. Without the assumptions, "highly available" and "exactly once" are adjectives and not claims.

---

## Fault Taxonomy Without a False Hierarchy

Textbooks often present failure classes as a ladder, from crash faults at the bottom to Byzantine faults at the top, with each level containing those below it. Real failure classes overlap, and a system hardened against a "stronger" class can still fall to a "weaker" one that it did not consider. We describe each class by what the other components can observe.

### Crash-stop and crash-recovery

In a crash-stop model, a failed process never returns. This is convenient for proofs and rare in production, where an orchestrator restarts the process within seconds. In the crash-recovery model the process comes back with its durable state, a new process identity, and no trustworthy volatile state.

Recovery raises questions that crash-stop abstracts away. The last log record may have been fully written, torn, or reordered by the storage stack. Some effects may have escaped to another system before the crash, and that system has no idea the sender died. A network response addressed to the old process can arrive after the restart and be mistaken for a reply to something new. A node restored from a snapshot may reuse an epoch, a nonce, or an operation ID that it had already used before the snapshot's successors were lost. And the node needs some way to prove that it has caught up before it serves anything.

Most of these come down to identity. A PID, container name, or hostname does not define an incarnation, because all three are reused. When authority spans restarts, we persist or obtain a monotonically increasing epoch, so that the rest of the system can tell this life of the process from the previous one.

### Omission, duplication, reordering, and delay

A send, channel, or receive path can omit a message. Transport and application layers may duplicate it during retry. Concurrent paths may reorder it. Delay can be unbounded in the asynchronous model.

The consequence is easiest to see from the caller's side. The entire observation available to it is:

```text
request sent -> timeout
```

That single observation is compatible with all of these histories:

1. the request never arrived;
2. it arrived but did not commit;
3. it committed and the response was lost;
4. it is still queued and will execute later;
5. a response is delayed behind a newer retry.

The caller cannot tell them apart, and the correct action differs among them. A timeout therefore creates suspicion and sets a latency boundary. It does not prove that the operation did not execute, and any design that treats it as proof will eventually duplicate or lose work.

### Timing and gray failure

A component can be correct and still too slow for its caller's deadline, which from the caller's position is a failure. Gray failure is the partial, observer-dependent form of this: a storage node serves cached reads but stalls writes, a proxy can reach one shard and not another, or a host answers a shallow health check while packet loss makes real work impossible. The component's own view of its health and its clients' view disagree, and the monitoring usually takes the component's side.

Resource exhaustion belongs in the failure model for the same reason. CPU throttling and long scheduler queues, stop-the-world pauses, memory pressure with paging or allocation failure, disk tail-latency spikes and full filesystems, exhaustion of sockets, threads, descriptors, connections, or ephemeral ports, and quota or admission rejection by a dependency all leave the process running and unable to do its job.

Slow components often cause more system-wide damage than stopped ones. A stopped component fails fast, and its callers move on. A slow one holds connections, threads, and memory in every caller until the timeout fires, and then attracts retries, so its trouble spreads upstream.

### Value, storage, and software faults

A component may respond on time with an incorrect value because of data corruption, a stale replica, nondeterministic state-machine code, misconfiguration, schema mismatch, numeric overflow, or software defect. Checksums detect some corruption but not a validly encoded wrong value.

Replicating corrupt state three times does not make it correct. Recovery needs an independent source of truth or invariant: end-to-end checksum, immutable log, reconciliation against domain facts, quorum comparison, or backup with verified provenance.

### Byzantine behavior

A Byzantine participant can behave arbitrarily, including sending conflicting statements to different peers. This model is relevant when participants cross trust domains, software/hardware compromise is in scope, or silent arbitrary faults must be tolerated.

The often-quoted $3f+1$ replicas to tolerate $f$ Byzantine faults belongs to a particular family of authenticated Byzantine consensus protocols and timing assumptions; it is not a universal sizing formula for every Byzantine system. PBFT, for example, uses quorums that intersect in enough correct replicas with $n \ge 3f+1$. Protocol choice must state authentication, synchrony, client, storage, and recovery assumptions.

For a single-organization data service, Byzantine consensus may add cost without addressing the dominant faults: bad deployments, stolen credentials, shared control planes, corrupt application writes, and correlated infrastructure loss. Cryptographic authentication and integrity checks are still valuable even when the consensus model is crash fault tolerant.

---

## Failure Domains and Correlation

Redundancy protects only against failures that do not affect all the copies at once. Every component sits inside a nest of failure domains, and a fault in an enclosing domain takes out everything beneath it.

```mermaid
flowchart TD
    ORG[Organization and operator control] --> CLOUD[Cloud account / identity / policy]
    CLOUD --> REG[Region]
    REG --> Z1[Zone A]
    REG --> Z2[Zone B]
    Z1 --> R1[Rack / switch]
    R1 --> H1[Host]
    H1 --> P1[Process]
    ORG --> CFG[Configuration and deployment pipeline]
    CFG --> P1
    CFG --> Z2
```

The physical hierarchy is the part of this picture that most designs get right. The dependencies that cut across it are where correlated failure usually comes from. Two replicas in different zones can still share the same IAM principal or an expiring certificate chain, a single DNS or control-plane dependency, a destructive configuration rollout that reaches both, the same software defect triggered by the same input, one regional service quota, a synchronized backup-deletion job, and capacity assumptions that stop holding once either zone is lost. Each of these is a domain in which the two replicas are one unit.

### Map each invariant to a domain

| Invariant | Threatened by | Containment mechanism |
|---|---|---|
| one active writer | partition, pause, stale lease holder | quorum lease plus downstream fencing token |
| acknowledged data survives one zone | correlated placement, premature ack | failure-domain-aware quorum and commit rule |
| tenant A cannot affect tenant B | shared queues/cache keys/control plane | isolation, quotas, cells, canonical tenant binding |
| rollout cannot corrupt all copies | simultaneous deployment | staged rollout, version skew testing, rollback artifact |
| backup is recoverable | shared credentials/key deletion | isolated account, independent key inventory, restore drill |

The blast radius is an architectural property. Cells and bulkheads reduce it only when routing, state, capacity, control plane, and operations respect the boundary; see [Cell-Based Architecture](../06-scaling/11-cell-based-architecture.md).

### Common-mode testing

Killing one process proves little about zone loss or a bad configuration push, because it exercises only the innermost domain. We test by domain, working outward: process pause and restart; host loss and disk corruption; loss of communication for a rack or zone; asymmetric regional routing; an outage of identity, keys, or policy; deployment of a deterministic bug to every instance; operator deletion followed by a restore from backup; exhaustion of a dependency's quota; and the loss of capacity at the same moment as a traffic spike. The later items are harder to stage, and they resemble the outages that make postmortem reading lists.

---

## Failure Detection Is Evidence, Not Truth

In a fully asynchronous system, a silent process cannot be distinguished with certainty from a correct process whose messages are delayed. Practical systems add timing assumptions and implement **failure detectors** that output suspicion.

### Timeout design

A timeout is a budget for everything that happens between sending a request and reading its response:

```text
queue wait + connection acquisition + network + remote queue
+ remote execution + response + scheduler variance
```

Set it from the caller's end-to-end deadline and measured distributions, not a round number copied into every client. A timeout longer than the useful deadline wastes capacity; one below normal tail latency causes false suspicion and retry amplification.

Use monotonic elapsed time for deadlines. Wall clocks can jump during synchronization and are needed only when the protocol explicitly reasons about real time. Propagate a decreasing deadline through calls rather than giving every hop the original duration.

### Heartbeats and accrual suspicion

Fixed “three missed heartbeats means dead” thresholds are easy to operate but adapt poorly to latency changes. Accrual detectors output a suspicion level derived from the observed arrival distribution; a higher threshold trades slower detection for fewer false positives.

Heartbeats need isolation from the work path only with care. A separate high-priority channel distinguishes a dead process from an overloaded work queue, but it can also report green while the application path is unusable. Pair process/liveness evidence with real readiness and end-to-end probes.

### Membership dissemination

All-to-all heartbeat traffic grows quadratically. Gossip membership protocols such as SWIM use randomized probing, indirect probes, and infection-style dissemination to scale, while incarnation numbers prevent an old suspicion from overriding a newer “alive” state.

Membership is not consensus. Two nodes can temporarily hold different membership views. Do not derive exclusive write authority from gossip alone.

### Asymmetric and partial connectivity

A can reach B does not imply B can reach A, and both reaching a health endpoint does not imply they can exchange protocol traffic. Detect from multiple vantage points and retain path-specific telemetry. During a partial partition, a node may reach the database but not the lease service, or the control plane but not peers.

The system should avoid declaring a partition topology from one observer's perspective. Authority should come from a quorum or fenced external coordinator, not from “I cannot see the old leader, therefore I am leader.”

---

## Authority, Epochs, Leases, and Fencing

Failover creates a new owner before the old owner is proven dead. If the old process resumes, two actors may perform side effects.

```mermaid
sequenceDiagram
    participant A as Worker A, epoch 41
    participant C as Coordinator
    participant B as Worker B, epoch 42
    participant S as Storage
    A->>C: lease 41
    Note over A: long pause
    C->>B: lease 42
    B->>S: write(value, fence=42)
    S-->>B: accepted, highest=42
    A->>S: late write(value, fence=41)
    S-->>A: rejected as stale
```

A lease reduces overlapping ownership only under its clock and communication assumptions. A **fencing token** makes the protected resource reject stale owners. If the downstream database, object store, or device ignores the token, the lease service cannot prevent a paused client from acting.

Fencing works only if a few rules hold together. Tokens increase monotonically within each protected scope. Acquisition and renewal are durable at the coordinator's commit point, so a coordinator failover cannot reissue an old token. The resource remembers the highest token it has accepted and rejects late messages that carry a lower one, even when the sender's credentials are still valid. Authority is never restored from a process snapshot, since the snapshot predates whatever happened after it. And expiration is computed with the protocol's stated clock-drift bound.

Leadership for replicated state machines is treated in [Leader Election and Fencing](../02-distributed-databases/09-leader-election.md); application locks in [Distributed Locks](./09-distributed-locks.md).

---

## Crash Consistency and Recovery State Machines

### Locate every irreversible boundary

A crash can interrupt an operation between any two steps, so we list the steps. For a mutation that touches a log, a database, a queue, and an external API, the crash points are:

```text
receive request
  -> persist operation identity
  -> commit domain state
  -> enqueue event
  -> call external effect
  -> record effect receipt
  -> send response
```

At each arrow, ask what survives and what a retry observes. A local transaction cannot atomically commit an external email, payment network request, or object-store mutation. Use durable workflow/effect protocols where required; see [Effect Commit Protocols for Workflows](../18-workflow-job-systems/06-retry-idempotency-compensation.md).

### Write-ahead recovery

The write-ahead log is the standard tool for making a crash at any point recoverable, and it requires more than a call to `append()`:

1. encode record length/version/checksum;
2. write the log before the dependent state mutation;
3. reach the documented durability boundary (`fsync`, replicated quorum, or equivalent);
4. make replay deterministic or idempotent;
5. detect and truncate an incomplete tail safely;
6. checkpoint without losing the replay prefix;
7. test filesystem and hardware ordering assumptions.

The storage mechanics are in [Write-Ahead Logging](../03-storage-engines/04-write-ahead-logging.md). Recovery must distinguish “record durable but not applied” from “applied but completion marker absent.”

### Rejoin, catch-up, and quarantine

A restarted replica should not serve merely because its process is healthy. Before it does, it validates its local identity, configuration, log, and snapshot. It compares its epoch and committed position with the current membership, since the cluster may have moved on without it. It repairs or replaces state that has diverged or is corrupt, catches up to within a bounded lag, and enters read or write service only once the invariants required for that service hold.

If the state fails its integrity checks, we quarantine it. Automatically copying a suspect replica to its peers can turn a local corruption into a cluster-wide one.

### Reconciliation

Retries prevent duplicate intent only when every consumer honors the operation identity, and some paths will always end in an unknown state. Reconciliation repairs the facts afterward by comparing independent records of what should be true:

```text
desired domain state
vs durable execution log
vs external provider receipts
vs derived projections
```

We run reconciliation continuously, partition it for scale, and make its repairs idempotent and auditable. "An operator will inspect the dead-letter queue" is not a recovery protocol.

---

## Network Partitions and Healing

A partition is a communication failure that separates some participants from others; it may be partial, asymmetric, protocol-specific, or indistinguishable from process delay. The CAP theorem constrains consistency and availability when messages needed for coordination cannot arrive; its precise scope is in [CAP Theorem: Scope, Proof, and Design Consequences](./03-cap-theorem.md).

Handling a partition in operation, as opposed to in a proof, involves five obligations:

1. **Authority:** only a quorum-backed/fenced side continues non-mergeable writes.
2. **Degradation:** define which reads, buffered writes, or local functions remain safe.
3. **Backlog:** bound hints, queues, leases, and retained versions while peers are absent.
4. **Healing:** exchange summaries, identify missing/divergent state, repair, and verify.
5. **Reintroduction:** rate-limit catch-up and prevent an old node from serving stale data.

### Stop, buffer, or diverge

For each operation we choose deliberately among a small set of behaviors:

| Strategy | Valid when | Cost after healing |
|---|---|---|
| Reject without quorum | invariant is not mergeable | reduced availability |
| Serve a labeled stale read | consumer accepts a bounded snapshot | session/monotonic-read handling |
| Buffer intent without claiming commit | durable local capacity is bounded | drain, expiry, duplicate handling |
| Accept concurrent writes | domain has deterministic merge/conflict semantics | conflict detection, user-visible resolution |

Do not classify the whole database as “CP” or “AP” and stop reasoning. A balance transfer, shopping-cart add, presence heartbeat, and profile photo update can have different safe behaviors.

### Repair protocols

Anti-entropy can compare Merkle trees, version vectors, log positions, or checksums to locate differences. Hinted handoff stores intended replicas' writes elsewhere temporarily, but hints need durability, limits, ownership, expiry, ordering, and a plan when the destination never returns.

Repair traffic competes with foreground traffic. After a long partition, sending every missing object at full speed can cause a second outage. Reserve capacity, prioritize safety-critical metadata, pace bulk transfer, and measure estimated time to convergence.

Conflicting application values require semantics described in [Conflict Resolution](../02-distributed-databases/04-conflict-resolution.md); byte-level replica repair must not silently choose last wall-clock timestamp when clocks and causality do not justify it.

---

## Cascading Failure and Overload

A component failure shifts traffic and work to the survivors. If the steady state uses 80% of fleet capacity, losing one of four equal cells raises the others' ideal load from 80% to roughly:

$$
80\% \times \frac{4}{3} \approx 107\%
$$

Failover without reserve capacity is load redistribution, not resilience.

### Retry amplification

Suppose a request fans out to five dependencies and each layer performs up to three attempts. If retries are nested without a shared budget, one top-level attempt can generate as many as:

$$
3^5 = 243
$$

downstream attempts along one failing chain. Real fan-out can make it worse. Retry at one owning layer, use a total deadline and attempt budget, add randomized backoff, and stop retrying overload or permanent failures. Detailed mechanics are in [Retries, Deadlines, and Hedging](../06-scaling/10-retries-timeouts-hedging.md).

### Admission before collapse

Queues convert a short overload into latency. Unbounded queues convert it into memory exhaustion and into work that is already stale by the time it is served. During a failure we would sooner refuse work early than accept it and fail it late. That means rejecting or shedding low-priority work at the edge, bounding concurrency at the scarce resource, and propagating backpressure to callers instead of accepting an infinite backlog. It also means reserving capacity for recovery, health, and control operations, so that the system retains the means to repair itself, and isolating tenants and critical paths from one another. An explicit degraded response is better than a timeout that arrives after the expensive work has been done.

See [Backpressure and Overload Control](../06-scaling/07-backpressure.md) and [Circuit Breakers as Adaptive Admission Control](../06-scaling/06-circuit-breakers.md).

### Backlog recovery math

If arrivals continue at $\lambda=8{,}000$ jobs/s, recovered capacity is $\mu=10{,}000$ jobs/s, and the outage accumulated 18 million jobs, ideal drain time is:

$$
T = \frac{18{,}000{,}000}{\mu - \lambda}
  = \frac{18{,}000{,}000}{2{,}000}
  = 9{,}000\ seconds \approx 2.5\ hours
$$

This ignores cache misses, retries, expired work, and database recovery load. If $\mu \le \lambda$, the backlog never drains. Recovery objectives need surplus capacity and expiry/compaction policies.

---

## Dependency and Control-Plane Failures

An end-to-end path is limited by its mandatory dependencies. If five independent serial dependencies are each 99.9% available, the simplified path availability is:

$$
0.999^5 \approx 99.50\%
$$

The independence assumption is often false, but the calculation exposes why adding a mandatory remote service to every request is an availability decision.

Not every dependency deserves to be mandatory, and we classify them by what happens to the operation when they are missing. A *hard data-path dependency* is one without which the operation cannot be correct. A *soft enrichment* can be omitted or served stale under explicit semantics. A *control-plane dependency* supplies configuration, membership, keys, policy, placement, or deployment state, and is needed to change the system more than to run it. An *observability dependency* must never block the operation, although its loss reduces our ability to operate safely.

A healthy data plane may continue on last-known-good immutable configuration during a control-plane outage. Bound how long, preserve expiry and revocation rules, and ensure a corrupt control-plane update cannot instantly reach all cells. Validate and sign artifacts, stage activation, and retain rollback state locally.

Fallbacks are real production paths. A stale cache, default policy, secondary provider, or local configuration needs the same correctness, load, security, and testing discipline as the primary. “Fail open” is not a neutral availability choice when it widens authority or accepts money-moving operations.

---

## Concrete Failure Traces

### The ambiguous payment timeout

1. The service sends provider request `op-73`.
2. The provider commits the charge.
3. The response is lost.
4. The service marks the attempt failed and retries with a new ID.
5. The customer is charged twice.

The fix is a stable operation identity accepted by the provider plus a status/reconciliation path. A longer timeout only changes how often ambiguity appears.

### Green health, unusable storage

1. A storage process answers `/health` from memory.
2. Its device has 30-second write stalls.
3. Load balancing continues sending full traffic.
4. Client timeouts trigger retries to every replica.
5. Connection pools fill and healthy replicas overload.

Use layered probes: process liveness, readiness based on bounded real work, and external SLO probes. Remove or down-weight a node based on observed service quality, with hysteresis to avoid flapping. Admission control must engage before retry load consumes survivors.

### Paused leader writes after failover

1. Leader A holds a lease and pauses for 45 seconds.
2. A majority elects B in a new term.
3. B updates an external blob store.
4. A resumes and completes its old upload.
5. The blob store accepts the stale overwrite.

Consensus protected the internal log but not the external side effect. Include the term/fencing token in the downstream conditional write or route the effect through a fenced single owner.

### Corruption copied by repair

1. A memory or software fault writes a validly checksummed but semantically invalid index segment.
2. The replica reports the newest version.
3. Anti-entropy treats version recency as truth.
4. The segment is copied to every replica.
5. The last good copy is deleted by compaction.

Use end-to-end/domain invariants, delayed garbage collection, immutable source logs, sampled recomputation, and quarantine. Version order proves recency, not correctness.

### Recovery overload

1. A zone loses connectivity for 40 minutes.
2. Clients shift to the surviving zones, consuming their reserve.
3. The zone returns and every replica starts catch-up.
4. Repair saturates network and storage.
5. Foreground latency triggers retries and the whole region fails.

Use coordinated reintroduction, repair budgets, priority classes, staggered restart, and convergence-time telemetry. “The zone is back” is the beginning of recovery, not its completion.

---

## Verification: Prove Invariants Under Faults

### State-machine and history testing

A fault campaign without a stated invariant can only show that the system stayed up. We write the invariant first:

```text
no two successful withdrawals spend the same balance
every acknowledged queue item is delivered or remains recoverable
no stale fencing token mutates the resource
read-your-writes holds for a session token
```

Then record invocation/completion histories with operation IDs and revisions. A checker can test linearizability, serializability, or domain-specific properties. Green service metrics do not prove these histories are valid.

### Test layers

1. **Deterministic unit/state-machine tests:** enumerate crash points and replay.
2. **Property and model tests:** generate operation/fault sequences and shrink failures.
3. **Component failpoints:** fail before/after log, commit, response, checkpoint, and cleanup boundaries.
4. **Distributed fault tests:** delay, drop, duplicate, reorder, partition, pause, restart, skew clocks, fill disks, and corrupt bytes.
5. **Load plus fault tests:** inject faults at peak traffic and cold-cache conditions.
6. **Recovery drills:** restore backups, rotate authority, rebuild indexes, reconcile external effects, and return capacity gradually.
7. **Controlled production experiments:** begin with contained cells and reversible faults after lower layers prove the mechanism.

Jepsen-style testing combines concurrent workloads, a fault “nemesis,” recorded histories, and a consistency checker. It proves only the tested workload, configuration, version, and fault set; a process-kill test does not establish behavior under asymmetric partition or disk corruption.

### Failure-injection safety

The experiment platform itself needs scope, approvals, automatic aborts, blast-radius limits, identity, audit, and a reliable cleanup path. Verify the fault actually occurred. A chaos test that failed to apply its network rule and reported green is worse than no test because it creates false confidence.

---

## Observability and Recovery Objectives

Failures are diagnosed from evidence gathered at boundaries, so that is where we instrument. For operations we record the outcome class (committed, rejected, ambiguous, fenced, or indeterminate) and the phase in which a timeout occurred: queue, connect, remote execution, or response. For detection and authority we record failure-detector suspicion with its false-positive and false-negative indicators, changes of leader or lease epoch, and rejections of stale tokens. For recovery we record replication, repair, replay, and reconciliation lag; queue age and estimated drain time, which matter more than queue length; the capacity that would remain after each failure domain is removed; corrupt-record quarantine and the verification backlog; and the version and last successful refresh of each dependency and control plane. Finally we record recovery milestones in terms of restored SLO, since a restarted process is not yet a recovered service.

Recovery time objective (RTO) and recovery point objective (RPO) are workload contracts. Define them per operation class and failure domain. A service can have a five-minute process restart but a six-hour data convergence time; the latter is the real recovery for affected reads.

Alert on user-impacting symptoms and invariant threats. “One heartbeat missed” is diagnostic evidence; “quorum at risk,” “ambiguous mutation rate,” “revocation fence rejected,” or “backlog will miss expiry” is actionable impact.

---

## Design Review Framework

For every important component and operation, we ask:

1. Which crash, delay, omission, partition, corruption, resource, software, security, and operator faults are in scope?
2. Which faults may be correlated through zone, region, account, deployment, identity, or control plane?
3. What exactly has committed at each acknowledgment boundary?
4. How does a caller resolve an ambiguous outcome?
5. What volatile state is lost on restart, and can restoring it reuse authority or identity?
6. Who grants new authority, and which downstream resource enforces fencing?
7. What work remains safe without quorum or a dependency?
8. How are divergence, corruption, and derived-state drift detected and repaired?
9. Is there enough surviving capacity for failover, retry suppression, and backlog drain?
10. What invariant checker and fault campaign demonstrate the claim?
11. How is a repaired component reintroduced without causing recovery overload?
12. Which metric proves recovery is complete from the user's perspective?

A resilience design is incomplete if its failure story ends at “retry, fail over, and add replicas.” It must define ambiguous effects, stale authority, correlated faults, overload, repair, and verification.

---

## References

- [Unreliable Failure Detectors for Reliable Distributed Systems](https://doi.org/10.1145/226643.226647): Chandra and Toueg's failure-detector abstractions
- [SWIM: Scalable Weakly-consistent Infection-style Process Group Membership Protocol](https://www.cs.cornell.edu/projects/Quicksilver/public_pdfs/SWIM.pdf): scalable probing, suspicion, and membership dissemination
- [The Phi Accrual Failure Detector](https://doi.org/10.1109/RELDIS.2004.1353014): adaptive suspicion rather than a binary fixed timeout
- [Gray Failure: The Achilles' Heel of Cloud-Scale Systems](https://www.microsoft.com/en-us/research/publication/gray-failure-achilles-heel-cloud-scale-systems/): observer-dependent partial failures
- [Practical Byzantine Fault Tolerance](https://www.usenix.org/conference/osdi-99/practical-byzantine-fault-tolerance): concrete $3f+1$ Byzantine replication design and assumptions
- [The Part-Time Parliament](https://lamport.azurewebsites.net/pubs/lamport-paxos.pdf): safety and progress under partial synchrony in Paxos
- [How Complex Systems Fail](https://how.complexsystems.fail/): Richard Cook's concise model of interacting failure and defense
- [Google SRE: Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/): overload, retries, deadlines, and load shedding
- [Jepsen analyses](https://jepsen.io/analyses): fault-injected history checking against consistency claims
- [RFC 8633: Network Time Protocol Best Current Practices](https://www.rfc-editor.org/rfc/rfc8633): operational clock synchronization and security considerations

---

## Related Chapters

- [Distributed Locks and Leases](09-distributed-locks.md): stale authority is a lock problem long before it becomes a corruption problem
- [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md): detection turns into traffic: a timeout decides how fast ambiguity becomes load
- [Leases, Heartbeats, and Recovery](../18-workflow-job-systems/08-leases-heartbeats-recovery.md): the recovery boundary drawn around a worker that may already be dead
- [Write-Ahead Logging](../03-storage-engines/04-write-ahead-logging.md): crash consistency, where the log decides which half of an interruption survives
- [Resilience Testing and Fault Injection](../11-observability/09-resilience-testing-and-fault-injection.md): the tier where each failure semantic gets executed instead of merely described
