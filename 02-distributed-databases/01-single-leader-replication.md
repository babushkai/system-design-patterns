# Single-Leader Replication

## TL;DR

Single-leader replication makes one node the serialization point for writes and turns its commit history into an ordered replication stream. The easy diagram (primary sends a log to replicas) omits the real protocol: which log position is durable before acknowledgment, how a follower proves it has the same history, which read positions are safe, how a new leader receives a higher epoch, and how the old leader is fenced. Asynchronous acknowledgment minimizes write latency but admits an RPO window. Synchronous acknowledgment protects only the stage actually acknowledged: received, flushed, or applied are different guarantees. Replica reads need a session or commit-position fence if freshness matters. Safe failover is not “pick the most responsive replica”; it is a state transition that preserves the committed prefix, issues a new term/timeline, prevents stale writers, resolves ambiguous client outcomes, and repairs followers without reintroducing divergent history.

---

## Scope: One Write Authority and Its Replicated Log

One node at a time accepts writes for a replicated dataset.

- [Multi-Leader Replication](02-multi-leader-replication.md) owns concurrent write authorities and cross-leader conflicts.
- [Leaderless Replication](03-leaderless-replication.md) owns quorum reads/writes without a distinguished writer.
- [Consensus Algorithms](08-consensus-algorithms.md) owns replicated state-machine agreement and majority safety proofs.
- [Leader Election](09-leader-election.md) owns the general mechanism for choosing and fencing an epoch owner.
- [Write-Ahead Logging](../03-storage-engines/04-write-ahead-logging.md) owns local crash recovery and log-record mechanics.

A database can implement primary/standby replication without making every transaction a consensus operation. It may use an external consensus service only for membership/failover, or rely on operator fencing. Therefore “there is a leader” does not imply “the data is consensus-replicated.” Document the actual promotion authority and acknowledgment contract.

---

## The Contract in Four Coordinates

Every write progresses through distinguishable stages:

```text
client command
    |
    v
primary append -> primary durable -> replica received -> replica durable -> replica applied
       LSN 841        LSN 841          LSN 841          LSN 841          LSN 839
```

Use the engine's native coordinate (LSN, binlog file/offset, GTID set, term/index, or timeline/position), but retain the distinction:

1. **generated:** the primary assigned the next ordered position;
2. **durable locally:** the primary's crash recovery can reproduce it;
3. **received remotely:** bytes reached a follower process or kernel;
4. **durable remotely:** a follower persisted the required log prefix;
5. **applied:** follower query state reflects the entry.

An SLA such as “synchronous replication means no data loss” is incomplete until it states *which followers, which stage, and what failure set*. Waiting for one remote process to receive bytes is not the same as waiting for a fault-independent replica to flush them. Waiting for flush protects failover durability but does not make a follower read see the write; that requires apply.

The topology also needs an **epoch** (term/timeline/generation). A log position alone is ambiguous after failover because two histories might both contain position 900. The safe coordinate is conceptually `(epoch, position)` plus history ancestry.

---

## Data Plane: From Transaction to Applied State

### Primary write path

A typical physical-log path is:

1. execute under local concurrency control;
2. append change records and a commit record to the primary log;
3. make the local log durable according to policy;
4. stream records to configured followers;
5. wait for the configured remote acknowledgment set, if any;
6. return the commit result to the client.

The key prefix invariant is:

> If the system acknowledges a transaction under durability policy D, every future writable epoch must contain that transaction or the system has violated D.

That invariant couples the request path to candidate selection during failover. A controller cannot promise remote-durable acknowledgment and later promote a replica that lacks the acknowledged prefix.

### Replication stream

Physical replication copies storage-engine log records or page changes. It preserves engine internals and is efficient for exact standbys, but often couples versions, page formats, and full-cluster topology.

Logical replication copies row/statement/domain changes. It supports selective replication, transformations, and some online upgrades, but needs stable identities, deterministic semantics, schema compatibility, and explicit handling for sequences, large objects, and DDL.

Statement replication is safe only when evaluation is deterministic and all implicit context is reproduced: time, randomness, collation, triggers, user-defined functions, auto-generated identifiers, and row-selection order. Row or logical-change replication trades larger records for less replay ambiguity.

### Follower receive, flush, and apply

Followers usually separate network receive from storage flush and replay. That pipeline lets a follower receive quickly while a single-threaded apply stage falls behind. Monitor all coordinates:

```text
primary generated:       2/AF00
follower received:       2/AF00  -> transport current
follower flushed:        2/AE80  -> small durability gap
follower applied:        2/A100  -> query-visible lag is large
```

One scalar “replication lag” hides which subsystem is failing. Time-based lag is also ambiguous during idle periods and clock skew; byte/position lag plus apply rate is usually more actionable.

---

## Acknowledgment Policies and Failure Semantics

### Asynchronous

The primary returns after local durability and does not wait for a remote replica. Normal-case latency stays local. If the primary and its local durable media become unavailable before a follower persists the suffix, acknowledged transactions can be lost on promotion.

For log generation rate $g$ bytes/s and replication delay $d$ seconds, the unreplicated data exposure is roughly:

$$
B_{exposed} \approx g d
$$

The business RPO is not bytes; it is acknowledged commands in that interval. Track both position lag and logical command count/value where possible.

### One synchronous follower

The primary waits for one eligible follower. This can survive primary loss if the failover controller is constrained to promote a node containing that durable acknowledgment. It may stop writes when no eligible synchronous follower exists, or silently/explicitly fall back to asynchronous mode, reintroducing an RPO window. That downgrade must be a visible state transition with an owner, alert, and recovery condition.

### Quorum or named synchronous sets

Policies may wait for any $k$ of $n$ followers or selected failure domains. Placement matters more than count: two replicas on one power/network boundary do not protect against that boundary. Candidate eligibility must reflect the same policy after failure.

Synchronous commit latency is approximately the relevant acknowledgment order statistic, not the mean follower RTT. For “any 2 of 3,” it tracks the second-fastest eligible acknowledgment plus primary work. Tail latency changes when a slow/failing replica changes which order statistic controls the request.

### Receive, flush, or apply acknowledgment

| Remote stage | Protects against | Does not guarantee |
|---|---|---|
| Receive/write | transient primary loss if follower process survives as assumed | follower host/power loss; readable freshness |
| Durable flush | primary plus independent follower process/host loss within placement assumptions | query visibility on follower |
| Apply | durable copy and read visibility through that position | application side effects outside database |

Name the stage in client-visible durability classes. A critical ledger and an ephemeral session update can use different classes if the API makes the trade explicit.

---

## Replica Reads Need a Position Contract

Adding followers increases read capacity only for reads allowed to observe their state. Four common contracts are:

### Eventual reads

Route to any healthy replica and accept arbitrary current lag within the operational policy. Suitable for feeds, search-like browsing, or caches where stale results are product-acceptable.

### Read-your-writes

Return the commit coordinate with a successful write. Before serving the next read, a replica must prove `applied_position >= required_position`; otherwise wait within a budget, route to the primary/current replica, or return a retryable freshness error.

```text
POST /profile -> 200, commit_token=(epoch 12, LSN 90AF)
GET  /profile with token
router selects replica whose applied coordinate covers (12, 90AF)
```

The epoch prevents comparing unrelated post-failover histories as plain integers.

### Monotonic reads

Track the greatest coordinate a session has observed. Never send it to a replica behind that coordinate. This prevents a user from seeing data appear and then disappear when load balancing changes replicas.

### Bounded-staleness reads

Serve a replica only if its applied position/time is within a documented bound. A time bound requires trustworthy commit/apply timestamps; byte lag alone cannot prove seconds of staleness under bursty writes. Define what happens when no follower meets the bound.

Replica reads can also conflict with replay. Long queries may require old row versions that apply wants to remove or may block DDL replay. Engines choose among canceling the read, delaying apply, retaining more versions on the primary, or routing the query elsewhere. Read scaling therefore creates retention and recovery coupling, not free replicas.

---

## Control Plane: A Safe Failover State Machine

Failure detection is suspicion, not proof. A safe failover needs ordered control-plane steps:

```text
1. suspect primary and stop assigning new client traffic
2. acquire a new promotion epoch from the authoritative coordinator
3. fence the old epoch at routers and, where possible, storage/network/power
4. identify candidates whose history contains the required committed prefix
5. choose the most advanced eligible history, not merely the lowest-latency host
6. promote it and create a new timeline/term
7. publish routing + epoch atomically enough that stale routes are rejected
8. reattach or rebuild followers against the chosen history
9. resolve in-flight client commands with durable command identifiers
```

The ordering between fencing and promotion is the safety boundary. If the old primary can still write while the new primary accepts writes, a network partition becomes split brain.

### Candidate selection

A candidate must satisfy:

- it has the acknowledged durability prefix required by policy;
- its log/history is internally valid and recoverable;
- it belongs to an eligible failure domain/version/configuration;
- it can obtain the new epoch and reject old-epoch traffic;
- its apply/recovery time meets the failover objective.

“Most advanced” is insufficient if a node has uncommitted or divergent local entries. The engine needs a history relation (timeline ancestry, GTID executed set, or term/index rules) to distinguish a valid extension from a fork.

### Planned switchover

A switchover can eliminate ambiguity:

1. drain or briefly quiesce writes;
2. record the final required coordinate;
3. wait for target flush/apply to reach it;
4. revoke/fence the old primary;
5. promote target with a new epoch;
6. move routing and verify session tokens;
7. demote the old node only after it proves it follows the new history.

Use this path for maintenance and topology changes. It exercises most failover machinery without a data-loss race.

---

## Split Brain, Divergence, and Rejoin

### Why timeout election is unsafe by itself

During a partition, the primary cannot distinguish “I lost the network” from “all replicas failed.” A replica cannot distinguish “primary failed” from “primary is isolated from me.” If both use timeouts to declare authority, both may write.

Use a majority-backed coordinator, storage-level reservation, cloud/virtual-machine fence, network fence, or manual procedure that establishes one promotion epoch. Clients and downstream systems include or validate that epoch where practical. A process-local role flag is not fencing.

### Divergent suffixes

After two writable histories exist:

```text
common prefix:  A B C
old primary:    A B C D E
new primary:    A B C F G
```

These are not ordinary replication lag. The suffixes encode potentially conflicting committed effects. Automatically rewinding `D E` is data loss; replaying them after `F G` may violate constraints or repeat effects. Freeze the losing history, preserve forensic copies, identify client-visible commits by command ID, and perform domain reconciliation. The prevention mechanism (fencing) is far cheaper than recovery.

### Rejoining an old primary

Never simply restart it as a follower. It must prove the new history is a descendant of its safe prefix or be rewound/reseeded from a trusted base backup plus log. The rejoin protocol verifies cluster identity, new epoch, checksum/history, and configuration before the node becomes read-eligible.

---

## Log Retention, Slots, and Rebuild Economics

A disconnected follower needs every log segment after its last durable coordinate. With retained log capacity $B$ bytes and generation rate $g$ bytes/s, the catch-up window is approximately:

$$
T_{retention} = \frac{B}{g}
$$

Size for peak sustained generation plus incident duration, not daily average. Schema changes, index builds, bulk loads, and vacuum/compaction can change log rate abruptly.

Retention mechanisms create opposing failure modes:

- **no reservation:** the primary recycles required log and the follower needs a full rebuild;
- **unbounded reservation/slot:** an abandoned follower pins log until the primary disk fills.

Put byte and age limits, ownership, and alerts on every retention reservation. Archive logs to independent durable storage when the recovery design requires catch-up beyond local retention.

Follower catch-up dynamics depend on production rate $\lambda$ and follower apply capacity $\mu$ (in a common unit such as log bytes/s). Backlog shrinks only when $\mu > \lambda$:

$$
T_{catchup} \approx \frac{B_{lag}}{\mu-\lambda}
$$

If apply capacity merely equals incoming rate, the follower remains perpetually behind. A rebuild also competes for primary/network/storage resources; throttle it against live SLOs while preserving an RTO estimate.

---

## Topology and Capacity

### Primary fanout

Directly streaming to $n$ followers multiplies primary network connections, encryption work, and log reads. Cascading followers reduce primary fanout and inter-region bandwidth, but downstream lag includes every upstream queue and cascading promotion rules become more complex.

For generated log rate $g$, direct egress is roughly $n g$ before protocol/compression overhead. A cascade can reduce cross-region copies but increases recovery dependencies. Place enough direct/fault-independent followers to meet promotion policy; use cascades for read/backup consumers that need not be immediate candidates.

### Write ceiling

All writes pass through one primary's concurrency control, log append, storage flush, and network acknowledgment path. More read replicas do not increase that write ceiling. Partition the data into independent single-leader groups when one authority is insufficient; [Database Sharding](../06-scaling/03-database-sharding.md) owns routing and movement between those groups.

### Read ceiling

Follower capacity is limited by apply work plus queries. Analytical reads compete with replay for CPU, cache, I/O, and version retention. Model a replica as two workloads, reserve apply headroom, and remove it from freshness-sensitive routing before lag runs away.

---

## Backups and Point-in-Time Recovery Are Separate Copies

Replication rapidly copies operator mistakes, corrupt writes, and malicious deletes. It is availability, not a historical backup.

A recoverable design needs:

- a consistent base snapshot with known start/end coordinate;
- an ordered, durable log archive from that coordinate onward;
- integrity checks and encryption/key-recovery procedures;
- a restore process that can stop before a bad transaction;
- periodic restore tests measuring actual RPO/RTO.

After failover, archive continuity must identify the new timeline/epoch. A point-in-time restore cannot blindly concatenate divergent logs.

---

## Security and Tenant Boundaries

Replication links carry the full change stream and often bypass ordinary query authorization. Use mutually authenticated channels, least-privilege replication identities, key rotation, network policy, and audited configuration changes. Treat base backups, archived logs, and temporary rebuild copies as production data with the same encryption and deletion controls.

A shared follower can expose all tenants even when an application query layer normally filters them. Row-level security behavior on replicas, privileged analytics access, logical publication filters, and backup restore access need separate review. Redact or tokenize sensitive logical-change payloads only if doing so preserves recovery and downstream contracts.

Promotion authority is a high-impact privilege. Separate it from routine database credentials, require an auditable epoch-grant path, and rehearse recovery when the coordinator or fencing mechanism is unavailable.

---

## Failure Modes

### Commit acknowledged, response lost

The primary commits, then the connection drops. The client cannot infer failure. Retrying with a new business identity duplicates the action. Persist a stable command ID and result in the same transaction; after reconnect/failover, resolve that ID before retrying. See [Idempotency](../01-foundations/08-idempotency.md).

### Synchronous mode silently degrades

The only synchronous follower disconnects. The primary continues asynchronously by policy, but monitoring still reports the cluster “healthy.” A later primary loss violates the assumed zero-RPO service. Expose the active durability mode on every response/metric where necessary and page on downgrade duration.

### Receiver current, applier stalled

Network lag is zero while a blocking query or incompatible schema change stops replay. Freshness routing sends reads to stale state because it watches receive coordinates. Gate on applied coordinate for read semantics and diagnose receive/flush/apply separately.

### Replication slot fills primary disk

A decommissioned consumer retains logs indefinitely. Free space falls until writes stop. Every slot/reservation needs a named owner, maximum retained bytes/age, and an explicit choice between dropping the consumer and protecting the primary.

### Failover chooses a low-lag but invalid branch

A controller compares only numeric offsets and promotes a node from a different timeline. Acknowledged history disappears. Candidate selection must validate history ancestry and durability policy, not scalar position alone.

### Old primary accepts background writes

Client routing moved, but a scheduled job connects directly to the old primary and writes after promotion. Routing changes are not fencing. Enforce the epoch at all ingress paths and disable/revoke the old writer at the resource or infrastructure boundary.

---

## Observability and Incident Evidence

Monitor by topology role and epoch:

- primary generated/durable coordinate and transaction commit rate;
- each follower receive, flush, and apply coordinate plus byte/time backlog;
- apply throughput, estimated catch-up time, replay conflicts, and paused state;
- active acknowledgment policy, eligible synchronous set, and downgrade time;
- commit latency split into local flush and remote acknowledgment;
- retained log bytes/age by slot or consumer and archive continuity;
- current writable epoch/timeline, promotion events, fence acknowledgments, and route propagation;
- replica-read waits/fallbacks for session tokens and bounded-staleness violations;
- ambiguous commands resolved after reconnect/failover.

Preserve promotion decisions, candidate coordinates/history, coordinator epoch, fence results, and client-routing versions. Without that evidence, a split-brain review becomes guesswork after the losing nodes are rebuilt.

---

## Migration and Verification

### Introducing a follower

1. take a verified base snapshot at a recorded coordinate;
2. restore it with encryption and cluster identity intact;
3. stream/archive every subsequent log record;
4. wait until receive, flush, and apply reach the target;
5. validate checksums or logical samples and schema/version compatibility;
6. add read traffic gradually; do not make it a promotion candidate until it passes failover criteria.

### Changing durability policy

Canary synchronous acknowledgment by workload class. Measure commit tail latency, eligible-set churn, downgrade behavior, and candidate guarantees. A rollback to asynchronous mode changes RPO; surface it as a semantic release, not a tuning toggle.

### Essential fault tests

- kill the primary before local flush, after local flush, after remote receive, after remote flush, and after commit-before-response;
- partition primary from replicas while clients can still reach both sides;
- pause receive and apply independently;
- exhaust log retention and verify bounded behavior;
- promote, then restart the old primary with direct clients/background jobs;
- lose the promotion coordinator or fencing provider;
- run long follower reads during replay/DDL;
- restore a base backup plus archived logs across a timeline change.

Assert both state invariants and acknowledged-command outcomes. A fast failover that loses a confirmed payment is not a successful test.

---

## Decision Framework

Choose single-leader replication when:

- one ordered write authority per shard fits throughput and locality;
- simple local transaction semantics matter more than multi-region write availability;
- clients can route writes to the current authority;
- RPO/latency can be expressed with an explicit acknowledgment policy; and
- the organization can operate fencing, promotion, backup, and rejoin protocols.

Reconsider when all regions must accept writes during inter-region partitions, one primary is the sustained write bottleneck, or client locality dominates consistency simplicity. Multi-leader and leaderless designs move rather than remove complexity: conflicts, quorum semantics, repair, and application merge become the new core.

For each deployment, write down:

1. the acknowledged durable stage and failure domains;
2. eligible promotion candidates and authority for the next epoch;
3. the old-primary fencing mechanism;
4. replica-read freshness contract and fallback;
5. maximum log-retention/catch-up window;
6. ambiguous-commit resolution key;
7. tested RPO and RTO from restore, not aspiration.

---

## Key Takeaways

1. Single-leader replication is an ordered-history and epoch protocol, not just log copying.
2. Receive, durable flush, and apply acknowledgments protect different outcomes.
3. The acknowledgment policy constrains which follower may safely become leader.
4. Replica reads require commit/session coordinates when freshness matters.
5. Failover must fence the old epoch before exposing the new one and must validate history ancestry.
6. Retention slots trade follower rebuild risk for primary disk risk; both need bounds.
7. Replication provides availability, while base backups plus archived logs provide historical recovery.

---

## References

- PostgreSQL, [High Availability, Load Balancing, and Replication](https://www.postgresql.org/docs/current/high-availability.html), current documentation.
- PostgreSQL, [Log-Shipping Standby Servers](https://www.postgresql.org/docs/current/warm-standby.html), current documentation.
- MySQL, [Replication](https://dev.mysql.com/doc/refman/8.4/en/replication.html), 8.4 Reference Manual.
- MySQL, [Replication with Global Transaction Identifiers](https://dev.mysql.com/doc/refman/8.4/en/replication-gtids.html), 8.4 Reference Manual.
- Diego Ongaro and John Ousterhout, [*In Search of an Understandable Consensus Algorithm*](https://raft.github.io/raft.pdf), USENIX ATC, 2014.
- Barbara Liskov and James Cowling, [*Viewstamped Replication Revisited*](https://pmg.csail.mit.edu/papers/vr-revisited.pdf), 2012.
- M. R. Eltabakh et al., [*Zephyr: Live Migration in Shared Nothing Databases for Elastic Cloud Platforms*](https://www.cs.purdue.edu/homes/csjgwang/pubs/SIGMOD2011_Zephyr.pdf), SIGMOD, 2011.
