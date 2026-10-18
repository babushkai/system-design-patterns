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

## Product Knob Map

The settings that decide this chapter's contract sit on three machines that three different reviews approve: the primary's commit path, each follower's replay path, and the coordinator that grants the write epoch. Audits read the first and inherit the rest. Read every value below off the running cluster before relying on it, because the shipped default of every product named here is asynchronous replication with unbounded log retention.

| Product and scope | Knob | Invariant it enforces | Default and how it misleads |
| --- | --- | --- | --- |
| PostgreSQL primary | `synchronous_commit`, `synchronous_standby_names` | a commit acknowledged to a client survives loss of the primary | streaming replication is asynchronous by default: `synchronous_commit` reads `on` while an empty standby list makes nothing wait for a follower, and `remote_apply`, `remote_write`, and `local` all collapse to the local-flush level of `on`. Loss at failover is proportional to replication delay. Set the name list first, then read who is synchronous. |
| PostgreSQL primary | `FIRST (num_sync)` versus `ANY (num_sync)` in the standby list | commits wait for a declared count of standbys, with named substitutes that can take over | FIRST waits until WAL reaches that many highest-priority members, so naming three zones and asking for two makes commit latency the second-fastest follower. Only directly connected followers count, so a cascaded standby added for bandwidth adds zero durability while looking like a synchronous candidate in configuration. |
| PostgreSQL primary | `max_wal_senders`, `max_replication_slots` | every consumer has a sender and a reservation | both default to 10 and both are start-time only. Lowering slots below the number of existing slots prevents the server from starting, so an inventory cleanup applied in a maintenance window can leave the primary unbootable. An abrupt client disconnect can hold a sender slot until timeout, which is why the documented advice is to sit slightly above the expected client count. |
| PostgreSQL primary | `max_slot_wal_keep_size`, `wal_keep_size`, `idle_replication_slot_timeout` | log retention has a byte bound and abandoned consumers expire | shipped values are -1 (a slot may retain unlimited WAL), no floor, and 0 (expiry disabled), which is the disk-full failure written into the default file. `wal_keep_size` is only a minimum and does not rein in a slot. The pair that actually bounds risk is a per-slot byte cap plus an idle timeout with a named owner who is paged on invalidation. |
| PostgreSQL follower | `hot_standby_feedback` | long-running standby queries are not canceled by primary cleanup | default off, so analytics on the follower die when the primary recovers dead tuples; turning it on moves that cost to the primary as bloat and passes feedback upstream through every cascade. Feedback is sent no more often than `wal_receiver_status_interval`, so raising that interval delays both symptoms in each direction. |
| PostgreSQL follower | `max_worker_processes` | a follower can run the queries it is routed | default 8, start-time only, and a standby must be at least the primary's value or queries are not allowed on it at all. A fleet template that sizes this per node role therefore produces followers that accept connections and refuse reads, and the same pool is what logical decoding and parallel work later borrow from. |
| PostgreSQL standby | `recovery_min_apply_delay` | a known-bad change stays reversible for a declared window | default zero. A delayed follower reports lag equal to the delay by design, so freshness alerts need an explicit exception or they page continuously, and the protection only covers what a human notices inside the window. It complements archived-log point-in-time recovery; it does not replace it. |
| MySQL 8.4 source | `rpl_semi_sync_source_enabled`, `_wait_point`, `_timeout` | an acknowledged transaction is durable somewhere else before the ack | the plugin ships disabled, so a "semi-sync" deployment is asynchronous until both ends load and enable it. After `_timeout` (default 10000 ms) with too few replicas replying, the source reverts to async and keeps serving commits; `Rpl_semi_sync_source_no_tx` counts what lost the guarantee, so downgrade duration belongs on the pager next to commit latency. |
| MySQL 8.4 | `innodb_flush_log_at_trx_commit`, `sync_binlog` | an acknowledged transaction survives an operating-system crash | both default to 1, one flush and one sync per commit. Any relaxed combination is a documented loss window on crash that no cross-site setting can cover: relaxing either locally makes a zero-RPO claim describe only the network hop. |
| Failover controller (Patroni-style) | `ttl`, `loop_wait`, `retry_timeout`, `maximum_lag_on_failover` | only a candidate within a declared log distance takes the epoch, and one holder has it at a time | reference values are ttl 30 s with loop and retry at 10 s. Raise `loop_wait` or `retry_timeout` without raising `ttl` and the lease expires while the controller believes it holds it. The shipped lag cap is 1048576 bytes, so one bulk load disqualifies every candidate and the next operator reads "no failover" as controller failure. |

```bash
# PostgreSQL primary: who is synchronous, and where each follower actually is
psql -X -Atc "SELECT application_name, state, sync_state, sent_lsn, write_lsn,
                     flush_lsn, replay_lsn,
                     pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS replay_backlog
              FROM pg_stat_replication ORDER BY application_name;"
# which consumer pins WAL, and how much of it
psql -X -Atc "SELECT slot_name, slot_type, active, restart_lsn,
                     COALESCE(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn), 0) AS retained
              FROM pg_replication_slots ORDER BY retained DESC;"
psql -X -Atc "SHOW synchronous_commit; SHOW synchronous_standby_names;
              SHOW max_slot_wal_keep_size; SHOW idle_replication_slot_timeout;
              SHOW wal_keep_size; SHOW max_wal_senders; SHOW max_replication_slots;
              SHOW max_worker_processes;"
# follower's own view: receive and replay positions are different numbers
psql "$FOLLOWER_DSN" -X -Atc "SELECT pg_is_in_recovery(),
                                     pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn();"

# MySQL 8.4: semi-sync participation, the downgrade counter, and the local durability pair
mysql -NBe "SHOW GLOBAL STATUS LIKE 'Rpl_semi_sync%';"
mysql -NBe "SELECT @@global.rpl_semi_sync_source_enabled AS enabled,
                   @@global.rpl_semi_sync_source_wait_point AS wait_point,
                   @@global.rpl_semi_sync_source_timeout   AS timeout_ms,
                   @@global.sync_binlog                    AS sync_binlog,
                   @@global.innodb_flush_log_at_trx_commit AS trx_commit;"

# controller policy: lease durations and the lag cap that disqualifies candidates
curl -s localhost:8008/patroni | jq '{role, timeline, optime, tags}'
grep -E "ttl|loop_wait|retry_timeout|maximum_lag_on_failover" /etc/patroni.yml
```

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

## Incident Playbook

Almost every single-leader incident is a disagreement between two numbers that a dashboard treats as the same number: sent versus applied, lag versus delay, eligible candidate versus healthy node. Establish which coordinate each claim rests on before changing configuration, because the durable actions in this subsystem are taken by whoever is trying to make replication look healthy fastest.

### Commit latency doubled right after a replica or policy change

Read `sync_state` and the standby list from the same query before touching either: a `FIRST (2 ...)` clause naming members of which only one is streaming makes every commit wait on an absent participant, so the p99 jump is waiting rather than slower storage. Confirm the missing member was ever eligible at all, because a cascaded standby never counts toward the synchronous set no matter what the topology diagram claims. Bound the change to the session or role that needs the durability level instead of moving the whole cluster's setting, and record the synchronous set size next to the latency sample so the next reviewer can see cause. Do not raise `num_sync` to "use the standbys we have", and do not set `synchronous_commit` off cluster-wide to make the dashboard quiet: both change the durability promise that applications were written against. Verify by watching commit latency return with the set size unchanged, and by confirming a test write still reports the acknowledgment level the service contract claims.

### A follower's receive coordinate is current while its reads are stale

The signature is `write_lsn` and `flush_lsn` advancing with `replay_lsn` frozen: the network is innocent and replay is blocked, usually by a long reader whose snapshot forces the standby to delay cleanup or by replay work starved of CPU and I/O. Freshness routing keyed on receive position is now sending reads to state that is old by an unmeasured amount, so move those reads first and diagnose second. Separate the two causes before choosing an action: a conflicting reader is a workload decision, a starved applier is a capacity decision, and adding another follower fixes neither. Do not enable `hot_standby_feedback` as a reflex, because it trades standby cancellations for primary bloat across every cascade, and do not restart the follower to unstick replay: it re-enters recovery from its own position and inherits the same blocker. Verify by watching `replay_lsn` advance inside one status interval and confirming stale-read fallbacks return to zero before restoring traffic.

### Primary free space is falling and one slot owns the difference

Rank slots by retained bytes against the free-space slope, then name the owner; a decommissioned consumer or an abandoned test subscription is the usual holder. Before releasing anything, prove that a base backup plus a continuous archive covers that slot's restart position, because that proof is what turns a dropped reservation from data loss into a rebuild. If continuity holds, invalidate the slot and let the owner catch up from archive; if it does not, buy time with an explicit byte cap, an expiry, and a throttled write path, and say out loud which of the two risks the team accepted. Do not delete the slot to reclaim space before archive continuity is confirmed, do not raise unlimited retention "until the release settles", and do not expect a restart to help: reservations are durable objects, not connections. Verify by re-measuring retained bytes and the free-space slope, and by actually starting the rebuild from archive rather than assuming it is possible.

### Failover did not happen even though a healthy replica exists

Get the controller's stated reason before promoting anything: lag cap, history or timeline mismatch, lost quorum in the lease store, or fencing that could not be confirmed are four different incidents with different fixes, and only the first two look like a broken controller. A one-megabyte lag cap refusing to promote a follower that is a terabyte behind is the configuration protecting acknowledged writes. Once eligibility is settled, the work is confirming the old epoch stopped accepting writes on every ingress path, including direct-address scheduled jobs that never consult routing. Do not force a promotion to end the page, do not widen the lag cap and promote in the same breath, and do not restart the old primary before fencing: each of those converts a recoverable outage into permanent divergence. Verify by recording which candidate was eligible on ancestry and lag, and by observing zero accepted writes attributed to the previous epoch after routing converges.

### After a failover an application read a value it had already overwritten

This is a violated session contract: routing moved to a node whose applied position does not dominate the coordinate the client last saw, so the read is legal for that node and wrong for that user. Decide per operation class rather than globally: reads that must follow a write wait on the applied coordinate or go to the writable epoch, and reads that tolerate staleness get an explicit bound plus a visible fallback. The evidence to keep is the token, the route's position at read time, and whether the wait expired, because without those three the report is indistinguishable from a lost write. Do not insert fixed client-side sleeps, and do not relax freshness for every reader because one flow cannot absorb the wait; that turns an incident into an unannounced product change. Verify by replaying the recorded token against the current route in a test run and confirming the fallback returns a declared "state unavailable" instead of a silently stale value.

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
- PostgreSQL, [Write Ahead Log](https://www.postgresql.org/docs/current/runtime-config-wal.html), current documentation.
- PostgreSQL, [Replication](https://www.postgresql.org/docs/current/runtime-config-replication.html), current documentation.
- Diego Ongaro and John Ousterhout, [*In Search of an Understandable Consensus Algorithm*](https://raft.github.io/raft.pdf), USENIX ATC, 2014.
- Barbara Liskov and James Cowling, [*Viewstamped Replication Revisited*](https://pmg.csail.mit.edu/papers/vr-revisited.pdf), 2012.
- M. R. Eltabakh et al., [*Zephyr: Live Migration in Shared Nothing Databases for Elastic Cloud Platforms*](https://www.cs.purdue.edu/homes/csjgwang/pubs/SIGMOD2011_Zephyr.pdf), SIGMOD, 2011.

---

## Related Chapters

- [Multi-Leader Replication](02-multi-leader-replication.md): what breaks when a second write authority is a feature rather than an accident
- [Consensus Algorithms](08-consensus-algorithms.md): the protocol that decides which single leader is the real one
- [Write-Ahead Logging](../03-storage-engines/04-write-ahead-logging.md): the log being replicated, and what it means for a replica to be caught up
- [Database Sharding](../06-scaling/03-database-sharding.md): the other axis of scale: many leaders over disjoint keys instead of one over all
