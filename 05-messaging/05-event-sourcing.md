# Event Sourcing and Domain Logs

Event sourcing stores accepted domain transitions as the authoritative record of an aggregate. Current state is obtained by folding an ordered stream of immutable facts, optionally starting from a verified snapshot. This is a domain persistence model, not a synonym for publishing integration events to a broker.

Scope: authoritative aggregate streams, optimistic concurrency, event-store transactions, snapshots, evolution, retention, and temporal reconstruction. [CQRS Projections](06-cqrs.md) owns query read models. [Outbox and Inbox](07-outbox-pattern.md) owns atomic publication to other systems. [Ordering](03-message-ordering.md) owns downstream gap/resequence protocols.

## Workload and contract

An event-sourced store supports:

```text
load_stream(aggregate_type, aggregate_id, from_version) -> events
append_stream(aggregate_id, expected_version, events, command_id) -> new_version
load_snapshot(aggregate_id, max_version) -> snapshot?
read_by_commit_position(from_position, filter) -> committed event batches
```

The command path is:

```text
command
  -> load aggregate at version v
  -> validate invariant and decide events
  -> append only if current version is still v
  -> return committed aggregate version and command result
```

Define:

- aggregate boundary and invariants protected by one append;
- expected-version conflict behavior;
- command idempotency scope and result retention;
- event and metadata schema ownership;
- stream and global commit ordering semantics;
- snapshot validity and rebuild path;
- temporal-query meaning under corrected/late information;
- retention, privacy erasure, legal hold, and integrity controls;
- integration publication and projection freshness contracts.

Events describe facts the domain accepted: `CreditLimitChanged`, not persistence implementation details such as `RowUpdated`. They carry stable identity, aggregate ID/version, event type/schema, command/causation/correlation IDs, recorded time, optional effective time, actor/authority provenance, and payload.

## State and invariants

The event store keeps stream metadata, immutable event records, transaction/commit records, command-result identities, snapshot records, schema/upcaster registry, and publication/projection checkpoints.

Enforce:

**Contiguous aggregate version.** A committed aggregate stream has versions `1..v` with at most one event at each version. A multi-event command reserves one contiguous range atomically.

**Expected-version compare-and-append.** Two commands based on version 12 cannot both append version 13. One wins; the other reloads and re-decides or reports conflict.

**Command identity is stable.** Retrying the same command ID with the same request returns the original committed result; reuse with different semantics is rejected.

**Commit closure.** A commit record never exposes only part of a multi-event append or references missing event bytes.

**Original event identity is immutable.** Upcasting creates a read representation; it does not silently alter stored evidence or event IDs.

**Snapshot is subordinate.** A snapshot accelerates replay but is never more authoritative than its exact stream prefix. It can be discarded and rebuilt.

**Derived copies are rebuildable.** Broker messages, indexes, caches, and query models are projections with checkpoints and reconciliation.

## Event store versus message broker

An event store is optimized for atomic append to one domain stream, expected-version conflicts, aggregate reconstruction, and long-lived semantic history. A broker is optimized for distribution, subscription progress, fan-out, and retention throughput. One product can implement both roles only if it satisfies both contracts.

A partitioned broker log alone may be insufficient as an aggregate store when:

- expected-version append cannot be enforced atomically;
- retention or compaction can erase required history;
- events for an aggregate can move partitions without a stream pivot;
- consumer offsets are mistaken for aggregate versions;
- restoring a broker archive does not restore command identity/snapshot metadata;
- administrative rewrite changes the authoritative record.

Conversely, making every integration consumer scan a transactional event table can overload the domain database and couple retention to subscribers. Keep the domain store authoritative, write an outbox/integration record in the same append transaction, and relay it to a broker with stable identity.

## Data plane and control plane

The **data plane** authenticates commands, loads streams/snapshots, checks command identity and expected version, appends events/commit metadata, reads committed positions, and constructs snapshots. It returns only committed batches.

The **control plane** owns aggregate/event types, schemas/upcasters, shard placement, retention/legal holds, encryption keys, snapshot policies, repair tools, and publication/projection registrations. Changes are immutable versions with staged rollout; readers pin a compatible schema/upcaster bundle.

The command service—not the store—normally owns business decision logic. The store enforces structural concurrency and uniqueness, while the aggregate code enforces domain invariants. Server-side stored procedures can combine them, but then domain-code deployment and replay compatibility move into the data tier.

## Append transaction

A relational layout might have:

- `streams(aggregate_id, current_version, shard, status)`;
- `events(aggregate_id, aggregate_version, event_id, type, schema, payload, metadata, commit_position)`;
- `commits(commit_position, event_count, checksum, recorded_at)`;
- `command_results(aggregate_id, command_id, request_digest, committed_version, result)`;
- `outbox(event_id, destination_contract, payload, created_at)`.

The append transaction:

1. check `command_results` for the command identity; return prior result on exact retry;
2. compare the stream’s current version with `expected_version` using a row lock or conditional update;
3. validate event IDs/types/sizes and assign contiguous aggregate versions;
4. write event rows and a commit position/batch checksum;
5. advance stream version;
6. write integration outbox records if required;
7. store command request digest/result;
8. commit once.

A unique constraint on `(aggregate_id, aggregate_version)` is the last line of defense, not the only concurrency protocol. A conflict means the aggregate decision was based on stale state; blindly retrying the same events at new versions can violate invariants. Reload the stream and re-run the command decision.

Global commit position provides a change feed but not necessarily a business total order across aggregates. Its allocation must not expose aborted transactions as unexplained gaps unless readers understand the gap policy.

Large aggregates can exceed transaction limits. Revisit the aggregate boundary rather than weakening atomic invariants. A command that legitimately emits many records should have bounded event count/bytes and one semantic summary where possible.

## Aggregate reconstruction

Loading folds events in aggregate-version order through deterministic transition functions:

```text
state_0 = initial_state
state_n = apply(state_(n-1), event_n)
```

`apply` does not call networks, read wall-clock time, generate random IDs, or consult mutable configuration. Any value that influenced the original decision belongs in the event or a versioned reference whose historical content is immutable.

Replay code validates aggregate ID, contiguous versions, event/schema identity, and stream checksum before applying. Unknown event types fail closed or use an explicit compatibility policy; silently skipping can produce plausible but false state.

Aggregate code evolves. Keep historical transition compatibility, use read-time upcasters to a supported internal representation, or perform an audited stream migration to a new stream/type with lineage. Do not require old commands to remain executable—only events need to remain interpretable for the retained history.

## Snapshots

A snapshot contains aggregate ID/type, exact last included version, state schema/version, serialized state, event-stream checksum/digest through the version or commit identity, code/upcaster compatibility, created time, and encryption metadata.

Snapshot creation is asynchronous:

1. load a verified stream prefix through version `v`;
2. fold it with a pinned reducer version;
3. serialize and checksum the snapshot;
4. write it immutably under `(aggregate_id, v, snapshot_schema)`;
5. publish it only after complete durability;
6. on load, select the newest compatible snapshot no later than current stream version and replay the suffix.

An event append never depends on snapshot success. Invalid, corrupt, or incompatible snapshots are discarded and rebuilt. Snapshot frequency follows measured replay cost and hot-aggregate access, not an arbitrary event count. Creating snapshots too often increases write/storage amplification; too rarely increases latency and recovery time.

Snapshots do not justify deleting events unless the product explicitly changes from event sourcing to a compacted-state model. A snapshot usually depends on current code and may not support audit, alternative projections, or corrected reconstruction.

## Event evolution

Separate wire schema version from semantic event type. Compatible additions can remain one type when absence has a stable meaning. Semantic changes use a new event type or version with explicit mapping.

Upcasters are pure, deterministic functions from old stored representation to a supported in-memory representation. Record the chain and test every historical schema. Avoid network lookups and “current defaults.” If an old event lacks data now required, model `unknown`, derive from immutable historical context, or perform an explicit migration—do not invent today’s value.

Event type deprecation proceeds by stopping new writes, ensuring all retained readers understand old/new forms, rebuilding projections, and retaining decoding support through the oldest kept event. Schema registries validate syntax; domain owners review semantics.

Stream migration is a new derived artifact:

1. pin source stream/checkpoint and migration code digest;
2. transform into a new stream namespace with source event lineage;
3. dual-project the source suffix or pause at a controlled pivot;
4. compare reconstructed state and invariant reports;
5. atomically switch command/read ownership;
6. keep source immutable through rollback/legal policy.

## Temporal queries, corrections, and audit

Replaying through commit position or effective time can answer “what did the system know?” or “what was believed effective?” only if the model distinguishes recorded time from domain effective time. Late facts and corrections create bitemporal questions. An event recorded today stating an address was effective last week must not disappear from the record of what the system knew yesterday.

Corrections are new events referencing the corrected fact; they do not overwrite history. Projections decide whether to present latest corrected truth or historical belief.

An append-only table is not automatically an audit trail. Audit requires authenticated actor/authority, tamper evidence, access logging, retention/hold, time provenance, and independent protection against privileged modification. Cryptographic hash chaining or signed checkpoints can detect some tampering, but key custody and external anchoring determine assurance.

## Retention and privacy

“Events are immutable forever” conflicts with cost, privacy erasure, and changing legal duties. Classify fields before storing them. Prefer events containing stable identifiers and necessary facts, not copied profiles or secrets. Store especially sensitive payload fields in separately encrypted blobs so key destruction or governed redaction can make data inaccessible while retaining non-sensitive event structure.

Retention can differ by aggregate/event class, but deletion changes reconstruction. Before pruning, create an authoritative genesis/compaction event or snapshot whose semantics explicitly replace the removed prefix, verify every required projection and audit obligation, and record the truncation boundary. After pruning, do not claim full temporal reconstruction before that boundary.

Backups, replicas, exported broker events, projections, snapshots, and debug stores are copies subject to the same deletion policy. Maintain a lineage registry and completion evidence.

## Sharding, replication, and recovery

Route an aggregate to one write shard/leader at a time. The shard term plus expected aggregate version fences stale writers. Hashing aggregate ID balances ordinary streams; tenant/range placement supports residency but needs hot-tenant splitting at aggregate boundaries.

Cross-aggregate transactions are possible only within a shared transactional shard and should be rare; otherwise model a durable workflow with explicit invariants and compensation. Event sourcing does not make a distributed transaction disappear.

Recovery validates the latest complete store checkpoint/snapshot, replays committed database/WAL state, then restarts publication and projections from their durable checkpoints. Align command-result identities with event recovery. Restoring events without idempotent command results can cause a retried client command to append a second logical transition.

## Capacity and cost model

Illustrative domain:

- 12,000 commands/s;
- 1.8 events/command;
- 1.1 KiB encoded event plus 180 bytes index/metadata;
- three database replicas;
- 20% of events produce a 700-byte integration outbox record;
- ten-year retained history;
- median aggregate has 14 events; p99 has 8,000.

Event rate is 21,600/s. One logical event stream grows about `21,600 * 1.28 KiB`, or 2.23 TiB/day. Three replicas are about 6.7 TiB/day before WAL, compaction, indexes, backups, and snapshots. Ten-year raw retention is economically enormous; compress measured payloads, tier cold history, minimize event size, and define retention by domain rather than assuming forever.

Outbox ingress adds roughly `21,600 * 0.20 * 700`, about 2.9 MiB/s logical. It is modest beside events but can burden the primary if polling indexes and cleanup are poor.

Replay cost is skewed. At a measured reducer speed of 80,000 events/s/core, reconstructing a p99 8,000-event aggregate consumes 100 ms CPU before I/O. A compatible snapshot at version 7,900 reduces the suffix to 100 events. Base snapshot policy on observed load/replay service time and snapshot write cost.

A full projection replay of one year at 21,600/s is 681 billion events; even 5 million events/s sustained takes about 37.8 hours before output writes. Rebuild capacity must be designed from day one.

## Concrete failure trace: snapshot published ahead of stream

A snapshot worker reads aggregate version 50 from a replica whose transaction view later rolls back/fails over, while the authoritative stream remains at 49. It publishes snapshot `(version=50)` without binding to a committed stream checksum. Loaders accept it and then append command decisions at expected version 49 based on state that includes an uncommitted event.

Containment disables the snapshot generation and reconstructs affected aggregates from authoritative events. Repair removes invalid snapshots and reconciles command results/projections. Prevention requires snapshot source reads from a committed generation, binds snapshot to exact stream version/commit identity and checksum, and rejects any snapshot beyond current authoritative version or with mismatched digest.

## Operations and observability

Track by aggregate/event type, shard, schema, snapshot version, and projection/publication:

- command rate, expected-version conflicts, duplicate command results, append latency/bytes;
- stream length/skew and unknown/invalid event rejection;
- event/global commit positions and replica lag;
- snapshot hit, age, build latency, incompatibility, checksum failure, and replay suffix;
- upcaster path usage, oldest schema, and deprecated writers/readers;
- outbox/publication and projection checkpoints/lag;
- replay throughput/ETA and read/write impact;
- retention/tier bytes, legal holds, privacy deletion progress, and integrity checkpoint verification.

Runbooks cover corrupt event, unknown schema, hot aggregate, conflict storm, invalid snapshot, projection/publication lag, replay overload, privacy erasure, and inconsistent disaster restore.

## Security and isolation

Authorize command handling, stream reads, temporal queries, replay/export, schema registration, snapshots, and repair independently. Tenant ID comes from authenticated context, not untrusted event payload. Encrypt payloads and backups with scoped keys; redact sensitive metadata from logs and traces.

Protect event/upcaster/model artifacts with provenance and review. Limit event/batch size, nesting, aggregate stream length, and replay rate. A crafted long stream or decompression payload can be a denial of service. Privileged repair never mutates silently; it emits audited corrective artifacts or a controlled migration with before/after digests.

## Verification strategy

- property-test aggregate reducers and invariants over generated event sequences;
- race concurrent expected-version appends and verify one winner/contiguous range;
- crash at each event/commit/outbox/command-result boundary;
- replay every retained historical schema through current upcasters;
- corrupt/advance snapshots and prove loaders fall back to the stream;
- compare snapshot-plus-suffix with full replay for sampled aggregates;
- rebuild projections and reconcile IDs/versions/digests;
- restore events, command results, snapshots, and outbox to skewed points;
- test privacy deletion across events, blobs, snapshots, backups, and projections.

## Decision framework

Use event sourcing when domain history is intrinsically valuable, invariants fit aggregates, alternative projections/reconstruction justify the complexity, and the organization can govern long-lived schemas. Do not use it merely to “have an audit log” or because events are already on a broker.

Answer:

1. Which aggregate invariants are protected by expected-version append?
2. Are events stable domain facts rather than persistence diffs?
3. Can every retained event be interpreted for its full lifetime?
4. What snapshot, replay, and rebuild bounds keep operations feasible?
5. How do integration publication and query projections remain derived and reconcilable?
6. How do privacy, retention, correction, and audit requirements coexist?
7. Can command identities and effects recover consistently with the event store?

## References

- [Martin Fowler: Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html)
- [Pat Helland: Immutability Changes Everything](https://www.cidrdb.org/cidr2015/Papers/CIDR15_Paper16.pdf)
- [EventStoreDB: Expected Version and Optimistic Concurrency](https://developers.eventstore.com/clients/grpc/appending-events.html)
- [Microsoft: CQRS Journey](https://learn.microsoft.com/en-us/previous-versions/msp-n-p/jj554200(v=pandp.10))
- [CloudEvents Specification](https://github.com/cloudevents/spec)
- [NIST SP 800-92: Guide to Computer Security Log Management](https://csrc.nist.gov/pubs/sp/800/92/final)
