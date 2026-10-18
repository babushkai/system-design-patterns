# CQRS and Projection Architecture

## TL;DR

Command Query Responsibility Segregation separates the model that validates writes from models built for specific reads. The write side owns invariants and produces committed changes; projection pipelines maintain replaceable read models with explicit freshness, publication, and rebuild state. CQRS is useful only when that separation buys measurable query, scale, or ownership advantages.

A CQRS projection needs an explicit read-model lifecycle covering query ownership, rebuild, atomic application, publication, freshness, and read-after-write behavior. [Event Sourcing](05-event-sourcing.md) owns authoritative domain logs; CQRS does not require event sourcing. [Outbox and Inbox](07-outbox-pattern.md) owns the publication and dedup atomicity patterns used here.

## Workload and Contract

Commands express intent and return command outcomes:

```text
SubmitCommand(command_id, aggregate_id, expected_version, payload)
  -> accepted/rejected, committed_source_version, result
```

Queries read a named projection:

```text
Query(read_model, query, consistency_token?, deadline)
  -> rows, projection_generation, applied_source_position, stale/incomplete
```

Define per read model:

- owned query shapes, filters, sort, pagination, aggregation, and authorization;
- authoritative source and event/change contract;
- freshness objective in position/version and elapsed time;
- read-your-writes policy using source/projection tokens;
- projection idempotency and ordering requirements;
- schema and generation lifecycle;
- rebuild source, duration, storage reserve, and cutover/rollback;
- behavior under missing events, poison data, and source retention gaps;
- reconciliation proof and repair authority.

Do not expose a “read database” as a shared data dump. A projection service owns a contract for one bounded set of queries; downstream teams do not write its tables or depend on undocumented columns.

## State and Invariants

A projection has:

| State | Purpose |
|---|---|
| read rows/indexes | denormalized query representation |
| inbox identities | proves which source records committed to this projection |
| source checkpoint | greatest contiguous applied source position per scope |
| per-entity version | detects stale/out-of-order changes where needed |
| generation manifest | schema/code/source checkpoint and publication status |
| rebuild state | snapshot position, catch-up position, validation evidence |
| query routing | active/canary/rollback generation |

Enforce:

**Single write authority.** Only the projection pipeline mutates a read model. Query handlers are read-only.

**Atomic apply.** Inbox insert, read-model mutation, entity version, and projection checkpoint commit in one transaction when they share a store. Broker acknowledgment follows that commit.

**Contiguous progress.** A checkpoint means all required source changes through that point are reflected or explicitly skipped by governed policy.

**Monotonic entity state.** An older source version cannot overwrite a newer projection row.

**Published generation is closed.** Query routing points only to a complete schema/index generation with a known source checkpoint and validation result.

**Projection is disposable.** It can be rebuilt from the authoritative source plus retained suffix without manual reconstruction from itself.

**Authorization is query-complete.** Access restrictions shape rows, filters, counts, aggregations, caches, and exports, not only final objects.

## Data Plane and Control Plane

The **write data plane** validates commands and commits authoritative state plus a durable change record. The **projection data plane** consumes that record, applies deterministic transformations, commits progress, and serves queries. The two can scale and deploy independently.

The **control plane** registers projection types, schemas, source contracts, checkpoints, generations, ownership, freshness/retention policy, build jobs, validation gates, and query routing. It publishes immutable generation manifests and atomically changes aliases/routes.

The query hot path should not ask the control plane per request. It consumes a cached signed/versioned routing snapshot and returns the active generation in response metadata. During a short control-plane outage, serving can continue from a pinned generation while builds/cutovers pause.

## Designing Read Models from Queries

Start with access patterns and invariants. A read model may be:

- a denormalized row/document per screen or entity;
- a search index for text/filter/ranking;
- a time-series or aggregate table;
- a graph/relationship index;
- a cache-like key/value view;
- an analytical materialization.

Store fields needed to answer the owned query without synchronous calls back to multiple services. Otherwise the “read model” is a distributed join at request time and inherits all dependencies’ latency/availability.

Denormalization duplicates data and creates maintenance obligations. Record source owner/version for copied fields. A customer-name change may update millions of order rows; alternatives include late binding from a local customer projection, grouping updates, or accepting a named freshness bound. Avoid copying sensitive fields unless the query requires them.

Indexes and partition keys follow measured filters/sorts. Include stable tie-breakers for pagination. Bound unbounded collections: “all followers in one document” creates hot rewrites and size limits. Use child/bucket rows with explicit query pagination.

One generic projection serving every consumer tends to recreate a coupled operational database. Prefer a small number of models aligned to coherent query workloads and ownership, while avoiding one model per trivial UI widget.

## Atomic Projection Application

For a transactional read store, consume one source record as:

1. validate event type/schema, source identity, and authorization/tenant metadata;
2. begin transaction;
3. insert `(projection, generation, event_id)` into an inbox with a unique constraint;
4. on exact duplicate, verify request/source digest and finish without reapplying;
5. check expected source/entity version and gap policy;
6. update/insert/delete every affected read row;
7. update entity version and contiguous source checkpoint;
8. commit;
9. acknowledge the broker.

A separate “has processed?” lookup then mutation then marker has two races: concurrent consumers can both pass, and a crash between steps can duplicate or lose the effect. Atomicity must include the actual projection mutation.

Some stores cannot transact the inbox and all derived structures together. For example, one transaction may not span a database row and a remote search index. Options are:

- make one durable projection table authoritative and derive the remote index via its own outbox/checkpoint;
- use versioned idempotent writes in the remote store and reconcile;
- build immutable batches/generations and publish a manifest;
- accept/document a repairable inconsistency window.

Calling two stores and then checkpointing is not an atomic projection.

## Ordering, Gaps, and Deletes

Projection semantics determine ordering needs. Replacement events with monotonic source versions can ignore stale arrivals. Deltas require ordered application or commutative identities. Cross-entity aggregates may require one source partition/checkpoint or a transactional batch marker.

On a gap, stop the affected strict scope, persist future records, fetch missing events from the source, or rebuild. Do not advance the global checkpoint through missing required work because other partitions are healthy. See [Message Ordering](03-message-ordering.md).

Deletes are versioned domain changes. Use tombstones carrying source version so a delayed update cannot resurrect a row. Tombstone retention must cover maximum replay/out-of-order window or the store must retain the highest source version independently after payload deletion.

Projection code is deterministic over event plus declared reference snapshots. Network lookups to “current” data during replay make rebuild results depend on when they run. If enrichment is necessary, consume a versioned local projection or record the chosen reference version.

## Freshness and Read-Your-Writes

Transport offset lag and time lag measure different things. Track:

- last source position available;
- last contiguous position applied;
- source recorded time at that position;
- oldest pending/gap time;
- read-model generation.

Client event time is not a reliable pipeline clock. Use source/broker recorded time plus domain effective time where needed.

For read-your-writes, the command response returns a consistency token such as `(source_stream, committed_version/position)`. A query can:

- wait until the selected projection has applied at least that token within a deadline;
- route to a write-side read path for that entity;
- overlay the command result locally in the client/UI;
- return a typed `not_yet_projected` response.

Sleeping for a fixed delay is neither correct nor efficient. A token scoped to one aggregate cannot imply global projection progress unless the source order provides that relation.

Waiters need bounds and cancellation. Register by position/entity, wake on checkpoint progress, expire on deadline, and cap per-tenant waiters. During projection outage, fail clearly rather than holding every request until the fleet exhausts connections.

## Rebuild and Generation Publication

Rebuilds are normal operations, not emergencies. Use blue/green generations:

1. register immutable projection code/schema generation `g+1`;
2. pin a source snapshot/checkpoint `S`;
3. bulk-build `g+1` from authoritative state/history through `S`;
4. consume the suffix after `S` with independent checkpoint/inbox;
5. reach the freshness gate;
6. reconcile counts, IDs, versions, aggregates, sampled fields, and business invariants;
7. shadow queries or mirror a stable sample and compare results/latency;
8. canary query routing, then atomically switch the alias;
9. keep `g` caught up or readable for a bounded rollback window;
10. stop and reclaim it only after rollback/evidence retention expires.

Do not truncate the active model and rebuild in place; it converts a data migration into an outage and removes rollback. Do not dual-write from projection code to old/new tables without independent checkpoints; a bug can mark both complete while omitting the same records.

Publication manifest includes projection type, generation, schema/code digests, source snapshot/checkpoints, compatible query API version, validation artifact, created time, and owner. Query nodes reject an incomplete/incompatible generation.

## Schema and Query Evolution

Additive read columns can be populated lazily only if queries define absence. Breaking changes use a new generation. Query APIs version semantics, not merely table columns; changing sort/tie-break, total-count accuracy, or authorization filtering can be breaking even if schema is compatible.

Projection consumers accept old/new source event forms through tested adapters. Deploy readers/consumers in a compatible order, then stop old event production after the documented window. Keep golden events from every retained schema.

When a projection changes ownership, transfer source contract, checkpoint history, rebuild artifacts, runbooks, privacy classification, and query SLO, not just database credentials.

## Product Knob Map

Read-model products ship defaults chosen for indexing throughput, and those defaults decide what a query is allowed to return. The knobs below are the ones that determine freshness, version safety, generation visibility, and whether a rebuild can be paused without an outage. Confirm each on your cluster: several differ between managed and self-hosted builds.

| Product and scope | Knob | Invariant it enforces | Default and how it misleads |
| --- | --- | --- | --- |
| Apache Kafka Streams projection | `commit.interval.ms`, `cache.max.bytes.mb` | progress is durable and each record contributes exactly once to an aggregate | commit interval trades write load against observable staleness, so "the projection is broken" is often the commit period. Caching suppresses intermediate updates per key, which silently changes aggregate results unless caching is disabled for those operators; correctness depends on a memory setting. |
| Elasticsearch or OpenSearch read model | `refresh_interval`, `if_seq_no` with `if_primary_term` | a visible document reflects a known source version, and a stale writer cannot overwrite it | visibility lags the refresh interval, so an immediately following query legitimately misses the row; without conditional writes the index is last-write-wins, which lets a delayed update resurrect a deleted document. Setting `refresh_interval=-1` for a bulk load and forgetting it produces a permanently stale model with green health. |
| AWS Lambda event source mapping (DynamoDB Streams, Kinesis, SQS) | `MaximumRetryAttempts`, `MaximumRecordAgeInSeconds`, `DestinationConfig` | a failing record is retried under a bound and then retained somewhere queryable | retries are unbounded and there is no failure destination by default, so one poison record either holds its shard forever or, once an age limit is added later, is discarded with no copy kept. Both outcomes look like transient consumer instability in metrics. |
| PostgreSQL materialized view as a read model | `REFRESH MATERIALIZED VIEW CONCURRENTLY` plus a unique index over the full output | readers never see a partially refreshed model and queries are not blocked by maintenance | concurrency requires a unique index covering the output rows; without one the statement fails, and the usual fallback to a non-concurrent refresh takes an exclusive lock that stalls every query on it. Materialized views are also not incremental: each refresh re-reads the base relation. |
| Multi-region read replica with last-writer-wins merging (for example DynamoDB global tables) | conflict resolution timestamp granularity | replicas converge to one value for a key | resolution uses coarse item timestamps, so two writes inside the same unit resolve by an implementation-defined tie-break. A projection that emits several updates per business event needs monotonic versioned writes or one writer region per key, not convergence-by-clock. |
| Redis or Memcached in front of a read model | `maxmemory-policy`, TTLs | a served answer is either correct or explicitly absent | all-keys eviction removes projection entries under memory pressure and the client sees a miss rather than degradation, so treating an evictable store as authoritative turns capacity pressure into phantom deletions. Keep it a cache with a defined source fallback. |
| Search index generation publication | alias swap in one atomic actions request | query traffic moves to a complete, validated generation only | pointing queries directly at a physical index exposes a half-built generation; an alias plus an atomic actions request is what makes cutover and rollback single-step operations rather than a data migration. |
| PostgreSQL read replicas for read-your-writes | `hot_standby_feedback`, `max_standby_streaming_delay`, LSN-based session staleness | a client reads its own write without waiting arbitrarily | replica backlog is measured in WAL bytes, not seconds, and feedback trades replication lag against query cancellations on the replica. Session-staleness checks also assume you reuse the same connection, which transaction-mode pooling breaks silently: the check passes against a different backend than the query. |

```bash
# read-only: what will a query actually see right now?
curl -s "$ES_URL/_alias/search-orders" | jq '.'                     # active generation
curl -s "$ES_URL/search-orders-*/_settings?pretty" \
  | grep -E 'refresh_interval|number_of_replicas'
aws lambda list-event-source-mappings --query \
  'EventSourceMappings[].{fn:FunctionName,retries:MaximumRetryAttempts,age:MaximumRecordAgeInSeconds,onFailure:DestinationConfig.OnFailure.Destination,state:State}'
psql "$READ_DB" -c "SELECT matname, populated FROM pg_matviews"
psql "$PRIMARY_DB" -c "SELECT client_addr, state, replay_lag,
                             pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn)) AS backlog
                      FROM pg_stat_replication"
redis-cli CONFIG GET maxmemory-policy                                # cache or accidental authority?
```

## Capacity and Cost Model

Illustrative read model:

- source produces 30,000 changes/s, average 900 bytes;
- each change updates 1.4 projection rows on average;
- measured transaction service time is 1.8 ms per 100-record batch plus 0.05 ms per row;
- 18 months of source history contains 1.2 trillion events;
- projection is 14 TiB per generation including indexes;
- query traffic peaks at 90,000/s.

Row mutation rate is 42,000/s. A 100-event batch updates 140 rows and consumes `1.8 + 140*0.05 = 8.8 ms` measured CPU/service work, or 0.088 ms per event under this batch shape. At 30,000/s that is 2.64 CPU-seconds/s before I/O/replication; shard by write locality and measure lock/index amplification.

Blue/green rebuild needs at least 28 TiB for two generations plus build spill, logs, backups, and reserve. If source replay sustains 4 million events/s, 1.2 trillion events take about 83 hours before catch-up and validation. A snapshot/export at source checkpoint can reduce rebuild time dramatically, but it must include exact stream positions.

At 90,000 queries/s and a measured 3.2 ms mean query CPU, demand is 288 CPU-seconds/s. At 55% target utilization, plan about 524 logical cores before cache misses and failure reserve. Read/write capacity and rebuild I/O need separate budgets so a rebuild cannot destroy query tails.

## Concrete Failure Trace: Checkpoint Commits Outside Projection

A consumer updates a search document, then writes its broker checkpoint to a separate metadata database. The search update times out but may have succeeded; the checkpoint commits. After restart, the event is skipped. Later a rebuild from the search index itself preserves the missing/ambiguous state.

Containment compares source versions and rebuilds the affected partition from the authoritative source. Repair introduces versioned idempotent search writes plus a durable local projection outbox/checkpoint, and makes rebuild read the source, not the derived index. Prevention fault-injects every cross-store boundary and requires reconciliation before checkpoint promotion.

## Operations and Observability

Track by projection, generation, source partition/schema, tenant, and query class:

- received/applied/duplicate/rejected rates and transaction latency;
- source available versus contiguous applied positions and time lag;
- gaps, buffered bytes, stale-version rejects, and tombstone age;
- inbox/checkpoint storage, unique conflicts, and retention horizon;
- read-model rows/bytes, index health, query latency/errors/count accuracy;
- consistency-token wait latency/timeouts and fallback path;
- rebuild snapshot/catch-up positions, throughput, ETA, validation failures;
- active/canary/rollback routing and result disagreement;
- source-to-projection reconciliation mismatches.

Runbooks cover lag, poison schema, permanent gap, corrupt generation, query regression, failed cutover, rebuild overload, and privacy deletion. Operators need per-entity repair and partition rebuild tools with audit/dry-run.

## Incident Playbook

Read-model incidents are usually freshness or completeness disputes rather than hard failures, which means the first job is deciding whether the projection is wrong or merely behind. That distinction changes the action completely: one justifies a rebuild, the other only justifies a routing decision while it catches up.

### Users report stale reads after successful writes

Separate transport lag from applied state: last available source position, last contiguous applied position, and the source recorded time at that position. If those advance normally, the pipeline is healthy and the product contract about freshness is what failed, so answer with the freshness bound rather than infrastructure work. If a command path promised read-your-writes, honor it explicitly for that entity (consistency token wait within a deadline, write-side read path, or a typed not-yet-projected response) and check that waiters are bounded per tenant, because unbounded waiting converts a freshness bug into a connection-exhaustion outage. Do not insert sleeps into the client or lower the projection commit interval under pressure: the first is wrong under load and the second trades every other consumer's throughput for one screen.

### A rebuild finished but the two generations disagree

Reconcile by identity sets, not counts: per-partition row counts agree far more often than ID sets do, and a missing record plus an extra record is invisible to `COUNT`. Diff sampled ID ranges, aggregate values, and entity versions between generations, then locate the first divergence in source position; that position tells you whether the cause is a skipped range, an enrichment lookup that changed mid-build, or a schema adapter. Keep the old generation readable and do not promote. Do not accept "the difference is under the tolerance" for a user-facing model: state the specific reconciliation result, and if enrichment was involved, pin it to a versioned reference before rebuilding, otherwise the next rebuild disagrees differently.

### Deleted rows reappear

Almost always an ordering artifact rather than a lost delete: check whether tombstones outlive the maximum replay and out-of-order window, and whether writes are version-gated so an older source version cannot overwrite a newer row. Absence is not a version, so a projection that represents deletion only by removing the row cannot distinguish "deleted" from "not yet arrived". Restore correctness with versioned tombstones or a retained highest-source-version column that survives payload deletion under privacy policy. Do not fix it by deleting again and rebuilding the affected partition immediately afterward: if the replay window still contains the older update, you have scheduled the resurrection a second time.

### Queries regressed after a generation cutover

Compare per-query-class latency distributions and cache hit ratios between generations before touching either; a mean that moved while p99 did not is usually noise plus a smaller result set. Real regressions come from physical layout: index or partition key chosen for the wrong filter/sort combination, missing stable tie-breaker forcing large sorts, unbounded collections rewritten on hot entities, or a new generation with fewer shards and the same write share. Verify by replaying recorded queries against both generations offline. Do not roll routing back and forth to "see if it gets better": every flip discards warm caches and invalidates the comparison you need, so pick one generation deliberately and keep it pinned while measuring.

### The projection is stuck on one record

Classify before acting: a schema or code defect will fail every occurrence of that shape, while a data or authorization defect fails one record and its siblings succeed. Read the failure's position and event type, then quarantine the specific identity with its source version and error so the pipeline advances past it under a recorded decision, and per-item inbox state keeps the rest of the batch durable. Restarting the consumer without changing anything is the most common non-action here and simply replays the same failure. Do not drop the record or advance the contiguous checkpoint through it to relieve lag: a skipped required change makes every later checkpoint claim false, and the eventual rebuild will not know which records were ever omitted.

## Security and Privacy

Projection consumers authenticate the source and derive tenant/authority from trusted envelope fields. Query authorization is enforced in the model/query plan; post-filtering top results leaks counts and can produce incomplete responses. Separate tenant partitions/stores when policy requires stronger isolation.

Read models multiply sensitive data. Minimize copied fields, encrypt stores/backups, propagate deletion/legal hold, and maintain lineage. Query logs, caches, rebuild snapshots, old generations, and shadow comparisons are additional copies.

Restrict projection publication, checkpoint seek, bulk export, repair, and generation deletion. Validate event and query sizes. A malicious high-fanout change can cause write amplification; enforce per-event mutation bounds or route large expansions through controlled backfills.

## Verification Strategy

- reducer golden/property tests for every event schema and delete/version edge;
- crash/concurrency tests proving inbox, rows, version, and checkpoint atomicity;
- differential replay against a simple reference model;
- snapshot-plus-suffix rebuild tests during concurrent source changes;
- blue/green cutover/rollback with query-result and invariant comparison;
- authorization tests across rows, counts, facets, caches, export, and old generations;
- load tests combining query peak, source peak, backlog catch-up, and rebuild;
- reconciliation drills that corrupt/drop/duplicate a partition and repair it.

## Decision Framework

Adopt CQRS when write invariants and read workloads genuinely need different models, read scale/query shape justifies denormalization, or independent projection ownership/rebuild adds value. Avoid it for ordinary CRUD where one transactional model answers queries adequately.

Before creating a projection:

1. Which exact query contract owns it?
2. What source is authoritative and retained long enough to rebuild?
3. How do inbox, mutation, entity version, and checkpoint commit atomically?
4. What freshness/read-your-writes behavior do callers need?
5. What storage/time reserve makes blue/green rebuild feasible?
6. How are generation publication, reconciliation, and rollback proved?
7. How are authorization, privacy deletion, and query semantics preserved across copies?

## References

- [Martin Fowler: CQRS](https://martinfowler.com/bliki/CQRS.html)
- [Microsoft: CQRS Journey](https://learn.microsoft.com/en-us/previous-versions/msp-n-p/jj554200(v=pandp.10))
- [Pat Helland: Data on the Outside versus Data on the Inside](https://www.cidrdb.org/cidr2005/papers/P12.pdf)
- [PostgreSQL: Materialized Views](https://www.postgresql.org/docs/current/rules-materializedviews.html)
- [Elasticsearch: Index Aliases](https://www.elastic.co/guide/en/elasticsearch/reference/current/aliases.html)
- [CloudEvents Specification](https://github.com/cloudevents/spec)
- [Apache Kafka Streams: cache.max.bytes.mb](https://kafka.apache.org/documentation/#streamsconfigs_cache.max.bytes.mb)
- [PostgreSQL: REFRESH MATERIALIZED VIEW](https://www.postgresql.org/docs/current/sql-refreshmaterializedview.html)
- [Amazon DynamoDB: Global Tables Conflict Resolution](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/globaltables.HOWITWORKS.html)

---

## Related Chapters

- [Event Sourcing and Domain Logs](05-event-sourcing.md): the write side that supplies the facts these projections are built from
- [Message Ordering](03-message-ordering.md): a projection is only as correct as the stream order feeding it
- [Workload-Driven Data Modeling and Derived Projections](../02-distributed-databases/10-data-modeling.md): read models are denormalized work, paid for once and maintained forever
- [Stream Execution](../13-data-pipelines/02-stream-processing.md): the machinery that keeps a projection current through restarts and replays
