# Publish-Subscribe Architecture

Publish-subscribe turns one committed fact into independent consumption streams. Its central abstraction is not “send to many services”; it is a durable event plus subscription state that allows each consumer to progress, pause, replay, filter, and fail without changing another consumer’s position.

A pub/sub contract must define topic semantics, fan-out architecture, subscription state, replay and bootstrap, filtering, and schema governance. It deliberately does not re-teach [Ordering](03-message-ordering.md) or [Delivery Guarantees](04-delivery-guarantees.md). Atomic domain publication belongs to [Outbox and Inbox](07-outbox-pattern.md). Multi-step business execution belongs to [Durable Workflows](../18-workflow-job-systems/04-durable-execution-workflow-engines.md) and [Retry, Idempotency, and Compensation](../18-workflow-job-systems/06-retry-idempotency-compensation.md).

## Workload and contract

A publisher emits a versioned fact:

```text
EventEnvelope {
  event_id
  event_type
  schema_id
  source
  subject
  source_version
  occurred_at
  recorded_at
  partition_key
  payload
  trace_context?
}
```

The broker exposes topic and subscription operations:

```text
publish(topic, envelope, durability_policy)
create_subscription(topic, start_policy, filter, retention_policy)
fetch(subscription, capacity, deadline)
checkpoint(subscription, partition, position, generation)
seek(subscription, replay_point)
```

An event is a statement in the past tense (`OrderAccepted`), not an imperative `SendShipment`. Commands have one intended owner and belong in a work queue or workflow. Facts can have zero, one, or many consumers, including consumers added later.

Define:

- who owns and may publish each event type;
- whether the topic is a transient notification channel or replayable event stream;
- subscription start policy (`latest`, earliest retained, timestamp, source version, explicit snapshot checkpoint);
- independent retention and replay limits;
- filtering semantics and whether filtered events advance progress;
- schema compatibility policy and deprecation window;
- partial regional behavior and tenant/residency boundaries;
- maximum fan-out, subscriber lag, and control-plane scale.

## State and invariants

The system maintains two different categories of durable state.

**Topic state** includes event records, partition/segment manifests, replication positions, schemas, retention watermarks, and producer authorization. **Subscription state** includes filter version, assigned partitions, delivered/in-flight positions, durable checkpoint, consumer generation, replay lease, and status.

Enforce these invariants:

**Topic bytes are independent of subscriber health.** One slow subscription does not corrupt or reorder another. It may affect retention only through an explicit bounded policy.

**Each subscription has one monotonic durable position per ordered scope.** Replays create a named new generation or reset operation; they do not silently move production checkpoints backward.

**Filtered records have defined position semantics.** A record excluded by a broker-side filter still advances the subscription scan position, or the filter change can unexpectedly resurrect historical data.

**Schema identity is immutable.** A schema ID always denotes the same canonical definition. Compatibility modes can change only as versioned topic policy.

**Publication authority is exclusive.** Only the service owning a fact may originate it. Relays may transport it without changing identity, source version, or schema.

**Replay is bounded and auditable.** Operators know who sought which subscription, to what point, with what destination/effect safeguards.

## Data plane and control plane

The **data plane** authenticates publishers, validates envelopes, selects partitions, appends/replicates records, evaluates bounded filters, serves subscriptions, and persists checkpoints. It uses cached versioned topic/subscription policy.

The **control plane** owns topics, schemas, publisher ACLs, subscription definitions, retention/quotas, partition maps, consumer-group membership, replay operations, and deprecation. A control-plane outage should not stop established data-plane flows immediately; serving continues under a pinned policy until its safety lease expires.

Schema registration and topic creation are not on the publish hot path. A producer deploy references pre-registered schema IDs and fails closed on unknown or incompatible types. Allowing publishers to auto-create topics or schemas under production traffic turns typos into permanent contracts.

## Fan-out architectures

### Shared append log with subscription cursors

The broker stores each event once per replica. Every subscription keeps positions into the same partitions. This is storage-efficient for many replaying subscribers and supports late consumers, but retention, read amplification, and subscription metadata become central concerns. Consumers with different filters still scan shared bytes unless the broker maintains filter indexes or routed topics.

### Write-time fan-out into per-subscription queues

Publication materializes one delivery record per subscription. Reads are simple and subscription retention is independent, but publish work and storage scale with fan-out. Creating a new subscription cannot recover old events unless an archive exists. A transactional fan-out stage must avoid partially populating subscriptions after a crash.

### Hierarchical fan-out

Large systems often append once to a regional/topic log, then durable fan-out workers populate subscription shards or edge regions. This absorbs enormous subscription counts and isolates slow consumers, but introduces a second checkpoint and freshness lag. The fan-out projection must be rebuildable from the shared log and idempotent by `(subscription, event_id)`.

Choose from event size, publish rate, subscription count, replay requirements, filter selectivity, retention independence, and cross-region topology. “Pub-sub” does not imply one physical architecture.

## Subscription lifecycle and checkpoints

A subscription state machine can be:

```text
CREATING -> BOOTSTRAPPING -> ACTIVE -> PAUSED -> DRAINING -> DELETED
                         \-> FAILED
```

Creation pins a start point and filter/schema policy. Bootstrapping may need a source snapshot plus stream suffix. Active consumers fetch under a membership generation. Pausing stops new delivery but preserves checkpoint and retention lease according to policy. Draining stops new work, waits for in-flight claims, then seals a final checkpoint. Deletion has a recovery grace period before metadata and retained data are reclaimed.

Checkpoint granularity controls duplicate replay and write load. Per-message checkpoints minimize replay but can dominate storage I/O. Batching reduces writes but replays the uncommitted suffix after failure. [Delivery Guarantees](04-delivery-guarantees.md) defines the effect boundary; the subscription records a durable scan position only after the consumer declares the batch complete.

Store both logical position and event/source time. Position is authoritative for progress; time supports lag and seek. Event `occurred_at` may be late or client-skewed, so use broker-recorded time for transport lag and source version/checkpoint for data completeness.

Membership changes carry a generation. Old members cannot checkpoint after partitions are assigned to a new generation. Cooperative handoff can reduce duplicate work by revoking and draining subsets, but the consumer must still tolerate overlap during crashes.

## Filtering and routing

Broker-side filters reduce egress and consumer CPU but add a query engine to the broker. Restrict them to typed, indexed envelope fields with bounded evaluation. Arbitrary payload scripts make broker latency and security dependent on untrusted code.

Filtering at publication by routing to many narrow topics makes reads efficient but couples producers to consumer taxonomy and can duplicate events. Filtering at consumption is flexible but pays network/scan cost. A stable compromise uses a small number of domain-owned topics plus subscription filters over approved envelope attributes.

Filter changes are migrations. Decide whether the new predicate applies only after activation, triggers replay from a point, or creates a new subscription. Mutating a predicate in place while retaining the old checkpoint makes it unclear which historical events were considered.

Do not use a filter as authorization unless the broker guarantees it is unbypassable for every fetch, replay, export, cache, and diagnostic path. Strong tenant isolation often requires separate topics/partitions/encryption domains.

## Schema and semantic governance

Syntax compatibility is necessary but insufficient. Adding an optional field can be wire-compatible while changing meaning. Maintain an event catalog with owner, purpose, field semantics, privacy classification, ordering key, source-version semantics, examples, and deprecation status.

Common evolution rules:

- add optional fields with explicit absence meaning;
- never reuse field identifiers/names for different semantics;
- preserve unknown fields where the serialization protocol requires round trips;
- use new event types for semantic changes rather than a flag that inverts meaning;
- allow consumers a documented upgrade window before removing production;
- validate producers against both schema and semantic policy in CI and at broker ingress.

Upcasting can present old payloads in a new in-memory shape, but the original bytes and schema ID remain available for audit. A chain of many runtime upcasters increases latency and makes replay depend on old code; compact into a new derived topic only with lineage and reconciliation, never by silently rewriting the authoritative event.

Consumer-driven contract tests are useful signals, but publishers cannot promise to preserve every accidental consumer dependency forever. Topic ownership and compatibility policy arbitrate changes.

## Bootstrap, replay, and new subscribers

A new subscriber to a large stream should not necessarily replay years of changes to reconstruct current state. Publish a versioned snapshot with:

- topic/schema generation;
- partition positions included in the snapshot;
- source checkpoint and creation time;
- object manifest, checksums, and encryption metadata.

The subscriber loads the snapshot idempotently, then consumes strictly after each recorded partition position. Events arriving during snapshot creation are covered by those positions. A snapshot without stream coordinates creates a gap or double-apply window.

Replay is a separate workload class. It can multiply broker reads and downstream effects. Require a destination, rate budget, start/end positions, dry-run estimate, idempotency plan, and cancellation. Prefer a new replay subscription or isolated consumer group so production progress is not moved backward. Throttle by bytes and downstream service capacity, not only messages/s.

## Cross-region topology

Replicate domain events according to residency and recovery policy. Active/passive replication gives a clear writer but requires subscription failover and position translation. Region-local topics with asynchronous inter-region relay reduce write latency but do not create one global order; duplicate identities and conflict ownership must be explicit.

The relay records source topic, partition, position, event ID, and source region. Destination publication is idempotent by source identity. A loop-prevention marker stops region A from re-exporting an event imported from B. Encryption and schema policy must be available before payloads cross regions.

During failover, a subscriber can either accept an RPO gap, wait for replication to catch up, or read both sources and deduplicate. Document the choice per event class. “Multi-region broker” is not itself a recovery contract.

## Capacity and cost model

Illustrative workload:

- 80,000 events/s at 900 encoded bytes;
- three broker replicas;
- 60 shared-log subscriptions;
- average subscription reads 70% of events after filters;
- two regional copies of consumer egress;
- seven-day topic retention.

Logical ingress is about 68.7 MiB/s; three replicas write about 206 MiB/s before indexes and compaction. Seven days of replicated topic bytes are roughly 119 TiB. Compression must be measured on real payloads and envelope repetition.

Consumer egress is `80,000 * 900 * 60 * 0.70`, about 2.8 GiB/s per regional copy before protocol overhead. Storage is independent of subscription count in a shared-log design, but egress is not. A per-subscription fan-out design would also materialize roughly 42 subscription copies per event, making write/storage amplification decisive.

Subscription metadata can dominate control-plane operations. With 60 subscriptions, 256 partitions, and checkpoints every 2 seconds, naive per-partition checkpoint writes produce 7,680 writes/s. Batch compactly by subscription/generation and measure recovery replay before lengthening the interval.

If one replay reads the full seven-day logical corpus in 12 hours, it adds about 1.3 GiB/s before replication and downstream writes. Allocate replay capacity explicitly.

## Concrete failure trace: abandoned subscription pins retention

A team creates a replayable subscription for an experiment, pauses it, and abandons the project. Retention policy says the oldest active subscription checkpoint protects data. The checkpoint remains six months behind, disk usage grows, and brokers approach full capacity. Deleting old segments would violate the subscription’s apparent replay contract; keeping them threatens the whole topic.

Containment stops low-priority publication or expands storage while identifying the owner. The control plane expires the subscription’s retention lease under audited policy, snapshots/archive if required, and advances the topic watermark. Prevention gives subscriptions owners, maximum retention leases, budget/quota, expiry, and alerts based on retained bytes attributable to each subscription, not merely consumer lag.

## Operations and observability

Track per topic, partition, subscription, schema, tenant, and region:

- publish rate/bytes, validation rejection, replication and append latency;
- subscription scan/delivery/checkpoint positions and lag in records, bytes, and time;
- oldest checkpoint, retained bytes attributable to each subscription, and expiry;
- filter selectivity/evaluation cost and filter-version distribution;
- schema/type volume, unknown-schema attempts, and deprecated-version traffic;
- fan-out backlog and source-to-subscription visibility lag;
- replay rate, estimated completion, downstream throttling, and cancellations;
- regional relay lag, duplicate imports, loops prevented, and RPO exposure.

Runbooks cover publisher schema regression, slow/abandoned subscription, accidental replay, corrupt snapshot, filter rollout error, control-plane outage, regional relay loop, and retention pressure.

## Security and privacy

Authorize publish and subscribe separately by topic, event type, tenant, region, and purpose. Schema registration, replay, seek, export, and subscription creation are privileged control-plane operations. Audit them immutably.

Minimize envelopes; routing headers are widely visible to broker infrastructure and logs. Encrypt sensitive payloads with scoped keys, but remember encryption can prevent broker-side filters. Apply retention and deletion policy to topics, snapshots, replay outputs, caches, and quarantines.

Reject spoofed `source`, subject, trace, and tenant fields; the authenticated publisher identity determines allowed values. Validate sizes, nesting, decompression ratio, and schema before fan-out to prevent one event multiplying resource abuse across every subscriber.

## Verification strategy

- model-test subscription create/pause/seek/checkpoint/delete and membership generations;
- crash fan-out workers at every checkpoint boundary and compare with the source log;
- verify schema compatibility and semantic golden events across producer/consumer versions;
- load-test subscription count, skewed filters, slow readers, replay, and control-plane churn;
- build snapshots while publishing and prove snapshot-plus-suffix has no gaps;
- partition regions and validate relay identity, loop prevention, and chosen RPO behavior;
- attempt unauthorized fetch, seek, export, filter change, and cross-tenant cache access;
- expire an abandoned retention lease and verify evidence plus notifications.

## Decision framework

Use pub-sub when independent consumers need the same fact, can own their checkpoints, and benefit from replay or isolation. Prefer direct synchronous calls when the caller needs an immediate response and both services share one availability budget. Prefer a work queue for one-owner tasks. Prefer a workflow when events are being used to hide an implicit multi-step state machine.

Before creating a topic, answer:

1. Is this a fact, command, or state snapshot, and who owns it?
2. Is replay required, and what snapshot/checkpoint bootstraps new consumers?
3. Which fan-out architecture fits subscription count, filters, and retention?
4. What schema and semantic compatibility contract governs evolution?
5. How are abandoned subscriptions, replay cost, and retention leases bounded?
6. What tenant/region/privacy boundaries apply to every copy?
7. How will subscriber effects tolerate the selected delivery semantics?

## References

- [CloudEvents Specification](https://github.com/cloudevents/spec)
- [AsyncAPI Specification](https://www.asyncapi.com/docs/reference/specification/latest)
- [Apache Kafka: Design](https://kafka.apache.org/documentation/#design)
- [Apache Avro Specification: Schema Resolution](https://avro.apache.org/docs/current/specification/#schema-resolution)
- [Google Cloud Pub/Sub: Subscription Properties](https://cloud.google.com/pubsub/docs/subscription-properties)
- [NATS JetStream: Consumers](https://docs.nats.io/nats-concepts/jetstream/consumers)
