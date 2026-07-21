# API Design and Evolution

## TL;DR

An API is a distributed compatibility contract among independently deployed producers, consumers, intermediaries, data models, and operators. Its correctness includes more than field names: resource identity, operation semantics, retries, error classification, concurrency, pagination consistency, asynchronous completion, webhook delivery, authorization scope, and migration behavior all become part of the contract.

Model stable resources and explicit state transitions. Make errors machine-actionable. Protect unsafe retries with atomically recorded idempotency. Protect concurrent updates with preconditions. Treat a pagination cursor as a signed, scope-bound continuation capability—not base64-encoded database state. Represent long work as a durable operation resource. Treat webhooks as an at-least-once outbound delivery system with authenticated messages, replay, fairness, and endpoint isolation.

Evolve additively where semantics permit, but do not assume every added field or enum value is harmless. Measure real consumers, compare contracts and behavior, stage adapters, publish deprecation and sunset metadata, and preserve a reversible expand/migrate/contract sequence.

Scope: the public service contract. [Edge Gateway](./02-api-gateway.md) covers edge enforcement and routing; [Idempotency](../01-foundations/08-idempotency.md) covers the general deduplication pattern; [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md) covers client attempt policy.

---

## The Contract Surface

An API contract spans:

| Dimension | Examples |
|---|---|
| Addressing | authority, resource name, path, method/RPC operation |
| Representation | media type, schema, field presence, numeric/time semantics |
| Behavior | state transitions, validation, side effects, defaults |
| Failure | status, problem type, retryability, partial completion |
| Concurrency | precondition token, conflict detection, ordering |
| Continuation | pagination snapshot, cursor scope, operation polling |
| Security | caller/resource authorization, data minimization, replay defense |
| Operations | limits, latency target, availability, retention, deprecation |
| Evolution | compatibility rules, version negotiation, migration timeline |

A generated schema captures only part of this surface. “The OpenAPI diff is additive” does not prove that a new validation rule, ordering change, authorization filter, or default is compatible.

### Core invariants

1. **Stable identity:** resource identifiers do not expose mutable storage placement or get reused for a different logical resource.
2. **Explicit transitions:** state-changing operations define allowed source states, resulting state, side effects, and concurrency behavior.
3. **Programmable failure:** clients branch on stable error types/codes, not message text.
4. **One logical effect:** a correctly retried unsafe request cannot create a second effect under the idempotency contract.
5. **No lost update:** a client can require that a mutation apply only to the version it observed.
6. **Scoped continuation:** cursors and operation handles cannot be moved across tenant, caller, query, sort, or API version.
7. **Durable async state:** accepting long work creates a durable operation identity before success is reported.
8. **Authenticated outbound delivery:** webhook receivers can verify source, covered message components, freshness, and event identity.
9. **Measured evolution:** removal follows observed migration and an explicit compatibility policy.
10. **Bounded work:** list size, query complexity, request/response bytes, polling, webhook fan-out, and retained deduplication state have limits.

## Resource and Operation Modeling

### Resources have identity and lifecycle

A resource is a domain concept with:

- an opaque stable identifier;
- tenant or parent scope;
- representation schema;
- lifecycle state;
- version/precondition token;
- creation and update time semantics;
- ownership and authorization rules; and
- supported operations.

Do not expose a table row merely because it exists. A public resource should remain coherent if the service later splits tables, changes storage, or moves regions.

~~~text
/tenants/{tenant}/invoices/{invoice}

collection operations:
  list, create

resource operations:
  get, update, delete/archive

state transitions:
  finalize, void, retry-delivery
~~~

Hierarchical names make authorization and uniqueness visible, but the server still validates that the child belongs to the named parent. Never trust a path parent while loading a child solely by globally unique ID.

### CRUD versus domain operation

Use a generic create/read/update/delete operation when the semantics match. Use an explicit domain operation when the transition has distinct authorization, validation, side effects, idempotency, or audit meaning.

~~~text
PATCH /invoices/{id}          change editable fields
POST  /invoices/{id}:finalize transition draft -> final
POST  /invoices/{id}:void     transition final -> void
~~~

Encoding *finalize* as a writable status field lets clients attempt impossible transitions and can bypass transition-specific policy. The operation contract states:

- permitted source states;
- required precondition/version;
- synchronous versus asynchronous completion;
- exactly which effect is committed;
- idempotency behavior;
- resulting resource/operation representation; and
- stable failure types.

### Method semantics matter

HTTP methods have standardized safety, idempotency, caching, and conditional semantics. Do not define a GET that creates durable state or a DELETE whose retry creates another unrelated effect. Custom RPC APIs need an equivalent per-operation declaration so intermediaries and clients can choose timeouts and retries safely.

“Idempotent” means repeating the same intended request has the same intended effect, not that every response byte or timestamp is identical. A PUT can be idempotent while still returning a different audit revision on replay; document observable behavior.

### Representation design

Prefer types that survive language and storage differences:

- opaque string IDs with no ordering promise;
- timestamps with explicit offset/UTC and precision;
- durations as typed units, not ambiguous integers;
- money as decimal/string or integer minor units plus currency under a stated rounding policy;
- explicit field presence distinct from null and default;
- bounded strings and collections;
- stable enum identifiers with unknown-value behavior;
- references for independent resources and bounded embedding for snapshots; and
- deterministic canonical form only when signing or hashing requires it.

### Update masks and patch semantics

Partial update needs an unambiguous operation:

- which fields are selected;
- whether omitted means unchanged;
- whether null clears a field;
- how maps and arrays merge or replace;
- whether immutable/output-only fields are rejected;
- how authorization applies per field; and
- which resource version is required.

Generic JSON merge can be appropriate, but domain-specific update masks often make intent and compatibility clearer. Reject unknown writable fields when silently ignoring them could make a client believe a security or monetary setting was applied.

## Errors as a Stable Protocol

Use a structured problem representation with a stable machine identifier:

~~~json
{
  "type": "https://api.example/errors/precondition-failed",
  "title": "Resource version is stale",
  "status": 412,
  "instance": "/operations/op_7f2",
  "request_id": "req_b1a8",
  "expected_version": "v18",
  "current_version": "v19"
}
~~~

The stable contract is the problem *type* and documented extension fields. Human detail may be localized or redacted.

### Error taxonomy

Distinguish at least:

| Class | Meaning | Typical client action |
|---|---|---|
| Malformed request | Syntax/media type cannot be parsed | Fix request; do not retry unchanged |
| Validation failure | Parsed request violates field/domain input rules | Fix indicated fields |
| Unauthenticated | Valid authentication is absent | Acquire/refresh credential |
| Unauthorized/hidden | Principal may not perform or observe action | Do not retry without changed authority |
| Not found | Resource absent or intentionally hidden | Reconcile identifier |
| Precondition failed | Client’s version/condition is stale | Refetch and resolve conflict |
| State conflict | Current domain state forbids transition | Change workflow, not transport retry |
| Quota/overload | Admission refused | Respect bounded retry guidance |
| Dependency unavailable | Service could not complete now | Retry only under operation and deadline policy |
| Internal failure | Unexpected server condition | Retry only if safe; report request ID |

Do not mark every server error retryable. An authorization service outage may yield an unavailable error but repeating immediately can worsen it. Return retry guidance only when it reflects server knowledge, and the client still applies the policy in [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md).

### Error safety

Problem details must not leak:

- existence of hidden resources;
- internal topology or dependency names;
- raw query/database errors;
- credentials or signing inputs;
- another tenant’s identifiers;
- policy text not authorized for the caller; or
- stack traces.

Log richer evidence under access control using the request ID. Keep the public error stable even if internal implementation changes.

## Idempotency for Unsafe Operations

Network ambiguity is unavoidable:

~~~text
client sends create-payment
-> service commits payment
-> response is lost
-> client cannot know whether retry is duplicate or recovery
~~~

An idempotency key names one logical operation. Its scope must include every boundary that prevents accidental collision:

~~~text
tenant
authenticated client or delegated principal class
operation/route identity
target resource where applicable
idempotency key
canonical semantic request fingerprint
API behavior version
~~~

### Idempotency record state machine

~~~mermaid
stateDiagram-v2
    [*] --> Reserved: atomic claim
    Reserved --> InProgress: execution starts
    InProgress --> Succeeded: effect and replay result committed
    InProgress --> FailedReplayable: terminal deterministic failure
    InProgress --> Unknown: outcome cannot be established
    Unknown --> Succeeded: reconciliation proves commit
    Unknown --> FailedReplayable: reconciliation proves no effect
    Succeeded --> Expired: retention contract ends
    FailedReplayable --> Expired: retention contract ends
~~~

Correct behavior:

1. Validate and authenticate before claiming a key that could be used for storage exhaustion.
2. Canonicalize the semantic request and compute a fingerprint.
3. Atomically reserve the scoped key or read the existing record.
4. If the same key has a different fingerprint, return a stable key-reuse conflict.
5. If an operation is in progress, return or wait under a bounded documented behavior.
6. Commit the business effect and durable idempotency outcome in one transaction where possible.
7. Replay the documented result for later identical requests.
8. Retain the record for at least the published client retry horizon and operation ambiguity window.

When effect and dedupe store cannot share a transaction, use a domain-unique operation ID in the effect store plus reconciliation. “Write effect, then write cache” still duplicates if the process fails between writes.

### Replay semantics

Document whether replay returns:

- the original status and representation;
- the current resource representation with an explicit replay marker;
- an operation resource; or
- a stable terminal problem.

Do not replay transient 5xx failures forever unless the contract declares them terminal for the logical operation. Do persist deterministic validation or conflict outcomes when doing so prevents a key from being reused with altered meaning.

### Cross-region idempotency

If the same client can fail over across regions, the idempotency scope needs a shared or deterministically routed authority. Region-local stores allow duplicate effects unless:

- the operation is pinned to a home region;
- the business store has a global uniqueness constraint;
- a globally consistent key claim is used; or
- the accepted duplicate bound is explicit and the effect is compensatable.

The complete storage and retry pattern is in [Idempotency](../01-foundations/08-idempotency.md).

## Optimistic Concurrency and Preconditions

Idempotency answers “is this the same logical request?” It does not answer “is the resource still the version I read?”

### Lost update

~~~text
client A reads version 7
client B reads version 7
client A writes change -> version 8
client B writes stale change without precondition -> silently overwrites A
~~~

Return a strong validator or resource version and require it for mutations where lost updates matter:

~~~http
GET /documents/doc_1
ETag: "v7"

PATCH /documents/doc_1
If-Match: "v7"
Content-Type: application/merge-patch+json
~~~

If current version differs, reject with the conditional-request status and current version metadata allowed by policy. The client refetches, merges with domain knowledge, and submits against the new version.

### Preconditions

- **If-Match** protects update/delete against stale state.
- **If-None-Match: \*** supports create-only-if-absent semantics.
- A domain version field can serve RPC protocols but needs the same strong comparison contract.
- Time-based validators are unsafe when multiple updates can occur within the timestamp precision or clocks differ.

For a multi-resource command, one version token may represent a transactional aggregate or explicit set of component versions. Do not imply atomicity across independent resources unless the service actually provides it.

### Conflict versus precondition failure

Use a precondition failure when the client’s stated version/condition is false. Use a state conflict when the request is current but the domain transition is disallowed—for example, voiding a settled transfer. Keeping them distinct lets clients choose refetch/merge versus workflow correction.

## Pagination as a Continuation Protocol

Offset pagination is simple but can scan deeply and shifts under concurrent inserts/deletes. Keyset pagination continues after a value in a total order:

~~~text
ORDER BY created_at, id
WHERE (created_at, id) > (:last_created_at, :last_id)
LIMIT :page_size
~~~

The unique tie-breaker is mandatory. A timestamp alone does not define a total order.

### Choose consistency semantics

| Mode | Guarantee | Cost/behavior |
|---|---|---|
| Live traversal | Each page reflects current data | Inserts/deletes may change total membership; keyset avoids many shifts but not a snapshot guarantee |
| Snapshot watermark | All pages constrained to a stable read revision/time | Requires MVCC retention, snapshot token, or materialized result |
| Materialized result set | Exact membership/order stored for traversal | Storage and lifecycle cost; useful for exports/search |
| Best effort offset | Simple page numbers | Duplicates/skips and deep-query cost accepted explicitly |

Do not promise “no duplicates or omissions” without defining snapshot behavior under mutation.

### Cursor envelope

A cursor should encode or reference:

~~~json
{
  "cursor_version": 3,
  "tenant": "tenant-a",
  "principal_scope": "account-42",
  "operation": "list-invoices.v2",
  "filter_digest": "sha256:...",
  "sort": ["created_at:asc", "id:asc"],
  "page_size_cap": 100,
  "snapshot": "revision:88421",
  "position": ["2026-07-18T08:00:00Z", "inv_9f2"],
  "expires_at": "2026-07-18T09:00:00Z",
  "key_id": "cursor-signing-7"
}
~~~

Base64 is encoding, not integrity or confidentiality. Authenticate the canonical cursor with an HMAC or digital signature. Encrypt it if its position, filter, or snapshot leaks sensitive information. Alternatively store opaque random cursor IDs server-side.

### Scope binding

On every continuation:

1. parse a bounded cursor format;
2. verify signature/MAC and key generation;
3. verify expiry and supported cursor version;
4. bind tenant and authenticated caller scope;
5. bind operation/API behavior version;
6. recompute normalized filter, sort, field projection, and relevant authorization digest;
7. enforce page-size cap rather than accepting a larger request;
8. validate snapshot availability; and
9. execute from the encoded position.

Without binding, a caller can reuse a valid cursor from another tenant or alter filters while retaining an authorized position. Authorization changes may require invalidating the cursor or applying the new restriction; never let an old cursor preserve access that has been revoked.

### Cursor key and schema rotation

Maintain verification overlap for old signing keys until issued cursors expire. Cursor format migration can:

- support old and new decoders during the maximum cursor lifetime;
- issue only the new version;
- reject unsupported/expired cursors with a stable restart-list problem; and
- avoid translating a cursor when the new ordering cannot preserve semantics.

Cursor lifetime is constrained by signing-key rotation, snapshot retention, authorization freshness, and storage/index compatibility.

### Pagination capacity

Bound page size, projected fields, filter complexity, sort allowlist, snapshot age, and total traversal rate. Index the exact filter/order prefix. Avoid exact total counts when they require a full scan; expose an estimate or omit the count under a stated contract.

## Long-Running Operations

If work cannot reliably finish within one request deadline, accepting it should create a durable operation resource:

~~~mermaid
stateDiagram-v2
    [*] --> Pending
    Pending --> Running
    Pending --> Cancelled: cancellation accepted
    Running --> Succeeded
    Running --> Failed
    Running --> CancelRequested
    CancelRequested --> Cancelled
    CancelRequested --> Succeeded: effect crossed cancellation point
    CancelRequested --> Failed
~~~

### Acceptance

The initial unsafe request uses idempotency. The service atomically persists:

- operation ID and tenant/owner;
- normalized request or protected reference;
- target resource and requested effect;
- idempotency scope;
- creation time and retention;
- current state/version;
- execution lease/attempt metadata; and
- result or stable problem reference.

Then return an accepted response with a location/name for the operation. If the response is lost, the same idempotency key returns the same operation.

### Operation contract

A get-operation response includes:

- stable operation identity;
- state and version/validator;
- progress only when meaningful and monotonic under defined units;
- creation/start/update/completion timestamps;
- cancellation capability and current cancellation state;
- terminal result resource or problem;
- recommended polling interval where useful; and
- retention/expiry after terminal state.

Use conditional polling so unchanged status does not retransmit large bodies. Clients apply backoff and jitter; the server may provide a retry hint. Better still, offer webhook/event completion plus polling as recovery.

### Cancellation semantics

Cancellation is a request, not proof of rollback. Define:

- before which commit point cancellation prevents the effect;
- whether in-progress external calls can be interrupted;
- which partial artifacts are cleaned up;
- whether compensation is attempted;
- terminal status when the effect completed during cancel; and
- idempotency of repeated cancel requests.

Never report *cancelled* while a durable effect may still commit later.

### Worker leases and duplicate execution

Operation workers can crash after performing work but before acknowledging completion. Use the same idempotency and fencing principles as any durable job:

- execution lease with attempt/fencing token;
- domain-unique operation ID passed to effects;
- checkpoint or reconciliation;
- atomic terminal transition; and
- bounded retry/dead-letter policy.

## Webhooks as Outbound Delivery

A webhook is an API call made by your system to a subscriber-controlled endpoint. It inherits all distributed delivery failures plus an SSRF and tenant-isolation boundary.

~~~mermaid
flowchart LR
    TX[Domain transaction] --> OUTBOX[(Transactional outbox)]
    OUTBOX --> PUB[Event publisher]
    PUB --> FAN[Subscription fan-out]
    FAN --> Q1[Endpoint queue A]
    FAN --> Q2[Endpoint queue B]
    Q1 --> D1[Signed delivery worker]
    Q2 --> D2[Signed delivery worker]
    D1 --> EP1[Subscriber A]
    D2 --> EP2[Subscriber B]
    Q1 --> DLQ[Dead-letter and replay store]
    Q2 --> DLQ
~~~

### Subscription model

A subscription records:

- tenant and authorized owner;
- verified endpoint and allowed redirect/network policy;
- selected event types and bounded filter;
- payload/schema version;
- signing key/secret generation;
- status, failure counters, and disable reason;
- concurrency and rate budget;
- retry/retention policy; and
- created/updated revision.

Validate endpoints against SSRF policy at creation and again at delivery after DNS resolution and redirects. Prevent access to loopback, link-local, metadata, management, or other tenant networks. Pinning a resolved address forever breaks legitimate changes; trusting every re-resolution enables rebinding. Apply the platform’s egress policy on each connection.

### Event and delivery identity

Separate the immutable domain event from one delivery attempt:

~~~json
{
  "event_id": "evt_7f2",
  "event_type": "invoice.finalized",
  "subject": "invoices/inv_42",
  "occurred_at": "2026-07-18T08:30:00Z",
  "schema_version": "invoice-event.v3",
  "tenant": "tenant-a",
  "data": {"invoice": "inv_42", "version": "v19"}
}
~~~

Each delivery also has subscription ID, delivery ID, attempt number, sent time, and signing-key ID. The receiver deduplicates by event or delivery identity according to the published contract.

### Authentication and replay defense

Sign the method, target authority/path where stable, covered content digest, event ID, timestamp, and key ID using a standard HTTP message-signature scheme or a precisely specified MAC canonicalization. Receivers:

1. preserve the received bytes needed by the signature;
2. select the key by key ID under the subscription;
3. verify signature and covered components;
4. enforce a bounded timestamp window;
5. deduplicate event/delivery ID beyond that window as required;
6. process asynchronously; and
7. acknowledge quickly after durable acceptance.

TLS authenticates the receiver endpoint and protects transit; it does not prove to the receiver which webhook producer created the body.

Rotate secrets/keys with overlapping verification, new-signing preference, per-subscription generation telemetry, and explicit compromise revocation.

### Delivery semantics

Unless a stronger protocol is built, promise **at least once**, not exactly once:

- a response can be lost after the receiver commits;
- delivery workers can crash after sending;
- retries can arrive after later events;
- two workers can race after lease expiry; and
- subscriber endpoints can be slow or inconsistent.

Define ordering scope, if any—often per subject or subscription partition—and make receivers robust to duplicates and out-of-order delivery. Thin events can tell receivers to fetch current state, but that trades payload staleness for an inbound API dependency and authorization requirement.

### Retry, disable, and replay

Classify outcomes:

- success acknowledgement: complete;
- explicit permanent rejection: dead-letter or disable under policy;
- authentication/signature mismatch: stop and alert rather than retry forever;
- rate limit/overload: honor bounded retry guidance;
- timeout or transient server failure: exponential backoff with jitter;
- endpoint/network policy violation: disable and alert.

Cap attempts, elapsed retry horizon, concurrent deliveries, and bytes. Isolate queues per endpoint or fair-share tenant so one failing subscriber cannot block all others. After exhaustion, retain a replayable record and notify the owner. A replay creates new delivery attempts while preserving the original event identity.

The durable publication path is described in [Transactional Outbox](../05-messaging/07-outbox-pattern.md), and poison handling in [Dead Letter Queues](../05-messaging/08-dead-letter-queues.md).

## Compatibility Is Semantic

### Compatibility categories

| Change | Often safe only if… |
|---|---|
| Add optional response field | Clients ignore unknown fields and signature/cache canonicalization tolerates it |
| Add request field | Old servers ignore or reject predictably; new semantics are optional |
| Add enum value | Every client has an unknown-value branch |
| Increase validation strictness | Existing valid traffic has been measured and migrated |
| Change default | Clients explicitly send value or behavior version is pinned |
| Change ordering | Contract did not promise order; pagination cursors are not invalidated |
| Remove field | All consumers stopped depending on presence and semantics |
| Widen numeric range | Client language/storage can represent it |
| Make operation asynchronous | Client handles accepted operation rather than immediate resource |
| Tighten authorization/filtering | Cursor/cache/session behavior cannot preserve old visibility |

Compatibility is directional:

- old client with new server;
- new client with old server;
- old persisted message with new reader;
- new message with old reader;
- old gateway adapter with new service; and
- rollback of code after new data or schema has been written.

### Unknown fields and enum values

Response consumers should usually ignore unknown fields and preserve an unknown enum branch. Request producers should not assume a server accepted an unknown field merely because it returned success; security-sensitive write APIs often reject unknown input to prevent false belief.

For Protocol Buffers:

- never reuse a field number;
- reserve removed numbers and names;
- add fields with compatible wire types;
- define an unspecified/unknown enum value;
- test old/new generated clients; and
- remember that semantic validation can still break even when wire parsing succeeds.

### Behavioral versioning

Choose a stable version negotiation surface:

- major version in resource path;
- negotiated media type;
- explicit version header/date; or
- per-client pinned behavior revision.

One surface is easier to operate than mixed route, query, header, and payload versions. A version selects a coherent behavior bundle: schemas, defaults, errors, pagination, authorization interpretation, and webhook event shapes.

Avoid permanent forks of domain logic. Adapt old contracts to one current internal model, but preserve old observable semantics until migration completes. The adapter itself is production code with tests, telemetry, and retirement criteria.

## Deprecation and Migration

### Expand, migrate, contract

1. **Inventory:** identify consumers, owners, versions, traffic, webhook subscriptions, SDKs, and batch jobs.
2. **Specify:** document old/new semantics and compatibility matrix.
3. **Expand:** deploy server/data support that accepts old and new without removing old behavior.
4. **Instrument:** measure use by authenticated client and feature, not only endpoint.
5. **Publish:** announce deprecation, replacement, migration guide, support window, and sunset metadata.
6. **Migrate:** update first-party clients, SDKs, examples, and high-volume external consumers.
7. **Constrain:** stop new adoption of the old version and reduce its privileges/capacity only under contract.
8. **Verify:** require a quiet window that covers infrequent jobs and regional failover.
9. **Contract:** disable, then remove old adapters/schema only after rollback and data compatibility are resolved.
10. **Audit:** retain evidence of consumer notification, usage, exception, and final turn-down.

HTTP deprecation and sunset fields can make lifecycle machine-readable, with links to migration documentation. They complement direct communication and usage telemetry; they do not replace them.

### Data and API migration

An API can be rolled back only if the data written under the new behavior remains readable by the old service. Coordinate with [Database Migrations](../15-deployment/03-database-migrations.md):

~~~text
expand storage -> dual/read-compatible service -> expose new API
-> migrate callers/data -> stop old writes -> remove old API
-> contract storage after rollback window
~~~

Dual-write is not automatically atomic. Prefer one authoritative write path with asynchronous projection or a database transaction where both representations share a store.

### Consumer exceptions

An exception names owner, client identity, old behavior, reason, traffic, expiry, and migration checkpoint. An exception with no expiry becomes an undocumented API version.

## Capacity Planning

API shape controls system load.

Let:

- $\lambda$ be accepted API requests per second;
- $b_{\text{req}}$ and $b_{\text{resp}}$ be average bytes;
- $p$ be average page size;
- $q$ be long-operation polling requests per operation;
- $e$ be events per second;
- $s$ be matching subscriptions per event;
- $a$ be mean webhook delivery attempts; and
- $d$ be retained idempotency bytes per unsafe operation.

Application-layer traffic is approximately:

$$
B_{\text{api}} \approx \lambda(b_{\text{req}} + b_{\text{resp}}).
$$

Polling load is:

$$
\lambda_{\text{poll}} \approx \lambda_{\text{operations}} q.
$$

Webhook delivery attempt rate is:

$$
\lambda_{\text{webhook}} \approx e s a.
$$

Idempotency storage over retention window $T$ is approximately:

$$
S_{\text{idempotency}} \approx \lambda_{\text{unsafe}} d T,
$$

before indexes, replicas, and stored response bodies.

### Control multiplicative dimensions

- maximum page size and projection;
- filter and sort complexity;
- expanded request/response size;
- idempotency result retention;
- operation polling frequency and terminal retention;
- events times subscribers;
- webhook retry attempts and elapsed horizon;
- API versions/adapters executed in parallel; and
- schema/contract validation cost.

Use per-tenant fair scheduling and route cost classes. Large exports should become long-running operations producing a bounded artifact rather than enormous list pages.

### Avoid count and polling traps

An exact total count can cost more than one page and contend with writes. Polling every operation at a fixed short interval creates load proportional to waiting time, not useful progress. Use estimates/omission, conditional requests, adaptive retry hints, jitter, or event notification with polling recovery.

## Failure Modes and Traces

### Effect commits before idempotency result

~~~text
service reserves key -> commits charge
-> crashes before recording success
-> retry sees “in progress” forever or executes again
~~~

**Controls:** same transaction where possible, domain-unique operation ID, reconciliation from business state, fenced execution, and explicit unknown outcome.

### Cursor is moved across tenant scope

~~~text
caller obtains valid cursor for tenant A
-> sends it on tenant B list route
-> server trusts encoded position/filter without scope binding
-> data ordering or identifiers leak across tenant
~~~

**Controls:** authenticated cursor envelope bound to tenant, principal scope, operation, filter, sort, projection, snapshot, expiry, and key generation.

### Live pagination duplicates or omits rows

~~~text
client reads page 1 by mutable score
-> scores change and new rows arrive
-> page 2 starts from old position
-> rows move across boundary
~~~

**Controls:** immutable/stable total order, snapshot watermark or explicitly best-effort contract, and cursor invalidation when ordering semantics change.

### Idempotency hides a lost update

~~~text
two clients submit different updates with different idempotency keys
-> both succeed against stale version
-> last writer silently removes first writer’s change
~~~

**Controls:** strong version/ETag precondition in addition to idempotency.

### Long operation reports cancellation too early

~~~text
cancel request sets status cancelled
-> worker has already crossed external commit point
-> effect completes after client was told it would not
~~~

**Controls:** cancel-requested intermediate state, fenced worker acknowledgement, effect reconciliation, and truthful terminal semantics.

### Webhook retry storm targets a recovering customer

~~~text
subscriber endpoint times out
-> many events retry on synchronized schedule
-> endpoint recovers into a backlog burst
-> new and old deliveries starve each other
~~~

**Controls:** exponential backoff with jitter, per-endpoint concurrency/rate, fair queue, retry-after support, backlog admission, and replay controls.

### Webhook signature covers the wrong representation

~~~text
producer signs JSON object before serialization
-> intermediary changes whitespace/content encoding
-> receiver canonicalizes differently or verifies only body
-> valid messages fail or unsigned method/path can be replayed
~~~

**Controls:** specified byte representation and covered components, content digest, standard message-signature algorithm, key ID, timestamp, and receiver conformance fixtures.

### “Additive” enum change crashes an old client

~~~text
server adds status=PAUSED
-> old generated client maps unknown to impossible branch
-> parsing succeeds but application crashes or treats as ACTIVE
~~~

**Controls:** unknown enum behavior in contract, compatibility fixture with old clients, staged exposure, and behavior-version fallback only when defined.

### Deprecated version is removed based on incomplete metrics

~~~text
dashboard shows no traffic for one week
-> monthly billing job and disaster-region client were absent
-> old route removed -> next scheduled/failover run fails
~~~

**Controls:** authenticated consumer inventory, observation window covering periodic/failover use, direct owner confirmation, deprecation metadata, exception tracking, and reversible disable phase.

## Observability and Audit

### Per request/operation

- request/trace ID, authenticated client class, tenant, and API behavior version;
- operation/route and schema revision;
- resource identity under redaction policy;
- idempotency state and replay indicator without logging the raw secret key;
- precondition supplied/current version and conflict outcome;
- cursor version/snapshot and verification failure class;
- long-operation identity, state transition, worker attempt, and cancellation point;
- stable problem type and retry guidance; and
- latency, bytes, and admission cost.

### Contract health

- traffic and errors by API version and authenticated consumer;
- unknown-field/enum observations where the codec exposes them;
- deprecated feature use by consumer and last-seen time;
- idempotency reservations, replays, fingerprint conflicts, unknown outcomes, and reconciliation age;
- precondition usage and failed-precondition rate;
- pagination depth, page size, cursor invalid/expired/scope mismatch, and snapshot expiry;
- operation age, polls per operation, stuck state, cancellation latency, and terminal retention;
- webhook event fan-out, queue age, attempts, signature/key generation, disable, DLQ, and replay; and
- schema/behavior candidate deltas during rollout.

High-cardinality IDs belong in secured traces/logs, not metric labels. Never log raw authorization credentials, idempotency keys, cursor plaintext, webhook secrets, or unredacted signed payloads by default.

## Verification Strategy

| Test layer | What to verify |
|---|---|
| Schema conformance | Request/response examples, bounds, presence/null/default, unknown fields, and media types |
| Semantic contract | State transitions, method safety/idempotency, error type, and authorization outcome |
| Old/new compatibility | Old client/new server, new client/old server, rollback after new data, and adapters |
| Idempotency fault tests | Lost response, crash at every state transition, concurrent duplicates, fingerprint conflict, and cross-region failover |
| Concurrency tests | Competing updates, stale delete, create-if-absent, aggregate versions, and merge behavior |
| Pagination model tests | Inserts, deletes, ties, sort changes, snapshot expiry, cursor tampering, scope substitution, and key rotation |
| Long-operation tests | Duplicate create, lease loss, worker retry, cancellation race, reconciliation, polling, and retention |
| Webhook tests | Outbox atomicity, SSRF/rebinding, signature bytes, key rotation, duplicate/out-of-order, retry, disable, DLQ, and replay |
| Evolution diff | Schema plus behavior, default, validation, ordering, error, authorization, and performance changes |
| Consumer replay | Recorded redacted requests and golden old-client fixtures against candidate versions |
| Load tests | Deep traversal, hot idempotency key, poll storm, large operation backlog, webhook fan-out, and failing endpoint |
| Security tests | Over-posting, hidden-resource oracle, cursor/key replay, signature confusion, tenant crossover, and data leakage |

Property examples:

~~~text
same scoped idempotency key + same semantic fingerprint
  => at most one committed logical effect

successful If-Match(version=v)
  => mutation linearizes only while current version equals v

valid cursor
  => tenant, principal scope, operation, query, sort, snapshot, and expiry match

webhook event accepted twice
  => receiver-visible domain effect is deduplicated by published event identity
~~~

## Decision Framework

### For each operation

1. What stable resource or domain transition does it represent?
2. Is it safe, idempotent, or unsafe with an idempotency contract?
3. What version/precondition prevents lost updates?
4. Which stable problem types can occur, and which are retryable?
5. What authorization depends on path, field, or current domain state?
6. Can it complete within one deadline; if not, what durable operation is created?
7. What are request, response, collection, and compute bounds?
8. How will old clients interpret new fields, enum values, defaults, and errors?
9. Which telemetry proves consumer behavior and migration readiness?
10. Can the server and data schema roll back after the change?

### For collections

Choose:

- live versus snapshot traversal;
- stable total order and index;
- opaque signed/encrypted versus server-side cursor;
- cursor lifetime and key rotation;
- principal/query/sort/projection binding;
- page/count limits; and
- restart behavior after expiry or incompatible migration.

### For asynchronous notification

Choose:

- polling, webhook, event stream, or combination;
- acceptance and terminal state model;
- idempotency and cancellation point;
- delivery guarantee and ordering scope;
- receiver authentication and message signature;
- retry/fairness/disable/replay contract; and
- retention and audit policy.

## Key Takeaways

1. An API contract includes behavior, failure, concurrency, continuation, security, operations, and evolution—not only schema.
2. Model explicit domain transitions when CRUD would hide authorization or side effects.
3. Stable problem types let clients branch safely while human details evolve.
4. Idempotency keys require scoped atomic records and reconciliation with the business effect.
5. Preconditions prevent lost updates; idempotency does not.
6. Cursor integrity, confidentiality, expiry, and tenant/query binding are security properties.
7. Long work returns a durable operation whose cancellation and duplicate-execution semantics are explicit.
8. Webhooks are at-least-once delivery systems with SSRF controls, message authentication, fair queues, retries, DLQ, and replay.
9. Additive wire changes can still break behavior, authorization, ordering, clients, or rollback.
10. Safe removal follows expand, instrument, deprecate, migrate, verify, disable, and contract.

---

## References

- [RFC 9110: HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110) — method semantics, conditional requests, status codes, and intermediaries
- [RFC 9457: Problem Details for HTTP APIs](https://www.rfc-editor.org/rfc/rfc9457) — stable machine-readable error details
- [RFC 8288: Web Linking](https://www.rfc-editor.org/rfc/rfc8288) — typed links and continuation relations
- [RFC 9865: Cursor-Based Pagination of SCIM Resources](https://www.rfc-editor.org/rfc/rfc9865) — opaque cursor request/response and query-binding semantics
- [RFC 9421: HTTP Message Signatures](https://www.rfc-editor.org/rfc/rfc9421) — covered components, signature parameters, and verification
- [RFC 9745: Deprecation HTTP Response Header](https://www.rfc-editor.org/rfc/rfc9745) — machine-readable deprecation metadata
- [RFC 8594: Sunset HTTP Header](https://www.rfc-editor.org/rfc/rfc8594) — planned retirement metadata
- [OpenAPI Specification](https://spec.openapis.org/oas/latest.html) — machine-readable HTTP API description
- [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12) — representation validation and vocabulary
- [Protocol Buffers: Updating a Message Type](https://protobuf.dev/programming-guides/proto3/#updating) — wire-compatible field evolution
- [CloudEvents Specification](https://github.com/cloudevents/spec) — interoperable event envelope attributes
- [Idempotency](../01-foundations/08-idempotency.md) — atomic deduplication, storage, TTL, and failure recovery
- [Transactional Outbox](../05-messaging/07-outbox-pattern.md) — atomic event publication after a domain transaction
- [Database Migrations](../15-deployment/03-database-migrations.md) — expand/migrate/contract across service and storage schemas
- [Edge Gateway](./02-api-gateway.md) — public routing, authentication, authorization, admission, and aggregation
