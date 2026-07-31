# Idempotency and Operation Identity

## TL;DR

Idempotency makes repeated attempts of one logical operation converge to one effect and one compatible result. It is not “ignore duplicate messages,” and it is not achieved by checking a cache before doing work.

A production protocol needs:

- a stable operation identity reused by every attempt;
- a scope that includes tenant, caller, endpoint/effect, and business epoch;
- a canonical request digest so the same key cannot mean two operations;
- an atomic boundary between deduplication state and the owned effect;
- explicit `IN_PROGRESS`, terminal, conflict, and expired states;
- a retention horizon at least as long as every possible retry/replay;
- fencing or downstream idempotency across non-transactional boundaries;
- query, reconciliation, and repair for ambiguous outcomes.

Idempotency does not make execution literally once. It makes retries, redelivery, failover, and replay safe within a declared scope and time horizon.

---

## 1. Semantics and System Model

Mathematically, an operation `f` is idempotent when:

```text
f(f(x)) = f(x)
```

For distributed effects, the useful contract is:

```text
execute(operation_key, semantic_request)
  -> one accepted effect
  -> same compatible outcome for duplicate attempts
```

Assume:

- requests and responses can be lost;
- a server may commit then crash before replying;
- clients, queues, and workflow engines retry;
- attempts can execute concurrently;
- stale workers can resume;
- delayed replay can occur after failover or restore;
- clocks cannot establish global ownership.

### 1.1 Core invariants

1. **Stable key:** all attempts of one logical operation use the same key.
2. **Unique scope:** the same key cannot collide across tenants, callers, operations, or business epochs.
3. **Parameter binding:** a key reused with a different semantic request is rejected.
4. **Atomic owned effect:** dedup state and any local effect commit together.
5. **Single terminal outcome:** terminal success/failure is monotonic.
6. **Concurrent convergence:** duplicate attempts do not execute unbounded parallel effects.
7. **Bounded guarantee:** retention is explicit and covers all valid repeats.
8. **No false completion:** `IN_PROGRESS` is not interpreted as success.
9. **Replay authorization:** returning a stored outcome does not bypass current tenant/resource access checks.
10. **Repairability:** ambiguous/stuck operations have a query and reconciliation path.

---

## 2. Natural and Keyed Idempotency

### 2.1 Naturally idempotent state transitions

Prefer setting a desired state over applying an unbounded delta:

```text
SET subscription_status = 'cancelled'  # naturally convergent
increment balance by -50               # repeats change state again
```

Conditional state machines can be idempotent:

```text
UPDATE orders
SET status = 'shipped', shipped_at = ?
WHERE order_id = ?
  AND status = 'paid'
```

A duplicate observes `shipped` and returns the stored transition result. The transition must still distinguish “already shipped by this operation” from “shipped by a different operation with incompatible parameters.”

### 2.2 Resource identity

`PUT /resources/{stable-id}` can be idempotent when the client chooses the resource identity and repeated payloads replace it consistently. `POST /resources` with a server-generated identity is not naturally idempotent; add an operation key or client-provided resource key.

HTTP method semantics are a protocol contract, not a database guarantee. A nominally idempotent `DELETE` can still send duplicate emails or ledger entries if its implementation is not.

### 2.3 Keyed effects

Non-idempotent effects (charge, shipment, notification, increment, external call) need a logical operation key. Examples:

```text
tenant-4/order-82/payment-intent-1
tenant-4/order-82/confirmation-email/v1
tenant-9/report/2026-07-18
workflow-51/step-reserve-inventory/sequence-3
```

Do not include retry attempt number or random value generated inside each attempt. That turns duplicates into distinct operations.

---

## 3. Key Scope and Request Identity

A raw caller string is not globally unique. Construct internal identity:

```text
internal_key = hash(
  tenant_id,
  authenticated_client_id,
  operation_namespace,
  caller_key
)
```

Include a business epoch when the same entity can legitimately undergo the operation again. `cancel-subscription/{subscription_id}` may be sufficient if cancellation is terminal; `charge/{order_id}` is insufficient if an order supports multiple payment attempts.

### 3.1 Canonical request digest

Bind the key to semantic input:

```text
request_digest = SHA-256(
  canonical_encode(
    tenant,
    operation_version,
    resource_id,
    amount,
    currency,
    destination,
    relevant_preconditions
  )
)
```

Canonicalization defines:

- field ordering;
- Unicode normalization;
- number/decimal representation;
- absent versus null;
- defaults;
- excluded transport-only fields;
- schema/operation version.

Do not hash raw JSON bytes if semantically equivalent encodings should match. Do not omit a field that changes the effect.

If an existing key has a different digest, return a conflict. Replaying the old outcome would falsely claim that the new parameters executed.

---

## 4. Deduplication State Machine

```text
ABSENT -> IN_PROGRESS -> SUCCEEDED
                    -> FAILED_FINAL
                    -> UNKNOWN

IN_PROGRESS -> EXPIRED/RECLAIMABLE
UNKNOWN -> RECONCILING -> SUCCEEDED | FAILED_FINAL | MANUAL_REPAIR
```

Record:

```text
idempotency_record:
  internal_key
  tenant_id
  operation_namespace
  request_digest
  state
  owner_epoch
  created_at
  lease_expires_at
  completed_at
  response_status
  response_schema_version
  response_ref_or_digest
  external_operation_id
  retention_until
```

### 4.1 First request

Atomically insert `IN_PROGRESS` if absent. A unique constraint or compare-and-swap elects the owner.

### 4.2 Concurrent duplicate

Policy options:

- wait/poll for the first outcome within the caller deadline;
- return `202 Accepted` plus status URL;
- return a specific “operation in progress” response;
- join a singleflight future inside one process as an optimization.

Do not immediately run the effect again.

### 4.3 Terminal duplicate

Verify tenant/caller authorization and request digest, then return the stored semantic outcome. The response may need re-encoding for a newer API version; preserve the original business result separately from transient headers.

### 4.4 Abandoned `IN_PROGRESS`

An owner may crash. Use a lease and monotonically increasing `owner_epoch`; a new attempt claims after expiry. Any local commit accepts only the current epoch. A timeout alone does not prove the old worker stopped, so fencing or effect-level idempotency is still necessary.

---

## 5. Atomicity With a Local Effect

When dedup record and effect share a database:

```text
BEGIN
  INSERT operation(internal_key, digest, state='IN_PROGRESS')
    ON CONFLICT -> load and verify

  apply business mutation guarded by internal_key

  UPDATE operation
    SET state='SUCCEEDED', response_ref=...
    WHERE internal_key=? AND owner_epoch=?
COMMIT
```

The unique operation identity can be embedded directly in a ledger/event row. A separate dedup table is not mandatory if the business table enforces the same invariant and stores enough result state.

### 5.1 Wrong order: effect then record

1. effect commits;
2. process crashes;
3. no dedup success exists;
4. retry executes effect again.

### 5.2 Wrong order: record then effect

1. record marked success;
2. process crashes;
3. effect never occurs;
4. retry returns false success.

Only a shared atomic transaction closes both gaps. Across external systems, use their idempotency contract, an outbox/inbox, or reconciliation; see [Effect Commit Protocols for Workflows](../18-workflow-job-systems/06-retry-idempotency-compensation.md).

---

## 6. API Contract

A write API should document:

- header/field carrying the key;
- maximum key length and allowed character set;
- uniqueness scope;
- operation types requiring it;
- parameter-reuse conflict behavior;
- concurrent in-progress behavior;
- terminal replay behavior;
- retention horizon;
- status-query endpoint;
- whether authentication/tenant changes invalidate replay;
- response fields that are stable versus regenerated.

Example:

```text
POST /payments
Idempotency-Key: order-82-payment-1

201 Created        first success
201 Created        compatible replay of success
409 Conflict       same key, different semantic request
202 Accepted       original attempt is still in progress
422/4xx            deterministic final rejection, if contract stores it
```

### 6.1 Which failures are cached?

Store deterministic terminal outcomes when repeating cannot change them under the same preconditions. Do not permanently cache transient infrastructure failure merely because the first attempt saw it.

Possible policy:

- validation/auth failure before operation ownership: not stored as operation outcome;
- deterministic domain rejection after ownership: store with domain version/preconditions;
- transient dependency failure: keep retryable or release ownership safely;
- ambiguous external timeout: `UNKNOWN`, reconcile;
- success: store.

Authorization can change. Always authenticate the duplicate request before returning a stored outcome, and decide whether current authorization is required to reveal it.

### 6.2 Status resource

```text
GET /operations/{key}

state: in_progress | succeeded | failed | unknown
result/reference
created_at
updated_at
```

Authorize this lookup like the underlying resource. A predictable key must not expose another tenant's operation.

---

## 7. Message Consumers and Inbox Transactions

Broker message IDs may identify deliveries, not business operations. Redelivery after republish or across topics can carry a new message ID. Prefer a producer-defined event/operation identity.

Consumer transaction:

```text
BEGIN
  INSERT inbox(consumer, event_id, digest)
    ON CONFLICT -> verify and return stored outcome

  apply local projection/effect
  append outgoing outbox events
  mark inbox complete
COMMIT

ack broker after commit
```

If the broker redelivers before ack, the inbox detects the committed event. If the database transaction fails, the broker redelivery retries.

Inbox scope includes the logical consumer/effect. Two independent projections may both legitimately process one event; a global `event_id` unique constraint across all consumers would suppress valid work.

Ordering and idempotency are separate. Deduplication does not repair out-of-order state transitions; use sequence/precondition handling from [Message Ordering](../05-messaging/03-message-ordering.md).

---

## 8. External Effects and Ambiguity

For a remote API with idempotency, pass your stable key and bind parameters. Persist the provider's operation ID/result.

For a remote API without idempotency:

1. commit a durable intent;
2. send a unique business reference;
3. on timeout, query/search provider by that reference;
4. reconcile callbacks/events;
5. retry only when evidence says no effect exists;
6. use manual repair if existence cannot be determined.

An application-side “processed keys” table cannot atomically cover a payment provider. Marking local completion before or after the call recreates the gap.

For message publication, use a transactional outbox. For local consumers, use an inbox. The canonical design is [Transactional Outbox, Inbox, and CDC Publication](../05-messaging/07-outbox-pattern.md).

---

## 9. Retention and Expiry

The guarantee exists only while identity is retained:

```text
retention >= max(
  client retry horizon,
  queue redelivery/retention,
  workflow replay,
  offline operation,
  disaster restore/replay,
  manual repair
) + safety margin
```

Document what happens after expiry:

- key may be treated as new;
- request must use a new business epoch;
- API rejects keys older than a timestamp;
- provider offers durable operation identity beyond the hot dedup tier.

Separate hot outcome response from long-lived identity. Terminal records can compact to key, digest, outcome code/reference, and audit metadata.

### 9.1 Cleanup race

Cleanup must not delete a record while an attempt can still commit. Use `retention_until` after terminal state, partition lifecycle, and compare state/epoch during deletion. Coordinate with backups: restoring a database snapshot without more recent dedup records can resurrect repeatable effects.

---

## 10. Multi-Region Design

Options:

### Home region per operation

Route by tenant/entity/key to one region. Regional store provides uniqueness. Failover transfers authority and restores dedup state before issuing effects.

### Globally consistent operation store

All regions perform compare-and-swap against one logical keyspace. Strong uniqueness; adds write latency and global dependency.

### Downstream-owned global idempotency

Regions may race locally, but the effect provider deduplicates the global key. Still coordinate local response/outcome state.

### Region-scoped identity

Safe only if the business effect itself is region-scoped. Prefixing a global payment key with region makes duplicates more likely, not safer.

Two eventually consistent regional `seen` sets do not guarantee global uniqueness. Both regions can observe absence and execute.

During disaster recovery, restore business state, operation records, outbox/inbox state, and external outcome references as one consistency set. Replay should query known outcomes before reissuing effects.

---

## 11. Capacity and Storage

Assume:

- 25,000 logical write operations per second;
- mean 1.06 attempts per logical operation;
- 1.1 KiB hot outcome record;
- 30-day retention;
- storage/index/replication factor 3.0;
- 2 percent of operations receive at least one duplicate status/read.

Attempt rate:

```text
25,000 * 1.06 = 26,500 attempts/s
```

Raw retained storage:

```text
25,000/s * 86,400 s/day * 30 days * 1.1 KiB
= about 66.4 TiB
```

With factor 3:

```text
about 199 TiB
```

This requires partitioning and compaction. Store large responses in object storage by digest/reference; keep enough immutable semantic result to reproduce the contract.

Hot keys can serialize repeated attempts. That is correct for one logical operation, but an attacker can create contention by replaying it. Rate-limit by authenticated client/tenant and avoid locks held during slow remote calls.

Dedup lookups add a write-path dependency. Provision N-minus-one capacity and define behavior during store outage. For effectful operations, “dedup unavailable, execute anyway” is usually unsafe.

---

## 12. Security and Privacy

- derive internal scope from authenticated tenant/client, not caller input alone;
- authorize before returning stored outcomes;
- reject cross-tenant key lookup;
- rate-limit key creation and polling;
- cap key/payload size;
- hash opaque random keys before storage when appropriate;
- avoid secrets and personal data in keys;
- encrypt sensitive response records;
- redact keys/digests from broad logs if they are correlatable;
- audit manual outcome overrides;
- prevent a client from probing whether another operation key exists.

An idempotency key is not an authentication credential. Possessing it should not grant result access or authority to repeat an action under another principal.

Parameter binding must include security-relevant fields: tenant, account, amount, destination, privilege context, and operation version. Omitting destination can replay a stored “success” for the wrong recipient.

---

## 13. Failure Traces

### 13.1 New key per retry

1. client times out after payment commits;
2. retry library generates a new UUID;
3. provider sees a new operation;
4. customer is charged twice.

**Prevention:** generate key once per logical operation above retry loop.

### 13.2 Check-then-act race

1. two workers query `seen(key)` and both see false;
2. both execute;
3. both insert completion.

**Prevention:** unique insert/transaction or atomic compare-and-swap before effect.

### 13.3 Key reused with new amount

1. first request under key charges 40.
2. caller changes amount to 55 but reuses key.
3. server returns prior 40 success.
4. caller records 55 as charged.

**Prevention:** canonical request digest conflict.

### 13.4 `IN_PROGRESS` treated as done

1. owner crashes before effect.
2. duplicate sees record exists and returns success.
3. effect is lost.

**Prevention:** explicit state and reclaim/reconciliation protocol.

### 13.5 Stale owner overwrites terminal result

1. epoch 7 pauses.
2. epoch 8 reclaims and succeeds.
3. epoch 7 wakes and writes failure.

**Prevention:** fenced epoch and monotonic terminal state.

### 13.6 Dedup expires before replay

1. broker retains messages seven days.
2. dedup records expire after one day.
3. delayed redelivery executes again.

**Prevention:** align retention horizons.

### 13.7 Backup restore repeats external effect

1. application database restores to yesterday.
2. provider still contains today's successful charge.
3. restored local dedup state lacks it.
4. workflow replays and charges again.

**Prevention:** provider query/reconciliation and DR-consistent outcome state.

### 13.8 Cross-tenant response leak

1. cache key is caller-provided key only.
2. tenant B guesses tenant A's key.
3. server returns A's stored result before authorization.

**Prevention:** internal tenant/client scope and authorization-first replay.

---

## 14. Observability and Repair

Track:

- logical operations and attempts;
- duplicate/replay rate;
- key-parameter conflicts;
- time in `IN_PROGRESS` and `UNKNOWN`;
- reclaim/fencing events;
- terminal outcome and response-replay rate;
- dedup lookup/write latency and saturation;
- storage growth/retention cleanup;
- external reconciliation backlog and age;
- expired-key repeats;
- cross-tenant/auth rejection;
- manual repair count.

Use operation key/digest in traces/logs under controlled cardinality and privacy rules, not metric labels.

Repair operations:

- query by internal key/business entity/external ID;
- attach provider proof;
- transition `UNKNOWN` to verified terminal outcome;
- reclaim abandoned work with new epoch;
- issue a genuinely new operation key linked to prior one;
- quarantine inconsistent records.

Never delete a dedup row simply to “retry.” That erases the safety boundary.

---

## 15. Verification

1. **Property tests:** same key/input converges; different input conflicts.
2. **Concurrency tests:** many simultaneous attempts produce one local effect.
3. **Crash injection:** before/after ownership, effect, outcome, response, and ack.
4. **Lease tests:** stale owner cannot commit after reclaim.
5. **Canonicalization vectors:** identical semantics across languages/SDKs.
6. **Retention tests:** replay near/beyond expiry and after backup restore.
7. **External ambiguity tests:** commit-with-lost-response, delayed callback, status outage.
8. **Message tests:** redelivery, republish with new broker ID, out-of-order event.
9. **Multi-region tests:** partition, concurrent absence, failover, stale replica.
10. **Security tests:** key guessing, cross-tenant reuse, changed principal, result leakage.
11. **Schema tests:** replay old stored outcome to new client/API version.
12. **Repair game day:** resolve a stuck/unknown high-value operation without manual database edits.

Fault injection at commit boundaries is essential. A happy-path duplicate unit test does not exercise ambiguity.

---

## 16. Decision Framework

| Operation | Preferred mechanism |
|---|---|
| Set resource to known state | natural idempotent transition + precondition |
| Create client-addressable resource | stable resource ID / PUT |
| Local database effect | unique operation identity in same transaction |
| Broker consumer effect | transactional inbox + local mutation |
| Publish after local commit | transactional outbox |
| Remote idempotent API | stable key + request digest + status query |
| Remote queryable but non-idempotent API | durable intent + reconciliation |
| Remote irreversible, non-queryable API | redesign/mediate or supervised execution |

Before accepting retries:

1. What is one logical operation?
2. Where is its stable key generated and retained?
3. What tenant/client/operation/epoch scope makes it unique?
4. Which fields bind its semantics?
5. Where is the atomic effect boundary?
6. What does a concurrent duplicate receive?
7. How does abandoned `IN_PROGRESS` recover?
8. How long can any retry, replay, callback, or restore repeat it?
9. Is uniqueness global across every region that can execute?
10. How is stored outcome access authorized?
11. What happens when the dedup store is unavailable?
12. How does an operator reconcile ambiguity?

If these answers are absent, “the endpoint supports idempotency keys” is an interface decoration, not a guarantee.

---

## Primary References

- [RFC 9110: HTTP Semantics, Idempotent Methods](https://www.rfc-editor.org/rfc/rfc9110#section-9.2.2)
- [Amazon Builders' Library: Making Retries Safe with Idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)
- [Stripe API: Idempotent Requests](https://docs.stripe.com/api/idempotent_requests)
- [PostgreSQL: Constraints](https://www.postgresql.org/docs/current/ddl-constraints.html)
- [Kleppmann: Designing Data-Intensive Applications, Transactions and Distributed Systems](https://dataintensive.net/)

---

## Related Chapters

- [Delivery Guarantees and Effect Boundaries](../05-messaging/04-delivery-guarantees.md)
- [Transactional Outbox, Inbox, and CDC Publication](../05-messaging/07-outbox-pattern.md)
- [Effect Commit Protocols for Workflows](../18-workflow-job-systems/06-retry-idempotency-compensation.md)
- [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md)
- [Distributed Locks](./09-distributed-locks.md)
