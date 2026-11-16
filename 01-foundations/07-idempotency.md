# Idempotency and Operation Identity

> The payment request timed out. Retry and you may charge twice; give up and you may not charge at all. How does a server tell a retry from a new request?

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

## Semantics and System Model

Networks lose messages, and the only remedy for a lost message is to send it again. The sender, however, cannot always tell a lost request from a lost reply. If a client asks a server to charge a card and hears nothing, the charge may or may not have happened, and the client faces a choice between two bad options. If it does not retry, some payments silently fail. If it does, some customers are charged twice.

Idempotency removes the dilemma by making the second request harmless. If an operation can be applied any number of times with the same effect as applying it once, the client may retry freely, and the question of whether the first attempt succeeded no longer needs an answer. Some operations have this property naturally, such as setting a value or deleting a record. Most interesting ones, such as charging a card or sending a message, do not, and have to be given it by attaching an identity to each operation so that the server can recognize a repeat. Nearly every reliability technique elsewhere in this book, including retries, queue redelivery, and workflow replay, is safe only because of what this chapter describes.

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

The protocol is designed against an environment that we assume to be hostile in ordinary ways. Requests and responses can be lost. A server may commit and then crash before it replies, which leaves the client unable to tell success from failure. Clients, queues, and workflow engines all retry, and their attempts can run concurrently with one another. A worker that was presumed dead can resume and finish what it started. A delayed replay can arrive long after the original, following a failover or a restore from backup. And clocks cannot be used to decide which attempt owns the operation, because no two machines agree on the time closely enough.

### Core invariants

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

## Natural and Keyed Idempotency

### Naturally idempotent state transitions

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

### Resource identity

`PUT /resources/{stable-id}` can be idempotent when the client chooses the resource identity and repeated payloads replace it consistently. `POST /resources` with a server-generated identity is not naturally idempotent; add an operation key or client-provided resource key.

HTTP method semantics are a protocol contract, not a database guarantee. A nominally idempotent `DELETE` can still send duplicate emails or ledger entries if its implementation is not.

### Keyed effects

Non-idempotent effects (charge, shipment, notification, increment, external call) need a logical operation key. Examples:

```text
tenant-4/order-82/payment-intent-1
tenant-4/order-82/confirmation-email/v1
tenant-9/report/2026-07-18
workflow-51/step-reserve-inventory/sequence-3
```

Do not include retry attempt number or random value generated inside each attempt. That turns duplicates into distinct operations.

The rule about where the key is generated is the one most often broken, and usually by accident. Many HTTP client libraries have a retry option, and a developer who generates an idempotency key inside the function that makes the request will find that each retry calls that function again and produces a fresh key. Every attempt then looks to the server like a new operation, and the protection silently disappears. The key has to be created once, at the point where the application decides to perform the operation, and stored or passed down so that every subsequent attempt carries the same value. For an operation that may be retried across a process restart, that means persisting the key before the first attempt.

---

## Key Scope and Request Identity

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

### Canonical request digest

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

The digest is only as good as the canonical form it is computed over. Two requests that mean the same thing must hash to the same value, and two SDKs in different languages must agree on what that value is. The canonicalization rules therefore have to settle field ordering, Unicode normalization, the representation of numbers and decimals, whether an absent field equals a null one, how defaults are applied, which transport-only fields are excluded, and which schema or operation version the request belongs to. Each of these is a place where a retry sent through a different client version could otherwise look like a different operation and be rejected, or worse, where a different operation could look like a retry.

Do not hash raw JSON bytes if semantically equivalent encodings should match. Do not omit a field that changes the effect.

If an existing key has a different digest, return a conflict. Replaying the old outcome would falsely claim that the new parameters executed.

The digest guards against a failure that looks harmless and is not. Suppose a client sends a payment of 40 under some key, and later, because of a bug or a reused variable, sends a payment of 55 under the same key. A server that only checks whether it has seen the key will reply that the operation already succeeded, and the client will record that 55 was charged when in fact 40 was. The server has told the truth about the first request and the client has drawn a false conclusion about the second. Comparing a digest of the request's meaning makes this case detectable, and a conflict response turns a silent accounting error into a visible bug.

---

## Deduplication State Machine

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

### First request

Atomically insert `IN_PROGRESS` if absent. A unique constraint or compare-and-swap elects the owner.

### Concurrent duplicate

The second attempt finds the record in `IN_PROGRESS` and must not start the effect again. What it tells its caller is a policy choice. It can wait or poll for the first attempt's outcome within the caller's deadline. It can return `202 Accepted` with a status URL, or a specific "operation in progress" response that the client knows to retry later. Inside a single process it can join a singleflight future for the first attempt, which is an optimization and not a substitute for the durable record, since the duplicate may land on another process.

Do not immediately run the effect again.

The concurrent duplicate is the case that check-then-act designs get wrong. Two attempts with the same key arrive within milliseconds of each other, typically because a client timed out and retried while the first attempt was still running. If each one checks whether the key has been seen, both find that it has not, and both proceed. The insert of an `IN_PROGRESS` record under a unique constraint is what makes the decision atomic: exactly one insert succeeds, and the other attempt learns that it is the duplicate before it has done anything. What it then does is a product choice. Waiting briefly for the first attempt to finish gives the cleanest experience to the caller, while returning a status the caller can poll keeps server resources free.

### Terminal duplicate

Verify tenant/caller authorization and request digest, then return the stored semantic outcome. The response may need re-encoding for a newer API version; preserve the original business result separately from transient headers.

### Abandoned `IN_PROGRESS`

An owner may crash. Use a lease and monotonically increasing `owner_epoch`; a new attempt claims after expiry. Any local commit accepts only the current epoch. A timeout alone does not prove the old worker stopped, so fencing or effect-level idempotency is still necessary.

This is the same problem that distributed locks face, and it has the same answer. An `IN_PROGRESS` record with a lease is a lock on the operation. If its holder stalls for longer than the lease, another attempt will take over, and the stalled holder may then wake and continue. The epoch number on the record lets the commit step reject work from a holder that has been superseded: the final update succeeds only if the epoch it carries is still the current one. Where the effect is in another system that cannot check epochs, the effect itself has to be idempotent, so that two holders racing to perform it produce one result.

---

## Atomicity With a Local Effect

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

### Wrong order: effect then record

1. effect commits;
2. process crashes;
3. no dedup success exists;
4. retry executes effect again.

### Wrong order: record then effect

1. record marked success;
2. process crashes;
3. effect never occurs;
4. retry returns false success.

Only a shared atomic transaction closes both gaps. Across external systems, use their idempotency contract, an outbox/inbox, or reconciliation; see [Effect Commit Protocols for Workflows](../09-workflow-job-systems/04-retry-idempotency-compensation.md).

The two traces above are mirror images, and together they explain why idempotency cannot be bolted on as a separate check. Whichever of the two writes happens first, there is an instant between them in which a crash leaves the system in a state that lies about what happened. One ordering lies by omission, losing the record that the effect occurred. The other lies by commission, recording an effect that never happened. The gap closes only when the deduplication record and the effect are written by the same transaction, so that they become durable at the same instant. That is why the simplest robust design is often to put the operation key directly on the business row, as a unique column on the ledger entry or the order, so that the database's own constraint enforces the guarantee.

---

## API Contract

Idempotency is a contract between client and server, and it fails when either side has to guess the other's behavior. A write API should therefore document where the key travels (which header or field), its maximum length and allowed character set, and the scope within which it must be unique. It should say which operation types require a key. It should state what happens in each of the three duplicate cases: a key reused with different parameters, a duplicate that arrives while the first attempt is in progress, and a duplicate that arrives after a terminal outcome. It should give the retention horizon and the status-query endpoint, say whether a change of authentication or tenant invalidates replay, and identify which response fields are stable across replays and which are regenerated.

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

### Which failures are cached?

Store deterministic terminal outcomes when repeating cannot change them under the same preconditions. Do not permanently cache transient infrastructure failure merely because the first attempt saw it.

One workable policy sorts failures by whether the operation had taken ownership of the key when they occurred. A validation or authorization failure that happens before ownership is not stored as an operation outcome, so the caller can correct the request and reuse the key. A deterministic domain rejection after ownership is stored, together with the domain version or preconditions it was decided against. A transient dependency failure leaves the operation retryable, or releases ownership safely. An ambiguous timeout from an external system is recorded as `UNKNOWN` and handed to reconciliation, because neither success nor failure can honestly be stored. Success is always stored.

Authorization can change. Always authenticate the duplicate request before returning a stored outcome, and decide whether current authorization is required to reveal it.

The distinction between deterministic and transient failures matters because a stored failure is permanent for the life of the key. If a payment attempt fails because a downstream service was briefly unavailable, and that failure is stored as the outcome, every retry under the same key will be told that the payment failed, and the only way to pay will be to start a new operation with a new key. That defeats the purpose of retrying. A rejection that will recur under the same conditions, such as insufficient funds, is worth storing. A failure that says nothing about the request itself is not.

### Status resource

```text
GET /operations/{key}

state: in_progress | succeeded | failed | unknown
result/reference
created_at
updated_at
```

Authorize this lookup like the underlying resource. A predictable key must not expose another tenant's operation.

---

## Message Consumers and Inbox Transactions

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

Message consumers need the inbox because brokers deliver at least once. A consumer that processes a message and crashes before acknowledging it will receive the message again, and from the broker's side there is no way to know that the processing completed. Recording the event identity in the same transaction as the consumer's own state change converts the redelivery into a no-op: the second attempt finds the event already in the inbox and does nothing. The acknowledgement to the broker comes after that commit, so the worst outcome of a crash is one redundant delivery that the inbox absorbs.

Ordering and idempotency are separate. Deduplication does not repair out-of-order state transitions; use sequence/precondition handling from [Message Ordering](../05-messaging/03-message-ordering.md).

---

## External Effects and Ambiguity

For a remote API with idempotency, pass your stable key and bind parameters. Persist the provider's operation ID/result.

For a remote API without idempotency:

1. commit a durable intent;
2. send a unique business reference;
3. on timeout, query/search provider by that reference;
4. reconcile callbacks/events;
5. retry only when evidence says no effect exists;
6. use manual repair if existence cannot be determined.

An application-side “processed keys” table cannot atomically cover a payment provider. Marking local completion before or after the call recreates the gap.

For message publication, use a transactional outbox. For local consumers, use an inbox. The canonical design is [Transactional Outbox, Inbox, and CDC Publication](../05-messaging/05-outbox-pattern.md).

---

## Retention and Expiry

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

The guarantee ends when the record does, so the contract has to say what happens after expiry. There are four honest answers. The key may be treated as new, which means a sufficiently late retry executes again. The request may be required to use a new business epoch. The API may reject keys older than a timestamp embedded in or associated with the key, which turns a silent duplicate into a visible error. Or the provider may offer a durable operation identity that outlives the hot deduplication tier.

Separate hot outcome response from long-lived identity. Terminal records can compact to key, digest, outcome code/reference, and audit metadata.

The guarantee lasts exactly as long as the record does, and this is the most common place where a correct design is undermined by an operational decision made later. A team sets a one-day expiry on idempotency records to save storage. Months later another team configures the message broker to retain messages for a week so that consumers can recover from long outages. Neither change is wrong on its own. Together they mean that a message redelivered on the third day finds no record and is processed again. The retention horizon therefore has to be derived from the longest path by which a request can be repeated, and it has to be revisited whenever any of those paths changes.

### Cleanup race

Cleanup must not delete a record while an attempt can still commit. Use `retention_until` after terminal state, partition lifecycle, and compare state/epoch during deletion. Coordinate with backups: restoring a database snapshot without more recent dedup records can resurrect repeatable effects.

---

## Multi-Region Design

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

The reason is the same one that makes multi-leader replication produce conflicts. Uniqueness is a property that requires knowing about every other attempt, and two regions that replicate asynchronously do not know about each other's recent writes. If a retry is routed to a different region from the original attempt, which is exactly what happens when the first region is slow or failing, both regions check their local copy, both find the key absent, and both execute. Global uniqueness requires either that every attempt for a key goes to one place, or that the decision is made by a store that is strongly consistent across regions, or that the final effect is deduplicated by a system that sees every attempt.

During disaster recovery, restore business state, operation records, outbox/inbox state, and external outcome references as one consistency set. Replay should query known outcomes before reissuing effects.

---

## Capacity and Storage

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

## Security and Privacy

A deduplication store holds stored responses indexed by a caller-supplied string, which makes it an attractive target. The internal scope of a key is derived from the authenticated tenant and client and never from caller input alone, and a lookup that crosses tenants is rejected. Authorization runs before a stored outcome is returned, because the caller's access may have changed since the outcome was recorded. A client must not be able to probe whether another operation's key exists.

The store also needs protection as a resource. We rate-limit key creation and polling, and cap key and payload size, so that it cannot be filled or used as free storage. Keys should not contain secrets or personal data; opaque random keys can be hashed before storage where that is appropriate, sensitive response records are encrypted, and keys and digests are redacted from broadly readable logs when they could be correlated with a customer. Manual overrides of an outcome are audited, since an override is a statement that money did or did not move.

An idempotency key is not an authentication credential. Possessing it should not grant result access or authority to repeat an action under another principal.

Parameter binding must include security-relevant fields: tenant, account, amount, destination, privilege context, and operation version. Omitting destination can replay a stored “success” for the wrong recipient.

---

## Failure Traces

### New key per retry

1. client times out after payment commits;
2. retry library generates a new UUID;
3. provider sees a new operation;
4. customer is charged twice.

**Prevention:** generate key once per logical operation above retry loop.

### Check-then-act race

1. two workers query `seen(key)` and both see false;
2. both execute;
3. both insert completion.

**Prevention:** unique insert/transaction or atomic compare-and-swap before effect.

### Key reused with new amount

1. first request under key charges 40.
2. caller changes amount to 55 but reuses key.
3. server returns prior 40 success.
4. caller records 55 as charged.

**Prevention:** canonical request digest conflict.

### `IN_PROGRESS` treated as done

1. owner crashes before effect.
2. duplicate sees record exists and returns success.
3. effect is lost.

**Prevention:** explicit state and reclaim/reconciliation protocol.

### Stale owner overwrites terminal result

1. epoch 7 pauses.
2. epoch 8 reclaims and succeeds.
3. epoch 7 wakes and writes failure.

**Prevention:** fenced epoch and monotonic terminal state.

### Dedup expires before replay

1. broker retains messages seven days.
2. dedup records expire after one day.
3. delayed redelivery executes again.

**Prevention:** align retention horizons.

### Backup restore repeats external effect

1. application database restores to yesterday.
2. provider still contains today's successful charge.
3. restored local dedup state lacks it.
4. workflow replays and charges again.

**Prevention:** provider query/reconciliation and DR-consistent outcome state.

### Cross-tenant response leak

1. cache key is caller-provided key only.
2. tenant B guesses tenant A's key.
3. server returns A's stored result before authorization.

**Prevention:** internal tenant/client scope and authorization-first replay.

---

## Observability and Repair

The health of an idempotency layer shows in the relationship between logical operations and attempts, so we count both, along with the duplicate and replay rate. A rising rate of key-parameter conflicts usually means a client bug. Time spent in `IN_PROGRESS` and `UNKNOWN`, and the number of reclaim and fencing events, show whether owners are dying mid-operation. We track terminal outcomes and the response-replay rate, the latency and saturation of deduplication lookups and writes, and storage growth against retention cleanup. For external effects we watch the reconciliation backlog and its age. Three further counters are worth alerting on because each indicates a hole in the guarantee: repeats that arrive after their key has expired, cross-tenant or authorization rejections on replay, and manual repairs.

Use operation key/digest in traces/logs under controlled cardinality and privacy rules, not metric labels.

Operators need a small set of supported repair operations, so that resolving a stuck operation never requires editing the database by hand. They need to query an operation by internal key, business entity, or external ID, and to attach proof from the provider. They need to move an `UNKNOWN` operation to a verified terminal outcome, reclaim abandoned work under a new epoch, issue a new operation key that is linked to the prior one when the work must be redone, and quarantine records that are inconsistent.

Never delete a dedup row simply to “retry.” That erases the safety boundary.

---

## Verification

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

## Decision Framework

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

## References

- [RFC 9110: HTTP Semantics, Idempotent Methods](https://www.rfc-editor.org/rfc/rfc9110#section-9.2.2)
- [Amazon Builders' Library: Making Retries Safe with Idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)
- [Stripe API: Idempotent Requests](https://docs.stripe.com/api/idempotent_requests)
- [PostgreSQL: Constraints](https://www.postgresql.org/docs/current/ddl-constraints.html)
- [Kleppmann: Designing Data-Intensive Applications, Transactions and Distributed Systems](https://dataintensive.net/)

---

## Related Chapters

- [Delivery Guarantees and Effect Boundaries](../05-messaging/04-delivery-guarantees.md): at-least-once delivery is why every consumer needs a convergence rule rather than a deduplication trick
- [Transactional Outbox, Inbox, and CDC Publication](../05-messaging/05-outbox-pattern.md): the outbox turns a split write and publish into one durable intent with an idempotent applier
- [Effect Commit Protocols for Workflows](../09-workflow-job-systems/04-retry-idempotency-compensation.md): where the per-operation contract becomes a multi-step protocol with compensation and human escalation
- [Retries, Timeouts, and Hedging](../06-scaling/09-retries-timeouts-hedging.md): hedged requests make duplicates a designed-in success, so convergence must hold under concurrency
- [Distributed Locks and Leases](08-distributed-locks.md): a lock is the alternative to idempotency, and usually the worse one across regions and partitions
- [Stripe](../21-case-studies/06-stripe.md): idempotency keys shipped as a public API contract, with the duplicate-callback trace attached
