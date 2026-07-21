# Retries, Timeouts, and Hedging: Deadline and Attempt Budgets

## TL;DR

Propagate one end-to-end deadline. Each hop decides whether enough budget remains to queue, connect, execute, retry, serialize, and return; work that cannot finish usefully should not start. A timeout bounds waiting but does **not** reveal whether a remote side effect committed.

Retry only a classified transient outcome when operation semantics make another attempt safe. Assign one layer as retry owner, use exponential backoff with jitter, and cap aggregate retry traffic with a budget, not only attempts per request. Hedging races an idempotent request against a distinct replica after a delay; its benefit depends on uncorrelated tails, prompt cancellation, and spare capacity. Disable or budget extra attempts under overload.

Deadlines bound time, retry/hedge budgets bound amplification, [rate limiting](./05-rate-limiting.md) bounds admission, [circuit breakers](./06-circuit-breakers.md) bound dependency state/concurrency, and [backpressure](./07-backpressure.md) bounds waiting. No one mechanism replaces the others.

---

## 1. Attempt Contract

| Field | Required answer |
|---|---|
| **User objective** | End-to-end latency/deadline and required success percentile. |
| **Operation semantics** | Read, idempotent write, idempotency-keyed effect, commutative update, or non-repeatable effect. |
| **Attempt phases** | Resolve, connect, handshake, acquire pool/queue, send, execute, receive, deserialize. |
| **Outcome classes** | Definitely not sent, ambiguous, retryable overload/transient, permanent, caller cancellation, and deadline exhausted. |
| **Retry owner** | Exactly one layer that has semantic context and remaining deadline. |
| **Attempt policy** | Maximum attempts, per-attempt bounds, backoff/jitter, endpoint diversity, and retryable results. |
| **Aggregate budget** | Retry and hedge concurrency/rate allowed per caller/dependency/priority. |
| **Hedge policy** | Eligible operations, trigger delay/signal, alternate placement, winner, and loser cancellation. |
| **Server behavior** | Deadline admission, cancellation support, idempotency storage, and overload guidance. |

### Invariants

1. No new attempt starts without enough remaining end-to-end budget to be useful.
2. A downstream receives one authoritative deadline/remaining budget, not a fresh full timeout per hop.
3. An ambiguous write is repeated only through an idempotency or reconciliation contract.
4. Retry amplification is bounded across the entire call graph.
5. Overload responses do not trigger immediate synchronized retries.
6. A hedge targets a different failure domain and its loser is canceled/released promptly.
7. Attempt work after caller cancellation/deadline is visible and bounded.
8. Metrics distinguish original operations, attempts, useful completions, and late/duplicate effects.

---

## 2. Data Plane and Control Plane

~~~mermaid
flowchart LR
    subgraph DP["Attempt data plane"]
        R["Logical operation<br/>deadline + idempotency"]
        A["Attempt admission<br/>remaining time + budget"]
        C["Concurrency / breaker"]
        D["Dependency replica"]
        W["Winner / terminal result"]
        X["Cancel loser and<br/>release resources"]
        R --> A --> C --> D
        D --> W
        A -->|delayed hedge/retry| C
        W --> X
    end

    subgraph CP["Policy and evidence"]
        P[("Versioned attempt policy")]
        B["Retry/hedge budget<br/>tokens + concurrency"]
        M["Latency/outcome model<br/>by operation and endpoint"]
        K["Runtime kill switch"]
        P --> B
        M --> P
        K --> P
    end

    P -.snapshot.-> A
    B -.permit.-> A
    D -.outcome + phases.-> M
~~~

Attempt decisions must remain local/fast from a policy snapshot. A synchronous policy lookup on the failure path creates another call that needs timeouts and retries.

---

## 3. Deadlines, Not Stacked Timeouts

A **timeout** bounds one wait or phase. A **deadline** bounds the logical operation end-to-end.

Suppose an edge has deadline <code>D</code> and calls services sequentially. If each service receives a fresh <code>D</code>, downstream work can continue long after the edge has returned failure. Instead propagate absolute deadline or remaining duration, accounting for clock/transport semantics. Use monotonic time within a process; do not compare unsynchronized wall clocks without a protocol designed for it.

At each hop:

1. Compute remaining budget.
2. Reserve return/serialization/network margin based on measurement.
3. Reject before queueing if useful completion is implausible.
4. Cap queue, connect, request, and per-attempt waits by remaining budget.
5. Propagate the reduced budget.
6. Cancel downstream when the caller abandons the operation.

### Sequential and parallel composition

For sequential phases:

> end-to-end latency = sum of phase/queue/call latencies on the critical path

For parallel fan-out:

> branch completion latency = maximum required branch latency + coordination overhead

Optional branches should have smaller budgets or be shed/degraded rather than consuming the whole user deadline.

### Per-attempt timeout derivation

Use phase-level production distributions under representative load:

- resolver/cache miss;
- pool acquisition;
- connection and TLS handshake;
- server queue and execution;
- response transfer/deserialize.

Choose a tolerated false-timeout probability and reserve enough time for any permitted later attempt plus backoff. A percentile copied from another service is not a derivation. Tight latency distributions may require margin for network change, deployment, and cold connection; wide distributions may need workload separation rather than a huge timeout.

Connection establishment often deserves a separate bound because failure and remediation differ from server execution. Ensure language/runtime timeouts actually cover DNS, connect, TLS, writes, reads, and pool waits as intended.

---

## 4. Timeout Does Not Mean Failure before Commit

Trace:

1. Client sends a payment request.
2. Server commits the charge.
3. Response is lost or arrives after client timeout.
4. Client sees <code>DeadlineExceeded</code>.

The outcome is **ambiguous**, not “failed.” Retrying with a new identity can charge twice.

Safe patterns:

- operation is naturally idempotent;
- client supplies a stable idempotency key and request fingerprint;
- server atomically stores effect plus idempotency result;
- caller queries operation status by stable ID;
- workflow records intent/outbox and reconciles.

The idempotency record must outlive the maximum client, queue, replay, and disaster-recovery retry horizon. See [Idempotency](../01-foundations/08-idempotency.md) for the canonical protocol.

Cancellation also does not undo a committed side effect. Servers should stop cancellable computation but complete or reconcile irreversible commit protocols safely.

---

## 5. Retry Classification

Classify from transport position, protocol status, operation semantics, and server guidance:

| Outcome | Default reasoning |
|---|---|
| Invalid/auth/permission/not-found/domain validation | Permanent until request or authority changes; do not blind retry. |
| Optimistic conflict | Retry only after re-read/replan under the domain’s conflict protocol. |
| Explicit rate/overload rejection | Honor guidance and budget; another immediate attempt usually worsens overload. |
| Connect refused/no route before send | May be safe to try another endpoint if deadline and policy permit. |
| Reset/timeout after send | Outcome may be ambiguous; writes need idempotency/status. |
| Server transient/unavailable | Retry only if server contract, capacity, deadline, and idempotency permit. |
| Caller cancellation/deadline | Do not start a new attempt. |
| Partial/streaming response | Resume only with a cursor/range/protocol that defines duplicate and ordering behavior. |

Do not use “all 5xx” or “all exceptions” as the semantic policy. Separate local-origin network failure from remote application response. A server may declare whether an attempt consumed quota or committed work.

### Retry at one layer

If layer <code>i</code> permits <code>a_i</code> total attempts, worst-case attempts at the deepest dependency are:

> maximum leaf attempts = product of a_i across retrying layers

Even modest policies multiply. Choose the layer closest to the user that understands operation semantics and can reselect endpoints while preserving deadline/idempotency. Inner layers should usually return classified failure quickly.

Transparent transport retries are safe only before application-visible commitment according to that transport’s protocol. Inventory SDK, proxy, mesh, load balancer, database driver, and application retries; “the code retries once” may be incomplete.

---

## 6. Backoff and Jitter

Without delay, a retry immediately recreates failed load. Exponential backoff produces a cap-limited window:

> window(k) = min(cap, base × 2^k)

The cap and base come from recovery dynamics and remaining deadline.

Jitter prevents clients aligned by the same failure from returning together:

- **full jitter:** delay uniformly from zero to the exponential window;
- **equal jitter:** retain part of the window plus randomness;
- **decorrelated jitter:** derive the next randomized delay from the prior delay and cap.

Full jitter spreads attempts broadly; variants trade minimum delay, completion time, and correlation. Seed/replay randomness in tests.

The complete schedule must fit:

> sum of attempt durations + backoff delays + return margin ≤ remaining deadline

Stop before the configured attempt count when the deadline cannot fund another useful attempt.

Honor server retry guidance as a lower-bound or policy input according to the API, then add client-side jitter and check the deadline. Never assume every client will obey guidance; the server still needs admission protection.

---

## 7. Aggregate Retry Budgets

Per-operation attempt caps do not bound fleet amplification. Let:

- <code>lambda_o</code>: admitted original operations/s;
- <code>lambda_r</code>: retry attempts/s;
- <code>beta</code>: allowed retry fraction under the policy.

Enforce:

> lambda_r ≤ beta × lambda_o
>
> total attempt rate ≤ lambda_o × (1 + beta)

Use a token bucket or concurrency pool that earns retry credit from original traffic and spends it on retries. Scope by dependency and priority so a failing optional call cannot consume critical retry credit.

Budget dimensions:

- rate of retry attempts;
- concurrent retry work;
- bytes/weighted cost;
- per-endpoint and aggregate dependency limits;
- regional failover/recovery reserve.

A small floor can help low-volume services recover from isolated failures, but it must be derived and bounded; a floor per process can multiply with fleet size. Prefer regional/cell accounting or leased credit when exact aggregate control matters.

When budget is exhausted, fail the logical operation or degrade; do not queue retries past their deadline.

### Retries and breakers

Every retry must reacquire:

- remaining deadline check;
- retry budget;
- dependency concurrency permit;
- circuit-breaker/probe permission;
- current routing/endpoint selection.

A retry is new load, not a privileged bypass.

---

## 8. Hedging Tail Latency

A hedge sends another equivalent attempt after the first remains incomplete for delay <code>h</code>, usually to a distinct replica/path. Return the first acceptable result and cancel the rest.

For independent identically distributed attempt latency <code>X</code>, two simultaneous attempts have:

> P(min(X1, X2) > t) = P(X > t)^2

Real tails are correlated by shared storage, network, queue, or workload; benefit can be far smaller. A delayed hedge adds an extra attempt only when the original exceeds <code>h</code>, so before cancellation effects:

> expected extra-attempt probability ≈ P(X > h)

Choose <code>h</code> from live per-operation/placement latency and spare capacity, not a universal percentile. Account for measurement lag and changing workload.

### Eligibility

Hedge only when:

- operation and result are idempotent/equivalent;
- distinct placement reduces the likely failure correlation;
- another attempt fits the deadline;
- hedge rate/concurrency budget has capacity;
- loser cancellation propagates to queue, server, and dependency;
- duplicate resource usage and side-channel effects are acceptable.

Do not hedge scarce writes, expensive model generation, long scans, or calls whose duplicate side effect is unsafe merely because the client returns one response.

### Cancellation

After a winner:

1. Signal cancellation/reset to losers.
2. Remove queued attempts.
3. Stop server work at safe cancellation points.
4. Release concurrency, memory, streams, and connection resources.
5. Record whether a loser nevertheless committed/completed.

Client task cancellation without wire/server support does not reclaim backend work.

### Alternatives

- tied requests: enqueue copies, start one, cancel other queued copies when execution begins;
- load-aware replica selection;
- partition-aware request splitting;
- latency-aware admission;
- fixing stragglers, caches, or hot shards.

Hedging treats tail symptoms and spends capacity; prefer root-cause removal when possible.

---

## 9. Metastable Failure and Goodput

An overload loop:

~~~mermaid
flowchart TD
    T["Brief slowdown / capacity loss"]
    L["Queueing and latency rise"]
    O["Timeouts and ambiguous outcomes"]
    R["Retries / hedges / user refresh"]
    W["More attempt work<br/>including abandoned calls"]
    G["Useful goodput falls"]

    T --> L --> O --> R --> W --> L
    W --> G
~~~

The trigger can disappear while the retry backlog and queueing sustain the bad state. Throughput may look high because servers finish work after callers leave.

Break the loop with:

- deadline admission and cancellation;
- aggregate retry/hedge budgets and kill switches;
- bounded queues and shedding;
- dependency concurrency limits/breakers;
- recovery drain controls;
- original-operation versus attempt/goodput telemetry.

Load tests must push beyond saturation, remove the trigger, and prove recovery without manual restart.

---

## 10. Concrete Failure Trace: Layered Retries

1. Edge, service A, and service B each independently allow multiple attempts.
2. The database slows and starts timing out, but continues some work after callers abandon it.
3. B retries each database call; A retries B’s already-amplified operation; edge retries A.
4. Attempt multiplication fills connection pools and queues.
5. Per-hop timeouts restart at each layer, so leaf work continues beyond the user deadline.
6. Original goodput approaches zero while database throughput remains high.
7. The database recovers physically, but queued/retry work keeps it overloaded.

Fix by assigning one retry owner, propagating one deadline, classifying ambiguous writes, enforcing aggregate attempt budgets, canceling abandoned work, and rejecting load before queues. This is a composition failure, not a need for a longer timeout.

---

## 11. Capacity and Cost Model

Let:

- <code>lambda_o</code>: original admitted operations/s;
- <code>beta_r</code>: retry budget fraction;
- <code>beta_h</code>: hedge budget fraction;
- <code>C_attempt</code>: mean resource cost/attempt;
- <code>S_attempt</code>: mean attempt service time;
- <code>p_late</code>: fraction of attempt work completed after logical termination.

Upper budgeted attempt rate:

> lambda_attempt ≤ lambda_o × (1 + beta_r + beta_h)

if budgets are independent and fully spent. Shared budgets can impose a tighter bound.

Expected attempt concurrency is approximately:

> lambda_attempt × S_attempt

plus cancellation propagation delay and queued attempts. Provision dependency concurrency from attempts, not only user QPS.

Wasted resource rate includes:

> late/loser/duplicate attempts × C_attempt

Hedging cost depends on tail probability after <code>h</code> and how quickly losers stop. Measure backend work, not only client task count.

Idempotency storage capacity is operation rate × retention horizon × record bytes, adjusted for replication/indexes. Retention must cover the maximum real replay horizon.

---

## 12. Operations and Migration

### Rollout

1. Inventory all retry/timeout/hedge layers and defaults.
2. Add attempt identity, logical operation ID, deadline, and outcome classification.
3. Propagate deadlines without enforcement and measure remaining budget.
4. Shadow retry decisions and amplification.
5. Establish idempotency for eligible writes.
6. Enable one retry owner with aggregate budget and kill switch.
7. Add cancellation and verify backend work reduction.
8. Canary hedging only on qualified read paths with spare capacity.
9. Remove conflicting legacy/mesh/SDK policies.

Changing timeout can change load: shorter timeouts may increase retries; longer timeouts increase in-flight resource occupancy. Canary against goodput and backend work.

### Runbooks

- retry storm: set attempt budget to zero/critical-only, shed, cancel expired work, then recover gradually;
- dependency gray failure: inspect endpoint/path diversity before hedging/retrying;
- ambiguous writes: query idempotency/status ledger and reconcile, never mass-blind-retry;
- deadline propagation failure: disable inner attempts and restore a safe edge bound;
- cancellation ineffective: cap concurrency and fix server/driver propagation;
- policy-control outage: use last verified bounded snapshot and local kill switch.

---

## 13. Security and Governance

- Authenticate policy distribution and restrict runtime kill-switch access.
- Prevent clients from extending trusted deadlines, attempts, or priority; clamp at the first trusted boundary.
- Bind idempotency key to authenticated subject, operation, and request fingerprint.
- Avoid exposing internal topology or capacity in retry guidance.
- Bound idempotency-key size/cardinality and protect lookup from abuse.
- Audit policy changes, retries of privileged effects, manual replay, and ambiguous-outcome reconciliation.
- Redact sensitive request/response data from attempt logs and idempotency records.
- Do not retry authentication challenges, authorization denials, or security checks through a permissive fallback.

An attacker can exploit retry/hedge multiplication as resource amplification; admission applies to attempts too.

---

## 14. Observability

By logical operation and dependency:

- original operations, total attempts, retry attempts, hedge attempts;
- attempts per operation distribution and amplification ratio;
- remaining deadline at every hop and deadline rejection;
- phase latency: queue/pool, DNS, connect, TLS, server, response;
- classified outcome and whether request bytes may have been sent;
- retry/hedge budget earned, spent, denied, and current concurrency;
- backoff delay and server-guidance compliance;
- hedge trigger, placement diversity, winner, cancel latency, loser backend work;
- idempotency hit/conflict/in-progress/retention expiry;
- completion goodput, late completion, abandoned work, and ambiguous effects;
- breaker/concurrency/rate-limit decisions correlated with attempts.

Do not infer user reliability from per-attempt success. The logical operation is the SLI boundary.

---

## 15. Verification

- prove deadline decreases across sequential and parallel call graphs;
- delay every phase independently, including DNS, pool, connect, TLS, send, server, and read;
- inject failure before send, after send, after commit, and after response;
- verify non-idempotent ambiguous writes are not repeated;
- run every SDK/proxy/mesh/application layer together and count leaf attempts;
- synchronize a fleet failure and verify jitter disperses attempts;
- exhaust retry/hedge budgets and prove ordinary admission remains bounded;
- make server retry guidance conflict with remaining deadline;
- hedge against correlated and independent replicas; measure extra backend work;
- cancel winners/losers at every stage and prove permit/resource release;
- expire idempotency records at the boundary and exercise reconciliation;
- push beyond saturation, remove load, and prove goodput recovers;
- disable policy control plane and operate from safe local snapshots/kill switches.

Use deterministic seeded timing tests plus production-scale stochastic load. Happy-path unit tests do not expose amplification.

---

## 16. Decision Framework

| Situation | Direction |
|---|---|
| No end-to-end latency objective | Define deadline before attempt policy |
| Permanent/domain/security failure | No blind retry |
| Transient pre-send/connect failure | Retry alternate endpoint if budget permits |
| Ambiguous write | Status/idempotency/reconciliation before retry |
| Dependency overload | Honor guidance, shed, and spend retry budget sparingly |
| Rare independent read stragglers with spare capacity | Delayed cancellable hedge |
| Correlated tail or saturated dependency | Hedging likely harms; fix/bound load |
| Multiple layers currently retry | Select one owner and remove multiplication |
| Server cannot cancel loser work | Price full duplicate backend cost |
| Remaining deadline cannot fund another attempt | Fail/degrade immediately |

Retries buy another sample from the failure distribution. Hedge only when a second simultaneous sample is worth its capacity. Neither changes an unsafe operation into an idempotent one.

---

## Primary References

- Jeffrey Dean and Luiz André Barroso, [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/).
- AWS Builders’ Library, [Timeouts, Retries, and Backoff with Jitter](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/).
- Marc Brooker, [Exponential Backoff and Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/).
- gRPC, [A6: Client Retries](https://github.com/grpc/proposal/blob/master/A6-client-retries.md), including hedging, retry throttling, and server pushback.
- Google SRE, [Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/).
- Nathan Bronson et al., [Metastable Failures in Distributed Systems](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s11-bronson.pdf).
