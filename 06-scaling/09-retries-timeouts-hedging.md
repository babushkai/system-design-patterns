# Retries, Timeouts, and Hedging: Deadline and Attempt Budgets

> A dependency slowed down, every caller retried three times, and a brownout became an outage. How do you retry without becoming the attack?

## TL;DR

We propagate one end-to-end deadline. Each hop decides whether enough of it remains to queue, connect, execute, retry, serialize, and return, and work that cannot finish usefully should not start. A timeout bounds how long a caller waits. It does **not** reveal whether a remote side effect committed.

A retry is appropriate only for an outcome classified as transient, and only when the semantics of the operation make another attempt safe. One layer is designated the retry owner. It uses exponential backoff with jitter, and aggregate retry traffic is capped by a budget in addition to a limit on attempts per request. Hedging races an idempotent request against a second replica after a delay. Its benefit depends on the tails of the two attempts being uncorrelated, on prompt cancellation, and on spare capacity. Under overload, extra attempts are disabled or tightly budgeted.

Deadlines bound time, retry and hedge budgets bound amplification, [rate limiting](./07-rate-limiting.md) bounds admission, [circuit breakers](./10-circuit-breakers.md) bound dependency state and concurrency, and [backpressure](./08-backpressure.md) bounds waiting. None of these replaces the others.

---

## Attempt Contract

Retrying a failed request is the most natural reliability technique there is, and in small doses it works very well. Most failures in a distributed system are transient: a connection was reset, an instance was restarting, a packet was lost. Trying again a moment later succeeds, and the user never knows.

The same technique is also one of the commonest causes of large outages. A retry is additional load, and it is sent at exactly the moment the receiving system has shown it is having trouble. If the trouble is that the system is overloaded, each retry makes the overload worse, which produces more failures, which produce more retries. A mechanism intended to hide small failures becomes one that turns a brief slowdown into a sustained collapse. Nearly everything in this chapter is a way of keeping the first effect while preventing the second.

The contract for attempts answers these questions:

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

## Data Plane and Control Plane

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

The diagram distinguishes a logical operation from its attempts. The user asked for one thing. The system may make several attempts to do it. Every budget, metric, and guarantee in this chapter depends on keeping those two counts separate.

Decisions about attempts are made locally and quickly, from a snapshot of policy. A synchronous lookup of retry policy on the failure path would be one more remote call needing its own timeout and its own retries, made at the moment things are already going wrong.

---

## Deadlines, Not Stacked Timeouts

A **timeout** bounds one wait or one phase. A **deadline** bounds the whole logical operation from end to end. The distinction looks pedantic and turns out to matter a great deal.

Consider a request that passes through three services in sequence, each configured with a sensible timeout of five seconds on its outbound call. The user's client also waits five seconds. Suppose the last service in the chain is slow. The client gives up at five seconds and shows an error. The first service's call to the second, however, started a little after the request arrived and has its own five seconds. The second service's call to the third started later still. Each layer's clock began when that layer started waiting, so the work deep in the chain can continue for several seconds after the user has gone. Under load, a large share of the system's capacity is then spent on requests that nobody is waiting for.

The remedy is to pass the deadline along with the request. The edge decides that the operation must complete by a certain time, and every hop receives that same deadline or the duration remaining. Within a process, monotonic time is used. Wall clocks on different machines are not compared unless the protocol was designed for it, since they may disagree by more than the deadline itself.

At each hop:

1. Compute remaining budget.
2. Reserve return/serialization/network margin based on measurement.
3. Reject before queueing if useful completion is implausible.
4. Cap queue, connect, request, and per-attempt waits by remaining budget.
5. Propagate the reduced budget.
6. Cancel downstream when the caller abandons the operation.

Step 3 is where the saving comes from. A service that receives a request with fifty milliseconds remaining, and knows the work takes two hundred, can refuse at once. Starting the work would consume resources for a result that will arrive too late to be used.

### Sequential and parallel composition

For sequential phases:

> end-to-end latency = sum of phase/queue/call latencies on the critical path

For parallel fan-out:

> branch completion latency = maximum required branch latency + coordination overhead

In a fan-out the slowest required branch determines the total. Optional branches are therefore given smaller budgets, or are shed or degraded, so that a slow optional call cannot consume the whole of the user's deadline.

### Per-attempt timeout derivation

A per-attempt timeout is derived from the measured distribution of each phase under representative load:

- resolver/cache miss;
- pool acquisition;
- connection and TLS handshake;
- server queue and execution;
- response transfer/deserialize.

The choice involves a trade. A timeout set at the ninety-ninth percentile of normal latency will fire falsely on one request in a hundred, each of which becomes a retry and therefore extra load. A timeout set far above normal latency rarely fires falsely and leaves callers waiting a long time on a server that has actually failed. We choose a tolerated false-timeout probability and leave enough time for any permitted later attempt and its backoff. A percentile copied from some other service has not been derived from anything. If the latency distribution is very wide, the better answer is usually to separate the fast and slow workloads, since no single timeout suits both.

Connection establishment generally deserves its own, shorter bound, because its failure means something different from slow execution and the remedy differs. It is also worth verifying what a runtime's timeout setting actually covers. Many libraries have separate and differently defaulted timeouts for DNS, connect, TLS, write, read, and waiting for a pooled connection, and a "request timeout" that omits pool wait can leave callers blocked indefinitely.

---

## Timeout Does Not Mean Failure Before Commit

The most consequential misunderstanding about timeouts concerns what they tell the caller. Consider:

1. Client sends a payment request.
2. Server commits the charge.
3. Response is lost or arrives after client timeout.
4. Client sees <code>DeadlineExceeded</code>.

From the client's side, this is indistinguishable from the case in which the request never arrived. In both the client sent something and heard nothing. The timeout reports that no answer came in time, and it carries no information about what happened at the server. The outcome is **ambiguous**. If the client treats it as failure and sends the payment again as a new request, the customer is charged twice.

The safe patterns all give the server a way to recognize a repeat:

- operation is naturally idempotent;
- client supplies a stable idempotency key and request fingerprint;
- server atomically stores effect plus idempotency result;
- caller queries operation status by stable ID;
- workflow records intent/outbox and reconciles.

With an idempotency key, the client generates a unique identifier for the operation and sends it with every attempt. The server records the key together with the result in the same transaction as the effect. A second request with the same key finds the record and returns the original result without acting again. The record must outlive the longest period over which a retry could arrive, including client retries, queue redelivery, replay, and disaster recovery. See [Idempotency](../01-foundations/07-idempotency.md) for the full protocol.

Cancellation has the same limit. Cancelling a request does not undo a side effect that has already committed. Servers should stop computation that can be cancelled, and should complete or reconcile an irreversible commit safely.

---

## Retry Classification

Whether to retry depends on what happened, and "an error occurred" is not enough to decide. Classification draws on where in the transport the failure arose, the protocol status, the semantics of the operation, and any guidance from the server:

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

The fourth and fifth rows differ only in whether the request had been sent, and that difference decides everything. A connection that was refused never carried the request, so the server cannot have acted on it and trying elsewhere is safe. A connection that was reset after the request was sent may or may not have delivered it.

Policies such as "retry all 5xx" or "retry on any exception" are too coarse to be safe. Failures that originate locally in the network should be distinguished from responses returned by the remote application. A well-designed server will say whether an attempt consumed quota or committed work.

### Retry at one layer

If layer <code>i</code> permits <code>a_i</code> total attempts, the worst case at the deepest dependency is:

> maximum leaf attempts = product of a_i across retrying layers

Three layers that each allow three attempts permit twenty-seven attempts at the bottom for one user request. Each layer's own setting looks modest, and the product is not something any single team chose.

The retry owner should be the layer closest to the user that understands the operation's semantics and can select a different endpoint while preserving the deadline and idempotency key. Inner layers generally return a classified failure quickly and leave the decision to that owner.

A transport may retry transparently only before the point at which the application could have seen the request, as defined by that transport's protocol. It is also worth taking an inventory. SDKs, proxies, meshes, load balancers, and database drivers frequently retry by default, and an engineer who believes "the code retries once" may be describing one of five layers.

---

## Backoff and Jitter

A retry sent immediately arrives while the condition that caused the failure is almost certainly still present, so it fails too and adds to the load. Waiting gives the problem time to clear. Exponential backoff lengthens the wait with each attempt, within a cap:

> window(k) = min(cap, base × 2^k)

The base and cap are chosen from how quickly the dependency recovers and how much deadline remains.

Backoff alone leaves a problem that is easy to miss. Suppose a server fails for a moment and a thousand clients' requests fail together. If every client waits exactly one second and retries, the server receives a thousand requests in the same instant one second later. If that overwhelms it, all thousand wait exactly two seconds and return together again. The clients were synchronized by the original failure and deterministic backoff keeps them synchronized, so that the load arrives in spikes separated by silence.

Jitter adds randomness to break the synchronization:

- **full jitter:** delay uniformly from zero to the exponential window;
- **equal jitter:** retain part of the window plus randomness;
- **decorrelated jitter:** derive the next randomized delay from the prior delay and cap.

With full jitter, each client picks a random delay between zero and the current window, and the thousand retries are spread evenly over the interval. The variants trade off minimum delay, time to completion, and correlation. Tests should seed the randomness so that runs can be reproduced.

The whole schedule has to fit within the deadline:

> sum of attempt durations + backoff delays + return margin ≤ remaining deadline

When the deadline cannot fund another useful attempt, the retries stop, whatever the configured maximum.

Where a server says how long to wait before retrying, the client treats that as a lower bound or as policy input according to the API, adds its own jitter, and checks the deadline. The server, for its part, cannot assume that every client will comply, and still needs its own admission protection.

---

## Aggregate Retry Budgets

A cap on attempts per operation does not limit the total load. Three attempts per request seems safe. If a dependency begins failing every request, however, every request uses all three attempts, and the dependency receives three times its normal traffic precisely when it is failing. The per-request cap bounds the worst case for one request and says nothing about the fleet.

A retry budget bounds the fleet. Let:

- <code>lambda_o</code>: admitted original operations/s;
- <code>lambda_r</code>: retry attempts/s;
- <code>beta</code>: allowed retry fraction under the policy.

We enforce:

> lambda_r ≤ beta × lambda_o
>
> total attempt rate ≤ lambda_o × (1 + beta)

With a budget of ten percent, retries may add at most a tenth to the original load. When failures are rare, almost every failed request can be retried, because few are failing. When everything fails, the budget is exhausted almost at once and the remaining requests fail without a retry. That is the desired behavior: retries are plentiful when they help and are cut off when they would do harm.

The usual implementation is a token bucket that earns credit from original requests and spends it on retries. Budgets are scoped by dependency and priority, so that a failing optional call cannot spend the credit that critical calls need. The dimensions are:

- rate of retry attempts;
- concurrent retry work;
- bytes/weighted cost;
- per-endpoint and aggregate dependency limits;
- regional failover/recovery reserve.

A small minimum allowance helps services with little traffic recover from isolated failures. It has to be derived and bounded, because a floor granted to each process is multiplied by the number of processes. Where the aggregate must be controlled precisely, accounting is done per region or cell, or credit is leased from a shared pool.

When the budget is exhausted, the logical operation fails or degrades. Queuing the retry until budget is available would only hold it past its deadline.

### Retries and breakers

Every retry must reacquire everything an original request would need:

- remaining deadline check;
- retry budget;
- dependency concurrency permit;
- circuit-breaker/probe permission;
- current routing/endpoint selection.

A retry is new load and gets no special standing.

---

## Hedging Tail Latency

Retries address failures. Hedging addresses slowness. In a large service most requests are fast and a small fraction are far slower than the rest, because of a garbage-collection pause, a busy disk, or a queue on one particular replica. For a user-facing request that fans out to many backends, the chance that at least one of them lands in that slow fraction is high, so the tail of the backends becomes the typical experience of the user.

A hedge sends a second, equivalent attempt after the first has been outstanding for a delay <code>h</code>, usually to a different replica or path. The first acceptable result is returned and the other attempt is cancelled.

For independent, identically distributed attempt latency <code>X</code>, two simultaneous attempts have:

> P(min(X1, X2) > t) = P(X > t)^2

If one request in a hundred takes longer than a second, the chance that two independent requests both do is one in ten thousand. That is the attraction. The formula assumes independence, however, and real tails are correlated. If both replicas are slow because they share a storage system, a network path, or an overloaded queue, the second attempt is slow for the same reason as the first and the benefit largely disappears.

Sending the hedge only after a delay keeps the cost low. A hedge is issued only for requests still outstanding at <code>h</code>, so before cancellation effects:

> expected extra-attempt probability ≈ P(X > h)

If the delay is set at the ninety-fifth percentile of latency, only about five percent of requests are hedged, for roughly five percent extra load, and those are the requests that were going to be slow. The delay is chosen from live latency for the specific operation and placement, together with the spare capacity available. A percentile applied universally will be wrong for most operations.

### Eligibility

A request is hedged only when:

- operation and result are idempotent/equivalent;
- distinct placement reduces the likely failure correlation;
- another attempt fits the deadline;
- hedge rate/concurrency budget has capacity;
- loser cancellation propagates to queue, server, and dependency;
- duplicate resource usage and side-channel effects are acceptable.

Scarce writes, expensive model generation, long scans, and calls with unsafe duplicate effects should not be hedged. The fact that the client returns only one response does not mean the work was done only once.

### Cancellation

Once there is a winner:

1. Signal cancellation/reset to losers.
2. Remove queued attempts.
3. Stop server work at safe cancellation points.
4. Release concurrency, memory, streams, and connection resources.
5. Record whether a loser nevertheless committed/completed.

The economics of hedging depend on this. If the losing attempt runs to completion on the server, every hedged request costs double. Cancelling the client-side task without any signal on the wire, or without server support for stopping, reclaims nothing at the backend.

### Alternatives

- tied requests: enqueue copies, start one, cancel other queued copies when execution begins;
- load-aware replica selection;
- partition-aware request splitting;
- latency-aware admission;
- fixing stragglers, caches, or hot shards.

Tied requests are a refinement worth knowing. The request is sent to two replicas at once, each told about the other. Whichever begins executing first notifies the other to drop its copy. The request is thus served by whichever replica has the shorter queue, and the duplicated work is limited to the brief interval before the cancellation arrives.

Hedging treats a symptom and pays for it in capacity. Where the cause of the tail can be found and removed, that is the better course.

---

## Metastable Failure and Goodput

The overload loop that retries create has a structure worth drawing:

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

The trigger at the top enters the cycle once. Everything below it feeds itself: latency causes timeouts, timeouts cause retries, retries add work, and the added work raises latency. The trigger can therefore disappear entirely while the system stays broken. The brief slowdown is over, the hardware is healthy, and the service remains saturated, because the load keeping it saturated is being generated by its own failure. This is called a metastable failure. The system has two stable states, one healthy and one overloaded, and a sufficient push moves it from the first to the second, where it stays.

Such a system is hard to diagnose from ordinary dashboards. Throughput may look high, since the servers are working flat out. Most of what they complete is for callers who have already timed out. Goodput, the rate of work completed in time to be useful, is the measure that shows what is happening.

The loop is broken with:

- deadline admission and cancellation;
- aggregate retry/hedge budgets and kill switches;
- bounded queues and shedding;
- dependency concurrency limits/breakers;
- recovery drain controls;
- original-operation versus attempt/goodput telemetry.

Load tests should push beyond saturation, remove the trigger, and confirm that the system recovers without a manual restart.

---

## Concrete Failure Trace: Layered Retries

1. Edge, service A, and service B each independently allow multiple attempts.
2. The database slows and starts timing out, but continues some work after callers abandon it.
3. B retries each database call; A retries B’s already-amplified operation; edge retries A.
4. Attempt multiplication fills connection pools and queues.
5. Per-hop timeouts restart at each layer, so leaf work continues beyond the user deadline.
6. Original goodput approaches zero while database throughput remains high.
7. The database recovers physically, but queued/retry work keeps it overloaded.

The trace combines every hazard in this chapter. Layered retries multiply the load in step 3. Stacked timeouts in step 5 keep work alive after the user has left. The database in step 2 continues queries that nobody is waiting for. And step 7 is the metastable state, in which the original cause is gone and the outage persists.

The first response to such an incident is often to lengthen the timeouts, on the reasoning that requests are timing out and so need more time. That makes it worse. Longer timeouts hold connections and threads for longer, so the pools fill sooner.

The repair is to assign one retry owner, propagate one deadline, classify ambiguous writes, enforce aggregate attempt budgets, cancel abandoned work, and reject load before it queues. The components were each configured reasonably, and the failure came from how they were composed.

---

## Capacity and Cost Model

Let:

- <code>lambda_o</code>: original admitted operations/s;
- <code>beta_r</code>: retry budget fraction;
- <code>beta_h</code>: hedge budget fraction;
- <code>C_attempt</code>: mean resource cost/attempt;
- <code>S_attempt</code>: mean attempt service time;
- <code>p_late</code>: fraction of attempt work completed after logical termination.

Upper budgeted attempt rate:

> lambda_attempt ≤ lambda_o × (1 + beta_r + beta_h)

if the budgets are independent and fully spent. Shared budgets can impose a tighter bound.

Expected attempt concurrency is approximately:

> lambda_attempt × S_attempt

plus the delay in propagating cancellation and any queued attempts. A dependency's concurrency has to be provisioned for attempts, since user request rate understates what it will receive.

The rate of wasted resources includes:

> late/loser/duplicate attempts × C_attempt

The cost of hedging depends on the tail probability beyond <code>h</code> and on how quickly losers stop. It should be measured as work done at the backend. Counting client tasks misses whatever continued after the client stopped listening.

Idempotency storage is the operation rate multiplied by the retention horizon and the record size, adjusted for replication and indexes. Retention must cover the longest real replay horizon.

---

## Operations and Migration

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

Step 3 is a cheap way to learn a great deal. Propagating the deadline and merely recording how much remains at each hop shows how much work is being done for requests that have already expired, before any behavior is changed.

A change to a timeout changes load, and in either direction. A shorter timeout may produce more retries. A longer one keeps resources occupied by in-flight requests for longer. Changes are canaried against goodput and backend work.

### Runbooks

- retry storm: set attempt budget to zero/critical-only, shed, cancel expired work, then recover gradually;
- dependency gray failure: inspect endpoint/path diversity before hedging/retrying;
- ambiguous writes: query idempotency/status ledger and reconcile, never mass-blind-retry;
- deadline propagation failure: disable inner attempts and restore a safe edge bound;
- cancellation ineffective: cap concurrency and fix server/driver propagation;
- policy-control outage: use last verified bounded snapshot and local kill switch.

The first runbook depends on something that has to exist beforehand: a switch that turns retries off across the fleet at run time. During a retry storm there is no opportunity to deploy a configuration change to every client.

---

## Security and Governance

- Authenticate policy distribution and restrict runtime kill-switch access.
- Prevent clients from extending trusted deadlines, attempts, or priority; clamp at the first trusted boundary.
- Bind idempotency key to authenticated subject, operation, and request fingerprint.
- Avoid exposing internal topology or capacity in retry guidance.
- Bound idempotency-key size/cardinality and protect lookup from abuse.
- Audit policy changes, retries of privileged effects, manual replay, and ambiguous-outcome reconciliation.
- Redact sensitive request/response data from attempt logs and idempotency records.
- Do not retry authentication challenges, authorization denials, or security checks through a permissive fallback.

The third item prevents a subtle attack. If an idempotency key is not bound to the caller, one user who guesses or observes another's key can submit a request with it and receive the stored result of the other user's operation.

Retry and hedge multiplication is also a means of amplification for an attacker, who can send requests designed to fail in ways that trigger retries deep in the system. Admission control therefore applies to attempts as well as to original requests.

---

## Observability

By logical operation and dependency, we record:

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

The amplification ratio, total attempts divided by original operations, is the most useful single early warning. In a healthy system it sits just above one. A rise means retries are increasing, and it usually shows before the outage does.

User-facing reliability cannot be read from the success rate of attempts. If every operation fails twice and succeeds on its third attempt, attempts succeed a third of the time and users succeed every time. The service-level indicator is defined on the logical operation.

---

## Verification

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

The fifth test finds what configuration review cannot. All the layers are run together as deployed, one failure is injected at the bottom, and the attempts that reach it are counted. The number is frequently much larger than anyone expected.

Deterministic tests with seeded timing are combined with stochastic load at production scale. Unit tests of the happy path do not reveal amplification.

---

## Decision Framework

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

A retry buys another sample from the distribution of outcomes. That is worthwhile when the failure was bad luck and of no use when the failure reflects the state of the system, since the next sample will be drawn from the same distribution. A hedge buys a second simultaneous sample, and is worth its capacity only when the two samples are reasonably independent. Neither one turns an unsafe operation into an idempotent one.

---

## References

- Jeffrey Dean and Luiz André Barroso, [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/).
- AWS Builders’ Library, [Timeouts, Retries, and Backoff with Jitter](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/).
- Marc Brooker, [Exponential Backoff and Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/).
- gRPC, [A6: Client Retries](https://github.com/grpc/proposal/blob/master/A6-client-retries.md), including hedging, retry throttling, and server pushback.
- Google SRE, [Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/).
- Nathan Bronson et al., [Metastable Failures in Distributed Systems](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s11-bronson.pdf).

---

## Related Chapters

- [Circuit Breakers](10-circuit-breakers.md): a retry budget is the difference between recovery pressure and a self-inflicted outage
- [Idempotency and Operation Identity](../01-foundations/07-idempotency.md): retries and hedging are safe only when the effect can be recognized a second time
- [Failure Semantics, Detection, and Recovery Boundaries](../01-foundations/03-failure-modes.md): a timeout turns an ambiguous outcome into load, which is why detection needs budgets
- [Load Balancing](04-load-balancing.md): the retry picks a host, and a naive pick turns one bad host into an outage
