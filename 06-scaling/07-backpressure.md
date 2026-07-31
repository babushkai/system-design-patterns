# Backpressure: Bounded Buffers, Flow Control, and Load Shedding

## TL;DR

Backpressure is the propagation of downstream capacity information toward producers. It is complete only when it changes **enqueue or production behavior**. Slowing workers while an upstream accepts an unbounded backlog merely moves the failure.

Every boundary needs a finite buffer, a full-buffer action, cancellation/deadline handling, and a signal the producer can obey. Options include blocking or asynchronous demand, explicit credits, cheap rejection, coalescing, sampling, degradation, or intentional loss. Durable queues absorb a bounded mismatch in time; they do not create service capacity.

Backpressure governs already admitted work and defines the overload policy at every buffer. [Rate Limiting](./05-rate-limiting.md) owns entitlement/admission budgets, [Circuit Breakers](./06-circuit-breakers.md) own dependency health and in-flight calls, [Auto-Scaling](./08-auto-scaling.md) owns delayed capacity changes, and [Retries, Timeouts, and Hedging](./10-retries-timeouts-hedging.md) owns repeated attempts.

---

## 1. Flow-Control Contract

For every producer → buffer → consumer edge, define:

| Field | Required answer |
|---|---|
| **Work unit** | Request, byte, message, frame, row, token, or weighted cost. |
| **Capacity** | Sustainable consumer rate and concurrency under the real workload mix. |
| **Buffer** | Location, count/byte limit, durability, ordering, and maximum residence time. |
| **Signal** | Credit, demand, blocked write, reduced receive window, queue-full response, or lag notification. |
| **Producer response** | Pause, slow, retry later, coalesce, sample, degrade, redirect, or discard. |
| **Loss policy** | Which work may be dropped and how consumers learn about gaps. |
| **Deadline** | When queued work becomes useless and who cancels it. |
| **Fairness** | Tenant, priority, key, or flow isolation. |
| **Recovery** | How backlog drains without overwhelming consumers or replaying expired work. |

### Invariants

1. Accepted but unfinished work is bounded by count, bytes, cost, and useful lifetime.
2. A full downstream boundary eventually changes upstream admission or production.
3. Producers cannot create unlimited hidden buffers outside the measured queue.
4. Work whose deadline or consumer has disappeared is canceled before expensive execution where possible.
5. Loss, coalescing, and sampling preserve a declared product semantic.
6. Critical traffic retains a bounded share during optional-traffic overload.
7. Recovery load is bounded; clearing a backlog cannot exceed downstream safe capacity.

---

## 2. Data Plane and Control Plane

~~~mermaid
flowchart LR
    subgraph DP["Data plane"]
        P["Producer"]
        A["Admission boundary"]
        Q[("Bounded buffer<br/>count + bytes + age")]
        C["Consumer pool"]
        D["Downstream"]
        X["Reject / drop /<br/>degrade / coalesce"]
        P --> A
        A -->|accepted| Q
        A -->|full or expired| X
        Q --> C --> D
        C -.credits / demand.-> A
        Q -.occupancy / age.-> A
    end

    subgraph CP["Control plane"]
        POL[("Versioned flow policy")]
        SCALE["Capacity controller"]
        OBS["Lag, goodput, drops,<br/>deadline and fairness"]
        POL --> A
        OBS --> SCALE
    end

    DP -.telemetry.-> OBS
    SCALE -.delayed capacity.-> C
~~~

The data plane must remain bounded without waiting for the autoscaler or configuration service. The control plane can change capacity and policy later; it cannot rescue an unbounded buffer during a fast burst.

---

## 3. Flow Control at Different Layers

### Transport flow control

TCP receive windows and HTTP/2 or QUIC stream/connection credits prevent a fast sender from overflowing protocol buffers at the receiver. They do not know whether the application’s database, worker pool, or user-level queue has capacity. If the application eagerly reads the socket into an unbounded heap queue, transport backpressure has ended too early.

### Application demand and credits

Pull/demand protocols let a consumer grant a bounded number of elements or bytes. Reactive Streams requires demand before a publisher sends elements. Credit must correspond to actual downstream capacity, not merely the ability to copy data into another buffer.

Track bytes or weighted work when item sizes vary. “Ten messages” can mean ten tiny events or ten multi-megabyte objects.

### Queue/broker flow control

A durable broker decouples availability and permits replay. It transfers the backlog to durable storage and extends the time available to recover. Producers still need:

- quota and queue-depth/age admission;
- broker publish failure handling;
- a maximum retained backlog;
- a policy for expiration, compaction, or dead-lettering;
- a consumer catch-up rate above new arrival rate.

Pausing a consumer does not backpressure a producer unless queue policy eventually rejects, slows, or changes producer behavior.

### Work scheduler

Concurrency permits bound work that left the queue. Separate pending work from running work. If every worker takes many items into a local prefetch buffer, the broker may look empty while a large, invisible backlog waits on workers; bound and observe prefetch.

---

## 4. Buffer Sizing and Queue Stability

Let:

- <code>lambda</code>: arrival work units/s;
- <code>mu</code>: sustainable completion units/s;
- <code>Q</code>: maximum queued units;
- <code>S</code>: mean service time;
- <code>D</code>: maximum useful queue delay.

During overload, backlog grows at:

> backlog growth rate = max(0, lambda − mu)

A free buffer of <code>Q_free</code> fills after approximately:

> time to full = Q_free / (lambda − mu), when lambda > mu

After arrivals fall to <code>lambda_recovery < mu</code>, a backlog <code>B</code> drains in:

> drain time = B / (mu − lambda_recovery)

If recovery arrival stays at or above completion, it never drains.

Little’s Law relates average queue occupancy <code>Lq</code>, admitted arrival <code>lambda_a</code>, and average wait <code>Wq</code>:

> Lq = lambda_a × Wq

Size the queue from useful wait and memory/disk cost, not from the largest integer supported. If callers have deadline <code>D</code>, capacity beyond roughly <code>mu × D</code> often stores work that cannot finish usefully; workload variance and priorities require simulation, not blind equality.

### Count and byte bounds

Enforce both. A count-only queue fails on oversized items; a byte-only queue can fail on per-item metadata, timers, file descriptors, or scheduler overhead. Weighted-cost bounds protect expensive operations.

---

## 5. Backpressure Mechanisms

### Blocking or asynchronous wait

The producer waits for capacity. This naturally propagates through a synchronous pipeline only if:

- waits have deadlines and cancellation;
- blocked producers do not hold scarce locks/transactions;
- thread or coroutine counts are bounded;
- cyclic dependencies cannot deadlock.

Blocking an event loop or holding a database connection while waiting turns flow control into resource starvation.

### Credit or pull

Consumers grant explicit demand. This gives a clear upper bound:

> outstanding elements ≤ granted credits + in-flight protocol race

Return credit only after the resource it represents is actually released. Granting credit when an item is merely copied to another queue breaks the bound.

### Explicit rejection

Reject before expensive work with an overload-specific response. This is appropriate for interactive requests whose callers can degrade or retry later under the canonical retry policy. A rejection path must avoid expensive authentication, logging, serialization, or dependency calls after the decision.

### Shaping

Delay work to a controlled rate when the caller deadline and buffer bound permit it. Shaping turns burst into queue delay; its capacity math still applies. Long shaping queues hide overload and make cancellation essential.

### Coalescing

Replace many pending updates for the same key with the newest state, or combine identical reads into one in-flight call. This is correct only when intermediate events have no required semantics. State replication may coalesce desired state; a financial ledger may not.

### Sampling and degradation

Reduce optional work:

- sample telemetry with known probability and weights;
- lower image/video quality;
- omit expensive enrichment;
- return a cached/stale result with provenance;
- aggregate many metrics into one summary.

Make the mode visible so downstream analytics and users do not treat sampled or degraded data as complete.

### Drop policy

- **Drop newest:** preserves accepted order and backlog but penalizes current traffic.
- **Drop oldest:** favors fresh work when old work loses value.
- **LIFO service:** improves chance that served interactive requests still have live callers, but can starve old work.
- **Priority drop:** preserves critical classes but needs quotas to prevent starvation.
- **Deadline drop:** removes objectively expired work.

Choose by product semantics, never by queue library default.

---

## 6. End-to-End Propagation and Hidden Buffers

Inventory every place work can accumulate:

- client retry queue;
- socket send/receive buffers;
- proxy pending requests;
- server accept backlog;
- runtime executor or event-loop queue;
- application channel;
- broker topic/subscription;
- consumer prefetch;
- connection pool;
- database lock/wait queue;
- fallback and dead-letter paths.

The end-to-end bound is the sum of these reservoirs, and the deadline cost is their sequential wait. A bounded queue followed by an unbounded executor is not a bounded system.

### Fan-out

One input can create <code>F</code> downstream work units. Backpressure must account for amplification before fan-out. If branches have different speeds:

- block all branches for strict aligned semantics;
- buffer each branch independently within bounds;
- drop/degrade optional branches;
- materialize once in a durable log and let branches own lag.

A slow optional consumer must not hold mandatory processing unless that coupling is deliberate.

### Cycles

Credit cycles can deadlock when A waits for B while B waits for A. Break cycles with bounded seed credit, separate control channels, or durable asynchronous boundaries. Never allow the backpressure signal itself to require capacity from the saturated data queue.

---

## 7. Fairness and Priority

One FIFO mixes cheap/expensive, interactive/batch, and tenants. A large low-priority backlog can make critical traffic miss deadlines even after capacity returns.

Use:

- separate queues or concurrency pools by priority/resource class;
- weighted fair scheduling;
- per-tenant queue and in-flight bounds;
- aging only where starvation is worse than missing fresh deadlines;
- reserved critical capacity plus controlled borrowing.

Priority must propagate. Marking a request high priority at the edge is useless if it waits behind bulk work in the database pool.

Guard against priority inflation: only trusted policy may assign classes.

---

## 8. Cancellation and Abandoned Work

When a client disconnects or deadline expires:

1. Stop admitting downstream fan-out.
2. Remove queued work if possible.
3. Propagate cancellation to active operations.
4. Release credits, permits, memory, and transactions once.
5. Decide whether a committed side effect must finish despite caller loss.
6. Record goodput separately from work completed too late.

Cancellation is not rollback. A write may already have committed. Use idempotency and effect protocols from [Idempotency](../01-foundations/08-idempotency.md); do not blindly retry an ambiguous operation.

For long jobs, cancellation may mean “stop future steps and compensate” rather than kill a process.

---

## 9. Concrete Failure Trace: Backpressure Stops at the Wrong Queue

1. A downstream database slows from <code>mu_normal</code> to <code>mu_degraded</code>.
2. Consumers reduce polling and their local queues remain bounded.
3. The API continues accepting every request into a durable broker with no age/size admission.
4. Broker depth grows; callers time out but messages remain valid to the worker.
5. Autoscaling adds consumers, but the database is still the bottleneck.
6. When the database recovers, old work plus live traffic and retries hit it together.
7. Most old results are no longer useful; goodput stays low and the database overloads again.

Backpressure existed only between broker and consumer. Fix the end-to-end contract: bound queue age/bytes, reject or degrade at enqueue, propagate deadlines into messages, discard expired work, reserve recovery capacity, and drain at a rate the database can sustain.

---

## 10. Backpressure versus Neighboring Controls

| Mechanism | Canonical question |
|---|---|
| Rate limit | May this subject/resource introduce this much work over time? |
| Concurrency limit | May another call be in flight to this dependency now? |
| Circuit breaker | Does recent evidence say calls should be suppressed/probed? |
| Backpressure | How does downstream saturation change upstream production and buffering? |
| Autoscaling | How should capacity change after delayed measurements? |
| Retry budget | How much extra attempt load may failures create? |

Compose them:

1. Reject invalid/unauthorized work cheaply.
2. Apply admission entitlement.
3. Check the deadline.
4. Acquire bounded queue/concurrency capacity.
5. Call through dependency protection.
6. Retry only within the remaining deadline and aggregate retry budget.
7. Feed sustained demand (not uncontrolled queue growth alone) to autoscaling.

The exact order depends on whether a token is charged for rejected or completed work, but the accounting boundary must be explicit.

---

## 11. Capacity and Cost Model

Include:

- buffer memory = queued items × per-item retained bytes plus allocator/index overhead;
- durable storage = arrival byte rate × retained backlog time × replication factor;
- enqueue/dequeue and acknowledgement operations;
- retry/dead-letter amplification;
- network and serialization copies;
- consumer idle headroom reserved for drain;
- cost of dropped work already partially executed.

For partitioned queues, total consumer parallelism may be bounded by partitions or ordering keys. Adding workers beyond eligible partitions does not raise <code>mu</code>. Hot keys create one slow partition while fleet averages look idle.

To meet a drain objective <code>T_drain</code> for backlog <code>B</code> with recovery arrival <code>lambda_r</code>:

> required completion rate ≥ lambda_r + B / T_drain

Verify the downstream can tolerate that rate. Otherwise change the objective, shed backlog, or isolate drain capacity.

Cost per **useful** completion matters. During overload, high CPU with many expired completions is poor goodput, not efficiency.

---

## 12. Operations and Migration

### Introducing bounds

1. Inventory and instrument every buffer.
2. Add byte, count, age, and deadline visibility.
3. Shadow the proposed full-buffer decision.
4. Teach producers to obey the signal.
5. Enable bounds for optional/low-priority work.
6. Expand while monitoring goodput and fairness.
7. Remove or cap hidden downstream buffers.

Changing buffer size or drop order changes observable semantics. Version queue messages and consumer behavior when adding deadlines, priorities, coalescing, or gap markers.

### Recovery runbook

- stop/limit new optional enqueue;
- cancel expired work;
- measure backlog by age, priority, key, and cost;
- establish safe downstream drain capacity;
- isolate live traffic from backlog where needed;
- raise consumers only while completion headroom remains;
- avoid retrying dropped work without a new admission decision;
- verify the system returns to low-lag stable state after the trigger is removed.

---

## 13. Security and Governance

- Authenticate before granting tenant/priority credit, with an earlier cheap abuse guard.
- Bound message/item size before allocation or deserialization.
- Prevent user-supplied priority, partition, or cost from bypassing fairness.
- Encrypt durable queues and inspect payload retention/privacy implications.
- Redact queued payloads from diagnostics and dead-letter samples.
- Apply deletion/retention policy to main, retry, dead-letter, spill, and local prefetch stores.
- Protect control/credit channels from spoofing and starvation.
- Audit emergency shedding, queue purge, replay, and priority-policy changes.

Backlogs often contain the most complete copy of recently submitted sensitive data.

---

## 14. Observability

At every boundary:

- offered, admitted, completed, rejected, dropped, coalesced, sampled, and expired units/bytes;
- queue count, bytes, oldest age, and residence-time distribution;
- producer blocked time, credit starvation, and full-buffer duration;
- consumer busy/idle time, service time, concurrency, and completion rate;
- prefetch/local hidden backlog;
- cancellation propagation and work completed after caller deadline;
- goodput versus throughput;
- fairness by bounded tenant/priority aggregation;
- retry and dead-letter creation/replay.

Queue depth without arrival and completion rate cannot predict whether lag is growing or draining. Oldest age is often closer to user impact than count.

---

## 15. Verification

- burst above capacity and prove every buffer remains within count/byte bounds;
- vary item size and processing cost independently;
- stop consumers, fill queues, and verify producer behavior;
- disconnect clients and verify queued/active cancellation and permit release;
- overload one tenant, key, and optional branch while critical work continues;
- create a hot partition and verify averages do not hide it;
- test every drop/coalesce policy against product invariants and gap signaling;
- partition credit/control channels without blocking emergency rejection;
- recover with a large backlog and prove drain stays under downstream safe capacity;
- add consumers beyond partition parallelism and verify capacity model;
- push beyond saturation, remove excess arrival, and prove autonomous recovery;
- reconcile accepted work with completed, expired, dropped, and still-pending outcomes.

Test long enough to expose retention, dead-letter, and replay loops, not only a short request burst.

---

## 16. Decision Framework

| Workload | Flow-control direction |
|---|---|
| Synchronous request, short deadline | Small/no queue, concurrency permit, early rejection |
| Lossless ordered stream | Credit/pull, bounded durable buffer, producer pause |
| Latest state supersedes old updates | Per-key coalescing |
| Telemetry where completeness is statistical | Sampling with recorded probability/weights |
| Durable jobs tolerating delay | Broker with age/size admission and drain plan |
| Optional enrichment | Shed/degrade before mandatory path |
| Cyclic topology | Explicit seed credit or durable cycle break |
| Producer cannot slow and loss is forbidden | Provision for peak or move buffer to a capacity/retention contract that can hold it |

If the producer cannot slow, work cannot be rejected, loss is forbidden, and storage is bounded, the requirements are inconsistent once arrival exceeds completion. Architecture cannot negotiate that arithmetic.

---

## Primary References

- Reactive Streams, [Specification](https://www.reactive-streams.org/).
- IETF, [RFC 9293: Transmission Control Protocol](https://www.rfc-editor.org/rfc/rfc9293.html), including receive-window flow control.
- IETF, [RFC 9113: HTTP/2](https://www.rfc-editor.org/rfc/rfc9113.html), including stream and connection flow control.
- Google SRE, [Handling Overload](https://sre.google/sre-book/handling-overload/).
- Google SRE, [Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/).
- Apache Kafka, [Consumer Configuration](https://kafka.apache.org/documentation/#consumerconfigs) and [Design](https://kafka.apache.org/documentation/#design).
- Apache Flink, [Monitoring Back Pressure](https://nightlies.apache.org/flink/flink-docs-stable/docs/ops/monitoring/back_pressure/).
