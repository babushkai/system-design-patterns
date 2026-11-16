# Backpressure: Bounded Buffers, Flow Control, and Load Shedding

> The queue grew for an hour before anything failed, and by then every request in it was already too late. Who should have been told to slow down, and how?

## TL;DR

Backpressure is the propagation of downstream capacity information toward producers. It is complete only when it changes **enqueue or production behavior**. If workers slow down while something upstream continues to accept an unbounded backlog, the failure has only been moved to another place.

Every boundary needs a finite buffer, an action to take when that buffer is full, handling for cancellation and deadlines, and a signal the producer can obey. The options include blocking or asynchronous demand, explicit credits, cheap rejection, coalescing, sampling, degradation, and intentional loss. A durable queue absorbs a bounded mismatch in time between arrival and completion. It does not create service capacity.

Backpressure governs work that has already been admitted and defines the overload policy at every buffer. [Rate Limiting](./07-rate-limiting.md) owns entitlement and admission budgets, [Circuit Breakers](./10-circuit-breakers.md) own dependency health and in-flight calls, [Auto-Scaling](./06-auto-scaling.md) owns delayed capacity changes, and [Retries, Timeouts, and Hedging](./09-retries-timeouts-hedging.md) owns repeated attempts.

---

## Flow-Control Contract

Every system has a rate at which work arrives and a rate at which it can be completed. When the first exceeds the second, the excess has to go somewhere, and there are only three places for it. It can wait in a buffer. It can be refused or discarded. Or the producer can be made to send less. No fourth option exists, and a system that has not decided among these has had the decision made for it, usually by an unbounded queue that grows until the process runs out of memory.

The unbounded queue is the default in most software, and it is attractive because it appears to make the problem go away. Under a brief burst it absorbs the excess and drains afterward. Under sustained overload it has a specific and unpleasant behavior. The queue lengthens, so each item waits longer. Callers time out and abandon their requests, but the requests are still in the queue and will still be processed, at full cost, for nobody. Throughput stays high and the amount of useful work falls toward zero. Eventually memory is exhausted, the process is killed, and everything in the queue is lost, including the work that could have been completed had the system refused the rest.

Backpressure is the discipline of making the decision explicitly. For every edge from a producer through a buffer to a consumer, we define:

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

The second invariant is the definition of the subject. A signal that reaches no producer, or reaches one that ignores it, is not backpressure.

---

## Data Plane and Control Plane

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

The dotted arrows that run backward, from the consumer and the buffer to the admission boundary, are the backpressure. Work flows left to right, and information about capacity flows right to left.

The diagram also separates two timescales. The data plane has to stay bounded by its own means, within milliseconds, without waiting for anything else. The control plane, including the autoscaler, works over minutes. It can add capacity and change policy after the fact. It cannot rescue an unbounded buffer during a burst that fills memory in seconds, and a design that relies on autoscaling to handle overload has no protection during the interval before the new capacity arrives.

---

## Flow Control at Different Layers

Flow control exists at several layers of a system, and each layer protects only its own buffers. A recurring mistake is to assume that because one layer has flow control, the system does.

### Transport flow control

TCP has had flow control since its beginning. A receiver advertises a window, the number of bytes it has room for, and the sender may not exceed it. When the receiving application stops reading, the kernel's buffer fills, the advertised window shrinks to zero, and the sender stops. HTTP/2 and QUIC add a similar scheme of credits for each stream and connection.

This protects the receiver's protocol buffers and nothing beyond them. TCP has no knowledge of whether the application's database, worker pool, or internal queue has capacity. If the application eagerly reads everything from the socket and appends it to an unbounded in-memory queue, the kernel's buffer never fills and the window never closes. The sender is told, in effect, that the receiver has unlimited capacity, and transport backpressure has ended one step too early.

### Application demand and credits

In a pull or demand protocol the consumer states how much it is prepared to receive, and the producer may send only that much. Reactive Streams is built on this rule: a publisher may not send an element until the subscriber has requested it. The credit must correspond to real downstream capacity. A stage that grants credit as soon as it has copied an item into its own buffer has only relocated the queue.

Where items vary in size, credits are counted in bytes or weighted work. Ten messages may be ten small events or ten objects of several megabytes each.

### Queue/broker flow control

A durable broker decouples the availability of producers and consumers and allows replay. In doing so it moves the backlog onto disk, which is larger than memory and survives restarts. That extends the time available to recover. It does not change the arithmetic: if arrival exceeds completion indefinitely, the disk fills as well, only later. Producers therefore still need:

- quota and queue-depth/age admission;
- broker publish failure handling;
- a maximum retained backlog;
- a policy for expiration, compaction, or dead-lettering;
- a consumer catch-up rate above new arrival rate.

The last item is easily forgotten. To drain a backlog, consumers must process faster than new work arrives. A system whose consumers can just keep up with normal traffic can never catch up after an outage.

Pausing a consumer does not, by itself, apply backpressure to a producer. The producer is affected only when queue policy eventually rejects it, slows it, or otherwise changes what it does.

### Work scheduler

Concurrency permits bound the work that has left the queue and is executing. Pending and running work should be counted separately. Prefetch is a place where a backlog hides: if each worker pulls many items from the broker into a local buffer for efficiency, the broker may report an empty queue while a large backlog sits invisibly in worker memory. Prefetch is bounded and observed like any other buffer.

---

## Buffer Sizing and Queue Stability

The behavior of a queue under overload follows from simple arithmetic, and the arithmetic is worth internalizing because intuition about queues is frequently wrong. Let:

- <code>lambda</code>: arrival work units/s;
- <code>mu</code>: sustainable completion units/s;
- <code>Q</code>: maximum queued units;
- <code>S</code>: mean service time;
- <code>D</code>: maximum useful queue delay.

During overload, backlog grows at:

> backlog growth rate = max(0, lambda − mu)

A free buffer of <code>Q_free</code> fills after approximately:

> time to full = Q_free / (lambda − mu), when lambda > mu

A buffer therefore buys time, and the amount is its size divided by the excess rate. If arrivals exceed capacity by a thousand items a second, a buffer of sixty thousand items delays the inevitable by one minute. Making the buffer ten times larger makes that ten minutes. No buffer size changes the outcome of sustained overload.

After arrivals fall to <code>lambda_recovery < mu</code>, a backlog <code>B</code> drains in:

> drain time = B / (mu − lambda_recovery)

The denominator is the spare capacity, the difference between what the system can do and what is still arriving. A system that normally runs at ninety percent of capacity has ten percent spare, so a backlog accumulated in one hour of total outage takes about nine hours to clear. If recovery arrival stays at or above completion, the backlog never drains. This is why incidents so often last far longer than their cause.

Little’s Law relates average queue occupancy <code>Lq</code>, admitted arrival <code>lambda_a</code>, and average wait <code>Wq</code>:

> Lq = lambda_a × Wq

Read in reverse, the law tells us how long items wait: queue length divided by throughput. A queue of ten thousand items in front of a system completing a hundred a second means each item waits a hundred seconds. If callers give up after five seconds, nearly everything in that queue has already been abandoned by the time it is served.

That gives the principle for sizing. A queue is sized from the wait that remains useful and from the cost of memory or disk, and the largest number the data type allows is irrelevant. If callers have a deadline <code>D</code>, capacity beyond roughly <code>mu × D</code> mostly stores work that cannot finish in time. Variance in the workload and priorities complicate this, and they call for simulation and not for blind use of the equality.

### Count and byte bounds

Both are enforced. A queue bounded only by count fails when items are oversized, since a thousand items of a megabyte each is a gigabyte. A queue bounded only by bytes can fail on per-item overhead such as metadata, timers, file descriptors, or scheduler state. Bounds on weighted cost protect expensive operations.

---

## Backpressure Mechanisms

### Blocking or asynchronous wait

The simplest mechanism is for the producer to wait until there is room. In a synchronous pipeline this propagates naturally, each stage waiting on the next, all the way back to the source. It is safe only under conditions:

- waits have deadlines and cancellation;
- blocked producers do not hold scarce locks/transactions;
- thread or coroutine counts are bounded;
- cyclic dependencies cannot deadlock.

The second condition is the one that causes trouble. A producer that blocks while holding a database connection or a lock is no longer merely waiting. It is withholding a scarce resource from everyone else, and flow control has become resource starvation. Blocking the event loop of an asynchronous runtime is worse, since it stops everything that loop serves.

### Credit or pull

Consumers grant explicit demand, which gives a clear upper bound on what can be outstanding:

> outstanding elements ≤ granted credits + in-flight protocol race

Credit is returned only when the resource it stands for has actually been released. If credit is returned when an item has merely been copied into another queue, the bound no longer means anything.

### Explicit rejection

The system refuses work before doing anything expensive and returns a response that says it is overloaded. This suits interactive requests, where the caller can degrade or try again later under a proper retry policy. It is often the kindest behavior available: a caller told immediately that the system is busy can do something useful, whereas a caller left waiting for a timeout cannot.

The rejection path itself must be cheap. If refusing a request involves authentication, detailed logging, serialization, or a call to a dependency, a flood of requests can overload the system through the cost of saying no.

### Shaping

Work is delayed so that it leaves at a controlled rate, where the caller's deadline and the buffer bound allow. Shaping converts a burst into queue delay, and the capacity arithmetic still applies. A long shaping queue hides overload and makes cancellation essential.

### Coalescing

Many pending updates for the same key are replaced by the newest one, or identical reads are combined into a single in-flight call. This is very effective where it applies, and it applies only where intermediate values carry no meaning. If a user's profile is updated five times while the queue is backed up, only the final state needs to be propagated. A replication system for desired state can coalesce. A financial ledger cannot, because each entry is a fact in its own right.

### Sampling and degradation

Optional work is reduced:

- sample telemetry with known probability and weights;
- lower image/video quality;
- omit expensive enrichment;
- return a cached/stale result with provenance;
- aggregate many metrics into one summary.

The degraded mode has to be visible. If telemetry is sampled at ten percent and the consumers of that data do not know it, they will conclude that traffic fell by ninety percent. Recording the sampling probability lets them scale the counts back up.

### Drop policy

When work must be discarded, the choice of which work is a product decision:

- **Drop newest:** preserves accepted order and backlog but penalizes current traffic.
- **Drop oldest:** favors fresh work when old work loses value.
- **LIFO service:** improves chance that served interactive requests still have live callers, but can starve old work.
- **Priority drop:** preserves critical classes but needs quotas to prevent starvation.
- **Deadline drop:** removes objectively expired work.

Serving the newest request first seems unfair, and under overload it is often right for interactive traffic. The oldest request in the queue is the one most likely to have been abandoned already by a caller who timed out. The newest has a caller who is still waiting. Processing in strict arrival order during overload means working through requests in decreasing order of their likelihood to matter.

The choice follows from product semantics. The default behavior of a queue library was chosen by someone who did not know the product.

---

## End-to-End Propagation and Hidden Buffers

A system is bounded only if every place where work can accumulate is bounded. There are more such places than most designs acknowledge:

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

The end-to-end bound is the sum of all these reservoirs, and the time a request spends is the sum of its waits in each. A carefully bounded application queue that feeds an unbounded thread-pool executor is an unbounded system, and the bounded queue has accomplished nothing.

### Fan-out

One input may create <code>F</code> units of downstream work, and backpressure has to account for that amplification before the fan-out occurs. When the branches run at different speeds, the options are:

- block all branches for strict aligned semantics;
- buffer each branch independently within bounds;
- drop/degrade optional branches;
- materialize once in a durable log and let branches own lag.

A slow optional consumer must not hold up mandatory processing unless that coupling was intended. If an order is written to the database and also sent to an analytics pipeline, a stall in analytics should not stop orders.

### Cycles

Where A waits for credit from B and B waits for credit from A, both wait forever. Cycles are broken with a bounded initial grant of credit, with separate control channels, or with a durable asynchronous boundary. One rule applies generally: the backpressure signal must never itself require capacity from the queue it is reporting on. A message saying "the queue is full" that has to pass through the full queue will not arrive.

---

## Fairness and Priority

A single first-in, first-out queue treats all work alike, and work is not alike. Cheap and expensive requests, interactive and batch work, and different tenants all share the line. A large backlog of low-priority work can then cause critical traffic to miss its deadlines even after capacity has returned, because the critical work is waiting behind it.

The remedies are:

- separate queues or concurrency pools by priority/resource class;
- weighted fair scheduling;
- per-tenant queue and in-flight bounds;
- aging only where starvation is worse than missing fresh deadlines;
- reserved critical capacity plus controlled borrowing.

Priority has to propagate through every layer to be worth anything. A request marked high-priority at the edge that then waits behind bulk work for a database connection has not been prioritized.

Priority also needs guarding against inflation. If callers can assign their own priority, every caller will in time mark everything as critical, and the classes lose their meaning. Only trusted policy may assign them.

---

## Cancellation and Abandoned Work

Work that nobody is waiting for is pure waste, and under overload it is the commonest kind. When a client disconnects or a deadline expires:

1. Stop admitting downstream fan-out.
2. Remove queued work if possible.
3. Propagate cancellation to active operations.
4. Release credits, permits, memory, and transactions once.
5. Decide whether a committed side effect must finish despite caller loss.
6. Record goodput separately from work completed too late.

Cancellation is not rollback. By the time the cancellation arrives, a write may already have committed. The caller does not know whether it did, and must not simply retry. The protocols for this situation are in [Idempotency](../01-foundations/07-idempotency.md).

For long-running jobs, cancellation may mean stopping future steps and compensating for those already taken. Killing the process is rarely the right implementation.

---

## Concrete Failure Trace: Backpressure Stops at the Wrong Queue

1. A downstream database slows from <code>mu_normal</code> to <code>mu_degraded</code>.
2. Consumers reduce polling and their local queues remain bounded.
3. The API continues accepting every request into a durable broker with no age/size admission.
4. Broker depth grows; callers time out but messages remain valid to the worker.
5. Autoscaling adds consumers, but the database is still the bottleneck.
6. When the database recovers, old work plus live traffic and retries hit it together.
7. Most old results are no longer useful; goodput stays low and the database overloads again.

Each component in this trace behaved defensibly. The consumers applied backpressure correctly and protected themselves. The broker stored messages durably, as brokers do. The autoscaler saw a growing queue and added workers, as it was configured to. The failure lies in where the backpressure stopped. It ran from the database to the consumers and from the consumers to the broker, and there it ended, because the API in front of the broker never learned that anything was wrong and went on accepting work.

Step 5 shows a common misdiagnosis. A long queue looks like a shortage of consumers, and adding consumers is the reflex. Here the bottleneck was the database, and more consumers meant more connections contending for a resource that was already saturated.

Step 6 is the second outage. When the database recovers, it is met with the entire stored backlog, the live traffic, and the retries of everyone who timed out, all at once, and it is overloaded again. Most of that backlog is from callers who left long ago.

The repair is to make the contract end to end: bound the queue by age and bytes, reject or degrade at the point of enqueue, carry deadlines inside messages so that expired work can be discarded, reserve capacity for recovery, and drain at a rate the database can sustain.

---

## Backpressure versus Neighboring Controls

Several mechanisms protect a system from overload, and they are often confused because they overlap. Each answers a different question:

| Mechanism | Canonical question |
|---|---|
| Rate limit | May this subject/resource introduce this much work over time? |
| Concurrency limit | May another call be in flight to this dependency now? |
| Circuit breaker | Does recent evidence say calls should be suppressed/probed? |
| Backpressure | How does downstream saturation change upstream production and buffering? |
| Autoscaling | How should capacity change after delayed measurements? |
| Retry budget | How much extra attempt load may failures create? |

A rate limit is concerned with entitlement: what a caller is permitted to send, whatever the current load. Backpressure is concerned with capacity: what the system can accept at this moment, whoever is asking. A caller well within its rate limit may still be refused because the system is saturated, and the two refusals mean different things.

They compose in sequence:

1. Reject invalid/unauthorized work cheaply.
2. Apply admission entitlement.
3. Check the deadline.
4. Acquire bounded queue/concurrency capacity.
5. Call through dependency protection.
6. Retry only within the remaining deadline and aggregate retry budget.
7. Feed sustained demand (not uncontrolled queue growth alone) to autoscaling.

The exact order depends on whether a token is charged for work that was rejected or only for work that completed. Either is defensible, provided the accounting boundary is explicit.

---

## Capacity and Cost Model

The cost of buffering includes:

- buffer memory = queued items × per-item retained bytes plus allocator/index overhead;
- durable storage = arrival byte rate × retained backlog time × replication factor;
- enqueue/dequeue and acknowledgement operations;
- retry/dead-letter amplification;
- network and serialization copies;
- consumer idle headroom reserved for drain;
- cost of dropped work already partially executed.

For partitioned queues, consumer parallelism may be limited by the number of partitions or ordering keys. Each partition is consumed by one worker at a time in order to preserve order, so adding workers beyond the number of partitions does not raise <code>mu</code>. Hot keys make this worse: one partition holding a very active key falls behind while the fleet's average utilization looks low, and adding consumers does nothing for it.

To meet a drain objective <code>T_drain</code> for backlog <code>B</code> with recovery arrival <code>lambda_r</code>:

> required completion rate ≥ lambda_r + B / T_drain

The downstream must be able to tolerate that rate. If it cannot, the objective has to change, the backlog has to be shed, or drain capacity has to be isolated from live traffic.

The quantity that matters is cost per **useful** completion. During overload, high CPU utilization spent completing work that has already expired is a sign of poor goodput and should not be mistaken for efficiency.

---

## Operations and Migration

### Introducing bounds

Adding bounds to a system that has none will cause requests to be rejected that were previously accepted, and so it is done in stages:

1. Inventory and instrument every buffer.
2. Add byte, count, age, and deadline visibility.
3. Shadow the proposed full-buffer decision.
4. Teach producers to obey the signal.
5. Enable bounds for optional/low-priority work.
6. Expand while monitoring goodput and fairness.
7. Remove or cap hidden downstream buffers.

Step 3 computes what the bound would have rejected without rejecting anything, which shows the effect of a threshold before it is enforced. Step 4 has to come before enforcement. A producer that receives a "queue full" response it does not understand will typically retry at once, which adds load at the worst moment.

A change to buffer size or drop order changes behavior that others can observe. Queue messages and consumer behavior are versioned when deadlines, priorities, coalescing, or gap markers are introduced.

### Recovery runbook

- stop/limit new optional enqueue;
- cancel expired work;
- measure backlog by age, priority, key, and cost;
- establish safe downstream drain capacity;
- isolate live traffic from backlog where needed;
- raise consumers only while completion headroom remains;
- avoid retrying dropped work without a new admission decision;
- verify the system returns to low-lag stable state after the trigger is removed.

The last item checks for a condition sometimes called a metastable failure. Some overloaded systems do not recover when the original cause is removed, because the overload sustains itself through retries, cache misses, or the backlog. A system that remains saturated after the trigger has gone needs load actively removed before it will return to normal.

---

## Security and Governance

- Authenticate before granting tenant/priority credit, with an earlier cheap abuse guard.
- Bound message/item size before allocation or deserialization.
- Prevent user-supplied priority, partition, or cost from bypassing fairness.
- Encrypt durable queues and inspect payload retention/privacy implications.
- Redact queued payloads from diagnostics and dead-letter samples.
- Apply deletion/retention policy to main, retry, dead-letter, spill, and local prefetch stores.
- Protect control/credit channels from spoofing and starvation.
- Audit emergency shedding, queue purge, replay, and priority-policy changes.

Queues are easily overlooked in security reviews because they are thought of as plumbing. A backlog, and especially a dead-letter queue, often contains the most complete copy of recently submitted sensitive data in the whole system, with long retention and broad read access for debugging.

---

## Observability

At every boundary we measure:

- offered, admitted, completed, rejected, dropped, coalesced, sampled, and expired units/bytes;
- queue count, bytes, oldest age, and residence-time distribution;
- producer blocked time, credit starvation, and full-buffer duration;
- consumer busy/idle time, service time, concurrency, and completion rate;
- prefetch/local hidden backlog;
- cancellation propagation and work completed after caller deadline;
- goodput versus throughput;
- fairness by bounded tenant/priority aggregation;
- retry and dead-letter creation/replay.

Queue depth is the metric everyone graphs, and by itself it says little. A depth of ten thousand may be a queue that is draining rapidly or one that is growing without limit, and only the arrival and completion rates distinguish them. The age of the oldest item is usually closer to what users experience: it states directly how long work is waiting, and it keeps rising when a consumer is stuck, even if the count is flat.

---

## Verification

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

The second-to-last test is the most informative. The system is driven past saturation, the excess load is removed, and we observe whether it returns to normal unaided. A system that passes has overload behavior that can be trusted. A system that fails will turn its next brief spike into a long outage.

Tests need to run long enough to expose retention, dead-letter, and replay loops. A short burst of requests reveals none of them.

---

## Decision Framework

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

The first row may be surprising. For a synchronous request with a short deadline, the best queue is very small or absent. A queue helps only if the work can still be useful after waiting, and an interactive request that has waited longer than its caller's patience is useless. A small concurrency limit with immediate rejection serves such traffic better than any buffer.

The last row marks the limit of what design can do. If the producer cannot slow down, work cannot be rejected, loss is forbidden, and storage is bounded, then the requirements contradict one another as soon as arrival exceeds completion. One of the four has to give, and it is better to choose which in a design review than to find out in production.

---

## References

- Reactive Streams, [Specification](https://www.reactive-streams.org/).
- IETF, [RFC 9293: Transmission Control Protocol](https://www.rfc-editor.org/rfc/rfc9293.html), including receive-window flow control.
- IETF, [RFC 9113: HTTP/2](https://www.rfc-editor.org/rfc/rfc9113.html), including stream and connection flow control.
- Google SRE, [Handling Overload](https://sre.google/sre-book/handling-overload/).
- Google SRE, [Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/).
- Apache Kafka, [Consumer Configuration](https://kafka.apache.org/documentation/#consumerconfigs) and [Design](https://kafka.apache.org/documentation/#design).
- Apache Flink, [Monitoring Back Pressure](https://nightlies.apache.org/flink/flink-docs-stable/docs/ops/monitoring/back_pressure/).

---

## Related Chapters

- [Message Queue Architecture](../05-messaging/01-message-queues.md): a queue is a buffer with an opinion, and this is what happens when it fills
- [Rate Limiting](07-rate-limiting.md): the same decision moved to the edge, where a client can be told to slow down
- [Network Transport Internals](03-network-transport-internals.md): flow control with a standard attached: windows, congestion control, and head-of-line blocking
- [Stream Execution](../10-data-pipelines/02-stream-processing.md): backpressure in a pipeline that also has to be correct across restarts
