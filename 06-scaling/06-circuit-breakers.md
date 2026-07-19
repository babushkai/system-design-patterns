# Circuit Breakers: Dependency State and Concurrency Control

## TL;DR

A circuit breaker is a **local admission state machine for one dependency scope**. It observes classified outcomes over a rolling sample, opens when calling is more harmful than failing fast, and later admits a bounded probe population. It is not a health check, retry loop, distributed consensus protocol, or universal fallback generator.

The breaker must control both **whether** calls may start and **how many** may be in flight. Error-rate state reacts after failures; a concurrency limit prevents slow calls from exhausting threads, connections, memory, or the dependency before enough errors exist. Recovery is progressive: open time only schedules a probe, and probe evidence must come from representative data-plane calls.

Prefer local enforcement because each caller sees its own route, deadlines, and endpoint set. Share policy and telemetry globally, but share live breaker state only when the protected resource is truly global and the coordination dependency is justified.

---

## 1. Dependency Contract

Define one breaker scope:

| Field | Required answer |
|---|---|
| **Caller scope** | Process, workload, cell, or region that owns the state. |
| **Protected dependency** | Cluster, endpoint, operation class, partition, or tenant-specific resource. |
| **Outcomes** | Success, business rejection, overload rejection, timeout, cancellation, reset, protocol error, and slow success. |
| **Sample** | Count- or time-based window, minimum evidence, weighting, and idle behavior. |
| **Open action** | Fail, use a bounded fallback, route elsewhere, or degrade optional work. |
| **Probe policy** | When probes may start, maximum concurrent probes, traffic mix, and evidence required to recover. |
| **Concurrency** | Fixed or adaptive in-flight limit, pending-queue policy, and per-priority isolation. |
| **Deadline interaction** | Which calls are never started because insufficient deadline remains. |
| **State lifecycle** | Configuration version, endpoint membership change, process restart, and manual override semantics. |

### Invariants

1. Breaker state for one dependency cannot suppress unrelated dependencies or operations.
2. Outcome classification reflects dependency health, not application business errors.
3. No transition is based on a rate until the configured evidence requirement is met.
4. Open state admits no ordinary calls; probes are explicitly bounded.
5. Concurrency and pending work are bounded in every state.
6. A fallback consumes its own declared budget and cannot create a hidden cascade.
7. Recovery traffic grows gradually and can be stopped by fresh failure evidence.
8. Configuration and manual overrides are observable, versioned, and reversible.

---

## 2. Data Plane and Control Plane

~~~mermaid
flowchart LR
    subgraph DP["Caller data plane"]
        R["Call with remaining deadline"]
        C["Concurrency permit"]
        B["Breaker state<br/>closed / open / probe"]
        D["Dependency"]
        F["Fast fail or bounded fallback"]
        O["Classified outcome<br/>latency + origin"]
        R --> C --> B
        B -->|admit| D
        B -->|deny| F
        D --> O --> B
        O --> C
    end

    subgraph CP["Control plane"]
        P[("Versioned policy")]
        T["Fleet telemetry<br/>outliers and incidents"]
        K["Kill switch / override"]
        P --> K
    end

    P -.snapshot.-> B
    B -.events.-> T
    K -.bounded override.-> B
~~~

The data plane makes decisions without a synchronous control-plane round trip. If the breaker depends on a remote state service for every call, that service becomes a new dependency on the failure path.

---

## 3. Scope and Outcome Classification

### Scope narrowly enough to isolate failure

A single breaker for “all outbound HTTP” lets a payment outage block search. One breaker per physical host can be too granular when calls are load-balanced across an elastic cluster and hosts churn. Useful scopes include:

- logical cluster plus operation class;
- endpoint for outlier ejection;
- shard/partition when failures are localized;
- tenant only when the downstream resource is tenant-isolated and cardinality is bounded;
- priority class when critical calls must not share concurrency with bulk work.

Membership changes should retire stale endpoint state. Do not let an old ejection follow a recycled address to a different instance.

### Classify causality

Count outcomes that imply the protected path is unhealthy:

- connect failure, reset, protocol failure;
- dependency timeout while useful deadline remained;
- server-declared overload;
- server error attributable to the dependency;
- slow success beyond a workload-derived limit, if slow calls consume dangerous capacity.

Usually exclude:

- validation, authorization, not-found, and domain conflicts;
- caller cancellation after its own deadline;
- failures introduced before the dependency call;
- deliberate rate-limit rejection from another policy scope.

A 500 is not automatically dependency failure and a 200 is not automatically healthy. A fallback response, partial result, or response after the caller abandoned it may be a failed service outcome.

Record local-origin and remote-origin failures separately. This distinguishes a bad network path from a dependency application failure.

---

## 4. State Machine and Rolling Evidence

~~~mermaid
stateDiagram-v2
    [*] --> Closed
    Closed --> Open: evidence says calls are harmful
    Open --> Probe: probe schedule permits limited traffic
    Probe --> Closed: representative recovery evidence
    Probe --> Open: probe failure or overload
    Closed --> Isolated: operator or policy override
    Open --> Isolated: operator or policy override
    Isolated --> Probe: controlled release
~~~

### Closed

Ordinary calls may start only if a concurrency permit exists. Outcomes update rolling statistics. “Closed” means calls are permitted, not that the dependency is certified healthy.

### Open

Ordinary calls fail fast or use a bounded fallback. The breaker continues to receive control-plane policy and may observe independent endpoint health, but the passage of time does not prove recovery.

### Probe

A small, capped number of representative calls test the real path. Probe concurrency, not just total probe count, matters: simultaneous probes can overload a barely recovered dependency.

Close only after enough successful/acceptable-latency evidence for the workload’s risk. Ramp concurrency rather than jumping from a handful of probes to the entire fleet.

### Count-based versus time-based windows

A count window reacts after a fixed number of outcomes; during low traffic it may contain old evidence for a long time. A time window ages evidence predictably but may have too few samples to estimate a rate. Both need:

- minimum sample count or weight;
- explicit treatment of no traffic;
- bucket rollover and monotonic time;
- configuration for slow-call and failure classification;
- a reset/rebase rule after topology or version changes.

Use additive bucket summaries—success, failure classes, slow outcomes, duration totals—rather than storing every call unless exact audit is required.

### Statistical caution

An observed failure fraction <code>f/n</code> is noisy when <code>n</code> is small. The minimum evidence should reflect the cost of false open versus delayed open, not a copied library default. For rare traffic, consecutive-failure logic or an active dependency-specific signal may be more useful than a percentage.

An exponentially weighted estimate reacts smoothly but does not provide a finite-window count and can hide a short severe incident. Document the estimator and test its response to burst, gradual degradation, and recovery.

---

## 5. Concurrency Limits and Bulkheads

An error breaker reacts too late to pure slowness. When latency rises, in-flight work grows:

> in-flight concurrency ≈ admitted calls/s × mean service time

A fixed concurrency limit bounds resource consumption before thread pools, connection pools, or the dependency collapse. Separate limits by dependency and priority; otherwise one slow backend consumes a shared pool and blocks healthy calls.

### Fixed limit

Derive a starting limit from:

- dependency-tested safe concurrency;
- caller memory/thread/connection budget;
- per-call cost and deadline;
- desired queue wait;
- replica count and failover skew.

A pending queue is optional and must be bounded by both count and deadline. Rejecting before queue insertion is better than completing work after the caller’s deadline.

### Adaptive limit

Adaptive algorithms infer queueing from latency. A gradient-style controller compares a long-term baseline latency with current sampled latency:

> gradient = clamp(baseline latency / current latency)
>
> proposed limit = current limit × gradient + small queue allowance

Then smooth and bound the update. When current latency rises above baseline, the limit contracts; when latency remains low, it explores upward.

This is a feedback controller, not magic:

- workload mix can change the baseline;
- client-side latency includes network and caller queueing;
- too little traffic provides no signal;
- retries and hedges contaminate samples;
- a high minimum limit can prevent protection;
- synchronized callers can oscillate against the same dependency.

Use exploration/probes and cap the rate of change. Observe rejected goodput as well as latency.

### Breaker and limiter ordering

Check deadline, acquire concurrency, then evaluate/call according to breaker state; exact ordering may differ to avoid holding a permit for an open circuit. The invariant is that probes consume permits and every terminal path releases them once.

Rate limiting in [Rate Limiting](./05-rate-limiting.md) controls admitted entitlement. Backpressure in [Backpressure](./07-backpressure.md) controls bounded waiting and producer flow. These mechanisms complement, not replace, dependency concurrency.

---

## 6. Probe and Recovery Design

An open interval should be backoff with jitter, not one fleet-wide timer. If thousands of clients all probe at the same instant, recovery traffic becomes another outage.

Probe choices:

- one elected process per cell;
- per-process probes with randomized schedule and very small concurrency;
- a shared advisory recovery signal followed by local verification;
- progressively increasing real traffic rather than synthetic health checks.

Probe the operation that failed. A shallow <code>/health</code> endpoint can succeed while database-backed requests fail. Conversely, an expensive write is a dangerous probe; use a safe representative read or a domain-supported idempotent transaction.

Recover in stages:

1. Confirm the connection/protocol path.
2. Admit a bounded representative sample.
3. Verify latency and success under rising concurrency.
4. Increase ordinary traffic gradually.
5. Reset old failure evidence only according to the state policy.

Keep the breaker able to reopen immediately during ramp.

---

## 7. Local versus Distributed Breakers

### Local state

Advantages:

- no coordination on the request path;
- reflects one caller’s route, network, endpoint set, and deadlines;
- failure is contained by process/cell;
- scales with callers.

Costs:

- callers disagree;
- low-volume callers have weak samples;
- every process may probe;
- restart loses recent evidence.

Disagreement is often correct: one zone can have a broken path while another is healthy.

### Shared live state

A central breaker can aggregate evidence and suppress a fleet quickly, but it creates:

- a hot coordination key;
- ambiguity about heterogeneous routes and operations;
- a global false-positive blast radius;
- stale-cache and partition semantics;
- synchronized open and probe transitions;
- a new dependency needed during incidents.

Use shared state when the protected resource is genuinely global and its operator supplies authoritative availability/maintenance state. Even then, enforce locally from a versioned snapshot and define what happens when the shared state is unavailable.

### Preferred hybrid

- local breaker and concurrency enforcement;
- fleet telemetry for anomaly detection;
- shared policy/configuration;
- endpoint outlier information scoped to routing domain;
- emergency global override with expiry, audit, and gradual release.

Global telemetry should advise, not erase, local evidence without a clear authority contract.

---

## 8. Fallbacks Are Dependencies

A fallback may return:

- cached data with stated age;
- a reduced feature set;
- an asynchronous receipt instead of synchronous completion;
- a semantically explicit unavailable response.

It must preserve safety. Never fabricate authorization, inventory, price, or payment success. Stale data needs maximum age and provenance.

Fallbacks consume capacity. If every failed recommendation call queries a shared database, the “fallback” transfers the cascade. Give it separate concurrency, admission, deadline, and observability budgets. A fallback failure must not recursively invoke the original dependency.

---

## 9. Concrete Failure Trace: Healthy Probe, Recovery Storm

1. A database-backed dependency slows and local breakers open.
2. Every caller schedules the same open duration.
3. The dependency’s shallow health endpoint remains fast.
4. At timer expiry, all callers send one health probe; all succeed.
5. Every breaker closes and releases its full concurrency limit.
6. The recovering dependency receives the fleet’s entire queued/retried demand.
7. Latency spikes, breakers reopen, and the system oscillates.

The breaker did not fail; its recovery contract was wrong. Jitter probe schedules, cap aggregate probe concurrency, use representative data-plane evidence, ramp the concurrency limit, discard expired queued work, and keep retries inside the budget owned by [Retries, Timeouts, and Hedging](./10-retries-timeouts-hedging.md).

---

## 10. Capacity and Cost Model

Let:

- <code>lambda</code>: offered calls/s at one breaker scope;
- <code>S</code>: mean dependency service time;
- <code>L</code>: in-flight limit;
- <code>Q</code>: pending-queue capacity;
- <code>n</code>: outcomes in the statistics window;
- <code>p_probe</code>: admitted probe calls/s while open;
- <code>C_fallback</code>: fallback resource cost/call.

Stable service without caller queueing requires dependency goodput above admitted rate. Little’s Law gives expected in-flight work near <code>lambda × S</code>; the limit <code>L</code> caps it. If work waits in a full queue:

> queue wait lower bound ≈ Q / completion rate

Reject requests whose remaining deadline is below plausible queue plus service time.

The fastest possible time to collect <code>n</code> samples at rate <code>lambda</code> is <code>n/lambda</code>. Thus a large evidence window delays opening for low traffic and may still be statistically noisy under bursty correlated failures.

While open, avoided dependency work/s is approximately ordinary attempted rate minus probe rate. Added fallback cost/s is approximately fallback rate × <code>C_fallback</code>. Include that capacity in incident planning.

Memory cost is scopes × rolling buckets plus endpoint/priority dimensions. Bound dynamic scopes; per-tenant breakers can become a cardinality attack.

---

## 11. Operations and Migration

### Rollout

1. Instrument classified outcomes without enforcing.
2. Compare proposed transitions against incidents and healthy tail behavior.
3. Enable concurrency limits with conservative shedding and a kill switch.
4. Shadow open decisions and inspect false positives by scope.
5. Enforce for a cohort/cell.
6. Introduce bounded probes and progressive recovery.
7. Expand while tracking goodput and fallback impact.

Changing scope or outcome classification resets the meaning of stored statistics. Version state and allow old/new policies to coexist during rollout rather than reinterpreting old buckets.

### Runbooks

- unexpected fleet-wide open: inspect shared configuration and classification before forcing close;
- breaker flapping: check probe representativeness, synchronized timers, and workload mix;
- no recovery: confirm traffic exists to produce probes and manual isolation is not stale;
- local divergence: compare network path, endpoint membership, version, and sample volume;
- fallback saturation: disable optional fallback and return explicit failure;
- control-plane outage: continue last verified local policy until its safety expiry.

Manual force-open/force-close must expire automatically. A forgotten force-close disables protection precisely during the next incident.

---

## 12. Security and Governance

- Do not treat authentication/authorization failures as dependency outages.
- Authenticate and sign dynamic policy/override distribution.
- Restrict who can force states, change classification, or raise concurrency.
- Audit state overrides, policy versions, and fallback activation.
- Avoid sensitive request attributes in breaker keys and metric labels.
- Prevent attackers from opening a global breaker with cheap intentionally failing requests; require authenticated scope and representative evidence.
- Keep security-critical dependencies fail closed unless the risk contract explicitly permits cached decisions.

A breaker is not a DDoS defense; admission and abuse policy belong at an earlier boundary.

---

## 13. Observability

Per scope, record:

- state and transition reason/config version;
- outcome counts by local/remote origin and classification;
- sample volume, failure/slow rate, latency distribution;
- admitted, rejected, queued, in-flight, and permit-wait counts;
- current adaptive limit, baseline/current latency, and update reason;
- probe scheduled/admitted/succeeded/failed and ramp stage;
- fallback volume, freshness, error, latency, and downstream cost;
- manual override and stale-policy age.

Fleet views need the distribution of local states, not only “percent open.” Correlate dependency goodput with caller rejection and retry attempts. A low downstream error rate can simply mean every circuit is open.

---

## 14. Verification

- classify every protocol/domain outcome with table-driven tests;
- inject isolated endpoint, zone-path, operation, and whole-cluster failures;
- test low-volume and bursty samples around the evidence boundary;
- drive pure latency degradation without errors and verify concurrency protection;
- kill calls and confirm every permit is released exactly once;
- restart processes and change endpoint membership while breakers are open;
- synchronize thousands of simulated callers and prove probe jitter/ramp prevent a herd;
- make shallow health succeed while representative traffic fails;
- saturate the fallback and prove it cannot cascade;
- partition the control plane and exercise cached-policy expiry;
- change policy/scope during traffic and verify state version semantics;
- push beyond saturation, remove the trigger, and prove automatic recovery.

Measure goodput and user-visible deadline success. Fast breaker rejection alone is not success.

---

## 15. Decision Framework

| Need | Mechanism |
|---|---|
| Bound simultaneous dependency work | Fixed or adaptive concurrency limit |
| Stop calling a persistently failing dependency | Rolling-outcome circuit breaker |
| Remove one bad replica | Endpoint outlier detection |
| Enforce user/tenant entitlement | Rate limiter |
| Bound producer queues and signal slowdown | Backpressure |
| Survive one caller’s network-path failure | Local breaker |
| Enforce authoritative global maintenance/isolation | Shared advisory/override plus local enforcement |
| Dependency is cheap, failure immediate, caller already bounded | A breaker may add no value |

Start with deadline propagation and concurrency bounds. Add an error-state breaker when failing calls waste meaningful capacity or amplify a cascade, and only at a scope whose outcomes have one interpretation.

---

## Primary References

- Michael Nygard, [Release It!](https://pragprog.com/titles/mnee2/release-it-second-edition/), for the circuit-breaker and stability patterns.
- Resilience4j, [CircuitBreaker](https://resilience4j.readme.io/docs/circuitbreaker), for count/time sliding windows and half-open state.
- Envoy, [Circuit Breaking](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/circuit_breaking).
- Envoy, [Outlier Detection](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/outlier).
- Netflix, [Concurrency Limits](https://github.com/Netflix/concurrency-limits), including gradient-based adaptive limits.
- Google SRE, [Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/).
- Nathan Bronson et al., [Metastable Failures in Distributed Systems](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s11-bronson.pdf).

---

**Next:** [Backpressure](./07-backpressure.md) follows admitted work through bounded buffers and propagates saturation to producers.
