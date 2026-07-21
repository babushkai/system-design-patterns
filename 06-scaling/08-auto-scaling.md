# Auto-Scaling: Delayed Feedback, Headroom, and Safe Scale-Down

## TL;DR

Autoscaling is a delayed feedback loop: observe a noisy signal, choose desired capacity, wait for provisioning, warm-up, and routing, then measure the effect. A burst can fill queues and exhaust concurrency long before new capacity is ready, so [admission](./05-rate-limiting.md) and [backpressure](./07-backpressure.md) must keep the data plane safe during actuation.

Choose a signal causally related to missing capacity, derive replica demand from per-replica service rate or target utilization, and include metric delay, decision period, provisioning, warm-up, and routing in the response budget. Use hysteresis, stabilization, and bounded step/rate changes to prevent oscillation. Scale down only after removing an instance from admission, draining or transferring work, and proving the remaining fleet can absorb load plus failure headroom.

---

## 1. Scaling Contract

| Field | Required answer |
|---|---|
| **Objective** | Latency/goodput SLO, queue-drain objective, utilization band, or cost under a reliability constraint. |
| **Scalable unit** | Process, pod, VM, node, shard, consumer, function concurrency, or vertically sized resource. |
| **Capacity model** | Useful work/s per ready unit for the actual workload distribution and dependency limits. |
| **Signal** | Metric definition, scope, aggregation, freshness, missing-data behavior, and causal relationship to demand. |
| **Actuator** | Minimum/maximum, step/rate limits, quota, placement, provisioning and warm-up distribution. |
| **Stability policy** | Deadband, scale-up/down evidence, stabilization windows, and conflicting-signal resolution. |
| **Scale-down protocol** | Readiness removal, connection/lease drain, state transfer, termination deadline, and rollback. |
| **Failure headroom** | Capacity retained for instance, zone, region, rollout, and dependency degradation. |
| **Ownership** | Exactly which controller owns desired replica/resource count. |

### Invariants

1. The system remains bounded while the controller observes and capacity starts.
2. Only ready, routed, and dependency-connected units count as serving capacity.
3. Desired capacity has explicit lower, upper, quota, and rate-of-change bounds.
4. One authoritative controller owns each scale field at a time.
5. Scale-down stops new work before removing capacity and does not abandon owned state.
6. The remaining fleet can serve admitted load plus the declared failure/rollout margin.
7. Missing or stale metrics cannot trigger unsafe scale-down.
8. Increasing replicas cannot silently multiply a global rate, retry, or connection budget.

---

## 2. The Feedback Loop

~~~mermaid
flowchart LR
    D["Demand"]
    A["Admission + bounded queue"]
    W["Ready workers"]
    Y["Goodput, latency,<br/>queue and utilization"]
    M["Metric pipeline<br/>sample + aggregate"]
    C["Controller<br/>estimate + policy"]
    P["Provision / initialize<br/>warm + route"]

    D --> A --> W --> Y
    Y --> M --> C --> P --> W
    C -.scale-down intent.-> W
~~~

The loop’s delay is:

> response delay = metric collection + aggregation/export + controller period + API/scheduler + provisioning + initialization + readiness/routing

During that interval, the offered workload continues. Design the bounded backlog or pre-warmed headroom to survive the measured delay distribution, including control-plane incidents.

The controller changes **desired capacity**. The actual plant includes scheduler constraints, image distribution, startup dependencies, cache warming, load-balancer convergence, and downstream bottlenecks. Desired replicas are not capacity.

---

## 3. Capacity Model before Metric Selection

Benchmark one ready unit with:

- realistic request/message cost distribution;
- production concurrency and batching;
- caches both cold and warm;
- real downstream limits;
- timeouts, retries, logging, encryption, and sidecars enabled;
- sustained load long enough to expose throttling, garbage collection, and memory growth.

Let <code>mu_unit</code> be sustainable useful completions/s per unit at the target tail latency, not the peak throughput at collapse. With arrival <code>lambda</code> and fractional headroom <code>h</code>:

> base units ≥ ceil(lambda / (mu_unit × (1 − h)))

Headroom must also cover the selected failure domain. If losing a fraction <code>f</code> of units is in scope, normal placement/capacity must satisfy useful demand after that loss; do not assume the autoscaler creates replacements during the failure.

Some workloads do not scale linearly:

- a database or external API caps total throughput;
- partition count caps active consumers;
- one hot key stays on one worker;
- coordination and cache miss rates grow with replicas;
- node/network/storage bandwidth becomes shared;
- each replica opens connection pools that overload the dependency.

Scale the bottleneck or shed work; adding workers beyond the constraint can reduce goodput.

---

## 4. Choosing Signals

### Resource utilization

CPU utilization works when CPU is the binding resource and per-unit work is stable. It fails when:

- requests wait on I/O while CPU is low;
- CPU throttling distorts observed usage;
- work is rejected before consuming CPU;
- one container/tenant is hot but an average is low;
- missing resource requests make the utilization denominator meaningless;
- a downstream bottleneck dominates latency.

Memory is often state rather than load and may not fall after traffic drops. Scaling replicas does not cure a leak.

### Concurrency

In-flight work maps directly to occupied permits for synchronous services. Scale from utilization of a tested per-unit concurrency limit, but separate running from queued and abandoned calls.

### Queue backlog and age

Backlog is useful for asynchronous workers. Depth alone ignores arrival rate, service cost, expired messages, partition parallelism, and drain objective. Oldest age is often closer to user impact.

For backlog <code>B</code>, new arrival <code>lambda</code>, per-worker sustainable completion <code>mu</code>, and desired drain time <code>T</code>:

> required workers ≥ ceil((lambda + B/T) / mu)

This requires <code>mu</code> measured for the current message mix and eligible parallelism. If a partition/key ordering constraint allows only <code>P</code> active consumers, useful workers are capped by <code>P</code>.

### Request rate

External request rate can be a leading demand signal when work per request is predictable. Derive desired units directly from tested <code>mu_unit</code>. If request cost varies, scale on weighted work or separate classes.

### Latency and error rate

Tail latency and errors are objectives but usually late, nonlinear scaling signals. By the time queueing raises p99, the fleet may already be saturated. Use them as guards/validation and pair with a leading signal.

### Multiple metrics

Compute a capacity recommendation per constraint and normally choose the maximum safe recommendation for scale-up. For scale-down, missing or conflicting metrics should retain capacity. Document aggregation: mean hides a hot replica, max can chase one pathological request, and percentile estimation has window delay.

---

## 5. Controller Algorithms

### Proportional target tracking

For a per-unit metric <code>m_current</code> with target <code>m_target</code> and <code>N</code> current units:

> desired units = ceil(N × m_current / m_target)

This assumes metric load approximately divides across replicas. It does not hold for hot keys or shared bottlenecks. Exclude or conservatively treat not-ready/missing-metric units so startup does not cause an unsafe reverse decision.

For a total external metric, derive units from total demand divided by target work/unit rather than averaging over current replicas.

### Deadband and hysteresis

Measurement noise near target can alternate scale-up/down. A deadband takes no action for small error. Hysteresis can use separate evidence for entering and leaving a capacity level. Derive it from metric variance, unit granularity, and cost/SLO tradeoff rather than a copied percentage.

### Stabilization

Retain recent recommendations:

- scale up using evidence fast enough to protect the SLO, subject to false-signal risk;
- scale down using the highest recent safe recommendation so a brief dip does not remove capacity;
- rate-limit additions/removals to what scheduler, dependencies, and drain can tolerate.

Stabilization is not a blind cooldown that ignores a worsening incident. Continue measuring and permit emergency scale-up or stop scale-down.

### Step scaling

Discrete steps suit large indivisible units or known thresholds, but every boundary needs hysteresis. Simulate step response with metric/actuation delay; an aggressive step based on stale backlog can overshoot after work already drained.

### Feedforward plus feedback

Scheduled or predictive scaling is feedforward: provision before known demand. It should not replace feedback because forecasts miss launches, incidents, and workload mix. Combine:

> desired = max(forecast capacity, feedback capacity, failure floor)

then apply bounds and placement. Record forecast error and its cost.

---

## 6. Cold Start and Scale-to-Zero

Cold-start path:

1. The controller detects need.
2. Quota and scheduler accept placement.
3. A node/VM exists or is provisioned.
4. The image/artifact downloads.
5. The process starts and loads configuration, keys, code, model, or cache.
6. Dependencies establish pools/leases.
7. Readiness passes.
8. Routing converges.
9. The new unit reaches sustainable goodput.

Measure time-to-first-ready and time-to-full-capacity separately. A process that passes readiness while loading a large cache can draw traffic and worsen the incident.

Options:

- keep a minimum warm fleet;
- maintain warm pools or pre-pulled artifacts;
- provision ahead from forecast;
- reduce initialization dependencies and artifact size;
- lazy-load optional state while readiness protects expensive routes;
- buffer only within the deadline/retention bound.

Scale-to-zero is appropriate when cold-start latency fits the product contract or an upstream durable queue can hold work safely. It is not appropriate for a synchronous path whose deadline is shorter than the measured cold start.

---

## 7. Safe Scale-Down

A capacity unit may own:

- active requests/streams;
- queue partitions and unacknowledged messages;
- leases, locks, or shard leadership;
- local state/cache needed by sessions;
- connection pools seen by upstreams.

Protocol:

1. The controller marks the unit draining and stops new assignment.
2. Service discovery/load balancing removes it from new traffic.
3. Producers observe the change.
4. Active work completes, transfers, checkpoints, or reaches a defined termination policy.
5. Leases, partitions, and leadership are released or fenced.
6. The remaining fleet is re-evaluated under the new load.
7. The unit terminates; forced termination is recorded and reconciled.

Scale-down capacity should include terminating units separately from ready capacity. A drain that takes longer than the controller’s scale-down interval can cause multiple overlapping removals.

### Stateful and partitioned workers

Rebalance can temporarily reduce goodput and amplify network/storage. Scale one step, observe rebalance completion, then continue. For sticky keys or caches, the cold-cache cost may exceed the saved compute.

### Multiple controllers

Deployment configuration, autoscaler, rollout controller, manual operator, vertical scaler, and node scaler can fight over replicas/resources. Establish field ownership. When introducing an autoscaler, transfer ownership without applying a stale static replica value that collapses the fleet.

---

## 8. Horizontal, Vertical, and Node Scaling

- **Horizontal:** adds parallel units; works for partitionable work and increases coordination/connection footprint.
- **Vertical:** changes CPU/memory; may require restart and has resource/host limits.
- **Node/cluster:** supplies placement capacity underneath workload scaling; usually slower and can make pending workload metrics misleading.

Nested loops need separated timescales and observability. Workload scale-up may create pending units, triggering node scale-up; once nodes arrive, workload metrics may already have changed. Scale-down loops can evict workload while another loop tries to add it.

Reserve node capacity for fast workload scale-up or model the full two-stage delay. Placement constraints, disruption budgets, and zone balance can make nominal free capacity unusable.

---

## 9. Concrete Failure Trace: Queue Metric Oscillation

1. A burst creates backlog <code>B</code>.
2. Queue metrics export after delay; the controller requests many workers.
3. Existing workers continue draining during provisioning.
4. New workers become ready after most backlog is gone.
5. They all prefetch, open dependency pools, and drive the queue metric near zero.
6. The controller immediately scales down from the stale low signal.
7. Draining/rebalance pauses consumption; new arrivals rebuild the queue.
8. Delayed high metrics trigger another large scale-up.

Fix it by calculating workers from arrival + drain objective, including actuation delay, bounding scale rate, using scale-down stabilization, observing ready/starting/draining separately, limiting prefetch and dependency concurrency, and retaining a minimum/failure floor.

---

## 10. Composition with Overload Budgets

Autoscaling changes capacity after other controls decide what survives:

- rate-limit global entitlement independent of replica count;
- bound per-replica and aggregate dependency concurrency;
- keep queues finite and propagate saturation;
- cap retries/hedges so they do not masquerade as new demand;
- retain multi-region failover headroom without depending on just-in-time scaling.

Decide which traffic metric counts:

> original demand ≠ attempts ≠ admitted work ≠ completed goodput

Scaling on retries can amplify a retry storm. Scaling on admitted work alone can hide rejected legitimate demand. Observe all four and use the objective’s correct one.

---

## 11. Capacity and Cost Model

Let:

- <code>N_ready</code>, <code>N_starting</code>, <code>N_draining</code>: actual lifecycle counts;
- <code>mu_unit</code>: useful completions/s per ready unit;
- <code>T_act</code>: chosen percentile of full scale-out actuation delay;
- <code>lambda_peak</code>: peak offered/admitted work rate as appropriate;
- <code>B_free</code>: safe free backlog capacity;
- <code>c_unit</code>: cost per unit time;
- <code>c_start</code>: one-time startup/warm/cache/dependency cost.

The backlog added during actuation is at least:

> B_added = max(0, lambda_peak − N_ready × mu_unit) × T_act

Require <code>B_free ≥ B_added</code> or shed enough work to satisfy it.

Scale-out must also respect downstream aggregate limits. If each replica can open <code>k</code> dependency calls, total potential dependency concurrency is <code>N_ready × k</code>; allocate a global/regional budget or reduce <code>k</code> as N grows.

Approximate capacity cost:

> steady cost = integral of ready + starting + draining unit-time × c_unit
>
> churn cost = scale events × c_start + cache/rebalance/egress work

Optimize cost per useful completion while meeting failure and latency objectives. A controller that flaps can appear to reduce average replicas while increasing startup and downstream cost.

---

## 12. Operations and Migration

### Rollout

1. Establish the per-unit capacity curve and actuation distribution.
2. Shadow recommendations against current capacity.
3. Set minimum, maximum, quota, and emergency manual bounds.
4. Enable scale-up only for a cohort.
5. Validate readiness, routing, connection budgets, and cold-cache impact.
6. Introduce conservative drain-aware scale-down.
7. Test rollout and failure-domain loss while scaling.
8. Transfer ownership from static configuration/controllers explicitly.

### Runbooks

- max reached: determine whether quota, placement, dependency, or true demand is limiting;
- pending units: inspect node capacity, affinity, images, secrets, and startup dependencies;
- ready but no goodput: inspect routing, warm-up, hot partitions, and downstream limits;
- oscillation: compare signal timestamp with action/effect timeline;
- unsafe scale-down: freeze removals, restore floor, reconcile interrupted work;
- metric outage: hold safe capacity and alert rather than infer zero.

Autoscaler config is production code. Version, review, canary, and roll it back.

---

## 13. Security and Governance

- Authenticate and authorize scale-policy and manual-bound changes.
- Prevent untrusted tenants from directly controlling global scaling metrics.
- Bound cardinality and validate external metric labels/values.
- Protect metric/control APIs from spoofing, replay, and stale writes.
- Audit changes to minimum, maximum, signal, target, and manual overrides.
- Restrict workload identity so new units receive only required secrets/resources.
- Ensure scale-out does not exceed licensed, privacy, regional, or budget constraints.
- Treat denial-of-wallet separately from denial-of-service: admission still caps abusive demand.

Scaling is not a security control; it can turn attack traffic into a cost incident.

---

## 14. Observability

Overlay on one timeline:

- offered/original demand, attempts, admitted work, completion goodput;
- signal value, timestamp, freshness, and missing samples;
- controller recommendation, clamp/rate/stabilization reason;
- desired, pending, starting, ready, draining, failed units;
- scheduler/provision/start/readiness/routing durations;
- per-unit throughput, concurrency, utilization, latency, and distribution skew;
- queue depth/bytes/oldest age and predicted drain time;
- downstream connection/concurrency and throttling;
- scale event/churn/startup cost and forecast error.

Alert on inability to meet the contract: actuation slower than buffer headroom, max/quota saturation, no goodput from new capacity, unsafe drain, stale metrics, or repeated oscillation.

---

## 15. Verification

- replay measured demand traces through an offline controller simulation;
- step, ramp, burst, periodic, and workload-mix changes;
- delay, drop, duplicate, and reorder metric samples;
- make provisioning and warm-up slower than normal;
- exhaust placement quota and node capacity;
- add replicas against a fixed downstream bottleneck;
- create one hot key/partition while fleet average remains low;
- scale during rollout and failure-domain loss;
- terminate units with active streams, leases, and unacknowledged work;
- run multiple controllers and verify field ownership;
- push beyond saturation, remove the trigger, and prove the loop converges without oscillation;
- compare predicted capacity, ready capacity, and actual goodput.

A load test that begins with a fully warm maximum fleet does not test autoscaling.

---

## 16. Decision Framework

| Workload property | Scaling direction |
|---|---|
| Parallel stateless work, stable per-request cost | Horizontal target tracking on leading demand/concurrency |
| Durable queue with known service rate | Arrival + backlog drain-time model |
| Known scheduled demand and long startup | Feedforward pre-scaling plus feedback |
| Large memory state/cache, poor partitionability | Vertical or shard-aware scaling |
| Scale unit starts after request deadline | Maintain warm floor; do not rely on reactive scale |
| Fixed downstream capacity | Admission/concurrency first; extra replicas may hurt |
| Rare workload tolerates cold start | Scale-to-zero may fit |
| Strong hot-key skew | Repartition/isolate key before adding fleet capacity |

Choose the simplest stable controller whose signal leads the objective and whose actuation fits the buffer/headroom envelope. Manual capacity with alerts is better than an unstable automatic loop.

---

## Primary References

- Kubernetes, [Horizontal Pod Autoscaling](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/), including the control loop and replica-ratio algorithm.
- Kubernetes, [Pod Lifecycle and Termination](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/).
- Kubernetes, [Disruptions](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/).
- KEDA, [Scaling Deployments and StatefulSets](https://keda.sh/docs/latest/concepts/scaling-deployments/).
- AWS Builders’ Library, [Static Stability Using Availability Zones](https://aws.amazon.com/builders-library/static-stability-using-availability-zones/).
- Google SRE Workbook, [Managing Load](https://sre.google/workbook/managing-load/).
