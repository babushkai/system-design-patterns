# Auto-Scaling: Delayed Feedback, Headroom, and Safe Scale-Down

## TL;DR

Autoscaling is a delayed feedback loop. It observes a noisy signal, chooses a desired capacity, waits for provisioning, warm-up, and routing, and only then sees the effect of its decision. A burst can fill queues and exhaust concurrency long before new capacity is ready, so [admission](./05-rate-limiting.md) and [backpressure](./07-backpressure.md) have to keep the data plane safe while the loop acts.

We choose a signal that is causally related to missing capacity, derive the required number of replicas from the per-replica service rate or a target utilization, and include metric delay, decision period, provisioning, warm-up, and routing in the response budget. Hysteresis, stabilization, and bounded steps prevent oscillation. Capacity is removed only after an instance has been taken out of admission, its work has drained or been transferred, and the remaining fleet has been shown able to absorb the load with its failure headroom intact.

---

## Scaling Contract

Autoscaling is often described as a way of matching capacity to demand automatically, as though it were a setting. It is better understood as a control system, of the same family as a thermostat or a cruise control, and it inherits the characteristic difficulty of that family, which is delay. A thermostat that measured the temperature of the room ten minutes ago and whose furnace takes ten minutes to produce heat will overshoot in both directions. An autoscaler works from metrics that are tens of seconds old and commands capacity that takes minutes to arrive.

Two consequences follow, and the rest of the chapter elaborates them. First, the autoscaler cannot protect a system from a sudden burst, because by the time it has reacted the burst has already been absorbed or has already caused the damage. Something faster must protect the system in the meantime. Second, a loop with delay can become unstable, adding and removing capacity in a cycle that is worse than leaving the capacity alone.

The contract makes the loop's parts explicit:

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

The seventh invariant guards against a failure that has emptied production fleets. If the metrics pipeline breaks and the autoscaler reads the absence of data as zero load, it concludes that no capacity is needed and removes it. Missing data must be treated as a reason to hold what exists.

---

## The Feedback Loop

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

The delay around the loop is the sum of every stage:

> response delay = metric collection + aggregation/export + controller period + API/scheduler + provisioning + initialization + readiness/routing

It is instructive to add these up for a real system. Metrics are scraped every fifteen or thirty seconds and averaged over a window. The controller evaluates on its own period. The scheduler must find a node, and if none has room, a new machine must be provisioned, which takes minutes. The container image must be pulled. The process must start, load its configuration, open connections, and warm its caches. The load balancer must learn that the new instance exists. A total of three to five minutes is ordinary, and during all of it demand continues to arrive at whatever rate caused the trigger.

The design therefore has to survive that interval by other means, either a bounded backlog that can absorb the excess or headroom that is already warm. The interval should be taken from the measured distribution, including its tail and including the occasions when the control plane itself is having an incident.

The controller changes **desired capacity**, which is a number in a configuration object. Actual capacity is whatever the scheduler, image distribution, startup dependencies, cache warming, load-balancer convergence, and downstream bottlenecks eventually deliver. A dashboard showing forty desired replicas says nothing about how many are serving.

---

## Capacity Model Before Metric Selection

Before choosing what to measure, we need to know how much work one unit can do. That figure comes from a benchmark of a single ready unit under conditions that resemble production:

- realistic request/message cost distribution;
- production concurrency and batching;
- caches both cold and warm;
- real downstream limits;
- timeouts, retries, logging, encryption, and sidecars enabled;
- sustained load long enough to expose throttling, garbage collection, and memory growth.

Let <code>mu_unit</code> be the sustainable rate of useful completions per unit at the target tail latency. This is well below the peak throughput at which the unit collapses, and the difference matters. With arrival <code>lambda</code> and fractional headroom <code>h</code>:

> base units ≥ ceil(lambda / (mu_unit × (1 − h)))

Headroom also has to cover the chosen failure domain. If the loss of a fraction <code>f</code> of units is in scope, such as one zone of three, the capacity present before the failure must be enough to serve demand after it. Counting on the autoscaler to create replacements during the failure is unsafe, because the failure that removed a zone may well have impaired the control plane, and because every other tenant of the same cloud region is asking for replacement capacity at the same moment. A system that already holds enough capacity to survive the loss, and needs nothing from the control plane in order to do so, is called statically stable.

Scaling out assumes that capacity grows in proportion to the number of units. Several common situations break that assumption:

- a database or external API caps total throughput;
- partition count caps active consumers;
- one hot key stays on one worker;
- coordination and cache miss rates grow with replicas;
- node/network/storage bandwidth becomes shared;
- each replica opens connection pools that overload the dependency.

The last of these means that scaling out can make matters worse. If a service is slow because its database is saturated, the autoscaler sees high latency or a growing queue and adds replicas. Each new replica opens its own pool of connections to the database, which now has more clients contending for the same capacity. Goodput falls and the autoscaler adds more. Where the bottleneck is elsewhere, the remedy is to scale the bottleneck or shed work.

---

## Choosing Signals

The signal is the hardest design choice. A good signal rises when capacity is short, rises before users are affected, and falls when capacity is added. Many popular signals fail at least one of these.

### Resource utilization

CPU utilization is the default, and it works when CPU is the binding resource and the work per request is stable. It fails in several situations:

- requests wait on I/O while CPU is low;
- CPU throttling distorts observed usage;
- work is rejected before consuming CPU;
- one container/tenant is hot but an average is low;
- missing resource requests make the utilization denominator meaningless;
- a downstream bottleneck dominates latency.

The first is the usual case for a service that spends its time waiting on a database or another service. Its threads are all blocked, its queue is growing, its users are timing out, and its CPU is nearly idle. An autoscaler watching CPU sees a system with ample spare capacity.

Memory is a poor signal for a different reason. Memory use often reflects state, such as caches and heap that the runtime has not returned, and it may not fall when traffic does. Adding replicas does not cure a leak.

### Concurrency

For a synchronous service, the number of requests in flight corresponds directly to occupied capacity. We scale on utilization of a tested per-unit concurrency limit, counting running calls separately from queued and abandoned ones.

### Queue backlog and age

For asynchronous workers, backlog is the natural signal, though depth alone leaves out the arrival rate, the cost of each item, expired messages, limits on parallelism, and how quickly the backlog should be cleared. The age of the oldest item is usually closer to what users experience.

For backlog <code>B</code>, new arrival <code>lambda</code>, per-worker sustainable completion <code>mu</code>, and desired drain time <code>T</code>:

> required workers ≥ ceil((lambda + B/T) / mu)

The formula has two terms. The first is the number of workers needed to keep pace with new arrivals. The second is the extra needed to clear the existing backlog within the target time. Scaling on depth alone amounts to using the second term without the first.

The value of <code>mu</code> must be measured for the current mix of messages and the parallelism actually available. If an ordering constraint allows only <code>P</code> active consumers, workers beyond <code>P</code> contribute nothing.

### Request rate

The external request rate is a leading signal, since it rises before latency does, and it works when the cost per request is predictable. Desired units follow directly from the tested <code>mu_unit</code>. Where request cost varies widely, we scale on weighted work or separate the classes.

### Latency and error rate

Tail latency and error rate are what we care about, and for that reason they are tempting as signals. They are late and nonlinear. Latency in a queueing system stays nearly flat as load rises and then climbs steeply near saturation, so by the time the ninety-ninth percentile has moved, the fleet is already saturated and the delay of the loop has yet to begin. These metrics are best used as guards and for validation, paired with a leading signal.

### Multiple metrics

With several signals, each yields its own recommendation, and for scaling up we normally take the largest safe one. For scaling down, missing or conflicting metrics are a reason to keep capacity. The aggregation needs to be written down, because each choice has a failure mode. A mean hides a single hot replica. A maximum can chase one pathological request. A percentile lags by the length of its window.

---

## Controller Algorithms

### Proportional target tracking

For a per-unit metric <code>m_current</code> with target <code>m_target</code> and <code>N</code> current units:

> desired units = ceil(N × m_current / m_target)

If ten replicas are at eighty percent utilization and the target is fifty, the result is sixteen. The formula assumes that load divides evenly across replicas, so that adding replicas lowers the metric in proportion. That fails for hot keys and shared bottlenecks. Units that are not ready, or that report no metrics, are excluded or treated conservatively, since otherwise a batch of starting replicas reporting zero load drags the average down and prompts a scale-down in the middle of a scale-up.

For a metric that describes total external demand, units are derived from the total divided by the target work per unit. Averaging it over current replicas would be a mistake.

### Deadband and hysteresis

Metrics are noisy, and a controller that reacts to every fluctuation around the target will add and remove capacity continually. A deadband ignores small errors. Hysteresis uses different thresholds for scaling up and scaling down, so that the metric has to move some distance in the opposite direction before the last action is reversed. Both are derived from the variance of the metric, the size of a unit, and the trade between cost and the service objective.

### Stabilization

The controller keeps a history of its recent recommendations and treats the two directions asymmetrically:

- scale up using evidence fast enough to protect the SLO, subject to false-signal risk;
- scale down using the highest recent safe recommendation so a brief dip does not remove capacity;
- rate-limit additions/removals to what scheduler, dependencies, and drain can tolerate.

The asymmetry reflects the costs of error. Too much capacity for ten minutes costs a little money. Too little costs an outage. So we scale up quickly on modest evidence and scale down slowly on strong evidence.

Stabilization differs from a blind cooldown, which ignores all input for a fixed period after any action. A cooldown would prevent a second scale-up while an incident grows. The controller keeps measuring throughout and remains free to scale up in an emergency or to stop a scale-down.

### Step scaling

Discrete steps suit large indivisible units and known thresholds, and every boundary between steps needs hysteresis. The step response should be simulated with realistic metric and actuation delay, because an aggressive step taken on a stale backlog reading can overshoot badly once the backlog has already drained.

### Feedforward plus feedback

Everything so far is feedback: the controller reacts to what has happened. Feedforward acts on what is expected. If traffic rises every weekday at nine, capacity can be added at half past eight, and the delay of the loop no longer matters. Scheduled and predictive scaling are of this kind.

Feedforward cannot stand alone, since forecasts do not anticipate launches, incidents, or changes in workload mix. The two are combined:

> desired = max(forecast capacity, feedback capacity, failure floor)

with bounds and placement applied afterward. Forecast error and its cost are recorded.

---

## Cold Start and Scale-to-Zero

The path from a decision to useful capacity has many steps:

1. The controller detects need.
2. Quota and scheduler accept placement.
3. A node/VM exists or is provisioned.
4. The image/artifact downloads.
5. The process starts and loads configuration, keys, code, model, or cache.
6. Dependencies establish pools/leases.
7. Readiness passes.
8. Routing converges.
9. The new unit reaches sustainable goodput.

Steps 7 and 9 are different moments, and the time to each should be measured separately. A process may pass its readiness check while its caches are empty and its runtime has not yet compiled its hot paths. It then receives a full share of traffic and serves it slowly. New capacity that is ready but cold can make an incident briefly worse.

The options for shortening or avoiding the path are:

- keep a minimum warm fleet;
- maintain warm pools or pre-pulled artifacts;
- provision ahead from forecast;
- reduce initialization dependencies and artifact size;
- lazy-load optional state while readiness protects expensive routes;
- buffer only within the deadline/retention bound.

Scale-to-zero removes all instances when there is no traffic, so that an idle service costs nothing. It suits workloads where the cold-start latency is acceptable to the product, or where a durable queue upstream can hold work safely until an instance exists. It does not suit a synchronous path whose deadline is shorter than the measured cold start, since the first request after any idle period will then always fail.

---

## Safe Scale-Down

Adding capacity is the safe direction. Removing it is where autoscalers cause outages, because the unit being removed is usually doing something. It may own:

- active requests/streams;
- queue partitions and unacknowledged messages;
- leases, locks, or shard leadership;
- local state/cache needed by sessions;
- connection pools seen by upstreams.

Removal follows a protocol:

1. The controller marks the unit draining and stops new assignment.
2. Service discovery/load balancing removes it from new traffic.
3. Producers observe the change.
4. Active work completes, transfers, checkpoints, or reaches a defined termination policy.
5. Leases, partitions, and leadership are released or fenced.
6. The remaining fleet is re-evaluated under the new load.
7. The unit terminates; forced termination is recorded and reconciled.

Units that are terminating are counted separately from ready capacity. If draining takes longer than the controller's scale-down interval, the controller may decide to remove another unit while the first is still draining, and then another. Several overlapping removals follow, each of which looked safe when it was decided.

### Stateful and partitioned workers

When a consumer leaves a group, its partitions are reassigned, and during the rebalance consumption may pause for the whole group. Removing several workers in quick succession triggers several rebalances. We remove one step, wait for the rebalance to complete, and then continue. For workloads with sticky keys or large caches, the cost of cold caches after rebalancing may exceed the saving in compute.

### Multiple controllers

The replica count of a workload can be written by the deployment configuration, the autoscaler, a rollout controller, a human operator, a vertical scaler, and a node scaler. If two of them believe they own it, they fight, each undoing the other's change. Field ownership has to be established. One hazard arises when an autoscaler is introduced: a deployment that still declares a static replica count will, on its next application, reset the fleet to that count, whatever the autoscaler had scaled it to.

---

## Horizontal, Vertical, and Node Scaling

- **Horizontal:** adds parallel units; works for partitionable work and increases coordination/connection footprint.
- **Vertical:** changes CPU/memory; may require restart and has resource/host limits.
- **Node/cluster:** supplies placement capacity underneath workload scaling; usually slower and can make pending workload metrics misleading.

These are separate control loops, and they nest. A workload autoscaler asks for more replicas. If the cluster has no room, the replicas wait as pending, and that prompts the node autoscaler to add machines, which takes minutes. By the time the nodes arrive, the workload's metrics may have changed. In the other direction, a node autoscaler that removes an underused machine evicts the workloads on it, and the workload autoscaler then tries to replace them.

Nested loops need to run on clearly different timescales, with the inner one faster, and need enough observability to tell which loop caused a given change. Either spare node capacity is reserved for fast workload scale-up, or the full two-stage delay is included in the response budget. Placement constraints, disruption budgets, and zone balance can also make capacity that is nominally free unusable.

---

## Concrete Failure Trace: Queue Metric Oscillation

1. A burst creates backlog <code>B</code>.
2. Queue metrics export after delay; the controller requests many workers.
3. Existing workers continue draining during provisioning.
4. New workers become ready after most backlog is gone.
5. They all prefetch, open dependency pools, and drive the queue metric near zero.
6. The controller immediately scales down from the stale low signal.
7. Draining/rebalance pauses consumption; new arrivals rebuild the queue.
8. Delayed high metrics trigger another large scale-up.

This is the oscillation that delay produces, and it has the same shape as the overshooting thermostat. At step 2 the controller acts on a reading that is already old. At step 4 its response arrives after the problem has largely resolved, so it has overshot. At step 6 it sees the result of the overshoot, an empty queue, and overcorrects. At step 7 the correction itself causes a disturbance, since rebalancing pauses consumption, and the cycle begins again. Each decision was reasonable given what the controller could see when it made it.

The repair addresses each stage. Workers are computed from arrival rate plus a drain objective, with the actuation delay included in the model. The rate of scaling is bounded and scale-down is stabilized. Ready, starting, and draining units are observed separately. Prefetch and dependency concurrency are limited, and a minimum floor is retained.

---

## Composition With Overload Budgets

Autoscaling changes capacity after the faster controls have decided what survives the interval:

- rate-limit global entitlement independent of replica count;
- bound per-replica and aggregate dependency concurrency;
- keep queues finite and propagate saturation;
- cap retries/hedges so they do not masquerade as new demand;
- retain multi-region failover headroom without depending on just-in-time scaling.

The first item deals with an interaction that is easy to miss. If each replica enforces a rate limit of a hundred requests a second for a tenant, scaling from ten replicas to twenty silently doubles that tenant's effective limit. Global budgets have to be defined independently of the number of replicas.

Which measure of traffic the controller watches also matters:

> original demand ≠ attempts ≠ admitted work ≠ completed goodput

Scaling on attempts, which include retries, can amplify a retry storm, since the controller reads the retries as demand. Scaling on admitted work alone can hide legitimate demand that is being rejected at the door. All four should be observed, and the controller should use the one its objective is actually about.

---

## Capacity and Cost Model

Let:

- <code>N_ready</code>, <code>N_starting</code>, <code>N_draining</code>: actual lifecycle counts;
- <code>mu_unit</code>: useful completions/s per ready unit;
- <code>T_act</code>: chosen percentile of full scale-out actuation delay;
- <code>lambda_peak</code>: peak offered/admitted work rate as appropriate;
- <code>B_free</code>: safe free backlog capacity;
- <code>c_unit</code>: cost per unit time;
- <code>c_start</code>: one-time startup/warm/cache/dependency cost.

The backlog that accumulates while waiting for new capacity is at least:

> B_added = max(0, lambda_peak − N_ready × mu_unit) × T_act

We require <code>B_free ≥ B_added</code>, or enough shedding to make it so. This inequality ties autoscaling to buffering. The excess of demand over current capacity, multiplied by the time it takes to add capacity, is the amount of work that has to wait somewhere. If the buffer cannot hold that much within its deadline, there are three remedies: more standing headroom, faster actuation, or load shedding.

Scale-out must respect the aggregate limits of downstream systems. If each replica can open <code>k</code> calls to a dependency, the total potential concurrency is <code>N_ready × k</code>. Either a global or regional budget is allocated, or <code>k</code> is reduced as the fleet grows.

Approximate capacity cost:

> steady cost = integral of ready + starting + draining unit-time × c_unit
>
> churn cost = scale events × c_start + cache/rebalance/egress work

The quantity to optimize is cost per useful completion, subject to the failure and latency objectives. A controller that flaps may show a lower average replica count and still cost more, once startup, cache warming, and the load its churn places on dependencies are counted.

---

## Operations and Migration

### Rollout

1. Establish the per-unit capacity curve and actuation distribution.
2. Shadow recommendations against current capacity.
3. Set minimum, maximum, quota, and emergency manual bounds.
4. Enable scale-up only for a cohort.
5. Validate readiness, routing, connection budgets, and cold-cache impact.
6. Introduce conservative drain-aware scale-down.
7. Test rollout and failure-domain loss while scaling.
8. Transfer ownership from static configuration/controllers explicitly.

The sequence enables scale-up before scale-down, in keeping with the asymmetry of risk. Running the autoscaler in shadow, where it records what it would have done without doing it, shows whether its recommendations are sensible before it is given control.

### Runbooks

- max reached: determine whether quota, placement, dependency, or true demand is limiting;
- pending units: inspect node capacity, affinity, images, secrets, and startup dependencies;
- ready but no goodput: inspect routing, warm-up, hot partitions, and downstream limits;
- oscillation: compare signal timestamp with action/effect timeline;
- unsafe scale-down: freeze removals, restore floor, reconcile interrupted work;
- metric outage: hold safe capacity and alert rather than infer zero.

Autoscaler configuration is production code. It is versioned, reviewed, canaried, and rolled back like any other.

---

## Security and Governance

- Authenticate and authorize scale-policy and manual-bound changes.
- Prevent untrusted tenants from directly controlling global scaling metrics.
- Bound cardinality and validate external metric labels/values.
- Protect metric/control APIs from spoofing, replay, and stale writes.
- Audit changes to minimum, maximum, signal, target, and manual overrides.
- Restrict workload identity so new units receive only required secrets/resources.
- Ensure scale-out does not exceed licensed, privacy, regional, or budget constraints.
- Treat denial-of-wallet separately from denial-of-service: admission still caps abusive demand.

Scaling is sometimes offered as a defense against traffic floods, on the reasoning that the system will simply grow to absorb them. An autoscaler cannot tell an attack from popularity, and it will provision for either. The service may stay up, and the attacker has then succeeded in running up the bill. Admission control is what limits abusive demand, and a maximum on scale limits the cost.

---

## Observability

Diagnosis of an autoscaler depends on seeing cause and effect on one timeline:

- offered/original demand, attempts, admitted work, completion goodput;
- signal value, timestamp, freshness, and missing samples;
- controller recommendation, clamp/rate/stabilization reason;
- desired, pending, starting, ready, draining, failed units;
- scheduler/provision/start/readiness/routing durations;
- per-unit throughput, concurrency, utilization, latency, and distribution skew;
- queue depth/bytes/oldest age and predicted drain time;
- downstream connection/concurrency and throttling;
- scale event/churn/startup cost and forecast error.

The third item records why the controller did what it did. When the recommendation was sixty and the fleet went to forty, the reason may have been a maximum, a rate limit, or a stabilization window, and without that record the controller's behavior is opaque.

Alerts are raised on inability to meet the contract: actuation slower than the buffer can cover, saturation at the maximum or at quota, new capacity that yields no goodput, unsafe drain, stale metrics, and repeated oscillation.

---

## Verification

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

Replaying recorded demand through a simulation of the controller is inexpensive and catches a great deal. It shows how the policy would have behaved on last month's traffic, including the oscillations, without any risk to production.

A load test that begins with a fully warm fleet at maximum size tests the capacity of that fleet. It does not test autoscaling, which is the process of getting there.

---

## Decision Framework

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

We choose the simplest stable controller whose signal leads the objective and whose actuation time fits within what buffers and headroom can cover. Where no such controller exists, fixed capacity with good alerting is the better arrangement. An unstable automatic loop is worse than no loop, because it generates its own incidents.

---

## References

- Kubernetes, [Horizontal Pod Autoscaling](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/), including the control loop and replica-ratio algorithm.
- Kubernetes, [Pod Lifecycle and Termination](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/).
- Kubernetes, [Disruptions](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/).
- KEDA, [Scaling Deployments and StatefulSets](https://keda.sh/docs/latest/concepts/scaling-deployments/).
- AWS Builders’ Library, [Static Stability Using Availability Zones](https://aws.amazon.com/builders-library/static-stability-using-availability-zones/).
- Google SRE Workbook, [Managing Load](https://sre.google/workbook/managing-load/).

---

## Related Chapters

- [Capacity Planning and Back-of-the-Envelope Estimation](../01-foundations/10-capacity-planning.md): the headroom target a controller is holding without being told what it is
- [Horizontal vs Vertical Scaling](02-horizontal-vertical.md): what the loop is actually allowed to add, and at what granularity
- [Metrics Systems and Monitoring](../11-observability/02-metrics-monitoring.md): sampling interval, clock skew, and signal causality decide the control loop
- [FinOps and Cost Engineering](../11-observability/06-finops-cost-engineering.md): capacity bought and unused is the cost side of the same decision
- [Compute Platform and Scheduling Architecture](../15-deployment/07-compute-platform-scheduling.md): placement constraints and headroom decide whether the scale-up can actually be scheduled
