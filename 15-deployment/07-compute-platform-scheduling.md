# Compute Platform and Scheduling Architecture

## TL;DR

Every service you design eventually runs as a process that someone else's scheduler placed on a machine with finite cores, memory, bandwidth, and file descriptors. The platform is not an implementation detail: it decides your latency tail (CPU quota periods, garbage collection under memory pressure, eviction), your blast radius (how one bad tenant or one bad node becomes an outage), and your bill (packing efficiency). A staff-level design states the platform contract explicitly: what each workload reserves, what it may burst into, what gets killed first, how it is told to stop, and how much headroom the fleet keeps for imbalance and failure.

Two mechanisms dominate everything else. **Resource controls** are cgroup v2 knobs: CPU bandwidth (`cpu.max`, a quota refilled on a fixed period) and memory ceilings (`memory.high`, `memory.max`), which the platform maps from your declared requests and limits. Their semantics are asymmetric and most teams learn that backwards: unused CPU request is a scheduling guarantee while an unused CPU limit silently taxes you in 100 ms periods, whereas memory is not elastic at all - exceed it and something dies. **Placement** is a bin-packing problem with constraints: spread for availability, pack for cost, and the gap between those two objectives is where utilization lives. **Inference**: most platforms fail at lifecycle discipline before they fail at capacity, because a pod that cannot stop cleanly turns a routine node rotation into user-visible errors.

## Workload Classification and Placement Contract

Placement decisions follow from what a workload is, not from what the cluster happens to have free:

| Class | Examples | Scheduling need | Resource posture |
|---|---|---|---|
| Latency-sensitive synchronous | API handlers, checkout, search serving | Spread across zones, low packing density, no burst-dependent neighbors | CPU request at p95 usage, limit unset or generous; memory request = limit |
| Throughput batch | Compaction, exports, video transcode | Tolerant of preemption; wants cheap capacity and high packing | Spot/preemptible, can be evicted with notice, checkpointed progress |
| Stateful singletons | Databases, brokers, leaders | Anti-affinity, topology awareness, storage class and zone pinning | Guaranteed QoS, no overcommit, dedicated node pool often wins |
| Long-tail async | Jobs, workflows, consumers | Queue depth as the scaling signal; fairness across tenants | Burstable, priority tiers, preemption of low tiers allowed |
| Isolated tenant workloads | Per-customer compute, notebooks | Hard isolation boundary, per-tenant quotas and node pools | Quotas plus separate failure domain, not shared kernel cache assumptions |

The contract you publish to application teams should answer seven questions in one page:

1. What does a request buy? (A scheduling guarantee, or a share of spare capacity?)
2. What does a limit do when hit? (Throttle, evict, kill, or throttle the tenant's traffic?)
3. Which node pool may it land on, and can it name one when it must?
4. How much disruption is allowed at once? (PodDisruptionBudget expressed in absolute replicas, not percentages that round to zero.)
5. How long does a graceful stop take, and who enforces the deadline?
6. What happens on interruption with 30 seconds of notice?
7. Who pays for headroom, and how is idle capacity reclaimed?

A platform where those seven answers are folklore will be tuned per-service forever. A chapter on scheduling owns placement, resource control semantics, lifecycle, and fleet-level failure behavior; it does not own deployment rollout mechanics (progressive delivery), autoscaling control loops (auto-scaling), or tenant identity and authorization (multi-tenancy).

## Resource Controls From the Kernel Up

Containers get their limits from cgroups v2. Knowing the layer below the YAML explains most "the platform is slow" tickets.

**CPU bandwidth.** `cpu.max` holds `quota period`, with a default 100 ms period in Kubernetes' kubelet configuration: a pod with a 500m CPU limit gets 50 ms of runtime per 100 ms period across all its threads. A process that burns its quota at the start of the period is descheduled until the next refill, so throttling appears as latency spikes with period 100 ms rather than as proportional throughput loss. The kernel exposes this directly in `cpu.stat` (`nr_throttled`, `throttled_usec`), and cAdvisor re-exports it as `container_cpu_cfs_throttled_periods_total`. **Documented**: CPU *requests* map to weight (`cpu.weight`) and only matter under contention, while *limits* install bandwidth control that applies whenever the container exceeds quota, even on an idle node.

Two consequences are worth internalizing:

- A CPU-limited process with a multi-threaded runtime (JVM with many GC threads, a Go program with GOMAXPROCS equal to host cores) can spend its whole quota in one scheduling period and then stall; capping the runtime's parallelism to the *effective* CPU allocation is usually worth more than raising the limit.
- A CPU request without a limit gives you guaranteed weight and unlimited burst, which is what latency-sensitive services want on a dedicated pool. The trade is fairness: one hot tenant can consume spare capacity that another tenant assumed it could borrow, so burst needs an admission story (pool separation or per-tenant quotas), not a limit set by superstition.

**Memory.** Memory is not compressible and cannot be throttled usefully, so the platform's only tools are early pressure (`memory.high` throttles allocations) and death (`memory.max` triggers the OOM killer inside the cgroup). Kubernetes QoS classes decide who dies first: Guaranteed pods get the most protected OOM scores, BestEffort the least, and Burstable scale between them. **Documented**: kubelet additionally evicts pods under node pressure (memory, disk, PID signals) with hard and soft thresholds, which is a different mechanism from a container limit and produces a different event (`Evicted` rather than `OOMKilled`).

Practical rules that hold across platforms:

- Set memory request = memory limit for latency-sensitive workloads. A request below actual usage means the scheduler placed you against a promise another pod is now breaking, and eviction is how the node resolves the lie.
- Size memory from working-set percentiles plus headroom for the worst legitimate burst (cache warm-up, a large compaction, one pathological query), not from the mean.
- Know what counts toward the limit: page cache is reclaimable, but tmpfs files, anonymous memory, and kernel slab allocations are not.
- Swap is off by default on most Kubernetes nodes, so "swap thrashing" is replaced by "OOM kill", which is faster and less forgiving.

**Everything else with a ceiling.** File descriptor limits (`ulimit -n`), PID limits (`pids.max`, hit by thread leaks), conntrack entries in the node's netfilter table (`nf_conntrack: table full, dropping packet` shows up as mystery 502s with no CPU signal), ephemeral storage for logs and image layers (disk-pressure eviction), and inode exhaustion. Each of these is a scheduler-adjacent capacity limit that teams discover through an incident because they never appear in the deployment manifest.

## Placement: Bin Packing With Availability Constraints

A scheduler answers two questions per pod: which nodes are eligible, and which eligible node is best. Kubernetes' scheduling framework splits this into filtering and scoring plugins; other schedulers differ in vocabulary but not in shape. The design decisions:

**Packing strategy.** Score functions that favor the least-allocated node spread workloads (better availability, worse cost, more small nodes); scores that favor the most-allocated node pack tightly (better cost, worse blast radius). Real platforms mix both: pack inside a failure domain, spread across them. Google's Borg paper is the canonical description of this tension at 10,000-machine scale, including admission control that checks reservations before accepting a job and eviction policies that keep priority classes intact when memory runs short.

**Failure-domain math.** Spreading replicas is only useful if failure domains are real: zone anti-affinity gives you AZ independence; topology spread constraints give you evenness across node pools; pod anti-affinity keeps two leaders off one host. The number of domains sets your availability budget - three zones tolerate one loss, five tolerate one with room to spare, and a "highly available" service with two replicas in two zones has no headroom during the second failure or during any node maintenance.

**Topology inside the machine.** NUMA matters for memory-bound services: cross-node memory access is materially slower than local access, so latency-sensitive workloads on large machines want CPU-manager and topology-manager alignment (Guaranteed QoS plus integer CPU counts is the precondition Kubernetes documents for that). Hyperthread siblings share a core's execution units, so "1 CPU" of quota on a sibling pair is not 1 physical core of throughput; benchmarks that ignore this read as platform flakiness.

**Gang and batch scheduling.** A distributed training job needs all-or-nothing placement: half a job holding GPUs while waiting for the rest is wasted money. That requires gang scheduling with reservations (Volcano, Kueue, YARN capacity queues) rather than per-pod best effort. GPU-shaped workloads add device plugins, MIG partitioning, and now structured parameters/DRA for claiming devices by capability.

**Consolidation and disruption.** Autoscalers that only scale up leave fleets half-empty. Consolidators (Karpenter's consolidation, cluster-autoscaler's scale-down) remove nodes when pods can be rescheduled elsewhere; they need PodDisruptionBudgets to respect, which is exactly where "the upgrade has been stuck for three days" comes from. Budget rules should be written so that allowed disruption never rounds to zero at your smallest replica count, and interruption handling should assume the cloud's documented notice window (AWS EC2 Spot gives a two-minute reclamation notice; GCP Preemptible VMs give 30 seconds) as a *drain deadline*, not a suggestion.

## Lifecycle: How a Workload Starts and Stops

Startup and shutdown are correctness-critical paths, not boilerplate.

**Starting.** A container that takes 40 seconds to warm its cache is not ready when the process is listening. Kubernetes gives you three probes with different meanings: startup gates the others, readiness controls whether endpoints include this pod, liveness decides whether to restart it. Getting those wrong produces the two classic incidents: liveness killing a service during a traffic spike (it was busy, not dead) and readiness too eager (traffic arrives before the plan cache or connection pool exists). Read the failure mode from the client's side: your service discovery propagation delay is part of your rollout risk, which is why readiness gates and target-group registration matter.

**Stopping.** The documented kubelet sequence is: run `preStop`, send SIGTERM, wait up to `terminationGracePeriodSeconds`, then SIGKILL. Four engineering consequences:

1. SIGKILL cannot be handled, so any cleanup that must happen has to fit inside the grace period with margin; make the in-flight request deadline strictly shorter than the grace period, and the grace period shorter than the platform's own drain timeout.
2. Endpoint removal races with termination. A pod can keep receiving requests after it has begun shutting down because load balancer deregistration is asynchronous; the pragmatic fixes are a `preStop` sleep plus readiness flip, or a load-balancer draining delay longer than your p99 in-flight time (Elastic Load Balancing target-group draining defaults to five minutes for exactly this reason).
3. PID 1 semantics inside containers: a shell wrapper (`sh -c "java -jar app.jar"`) can swallow signals unless `exec` or `init: true` is used, so your graceful shutdown never runs. This is one of the most common invisible defects in container images.
4. Draining means *finishing* work, not just refusing new work: flush write buffers and consumer offsets, release leases (or let them expire, which costs a lease interval), stop consuming from queues you cannot finish, and close pools after in-flight work settles.

Shutdown order for a stateless synchronous service, as a reference design worth copying:

```
1. flip readiness to false           # stop new traffic admission
2. sleep >= discovery/LB propagation  # bounded, measured, not guessed
3. stop accepting new requests        # return 503 with Retry-After where honest
4. drain in-flight up to deadline     # log what got cut, count it as an error
5. flush sinks (metrics, logs, offsets)
6. close dependency pools and exit 0   # exit non-zero only if something failed
```

**Stateful workloads** add one rule: never let a stopping process keep a lock or lease that others are waiting on. Release deliberately, and make the release path as tested as the happy path (see the leases chapter).

## Sizing Model From Traffic to Replicas

The sizing chain is arithmetic; the platform's job is to make the arithmetic visible.

Take an API at 2,000 RPS per region with a 40 ms mean response time, an async server that handles up to 64 concurrent requests per pod, and a target utilization of 0.7:

- In-flight concurrency (Little's law): `2,000 x 0.040 = 80` requests.
- Pods at the concurrency ceiling: `ceil(80 / 64) = 2`; at 0.7 target utilization: `ceil(80 / (64 x 0.7)) = 2`. Two pods is not an availability answer, so floor by failure-domain rules instead: with three zones and a one-zone loss tolerance you need at least four pods so that three remain and each carries 33% more load without exceeding the concurrency ceiling.
- CPU request per pod from measurement: if p95 usage is 1.4 vCPUs at this concurrency, request 1.5 vCPUs; with memory request = limit at 2 GiB working set plus 30% headroom.
- Scale-out trigger: prefer the signal that saturates first. CPU utilization is a fine proxy only when threads are the constraint; for async I/O-bound services, in-flight concurrency or queue depth is the honest signal, and lag-based scaling (queue length per consumer) beats CPU by minutes during a burst.
- Autoscaler behavior: keep scale-up fast (no stabilization window) and scale-down slow (the documented default stabilization window is five minutes), because premature scale-in converts a two-minute spike into a cold-start storm.

**Inference**: the most common sizing error in Kubernetes-hosted services is not too few replicas but too small a CPU *limit* relative to per-request parallelism, which caps single-request latency no matter how many replicas exist. Check throttling counters before adding capacity; a fleet at 35% utilization with a terrible p99 is usually throttled, not undersized.

## Fleet Economics and Headroom

Utilization is a budget decision, not a virtue. Three curves trade against each other:

- **Packing efficiency** rises with tighter placement and looser spread constraints. Every point of wasted capacity is money at scale; Google reported that fleet-wide profiling and placement work recovered resources large enough to justify the effort of instrumenting everything.
- **Failure headroom** falls with packing. A cluster at 90% steady utilization cannot absorb a node-pool loss, a bad deploy needing 30% extra capacity, or a rebalance after an AZ failure - which is when capacity is most expensive and slowest to obtain.
- **Recovery speed** determines how much headroom you can safely give back. If new nodes appear in three minutes and pods become ready in two, a burst needs less standing reserve than it does on a platform that takes twenty-five minutes to provision a GPU-backed node.

Practical structure that keeps all three honest: split the fleet into a **baseline pool** (reserved, protected, sized for steady load plus one failure domain), an **elastic pool** (cheap or spot-shaped, sized for variance, drains gracefully), and a **burst allowance** (quota you can claim in a known-bad hour, tested quarterly). Charge each team its working-set p95 rather than its requests, because billing on requests is how a fleet ends up with 40% phantom reservations.

## Security, Isolation, and Multi-Region Notes

- Namespace isolation is an API-server concept, not a hardware boundary. Node pools, RuntimeClass/sandboxed runtimes, and per-tenant quotas are the mechanisms that bound blast radius when a tenant executes arbitrary code (see multi-tenancy for the tenant-lifecycle half).
- Kernel-shared resources leak information and interference: last-level cache and memory bandwidth are shared unless you partition them (Intel RDT-class technologies or dedicated pools), so a noisy co-resident workload can degrade a latency-sensitive neighbor with no CPU signal at all.
- Node identity matters: kubelet credentials, node service accounts, and the container runtime socket (`/var/run/docker.sock` or containerd socket mounted into a pod is a root-equivalent grant) belong in any threat model of the platform.
- Multi-region placement changes resource semantics: cross-region scheduling decisions are usually made by DNS or global load balancing rather than by the cluster scheduler, and a regional pool sized at 100% of its own traffic guarantees failure when it must carry a neighbor's share. Size each region for `1 / (N - 1)` of global peak if N regions must absorb one loss.

## Failure Traces

**Throttled at 35% utilization.** A Go service migrates to Kubernetes and p99 doubles while dashboards look healthy. GOMAXPROCS equals the host's core count, CPU limit is 500m, and `cpu.stat` shows most periods throttled. Fix: align runtime parallelism with effective CPU (auto-GOMAXPROCS or an explicit value), then re-measure; no replicas added.

**The cache that killed itself.** A service sets memory request 1 GiB and no limit, runs on a Burstable QoS node alongside batch jobs, and grows its in-process cache to 3 GiB as designed. The node hits its memory-pressure threshold and kubelet evicts the lowest-priority pods; the cache-cold restart storms the database behind it. Fix: request = limit for caches, bounded eviction policy inside the process, and a warm-up path that does not stampede (see the stampede chapter).

**PDB deadlock during node rotation.** A four-replica statefulset declares `maxUnavailable: 1` while its pods must all be ready for quorum; the node updater drains one pod, the remaining three never become "ready" because they are recovering, and the rollout stalls. Nothing is broken per pod; the budget assumed independent readiness.

**502s with no CPU, no errors, no pattern.** A burst fills the node's conntrack table; the kernel drops packets for new connections while existing ones are unaffected, so error rates look random and only rise on fresh clients. Fix: raise `nf_conntrack_max` deliberately, shorten established timeouts, keep-alive aggressively, and alert on conntrack drops as a first-class saturation signal.

**Drain storm on a Tuesday.** A consolidator removes three nodes at once during a normal hour; the pods reschedule onto nodes already at their concurrency ceiling, latency triples, and the autoscaler responds by adding capacity that takes six minutes to arrive. Fix: disruption budgets that bound concurrent eviction by *capacity*, not just pod count, plus scale-up policy that pre-claims replacement capacity before draining.

**Signals lost at shutdown.** A consumer commits offsets after a 30-second drain but receives SIGKILL at the grace period's end because `terminationGracePeriodSeconds` was left at the 30 s default and in-flight work runs to 45 s. Duplicate processing downstream is attributed to the message broker. Fix: derive the grace period from measured in-flight p99 plus flush time, and assert on "terminated with SIGKILL" as a release gate.

## Verification Strategy

Prove the platform contract with evidence rather than with a diagram:

1. **Saturation counters wired**: CPU throttling ratio, working set versus request, `oom_kill_total`, eviction events, conntrack usage and drops, file descriptor and PID usage, ephemeral storage, per-zone pod counts. Each has an alert or an SLO-relevant dashboard panel; the USE method applies to resources, not just services.
2. **Throttle test**: run a load profile that pushes one pod's quota utilization from 50% to 110% and confirm what you see in metrics matches what users see in latency. If throttling is invisible in your dashboards, fix the dashboards before tuning limits.
3. **Kill drills**: kill one pod, one node, one zone; assert that client-visible errors stay inside the budget predicted by your replica and failure-domain math. A service claiming zone redundancy must pass this with a measured number attached.
4. **Drill the stop path**: send SIGTERM under load and verify zero dropped requests beyond the documented allowance, no SIGKILL terminations, and correct offset release; make "SIGKILL on shutdown" fail a CI or soak test.
5. **Consolidation rehearsal**: force a node deletion during business hours with disruption budgets enabled and confirm the autoscaler replaces capacity before eviction, not after.
6. **Quota realism check**: compare requests to measured p95 usage per namespace monthly; unexplained reservations are reclaimed, and the report is the cheapest capacity you will ever find.

## Decision Framework

| Question | Choose this | Avoid |
|---|---|---|
| CPU limit for latency-sensitive service? | Request at p95 usage; limit unset or well above burst, on a protected pool | A limit near average usage, which buys 100 ms scheduling stalls |
| Memory limit? | Request = limit, sized from working set plus worst legitimate burst | No limit with a growing cache, which converts growth into eviction of neighbors |
| Packing vs spread? | Pack within a failure domain, spread across them; floor replicas by zone-loss math | Even spread everywhere (cost) or tight packing everywhere (blast radius) |
| Scaling signal? | The resource that saturates first: concurrency, queue depth, then CPU | CPU utilization for async I/O-bound services |
| Cheap capacity? | Elastic pool with interruption budgets and checkpointed work | Betting synchronous latency-sensitive traffic on preemptible nodes |
| Isolation for untrusted code? | Separate node pools, quotas, sandboxed runtime, no host sockets | Namespace-only separation |
| Grace period? | Measured in-flight p99 plus flush time, below the platform drain timeout | The 30 s default, and shell wrappers that eat SIGTERM |

## Key Takeaways

1. Requests are promises, limits are enforcement, and they are enforced by different mechanisms: weight versus bandwidth control for CPU, ceiling plus OOM policy for memory.
2. CPU limits show up in the p99, not the average, because quota refills on a fixed period; check throttling counters before buying replicas.
3. Memory is inelastic: set request = limit for anything with a working set, or accept that node pressure decides who dies.
4. Availability is failure-domain arithmetic - replica count divided by zone count, with room for the loss of one domain plus one maintenance window.
5. Placement is bin packing under constraints; pack inside a domain, spread across domains, and pay for headroom explicitly instead of discovering it during a rebalance.
6. The stop path is a correctness path: readiness flip, bounded drain, flush, exit - with the grace period derived from measured in-flight latency rather than a default.
7. Interruptible capacity is fine when interruption is designed for: notice window treated as a drain deadline, checkpointed progress, and disruption budgets that bound concurrent eviction by capacity.
8. Every platform promise needs a counter and a drill: throttling ratio, OOM kills, evictions, conntrack drops, SIGKILL terminations, and one killed node per quarter.

## References

- Verma, A., Pedrosa, L., Korupolu, M., et al. "Large-Scale Cluster Management at Google with Borg." EuroSys 2015. Admission control against reservations, packing versus fairness tension, eviction and priority classes at datacenter scale.
- Verma, A., et al. "Cluster Management with Homogeneous Clusters." EuroSys 2015. Why homogeneous machines plus constraint-aware placement beat ad hoc sizing.
- Kubernetes documentation: "Manage Resources for Containers", "Pod Quality of Class", "Pod Lifecycle and Termination", "Eviction API", "Pod Disruption Budgets", "Topology Manager", and "Scheduling Framework".
- Linux kernel documentation: Control Group v2 (cpu.max, cpu.weight, memory.high, memory.max, pids controller) and "Complete Load Balancer"/CFS bandwidth control notes describing quota periods and throttling statistics.
- Kubernetes documentation: "Horizontal Pod Autoscaler" behavior policies (scale-down stabilization window) and "Runtime Class".
- AWS documentation: "Rebalance recommendations" and Spot Instance interruption notices (two-minute reclamation notice); Elastic Load Balancing "Connection draining" defaults.
- Google Cloud documentation: Preemptible VM termination notice window and node autopilot sizing semantics.
- Dean, G., and Barroso, L. A. "The Tail at Scale." Communications of the ACM 2013. Why per-request latency depends on shared-resource interference.
- Cummins, J., et al. "Profiling a Datacenter." SIGMETRICS 2021. Fleet-scale evidence that resource accounting pays for itself.

---

## Related Chapters

- [Auto-Scaling](../06-scaling/08-auto-scaling.md): the control loop that consumes this chapter's placement rules and headroom budget
- [Horizontal vs Vertical Scaling](../06-scaling/02-horizontal-vertical.md): decides whether the scheduler is choosing machines for you or buying fewer of them
- [Progressive Delivery and Deployment Strategies](01-deployment-strategies.md): rollout mechanics that assume the stop path and readiness semantics described here
- [Software Delivery Control Planes and GitOps Reconciliation](04-cicd-gitops.md): where desired resource requests and limits become cluster state again
- [Multi-Tenant Isolation and Tenant Lifecycle Architecture](../06-scaling/12-multi-tenancy.md): the tenant-facing half of quotas, pools, and blast-radius boundaries
- [FinOps and Cost Engineering](../11-observability/06-finops-cost-engineering.md): turns packing efficiency and phantom reservations into a bill someone can act on
- [Background Jobs and Worker Pools](../18-workflow-job-systems/02-background-jobs-worker-pools.md): queue-depth scaling and worker pool sizing built on the same concurrency arithmetic
- [Performance Engineering and Profiling](../11-observability/08-performance-engineering-and-profiling.md): the measurement discipline that proves whether a limit is throttling your tail
