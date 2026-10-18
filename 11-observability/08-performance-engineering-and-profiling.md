# Performance Engineering and Profiling

## TL;DR

Performance work is measurement plus arithmetic, not a library contest. The measurable claim is always the same shape: for workload W at arrival rate lambda, endpoint E's latency distribution stays under budget B at utilization U, on hardware H, with cost C per million requests. Anything you cannot restate in that shape is an opinion about code, not a result. Two disciplines carry almost all of the value: **budget decomposition**, which turns a target p99 into per-hop allowances before anyone writes code, and **attribution by sampling**, which turns "it feels slow" into a named frame, syscall, lock, or queue with a percentage attached. Both are cheap; both are skipped when a team is in a hurry.

The order of operations matters more than the tools. Establish steady-state evidence first (a profile of the system as it runs today, plus the queueing arithmetic that says how much headroom exists), then reduce work rather than adding parallelism, then re-measure with a method that survives review. Amdahl's bound decides whether an optimization is worth scheduling at all: removing a component that is 8% of service time cannot buy more than 8%, no matter how clever the rewrite. **Inference**: on most production services, the largest available wins are not in hot loops but in duplicated work across boundaries (unbatched calls, unindexed lookups, re-serialization, retries of avoidable failures), and each is found by measurement rather than by intuition about code.

## Workload and Measurement Contract

A performance chapter without a workload description is a set of anecdotes. Write the contract down before choosing a tool:

| Field | What it pins down | Why it changes the answer |
|---|---|---|
| Workload mix | Read/write ratio, payload size distribution, endpoint share | A profile of the write path says nothing about the read fan-out |
| Arrival process | Mean rate, burstiness (coefficient of variation), seasonality | Latency is a function of burstiness at fixed mean load |
| Concurrency target | In-flight requests per instance and per client | Determines whether you are CPU-bound, queue-bound, or lock-bound |
| Target distribution | p50, p99, and the hard deadline, not "average" | Averages hide the tail where users churn and timeouts fire |
| Cost envelope | USD per million requests, or cores per thousand RPS | Converts every optimization into a budget line a manager can read |
| Environment | Instance type, kernel, runtime version, cache state, region | A result that does not name its environment is not reproducible |

Three definitions keep arguments honest:

- **Service time** is the work a request causes; **response time** is service time plus every wait in front of it. Most "the code got slower" reports are waits, and no profiler that samples only on-CPU time can see them.
- **Utilization** is the fraction of a resource's capacity consumed. It is not saturation: a node at 95% CPU utilization with perfect scheduling still has queueing delay, and a node at 40% with one hot core does not.
- **Throughput ceiling** is what the system sustains while meeting its latency target, not the peak it survives once.

The chapter owns measurement, attribution, budget arithmetic, and benchmark methodology. It does not own target setting (see the SLO chapter), per-request causal attribution across services (the tracing chapter), or fleet sizing (capacity planning). Those three neighbors supply the numbers this chapter spends.

## Anatomy of a Latency Distribution

A request's latency is a sum of named intervals, and each interval has its own distribution and its own failure behavior:

```
client think time
  + DNS resolution and connection setup           (amortized by keep-alive)
  + TLS handshake                                 (0-RTT or session resumption changes it)
  + edge/LB queueing                              (accept queue, worker pool wait)
  + application queue                             (Little's law decides how long)
  + dependency calls in series                    (each with its own tail)
  + fan-out to N dependencies                     (tail amplification: max of N draws)
  + serialization and compression                 (allocation-heavy, easy to shave)
  + garbage collection or runtime pauses          (concurrent, but not free)
```

Three mechanisms produce most tails:

**Fan-out amplification.** If a request touches N independent services, each with a 1% chance of being at its p99, the probability that at least one is slow is `1 - (1 - 0.01)^N`: about 63% for N = 100. This is the documented arithmetic behind "the tail at scale": fan-out converts many small tails into a common-case problem, which is why hedging and per-hop deadlines exist rather than bigger servers.

**Queueing near saturation.** With utilization rho and mean service time E[S], an M/M/1 queue's expected delay is `E[S] / (1 - rho)`: 2x idle latency at rho = 0.5, 5x at 0.8, 10x at 0.9. Real services are worse than M/M/1 when arrivals are bursty or service times are heavy-tailed; the Kingman approximation multiplies that term by `(c_a^2 + c_s^2) / 2`, where `c_a` and `c_s` are the coefficients of variation of inter-arrival time and service time. Two engineering consequences follow: smoothing arrivals (batching, request coalescing, jittered retries) buys latency without capacity, and reducing variance in service time is worth as much as reducing its mean.

**Interruption.** Garbage collection, CPU throttling, page cache misses, transparent huge-page compaction, and disk writeback all add waits that are invisible to on-CPU profiles. Go's collector runs concurrently but adds allocation assist cost to the mutator; the JVM's default maximum pause target (`-XX:MaxGCPauseMillis`) is 200 ms, which is a disaster for an endpoint with a 100 ms budget and no one reads that default until it fires. **Documented**: Linux CFS bandwidth control refills quota on a fixed period (default 100 ms), so a container that exhausts its CPU quota inside a period is descheduled until the next one, which is why CPU limits show up as p99 spikes rather than as a small throughput tax.

Measure the intervals, not just the total. A histogram of end-to-end latency tells you that you have a problem; five histograms of queue time, dependency time, serialization time, and pause time tell you which one.

## Queueing Math for Servicing Budgets

Four formulas cover most design reviews. All of them are arithmetic you can do in a spreadsheet before writing code.

**Little's law: `concurrency = arrival_rate x latency`.** It sizes everything: 2,000 RPS at 40 ms needs 80 requests in flight somewhere; if one worker handles one request at a time and its service time is 8 ms, you need at least 16 busy workers plus headroom for burstiness. When latency rises to 200 ms during an incident, the same 2,000 RPS needs 400 in flight, which is how a slow dependency turns into pool exhaustion.

**Utilization to replicas.** `replicas = ceil(arrival_rate x service_time / (concurrency_per_instance x target_utilization))`. Choose the target with the queueing curve in mind: 0.7 is a common choice for bursty HTTP fan-in precisely because 0.85 leaves no room for a two-minute imbalance.

**Amdahl's bound.** `speedup <= 1 / (1 - p)` where p is the share of total service time you can remove. Removing 30% of a component that is 20% of latency yields 1 / 0.94 = 6.4% end to end. This kills more bad proposals than any review meeting.

**Deadline propagation.** Each hop's allowance must be smaller than its caller's remaining budget, and the timeout you set is the deadline your dependency will plan against. If a gateway has a 150 ms budget and spends 20 ms on itself, downstream services get 130 ms total; if three of them run in parallel with their own fan-out, each needs its own sub-budget plus a hedge margin. A request that keeps trying after the caller gave up consumes capacity to produce a response nobody reads: charge for it explicitly, and drop it as soon as the deadline passes.

Worked example - a search API with a 120 ms p99 target:

| Segment | p99 today | Share | Action | Expected p99 after |
|---|---|---|---|---|
| Edge queue and TLS | 6 ms | 5% | keep-alive reuse, connection pooling | 6 ms |
| Application queue | 34 ms | 28% | raise per-instance concurrency, add replicas at rho > 0.7 | 12 ms |
| Ranking service (fan-out 12) | 41 ms | 34% | hedge at p90, cache hot queries | 26 ms |
| Document fetch | 18 ms | 15% | batch the N+1 into one multi-get | 9 ms |
| Serialization and compression | 14 ms | 12% | schema change to skip unused fields | 7 ms |
| Runtime pauses (GC) | 7 ms | 6% | lower allocation rate per request | 5 ms |

**Reference design**: the total falls from about 120 ms to about 65 ms, and roughly half of that comes from removing duplicated work in two places rather than from any capacity change. The queue line is the one worth checking first in your own system, because it is the only line that gets worse on its own as you grow.

## Profiling Toolbox by Question

Pick the tool by the question, not by availability. Each row names the failure mode that the profile answers and the one it cannot.

| Question | What to collect | What it can miss |
|---|---|---|
| Where does CPU time go? | Sampling CPU profiler at 100 Hz or more; flame graphs for reading, differential profiles for regressions | Everything off-CPU: waits, locks, throttling |
| Why is a request blocked? | Wall-clock or latency profiles, off-CPU and scheduler-latency analysis | Work that never blocks but is slow |
| Who allocates memory? | Heap allocation-site sampling; live-heap (in-use) profiles for leaks | Objects retained by a single long-lived reference look like their allocator, not their owner |
| Is there contention? | Mutex and lock-contention profiling, scheduler runqueue delay | Contention that shows up only at higher concurrency |
| Are we thrashing memory or cache? | Hardware counters: instructions per cycle, last-level cache misses, TLB misses, stalled-cycles | Anything on shared or virtualized hosts where counters are unavailable |
| What does the syscall path cost? | Syscall counts and latency histograms, block-IO and network retransmit counters | Application-level meaning of those syscalls |
| Is a specific request slow? | Traces with spans, sampled at a rate that keeps rare traces available | Aggregate cost; tracing samples too sparsely for arithmetic |
| What changed in the fleet? | Continuous always-on profiles per service and version, retained for weeks | Nothing fundamental; it is expensive to run badly |

**Documented**: Google reported continuous profiling of all production jobs at less than 2% of total datacenter resources (SIGMETRICS 2021, "Profiling a Datacenter"), which is the number that ended the argument that profiling had to be an incident-time activity. Always-on profiles are how you find the service whose regression cost four percent of a fleet's cores in a Tuesday deploy nobody watched.

Sampling semantics deserve a paragraph because they mislead smart people. A CPU profile at 10 ms resolution assigns samples to whatever frame is on top of the stack, so inlined functions and tail calls move work onto neighbors; that is why profilers need frame pointers or unwinding data. **Documented**: Go enabled frame pointers by default on linux/amd64 and linux/arm64 in Go 1.24, which made kernel-based (eBPF) unwinding of Go stacks practical without DWARF decoding. Allocation profiles are biased toward the allocating site unless corrected: sampling every Kth allocated byte tells you where bytes were created, not what is live now, and only live-heap profiling with dominator or retention analysis identifies a leak's owner.

## Reading Profiles Without Lying to Yourself

Six mistakes account for most wrong conclusions:

1. **Reading a flame graph as a timeline.** Width is sample share, horizontal position is sorted, not chronological. A wide frame means "this subtree held the CPU in many samples", which may be one slow call or a million cheap ones; split by function to tell them apart.
2. **Comparing profiles across workloads.** A profile taken after a cache flush is a different program. Compare identical mixes, same concurrency, same cache state.
3. **Attributing cost to the caller.** If `Serialize()` is 30% of samples and half of those calls are made twice per request by an accidental second encode, the fix is upstream of the frame you are looking at. Count calls, do not just read percentages.
4. **Trusting a profile from a profiler that perturbs the system.** Lock profiling in some runtimes changes allocation behavior; debug builds change inlining. State the build flags next to every number.
5. **Ignoring the machine.** A service that looks CPU-bound may be waiting on a shared dependency; a node at 100% of one core may look idle at the node level. Profile the resource you suspect, not the one you can see.
6. **Concluding from one profile of the p50 path.** Slow requests usually have a different code path (cache miss, retry, cold plan cache). Filter your profiler to the slow cohort; latency profiling by request class is how you find the 1% instead of describing the 99%.

For regressions, use differential profiles: subtract version A's samples from version B's and read the difference plot. Absolute profiles argue about code; differences point at the change that shipped.

## Benchmark Methodology That Survives Review

A benchmark is an experiment with a hypothesis, a control, and a stated environment. The checklist that keeps results believable:

- **Prevent the compiler from deleting your work.** Optimizing compilers remove dead stores and fold constants; runtimes need opaque sinks (a `Blackhole`-style barrier) so measured work actually happens.
- **Warm up deliberately.** JIT tiering, plan caches, connection pools, and page cache all take seconds to settle. Report the warmup policy, not just the result.
- **Isolate or randomize cores.** Frequency scaling, turbo budgets, and noisy neighbors create 10-30% swings. Pin threads, disable simultaneous multithreading when measuring single-thread work, or accept a wide confidence interval.
- **Report distributions for latency, minimums plus medians for throughput ceilings.** A mean hides the tail; a median of a closed-loop load test understates it worse than that.
- **Avoid coordinated omission.** If your load generator waits for a response before issuing the next request, it silently slows its own arrival rate exactly when the system is slow, deleting the worst samples from your histogram. Keep arrivals on schedule and count queue time as latency.
- **Compare with a statistical test, not eyeballs.** Alternating A/B/A runs on the same hardware with a fixed schedule plus a rank-based or bootstrap comparison of distributions detects a 2% regression that a single run pair will hide.
- **Confirm in production before claiming it.** End-to-end claims need same-hardware canaries under real traffic, because staging rarely reproduces cache state, connection reuse, and dependency tails.

## Optimization Order and Cost of Each Layer

Optimizations have a natural cost ordering, and skipping down the list is almost always a mistake:

1. **Delete work.** Cache the result, batch the calls, coalesce the writes, drop the unused field, remove the retry that fires on a permanent error. Highest leverage, lowest risk, no new machinery.
2. **Improve complexity and locality.** Index instead of scanning, shrink the working set, iterate contiguously, avoid hash lookups in inner loops, prefer arrays over pointer graphs so the prefetcher has something to do.
3. **Reduce data movement.** Fewer copies, streaming instead of buffering, compression tuned per payload, zero-copy paths (`sendfile`, `splice`, `io_uring`) where syscalls dominate.
4. **Bound and reuse expensive objects.** Connection pools, buffers, prepared statements, compiled plans. Pool sizing is a queueing question: too small creates waits, too large moves the queue into the dependency.
5. **Add concurrency carefully.** Parallelism multiplies throughput until it hits a shared limit (locks, single-writer shards, memory bandwidth), then it multiplies contention. The universal scaling law's knee arrives earlier than expected on NUMA machines and shared caches.
6. **Change runtime or hardware.** SIMD, compression accelerators, GPUs for batch inference, a compiled language for a numeric kernel. Real, expensive, and last: the migration cost is measured in quarters, and steps 1-4 usually return most of the win this quarter.

Two rules keep step 5 honest. Never parallelize work that shares a lock you have not measured. Never add concurrency to a system whose downstream dependency is already near saturation, because you will convert its latency into your outage (see the backpressure chapter).

## Regression Defense in CI and Production

Performance regresses for boring reasons: an added field, a new middleware layer, an index dropped by a migration, a dependency that started allocating. Defenses worth their cost:

- **A narrow, deterministic benchmark suite** covering the top three endpoints and the two data structures that dominate service time, run on pinned hardware with fixed input corpora. Gate on medians and throughput with a bootstrap confidence interval; alert on trend, not on single runs.
- **Differential flame graphs per release**, generated in the pipeline from a standard load test. Reviewers should see what got wider between versions without asking.
- **Allocation budgets as tests.** A request path that doubles its allocations per request will pass today's latency test and fail next quarter's cost model. Assert on bytes-per-request, which is stable and cheap to measure.
- **Always-on profiles retained per version**, so an incident investigation starts from "compare this hour against the same hour last week" rather than from zero.
- **A latency budget tree in the repository** as a checked-in file that CI can validate: each span's measured p99 must fit inside its allowance, and exceeding it fails the build the way a failing test does.

## Capacity and Cost Model

Performance work is worth doing only when it changes a decision, so price it. Take a service at 2,000 RPS per region, 40 ms mean service time, eight vCPUs per instance, and a target utilization of 0.7:

- In-flight concurrency from Little's law: `2,000 x 0.040 = 80` requests.
- If one instance sustains 50 concurrent requests at the latency target, replicas needed: `ceil(80 / (50 x 0.7)) = 3`, and a fourth for imbalance or an AZ loss.
- Cost per million requests at USD 0.10 per vCPU-hour: `(4 x 8 vCPU x USD 0.10) / (2,000 x 3,600)` is about USD 0.00044 per request, or USD 444 per million requests.
- Cutting mean service time by 25% to 30 ms drops concurrency to 60, replicas to three including headroom, and cost per million requests by roughly a quarter - assuming the removed work is real CPU rather than queue time.

That last assumption is where performance accounting goes wrong. **Inference**: reductions in service time convert into fewer replicas only if utilization was the binding constraint; reductions in *wait* time convert into better latency at the same fleet size. State which you bought, because they are different purchases and only one of them shows up on the invoice.

## Security, Privacy, and Multi-Region Notes

- Profiles are sensitive artifacts. Heap dumps contain tokens, keys, and customer rows; a continuous profiler that captures allocation stacks plus values is a data-retention decision, not an ops detail. Store profiles with the same access control as production data and expire them on a schedule.
- Sampling changes the confidentiality surface: symbol names in a profile reveal internal endpoints and feature flags. Treat profile files as internal-only artifacts.
- Never disable authentication or authorization checks to "measure the real cost". Measure with the check enabled and optimize the check (token caching, symmetric verification, cached decisions) so the security control stays on the measured path.
- Multi-region adds a floor you cannot optimize away: cross-region round trips are tens of milliseconds at minimum, so an architecture that fans out synchronously across regions has a latency budget set by physics. Reduce round trips rather than tuning code (see multi-region architecture).

## Failure Traces

**A limit, not a workload, sets the p99.** A Java service's p99 oscillates on a 100 ms period; CPU is at 35%. The container has a CPU quota sized to average demand, and CFS throttles every period boundary for the busiest threads. Removing the limit (or raising it above burst need) flattens the tail; utilization barely moves. Lesson: p99 spikes on a periodic clock are a scheduling story, not a code story.

**The optimization was real and irrelevant.** A rewrite cuts JSON serialization from 12 ms to 3 ms; end-to-end p99 moves 2 ms. Serialization was 12% of the path and the ranking fan-out dominated. Lesson: run Amdahl's bound before scheduling the work, then measure the end-to-end distribution rather than the component.

**Coordinated omission hides a bug.** Load tests report p99 of 80 ms with zero errors; production reports 1,400 ms for the same build. The load generator was closed-loop: when a worker stalled behind a slow dependency it stopped issuing requests, deleting exactly the samples that showed the stall. Lesson: open-loop arrival generation and queue-time accounting, or no tail claim at all.

**Allocation regression survives review.** A new observability middleware allocates a map per request; latency is unchanged at current load, memory bandwidth rises 40%, and two quarters later the service cannot absorb a normal traffic spike without OOM kills. Lesson: bytes-per-request as a CI assertion catches what latency gates miss.

**Profiling during the incident makes it worse.** An engineer attaches a lock-profiling agent to every replica of a saturated service; the added overhead plus a full heap scan causes a restart storm and a SEV1. Lesson: profile one instance at reduced resolution, or read the always-on profile you kept last week.

## Verification Strategy

An optimization is done when the evidence would convince a skeptic who did not write it:

1. **Before**: an always-on profile of the current version plus the budget tree line you intend to move, with the arithmetic that predicts the win (Amdahl share, queueing term, or allocation delta).
2. **Reproduce**: a benchmark that reproduces the production shape within 10% on the metric you will optimize, with the environment pinned and recorded in the pull request.
3. **Compare**: alternating A/B/A runs long enough for a rank-based test to reject "no difference" at your stated effect size; report p50, p99, throughput, CPU per request, and bytes allocated per request.
4. **Canary**: same-hardware canary under real traffic behind an error-budget gate; confirm the budget tree line moved and that no other line got worse (the common failure is moving cost into a shared dependency).
5. **Soak**: at least one full traffic cycle, because GC fragmentation, connection-pool drift, and cache warm-up hide multi-hour effects.
6. **Attach**: differential flame graph, before/after histograms, and the updated budget file. The next engineer inherits your method, not just your conclusion.

## Decision Framework

| Situation | First move | Second move | Stop when |
|---|---|---|---|
| Users report slowness, no metric breach | Latency profile of the slow cohort by request class | Check queue time and dependency tails | You can name the interval that moved |
| p99 breaches an SLO now | Kill the extra work: shed load, drop optional fan-out, raise concurrency within safety limits | Then profile; do not optimize during the page | The error budget stops burning |
| Cost per request too high | Allocation and CPU profiles at fleet scale, ranked by service cost share | Delete work with the worst cores-per-request | Cores per million requests falls measurably |
| Feature will not fit its budget | Budget tree: which line can absorb it | Batch or cache the added call | Every line still fits with headroom |
| Regression after a deploy | Differential profile against last week's same-hour profile | Bisect by config flag, not by guesswork | One change explains the delta |
| Benchmark looks great, production flat | Check whether the benchmark exercises the production path (cache state, payload sizes, concurrency) | Instrument the real path | The two agree within 10% |

## Key Takeaways

1. No profile, no optimization: a claim about performance without a distribution and an environment is a preference about code.
2. Buy work removal before capacity, and check Amdahl's bound before you schedule anything: the ceiling on your clever rewrite is its share of service time.
3. Tails come from fan-out, queueing near saturation, and interruptions - so measure queue time and pauses, not just CPU.
4. Little's law turns a latency regression into a capacity incident; size pools and replicas with `lambda x W` plus headroom for burstiness, and re-check after every latency change.
5. Always-on profiling at low single-digit percent overhead is now cheap enough that "we could not reproduce it" is a process gap, not a limitation.
6. Closed-loop load tests delete your worst samples; keep arrivals on schedule or do not publish a p99.
7. Price every result in cores and bytes per million requests, and say whether you bought service time (money) or wait time (latency).
8. Leave the system with a budget tree and a benchmark that fails when the budget breaks, so the win survives the next six months of commits.

## References

- Dean, G., and Barroso, L. A. "The Tail at Scale." Communications of the ACM 56(2), 2013. Fan-out tail amplification, hedged requests, and deadline propagation.
- Cummins, J., et al. "Profiling a Datacenter." SIGMETRICS 2021. Fleet-wide continuous profiling reported at under 2% of total datacenter resources, with case studies of the savings it produced.
- Curtsinger, C., and Berger, E. D. "MemProf: Sampling Allocator Profiling Done Right." USENIX ATC 2012. Why naive allocation sampling misattributes memory cost.
- Gregg, B. *Systems Performance: Enterprise and the Cloud*, 2nd ed. Addison-Wesley, 2020. Service time versus wait time, off-CPU analysis, USE method.
- Gregg, B. "Differential Flame Graphs" and "Flame Scope" (performance-analytics.com). Reading regressions as profile differences.
- Tene, G. "How NOT to Measure Latency." Strange Loop 2015. Coordinated omission, percentile misuse, and the HDR histogram method.
- Kingman, J. F. C. "On Queues in Heavy Traffic." Mathematical Proceedings of the Cambridge Philosophical Society, 1962. The diffusion approximation behind variance-scaled queueing delay.
- Go project documentation: `runtime/pprof`, garbage collector tuning guide, and the Go 1.24 release notes on frame pointers by default.
- Linux kernel documentation: Control Group v2 (cpu.max bandwidth control periods and quota) and block/IO accounting.
- Amazon EKS / Kubernetes documentation: container CPU limits, `container_cpu_cfs_throttled_periods_total`, and memory QoS classes.

---

## Related Chapters

- [Capacity Planning and Back-of-the-Envelope Estimation](../01-foundations/10-capacity-planning.md): supplies the arrival rates and service-time estimates this chapter's arithmetic spends
- [SLOs and Error-Budget Control](05-slos-error-budgets.md): defines the target distribution that turns a latency number into a decision
- [Distributed Tracing and Telemetry Pipelines](01-distributed-tracing.md): attributes latency to services and spans while profiling attributes it to code paths
- [Metrics Systems and Monitoring](02-metrics-monitoring.md): histogram bucket choices decide whether your tail claim is measurable at all
- [Retries, Timeouts, and Hedging](../06-scaling/10-retries-timeouts-hedging.md): the mechanisms that trade tail latency against load, priced with the same arithmetic
- [Horizontal vs Vertical Scaling](../06-scaling/02-horizontal-vertical.md): for the case where the profile says the code is fine and the machine is the constraint
- [FinOps and Cost Engineering](06-finops-cost-engineering.md): converts cores per million requests into a budget line that survives a reorg
- [Incident Command and Learning](07-incident-management.md): why you profile one replica during a page rather than every replica at once
