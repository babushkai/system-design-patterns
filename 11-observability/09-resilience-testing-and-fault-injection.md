# Resilience Testing and Fault Injection

## TL;DR

Failure behavior is a specification you either test or discover. Every system in this book fails in ways its authors did not intend, and the difference between a team that ships a five-nines control plane and a team that learns about its failover bug from a customer ticket is almost entirely whether anyone executed the failure before customers did. The measurable claim resilience testing produces is narrow and testable: under fault F, sustained for duration D, in scope S, the system keeps error budget burn inside B and recovers without human action or with a runbook that a named on-call engineer has actually executed once.

Three tiers of fidelity exist and they are not interchangeable. **Documented**: deterministic simulation at the design layer finds the interleaving bugs that no amount of staging traffic reproduces - FoundationDB reports correctness bugs found by simulation runs measured in CPU-years before deployment, and the MACE fault-injection work found dozens of real defects in Kafka, Cassandra, and ZooKeeper from a harness smaller than the systems it tested. Model checking finds protocol-level impossibilities. Production fault injection finds the faults that only production has: the misconfigured DNS resolver, the certificate nobody renewed, the quota shared with a sibling team. Teams that only do the third tier are permanently surprised by the first two; teams that only do the first tier are surprised by reality.

Two disciplines separate this from "breaking things on purpose". Every experiment states a hypothesis with a pass criterion and an abort criterion *before* it runs, with steady state defined as an error-budget burn rate rather than a gut feeling that traffic looks normal. And every finding becomes a permanent regression: either an automated test in the simulator, a drill script, or an architectural change that removes the fault class. **Inference**: on most teams the highest-yield first quarter is not exotic chaos tooling but restore drills, kill-leader automation, clock and disk-fault simulation, and one honest game day per high-severity path.

## What Resilience Testing Has to Prove

Write the claim before choosing the tool, because each tier can prove only some of it:

| Claim | Shape | Tier that can prove it |
|---|---|---|
| Safety under interleaving | No lost write, no double commit, no two leaders, no stale read after a successful ack | Deterministic simulation, model checking |
| Liveness under fault | Requests eventually complete or fail explicitly within deadline X | Simulation with timeouts, production injection |
| Recovery without invention | The documented procedure works, in the current release, in this region | Drills executed by on-call staff |
| Blast radius bound | Fault confined to S; peers degrade within agreed margin | Production injection with control group |
| Capacity under real mix | Sustained target load at latency target with headroom H | Load and soak against production-shaped traffic |

Three distinctions keep the vocabulary honest:

- **Fault, failure, error**: a fault is injected (latency, a killed process, a full disk); a failure is when the system stops delivering its service; an error is the observable deviation. Testing injects faults to provoke failures and checks that errors surface where they can be seen and contained.
- **Resilience testing is not load testing.** Load tests ask how much the system takes; resilience tests ask what happens when a part of it lies, stalls, or disappears. A system can pass both and still fail over incorrectly.
- **Chaos engineering is not the whole field.** Production experiments are the highest-fidelity and most expensive tier. The cheapest bug fixes in this area come from simulator and spec work done before code review, which is where most of the fault classes in this book actually get caught.

Scope ownership follows the same rule as everything else in the book: the team that owns the failure mode owns the test for it. A platform team proves node loss and AZ loss; an application team proves "dependency returns 500 with a 30-second timeout" and "the queue redelivers twice". Nobody proves cross-cutting claims except in a joint drill, which is one reason drills are load-bearing rather than ceremonial.

## Three Tiers of Fidelity

The tiers differ in cost per bug found, fidelity to production, and what they can even express:

| Tier | Cost per run | Fidelity | Can express | Cannot express |
|---|---|---|---|---|
| Model checking (TLA+, TLC) | Seconds to hours per property | The model only | Protocol logic, invariants, fairness | Code, I/O, real timing |
| Deterministic simulation | Minutes to days of CPU | High for logic, none for hardware or OS | Interleavings, timeouts, partitions, upgrades | Kernel behavior, NIC quirks, neighbor interference |
| Staging and canary injection | Minutes per experiment | Medium; wrong traffic, wrong scale | Deployment mechanics, resource limits, config drift | Rare interleavings, real adversarial load |
| Production injection | High (requires trust plus guardrails) | Total | Everything, including organizational failure | Bugs whose blast radius exceeds the budget |

Read the last column of each row as the reason all tiers exist. A simulator cannot find that a cloud provider's metadata endpoint returns 429 under concurrent credential refresh; production cannot economically reproduce the one in ten million message-ordering bug that a simulator finds before the protocol ships.

Sequence them by cost of change: spec during design, simulation continuously in CI, staging injection per release, production experiments per quarter against the highest-severity paths plus any path where a drill just failed.

## Deterministic Simulation at the Design Layer

Deterministic simulation rests on one inversion: instead of writing tests and running them on real infrastructure with real time, you write a *model* of the system's nondeterminism - random number generator, clock, network scheduling, disk behavior, process lifetime - and run the real logic against it in a loop where every source of nondeterminism is seeded. The whole run becomes a pure function of a seed, which means any failure is replayable, shrinkable, and parallelizable across thousands of cores.

The mechanics that make it work:

1. **Single-threaded event loop with virtual time.** Actors or coroutines exchange messages through an in-memory scheduler; the clock advances only when the scheduler decides, so a simulated week of heartbeats costs milliseconds and timeouts fire on demand rather than never.
2. **Fault injection as first-class random choices.** At each scheduling point the harness may reorder a message, drop it, delay it past a timeout, duplicate it, crash a process and restart it with a different clock offset, corrupt a disk write, or pause a process for an hour. Faults are drawn from a schedule that is itself part of the seed.
3. **A correctness oracle.** Either an invariant checked continuously (at most one primary, no committed record lost, version monotonicity) or an end-to-end comparison against a simple reference model - the classic trick is to run every transaction against a naive in-memory database and assert identical observable results.
4. **Shrinking.** When a seed fails, an automated reducer removes events until it finds a minimal sequence that still fails. A four-line reproduction is a fix; a forty-thousand-event trace is a week of reading.
5. **Coverage guidance.** Random seeds find long-tail combinations; instrumentation that records which branches and fault classes were exercised tells you when the distribution is too narrow. Both are needed, because pure random search on a deep state space mostly explores the shallow part.

**Documented**: FoundationDB's team has described running correctness simulation continuously, with bugs found in simulation treated as normal course-of-business rather than incidents, and Apple's "Deterministic Simulation 2.0" work reports dropping per-test cost by orders of magnitude by simulating at a higher level and reusing prior runs - the practical lesson being that the *cost per run*, not the technique, decides whether your team actually runs it.

What simulation cannot do: model your operating system's TCP buffer behavior, a specific disk firmware that returns EIO on trim, or the fact that two services share one connection-tracking table. It also cannot save you from modeling the wrong thing - if the spec says retries are idempotent and they are not in production, the simulator agrees with the bug-free fiction. That gap is exactly what the production tier exists to close.

## Model Checking and Protocol Invariants

Model checking answers a different question: not "does this implementation survive many random bad days" but "is this design capable of violating the property, at all". You write a formal-ish state machine plus the properties that must hold, and an exhaustive or bounded search either proves them within a scope or produces a counterexample path.

Where it earns its keep in this book's subject matter:

- **Consensus and replication**: single writer, no committed log entry lost, monotonic reads, leadership exclusivity. TLA-plus style specifications of Raft-like protocols have surfaced subtle liveness and configuration-change defects that testing missed for years.
- **Effect commit and saga logic**: exactly-once claims, compensation ordering, "no orphan reservation after a crash at any point in the sequence". Enumerating crash points is precisely what a checker does well and what an integration test does badly.
- **Cache invalidation and versioning**: interleavings where a reader observes a newer value then an older one, or a writer's delete loses a concurrent write.

Practical guidance from teams that adopt it: write the spec *before* implementation, because the value is in the arguments you have while writing it; keep the model small by abstracting away everything not implicated in the property (a five-variable model checks in seconds, a fifty-variable one never finishes); bound the state space with symmetry reduction and a small number of nodes; and treat every counterexample as a design conversation, not a test failure.

Property selection matters more than tool choice. The properties that catch real incidents are usually about *authority* (who may write), *durability boundaries* (what is safe to acknowledge), and *time* (what happens when a lease expires while its holder is paused). Jepsen-style testing occupies the productive middle between model checking and production injection: it drives a real deployment with a workload generator, injects partitions and kills, then checks a history for anomalies such as stale reads or lost writes - which means it tests the implementation and the deployment together, and reports in terms of guarantees customers actually care about.

## Fault Injection in Production Against an Error Budget

Production experiments are controlled scientific studies, and the discipline that separates them from outages is the write-up before the run. The canonical loop has five steps and every one of them is a place where teams quietly cut corners:

1. **Hypothesis with steady state defined.** "Steady state" is a measured range on indicators that matter - error-budget burn rate, p99 within X milliseconds of baseline, saturation below Y - not "the dashboard is green". Compare against a control group (an untouched cell, region, or tenant cohort) so that a coincident degradation elsewhere does not become a false result.
2. **Blast radius design.** Smallest meaningful scope: one pod, one shard, one availability zone, one percent of traffic, one non-critical tenant cohort with consent. The scope is also the abort threshold: if indicators outside the scope move, the experiment has already failed and must stop.
3. **Automated abort.** A kill switch that reverts injection on burn-rate breach, on a real page firing, or on elapsed time exceeding the plan - automatic, not "the on-call engineer will notice". Manual abort assumes attention is available, which is the one resource experiments are designed to stress.
4. **Run with witnesses and a clock.** Somebody who can abort, somebody who is not typing during the injection, and a written timeline of what was injected when. Experiments run during change freezes, releases, or peak events produce results nobody trusts and incidents nobody needs.
5. **Conclusion recorded as a design decision.** Either the system behaved as claimed (the claim becomes documented behavior with a test that keeps it true), or it did not (a finding enters the same queue as a bug, with severity derived from what the experiment proved about customer impact).

**Documented**: the original chaos engineering principles paper (Basiri et al., ICSE-SEIP 2017) frames this precisely: state steady state as measurable behavior, vary real-world events, run experiments in production, automate to minimize blast radius - and it explicitly ties "long-running continuous experiments" to the point that a one-off game day proves very little about a system that changes weekly.

Two failure modes of programs rather than experiments are worth naming. The **theatrical experiment** injects a fault the architecture already handles (kill one stateless pod in a five-replica deployment) and reports success; it is not worthless - it validates the harness, which matters - but it must be scheduled upward toward faults that could plausibly break something. The **unbounded experiment** injects without an automatic abort because "we know this is safe", and discovers that the safety argument had a hole in exactly the place the experiment was designed to probe.

## A Catalog of Injectable Faults

The catalog below is organized by what the fault teaches, not by which tool injects it. Each row is a candidate entry in a coverage matrix whose columns are your high-severity paths.

| Fault | Injection method | What it tests | Typical finding |
|---|---|---|---|
| Dependency latency (ms to seconds) | Proxy or egress rule adding delay | Timeout ordering, circuit breaker, retry budget | Client timeout longer than caller's deadline; retries fired on a slow-but-successful call |
| Dependency errors and 5xx bursts | Fault-injecting sidecar or mock | Error classification, retry vs fail-fast | Retryable and permanent errors treated identically; user sees raw dependency error |
| Network partition (asymmetric) | Packet drop rules between subsets | Leader election, quorum behavior, split-brain writes | Two writers after heal; stale cache served as fresh after rejoin |
| Clock skew and jumps | Namespace time offset, suspended process | Lease expiry, token validation, backoff arithmetic | Lease held past expiry by a paused process; negative durations from wall-clock subtraction |
| Disk full on the writer only | Fill volume or set quota | Backpressure, WAL handling, error propagation | Silent write loss, or crash loop that cannot start because the disk is still full |
| Slow fsync and IO stalls | cgroup IO throttle, dm-delay | Group commit assumptions, durability claims | Acknowledged writes lost after failover; latency cliff at a specific queue depth |
| DNS failure or stale answers | Block resolver, return SERVFAIL, long TTL | Connection reuse assumptions, retry storms | Total outage while the resolver is down; thundering herd on recovery |
| Connection and conntrack exhaustion | Lower limits, hold sockets open | Pool sizing, keep-alive assumptions, health checks | Health checks share the exhausted pool and report unhealthy, amplifying the event |
| Leader or primary kill (clean and SIGKILL) | Process termination with and without drain | Failover time, fencing, client reconnect logic | Split brain without fencing; clients retry forever against the old leader's cached address |
| AZ loss | Detach a zone from load balancer, scale to zero | Replica arithmetic, cross-AZ data locality, quota | Control plane cannot rescale because capacity reservation is per-zone |
| Region loss and degraded peer region | Simulated at routing layer in a drill | Failover runbook, DNS TTL reality, data lag assumptions | Runbook steps reference a console that is unreachable from the failed region |
| Credential and certificate expiry | Short-lived fixtures issued deliberately | Rotation automation, alerting before expiry | Expiry detected by clients; rotation requires an interactive login nobody remembers |
| Poison message in a queue | Publish a payload that fails validation | Dead-letter handling, consumer isolation | One bad message blocks the partition or restarts forever with no visibility |
| Duplicate and out-of-order delivery | Redeliver, reorder in harness | Idempotency keys, dedupe windows | Double charge, double shipment, or a state machine that regresses |
| Version skew during rollout | Hold one replica on the previous build | Wire compatibility, migration safety, cache format | New reader cannot parse old writer's payload; flag defaults differ between versions |
| Recovery thundering herd | Restore all replicas at once, cold caches | Backoff with jitter, warmup, load shedding | Recovery causes the next outage; retry amplification visible only in the second wave |

Three rules for using it. Inject faults at the boundary where they occur in reality (network-level latency is not the same experiment as application-level sleep, because only one of them fills sockets and buffers). Inject combinations once a fault class is individually passing, since coupling - disk full plus leader failover plus retry storm - is where real incidents live. And always inject the *recovery* path, not just the failure: most pain in this book's failure traces arrives on the way back, when caches are cold, connections are unthrottled, and every client retries at once.

## Game Days and Recovery Drills

A game day is a resilience test whose subject is the organization. Two formats are worth distinguishing because they find different defects:

- **Tabletop** (sixty to ninety minutes, no hands on keyboard): read a scenario aloud from a real incident template, then walk decisions against the actual runbook. Finds missing roles, ambiguous authority to fail over, contact lists that point at people who left, and the step that says "restore from backup" without saying from which backup or how long it takes.
- **Live drill** (scheduled, guarded, executed): inject a real fault with real on-call staff following the current runbook, measured against time-to-detect, time-to-mitigate, and whether mitigation matched the document. Finds tooling that only works on the author's laptop, dashboards nobody can find under load, and the fact that "the database team" has one person who knows the procedure.

Design drills around your highest-severity paths, defined as the alerts that have paged someone to a customer-visible impact in the last year plus any path where data loss is possible. Keep a coverage matrix: rows are failure modes from the catalog and the failure-semantics chapter, columns are quarters, cells record which tier proved the behavior and when. A matrix with twelve rows and three empty columns is a better planning artifact than any resilience roadmap document.

Restore drills deserve their own cadence because they fail differently from everything else. Requirements: restore into an environment that cannot write to production, verify against application-level invariants rather than "the database started", measure elapsed time honestly (including the wait for someone with permission), and rotate who performs it so the knowledge is not a person. A backup you have never restored is a hypothesis about a file format.

## Load, Soak, and Saturation Boundaries

Load testing belongs to capacity planning; soak testing belongs to resilience because most of what it finds is time-dependent failure. The distinction is duration and what you watch:

| Mode | Duration | What it is for | What it finds that short tests miss |
|---|---|---|---|
| Baseline load | Minutes | Latency and throughput at expected traffic | Nothing new; it calibrates the comparison |
| Stress to failure | Minutes to an hour | Where the ceiling is and how it fails | Whether degradation is graceful or a cliff, and which resource saturates first |
| Spike | Seconds | Autoscaler reaction time and cold-start cost | Queue buildup that outlives the spike; scale-down oscillation |
| Soak | 12 to 72 hours sustained | Behavior over time at steady load | File descriptor and connection leaks, memory fragmentation, log volume exhaustion, index bloat, token expiry boundaries, cache eviction storms |
| Shadow replay | Continuous | Real mix against a candidate | Coverage gaps between synthetic load and actual traffic shapes |

Method notes that keep results trustworthy: drive load with closed-loop generators only when concurrency is the fixed parameter, and prefer open-loop arrival patterns when measuring latency under load (a generator that waits for a response before sending the next request hides queueing delay, the classic coordinated-omission error); include thinking time or per-client pacing so the test is not accidentally a self-inflicted denial-of-service; warm caches and connection pools deliberately and report whether results are warm or cold; and treat saturation as a fault class - the run where CPU hits one hundred percent is an experiment about interference, so measure the noisy neighbors on the same host rather than assuming dedicated hardware.

Soak tests earn their calendar slot only if someone watches the time-dependent signals: descriptor counts, connection pool waits, garbage collection frequency drift, page-fault and swap trends, write amplification, queue depth at fixed load, and log storage remaining. A soak run whose only output is "p99 was stable" proves nothing about the failures it exists to catch.

## Measuring the Program

Resilience programs die from a metric that counts activity: experiments run, game days held, chaos tooling deployed. Those numbers go up while customer risk is unchanged. The metrics that correlate with fewer pages:

- **Coverage of high-severity paths**: fraction of failure modes attached to a real paged incident (plus every data-loss-capable path) that has a passing test at any tier, with the tier recorded. A drill covering a fault nobody can trigger is a zero, not a one.
- **Findings per experiment, trended**: a program whose last twelve experiments found nothing is either exceptionally mature or injecting faults it already passes. Track where findings were found (simulator, staging, production) to see which tier is underused.
- **Time-to-detect and time-to-mitigate from tests, not incidents**: run the drill, measure against the SLO chapter's error-budget math, and compare quarter over quarter. These two numbers are the only honest summary of organizational resilience.
- **Recovery regression rate**: how many restore drills failed, and how many failures were caused by something that changed in the last thirty days. This is the leading indicator for "our backup has been broken since the March migration".
- **Actions shipped against actions written**: drill findings that become code or architecture changes within a quarter. A backlog of resilience actions with owners and no shipping is a queue, not a program.
- **Escape correlation**: for each incident, was this failure mode tested? If yes, the test asserted the wrong thing (fix the oracle); if no, the coverage matrix was wrong (add the row). Both are cheap corrections; neither happens without the question being asked in the review.

## Security, Privacy, and Multi-Region Notes

Fault injection is a security control as much as an availability one. Fail-open behavior shows up under stress: authorization checks that default to allow when the policy service times out, feature flags that fall back to permissive defaults when configuration fetch fails, rate limiters that disable themselves when their backing store is unreachable. Inject dependency failure into every security-relevant decision path and assert that writes refuse to proceed while reads degrade predictably. Pair this with the threat-modeling chapter: each abuse case that assumes a control is available needs an experiment showing what happens when that control is not.

Privacy constrains the fidelity of load testing. Shadow traffic that copies production requests to a test environment copies customer data, so it needs field-level minimization, a retention limit, and an explicit decision about whether replay crosses a purpose limitation boundary. Synthetic generators built from a data-shape specification rather than a payload copy are less realistic and much cheaper to defend.

Multi-region systems add three systematic surprises worth pre-committing to drills for:

1. **DNS TTLs outlive your failover.** Clients and resolvers cache answers far longer than the configured TTL, so "we flipped DNS" is not a mitigation with a known completion time. Measure the actual drain curve during a drill instead of assuming sixty seconds.
2. **Control planes are regional and share quota with the thing you are recovering.** Scaling out in region B often fails because the instance quota, IP range, or credential issuance path is scoped to region A - discovered at the worst possible moment unless it has been drilled.
3. **Replica lag makes failover a data decision, not a switch.** The runbook must state how much acknowledged write loss is acceptable, who authorizes it, and how divergence is reconciled afterward; that is disaster-recovery arithmetic, and it belongs in the drill script rather than in a phone call at three in the morning.

## Failure Traces

**Trace: the failover that lost eleven minutes.** A managed database promotes a replica after a primary kill. Failover completes in forty seconds; clients take eleven minutes to stop writing to the demoted primary because their connection pools cached the old topology and their retry policy treats "not primary" as retryable-without-refresh. Detection: an SLO burn alert, then a second one for the resulting queue backlog. A drill that killed the primary would have measured exactly this, and the fix (refresh-on-not-primary plus a bounded stale-topology window) is a day of work.

**Trace: the retry storm behind a successful drill.** An experiment injects two hundred milliseconds of latency into a payment dependency. Circuit breakers open as designed, error rates stay inside budget, and the team declares success. Thirty seconds after injection stops, every client retries simultaneously against a cold connection pool and error rate triples for four minutes. The experiment tested failure and shipped recovery untested; jittered backoff plus a warmup ramp on half-open closure close it.

**Trace: clock jump and the expired lease.** A virtual machine is suspended during a host maintenance event and resumes with its clock eleven minutes ahead. Its lease expired while paused, another worker acquired it, and both process the same job sequence; downstream dedupe suppresses the duplicate charge but not the duplicate shipment. Simulation catches this class cheaply because pausing a process for an hour costs nothing in virtual time; production catches it once per fleet maintenance window whether you plan for it or not.

**Trace: disk full on the leader only.** The write-ahead log volume fills. Followers continue serving reads, health checks pass, and writes fail with a storage error translated into a generic internal error at the second hop. Nobody connects the two because the disk metric is on a different dashboard than the error rate. Detection took nine minutes; the fix is three alerts and one error-classification change, all discoverable by an experiment that costs a filled volume in staging.

**Trace: the certificate nobody renewed.** A mutual-TLS certificate between two internal services expires. Both services started successfully because renewal runs on a sidecar that had been failing silently for sixty days, and the alert was written against process health rather than certificate age. The drill that finds this is not a chaos experiment but a fixture: issue deliberately short-lived credentials in a staging environment once a quarter and watch whether anything notices before clients do.

## Verification Strategy

Verification here is meta-verification: how do you know the resilience testing itself works?

- Assert on **customer-observable outcomes**, not on internal signals. A test that passes because "the failover completed" while customers saw eleven minutes of errors is an oracle bug, and oracle bugs are the dominant failure mode of resilience suites.
- Require a **negative control** in every automated fault test: inject a benign variant that must pass and verify the harness detects a deliberately introduced regression at least once per release. A fault harness that cannot fail is worse than none because it manufactures confidence.
- Keep fault tests in the **same CI pipeline as functional tests**, run on every change to the component they protect, with failure treated as a release blocker rather than a flaky-test backlog item. Flakiness in fault tests is usually a genuine nondeterminism finding; triage it as one.
- Version the **coverage matrix next to the design doc** for each system, and make adding a new high-severity path to the alerting rules require a matching row. That single gate is what keeps the matrix from decaying after the initial enthusiasm.
- Review **experiment safety on the same cadence as experiments**: abort paths tested, kill-switch credentials valid, blast-radius scoping still correct after topology changes. Guardrails rot faster than the systems they protect.

## Decision Framework

| Situation | Where to start | What "done" looks like for the quarter |
|---|---|---|
| New service, no test history | Catalog rows for its dependency set; staging injection per release plus one live drill | Every high-severity path has a row and an owner |
| Stateful or consensus-based component | Model the invariants first, then deterministic simulation or Jepsen-style checks in CI | Safety properties machine-checked; findings fixed before launch |
| Repeated incidents from the same class | Drill that specific class until it stops recurring; fix architecture rather than adding alerts | Two consecutive drills pass without human intervention |
| No production injection permission yet | Simulator plus staging injection plus tabletop drills; earn the right with guardrails documented and abort paths tested | Safety case accepted, first bounded production experiment scheduled |
| Mature platform, low incident rate | Move up the composition ladder: multi-fault experiments, region loss, control-plane failure, supply-chain faults | Findings come from combinations rather than single faults |
| Regulatory or contractual resilience claims | Test the claim as written (RTO and RPO), with evidence a reviewer can reproduce | Documented runbook plus dated drill evidence per claim |

Refuse to schedule an experiment whose abort condition you would not accept as an incident. Refuse to declare a system resilient on the basis of tests that assert internal events. And refuse, politely and permanently, the argument that resilience work is finished: the system changes weekly, so yesterday's proof has an expiry date whether or not anyone wrote one down.

## Key Takeaways

1. Failure behavior is specification: either you execute it in a test or a customer executes it for you, and only one of those produces a written result.
2. Three fidelity tiers, three different bug populations - simulation finds interleavings, model checking finds impossibilities, production injection finds reality; none substitutes for another.
3. Deterministic simulation makes a week of heartbeats cost milliseconds and every failure replayable by seed, which is why the cost per run rather than the technique decides adoption.
4. Write invariants before faults: single writer, no acknowledged write lost, lease authority exclusive. Fault schedules without an oracle produce animations, not evidence.
5. Production experiments are hypotheses with a measured steady state (error-budget burn), a bounded scope, and an automatic abort - never a human noticing.
6. Inject the recovery path too; cold caches, unthrottled reconnects, and synchronized retries turn a clean failover into the second outage.
7. Soak tests earn their slot only when someone watches time-dependent signals: descriptors, pool waits, collection drift, write amplification, log volume, token expiry boundaries.
8. Drills test the organization; restore drills especially, because a backup nobody has restored is a hypothesis about a file format.
9. Measure coverage of paged failure modes, findings per experiment, and drill-measured time-to-mitigate - never experiments run.
10. Fail-open security controls are availability tests in disguise: inject dependency failure into every authorization, flag, and limiter path and assert that writes refuse to proceed.

## References

- Basiri, A., Behnam, N., de Oca, S. M., et al. "Chaos Engineering." ICSE-SEIP 2017. The five-principle formulation: measurable steady state, real-world event variation, production execution, automated experiments, and minimized blast radius.
- Palish, T., et al. "Deterministic Simulation 2.0: Greatly Reduced Costs." arXiv:2102.10302, 2021. Making simulation cheap enough to run continuously by simulating at a higher level and reusing prior runs.
- Kraft, P., et al. "Operation and Correctness of a Large-Scale Distributed Storage System." USENIX ATC 2017. How FoundationDB combines deterministic simulation with a correctness oracle in ordinary engineering practice.
- Kingsbury, C., Boyer, H., Kallberg, M., Lee, S., Wong, H.-F. "Failure Analysis of Crackly Systems: A Study into the Kafka Community's Bug Reports." OSDI 2020. The MACE harness and the dozen-plus real defects it surfaced across Kafka, Cassandra, and ZooKeeper.
- Lamport, L. "Specifying Systems: The TLA+ Language and Tools for Hardware Engineers." Addison-Wesley, 2002. Writing the state machine and properties that a checker can search exhaustively.
- Herlihy, M. P., and Wing, J. M. "Linearizability: A Correctness Condition for Concurrent Data Structures." ACM TOPLAS 1990. The correctness condition most anomaly checkers (including history checkers) measure against.
- Yin, Z., Wu, X., Yang, Y., Wang, J., and Chen, G.-H. "An Empirical Study of Chaos-Induced Incidents in Large-Scale Cloud Systems." ICSE 2021. What goes wrong when chaos tooling itself becomes the incident source.
- Zheng, C., et al. "SSE: Evaluating Resilience for Tree-of-Replicas Services." SOSP 2023. Synthetic end-to-end resilience testing of a large production service tree, with continuous fault injection at scale.
- Kingsbury, C. Jepsen system reports (jepsen.io). Anomaly-based verification of real deployments under partitions and kills, reported as guarantees rather than test outcomes.
- Beyer, B., Jones, C., Petoff, J., and Murphy, N. "Site Reliability Engineering" and "The Site Reliability Workbook." O'Reilly, 2016 and 2018. Emergency-response organization, playbook and wrangler roles, and load management primitives that drills exercise.

---

## Related Chapters

- [Failure Semantics, Detection, and Recovery Boundaries](../01-foundations/06-failure-modes.md): the taxonomy this chapter's catalog injects, one row at a time
- [SLOs and Error-Budget Control](05-slos-error-budgets.md): supplies the steady-state definition and abort thresholds every experiment needs
- [Incident Command and Learning](07-incident-management.md): the organization that drills rehearse and incidents actually exercise
- [Performance Engineering and Profiling](08-performance-engineering-and-profiling.md): separates queueing wait from real work before you call a degradation a resilience bug
- [Disaster Recovery and Data Reconstruction](../15-deployment/05-disaster-recovery.md): the recovery arithmetic that restore drills measure against reality
- [Raft](../09-whitepapers/07-raft.md): the leader election and fencing invariants worth checking before you kill a primary
- [FoundationDB](../09-whitepapers/13-foundationdb.md): the reference implementation of correctness-by-simulation as daily practice
- [Effect Commit Protocols for Workflows](../18-workflow-job-systems/06-retry-idempotency-compensation.md): where duplicate delivery and compensation ordering become testable claims
