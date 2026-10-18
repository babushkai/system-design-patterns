# Third-Party Dependency and Supply Chain Architecture

## TL;DR

The dependency graph is the architecture that actually runs. The diagram on the wiki shows twelve services; `cargo tree`, `pip freeze`, or a lockfile diff shows four thousand components, and every one of them carries failure semantics, an upgrade obligation, a latency contribution, a data-flow obligation, and a legal term that somebody outside engineering can change on a Tuesday. **Documented**: the incidents that define this area are all availability or integrity failures at a distance. A single maintained-by-one-person left-pad package disappearing from npm took down builds across the internet in 2016. event-stream (2018) and ua-parser-js (2021) showed account takeover producing targeted payloads inside widely-installed packages. Log4Shell (2021) demonstrated that a logging library reachable through transitive dependencies could put an organization into incident response for software it never installed directly. SolarWinds (2020) moved the attack into the vendor build pipeline, and the xz-utils backdoor attempt (2024) showed a years-long social engineering campaign aimed at a compression library that sshd links against.

None of those failure classes are exotic, which is why dependency choices belong in design review rather than in a pull request diff. **Inference**: three properties make third-party components structurally harder than first-party ones. You inherit their failure modes without inheriting their runbook. Their upgrade cadence is uncorrelated with your release schedule. And their availability objective, if published at all, is measured against a definition of "available" that does not include your tail latency or your dependency's own dependencies.

This chapter covers the whole surface: selection criteria with real weights, lock-in as a measurable architectural property, boundary design that keeps an exit path open without paying an abstraction tax for it, supply chain integrity from registry to signed artifact, runtime containment of a component you do not operate, and the standing upgrade program that keeps the whole thing from becoming a security liability with a roadmap. **Reference design**: every production dependency has a named owner, a pinned digest, a declared failure budget, an enforced timeout and retry policy, and one written sentence describing how you would leave it.

## What a Dependency Actually Imports

Engineers evaluate functionality and adopt a package; the organization inherits six things at once:

| What you inherit | Concrete form | When it surfaces |
|---|---|---|
| Failure modes | Retry storms, unbounded queues, blocking DNS, hidden connection pools, clock sensitivity | First dependency degradation, usually during peak traffic |
| Upgrade obligation | Transitive graph growth, abandoned APIs, deprecations, runtime EOL | Months later, in the worst possible sprint |
| Security surface | Postinstall hooks, deserialization paths, secrets handling, network egress | At the vendor, not at you: build-time code execution is arbitrary code execution |
| Latency and capacity budget | Their p99 plus your retries, plus their own retry policy multiplied against yours | Load test, or production |
| Data flow | What they store, where, for how long, and whether subprocessors see it | During the customer security questionnaire, always too late |
| Legal and commercial terms | License obligations, pricing metric changes, acceptable-use limits | Renewal, or a ToS email with 30 days notice |

**Inference**: the reason experienced engineers are boring about dependencies is asymmetry. The upside of adopting a package is a sprint saved; the downside is an incident whose root cause lives in a repository you cannot push to, owned by someone who owes you nothing. That asymmetry says the bar for adoption should be higher than the bar for writing fifty lines yourself, and lower than the bar for building a competing system.

The transitive graph compounds this. A direct dependency is a decision; a transitive one is an accident with the same blast radius. Any inventory that counts only direct dependencies understates exposure by an order of magnitude on typical application stacks, which is why the useful metrics in this chapter count the whole resolved graph.

## Choosing a Dependency

Score candidates against the failure you are most exposed to rather than against a generic rubric. The weights below describe a component on a request path; a build-time tool should trade maintenance rigor for simplicity, and an analytics SDK should trade performance for data-flow restraint.

| Criterion | What to look at | Strong signal | Disqualifier |
|---|---|---|---|
| Maintenance health | Time to merge pull requests, release cadence, open issue age, number of maintainers with commit access | Regular releases plus a published security policy | Bus factor of one and a "help wanted" issue from two years ago |
| Failure behaviour | Timeout defaults, retry policy, connection pooling, error taxonomy | Explicit budgets the caller controls | Silent infinite retry, or exceptions swallowed into a log line |
| Surface area | Resolved graph size, optional features, native extensions, install-time scripts | Small resolved graph; features opt-in | Pulls in a framework to parse a date |
| Security posture | Public advisories and time-to-patch, fuzzing, formal review, artifact signing | Advisories published *and* fixed quickly with backports | Vulnerability reports handled by email forever |
| Operational fit | Metrics exported, trace propagation, structured logs, deterministic configuration | Emits RED metrics and honours context cancellation | No way to observe it from outside |
| Exit cost | Interface shape, data portability, whether their types cross your domain boundary | Thin adapter with contract tests | Their entity classes embedded in your domain model |

Two heuristics resist argument. First, **prefer boring and old** on any request path: a library that has been stable for six years has already paid for its surprises, while a fast-moving one transfers its learning curve to your on-call rotation. Second, **count the resolved graph, not the manifest entry**: adopting one package that pulls in forty is a different decision from adopting one, and the diff is where maintainability either appears or does not.

For infrastructure-class components (queues, stream stores, workflow engines, vector indexes, model servers) add three questions that application libraries can skip: who operates the control plane during our failover test; what is the operational model at our scale rather than at their benchmark scale; and what does an upgrade look like when it cannot be done with a rolling restart.

## Lock-in as a Measurable Property

Lock-in is not emotional discomfort about switching; it is exit cost, and exit cost decomposes into five independent parts that are easy to price separately:

| Lock-in type | Mechanism | Cheapest mitigation |
|---|---|---|
| Data format | Proprietary storage format, no export of logical state, or export that loses fidelity | Keep a canonical representation you own; treat the vendor store as a projection (chapter on derived projections) |
| API shape | Vendor request and response types spread through business logic | Port at the seam: your interface, their adapter |
| Operational | Custom tooling, dashboards, and runbooks only valid for one provider | Generate them from declarative definitions that name the provider as a parameter |
| Economic | Egress pricing, per-request billing, or storage classes that reward retention | Model cost at 10x traffic before adopting; put egress on the design-review checklist |
| Skills and staffing | Whole team certified on one proprietary system | Accept it deliberately when the skill is scarce and durable; resist it when it is a wrapper around SQL |

**Documented**: the recurring pattern in infrastructure licensing disputes of the last decade (open-core search, observability, and database projects moving from permissive licenses to source-available or server-side terms, and the forks that followed) is a reminder that license risk is architectural risk with a legal calendar. A dependency decision made on technical merit alone can still be reversed by a party with no interest in your latency budget.

The discipline this implies is cheap: keep an *exit note* per significant dependency, three sentences naming the replacement candidate, the data you would have to move, and the interface that would have to change. Exit notes are also excellent design-review artifacts, because "I cannot name the seam" is a faster signal than any abstraction argument.

One warning about the obvious cure. Building a provider-agnostic abstraction for its own sake produces a lowest-common-denominator interface that models nobody's semantics well, and then leaks on every feature you actually use. Abstractions earn their tax when at least two of three hold: you have a real second implementation now, you have a plausible regulatory or commercial reason to need one soon, or the seam is where your domain language gets protected from someone else's vocabulary. Otherwise, depend on the vendor directly and put the effort into the exit note instead.

## Supply Chain Integrity From Registry to Runtime

The attack surface that matters most is not the library's code path; it is the machinery that turns a version string into running bytes. Every step below closes a documented class of incident, and they compose: nothing here is optional in one place and unnecessary in another.

```text
source            developer machine / upstream release
  |  commit signing, protected tags, release from CI not laptop
  v
build             hermetic builder, no ambient credentials, egress allow-list
  |  reproducible inputs, pinned toolchain, dependency hashes enforced
  v
artifact          image or package produced once, never rebuilt per environment
  |  content digest, provenance attestation, SBOM generated at build time
  v
sign              key held by the build system, short-lived identity, no repo secret
  v
registry          private proxy; upstream mirrors snapshotted and immutable
  |  signature + attestation verification at pull time, not report time
  v
deploy            admission control rejects unsigned or unpinned images
  |  digest pinning in the manifest, mutable tags rejected
  v
runtime           no registry credentials, minimal egress, seccomp/AppArmor profile
```

| Control | What it stops | Notes on doing it properly |
|---|---|---|
| Lockfiles with hashes, committed | Substituted artifact, silent upstream republish | Resolve once; forbid lockfile drift in CI rather than fixing it at deploy time |
| Pin by digest, not tag | Tag mutation, which is legal and common for container images | Digest pinning plus a scheduled job that opens an update change when the tag moves |
| Private registry proxy with allow-list | Typosquatting, dependency confusion, name-collision installs across internal and public scopes | Namespace separation must be enforced on the client, not just the proxy |
| Signature and provenance verification | Build compromise, unsigned upstream artifacts | Verify at pull or admission time; a report nobody reads is detective control for auditors only |
| SBOM capture (SPDX or CycloneDX) | Not being able to answer "are we affected" in an hour | Generate during build so it reflects the resolved graph, not the manifest |
| Hermetic, egress-restricted builds | Install-time scripts reading credentials from the build environment | Build-time code execution is the highest-privilege moment in your pipeline; treat it that way |
| Least-privilege pipeline identity | Registry publish from a compromised pull request | Publish permissions separated from test permissions; fork pull requests cannot reach production secrets |
| Reproducible builds where feasible | Being unable to prove which source produced the running binary | Realistic for compilers and static toolchains; aspirational for most application images |

**Inference**: ordering matters. Signature verification without registry control is theatre, because an attacker who can make you install by name will make you install something signed by them. Egress-restricted builds plus digest pinning plus an allow-list proxy catch materially more than any scanner, because they remove capability instead of detecting intent. Scanners remain worth running for prioritization: their value is a defensible patch queue, not prevention.

On that patch queue: severity-only triage produces noise, because most advisories cannot be reached in your deployment. Reachability analysis (is the vulnerable function actually called with attacker-controlled input?) turns "1,400 known vulnerabilities" into a list someone can finish, and it converts an unpayable debt into a maintenance task with a service-level target attached.

## Runtime Containment of Components You Do Not Operate

A third-party service is an untrusted dependency with better branding. The failure budget math is unforgiving: if the vendor commits to four nines a month and your objective is three, then the vendor alone consumes most of your error budget before your own code contributes anything. **Inference**: any synchronous third party on the request path either gets its own budget line in the SLO decomposition or becomes an asynchronous projection with stale-read semantics.

Containment is a specific set of mechanisms, all of which are cheap and all of which get skipped:

- **Timeouts first, always.** Set a per-call timeout below your own deadline, and propagate cancellation. A default timeout of "none" converts a vendor latency spike into thread-pool exhaustion, which is an outage of your service caused by their slowness.
- **Bounded retries with jitter and a retry budget**, coordinated with the fact that the vendor may also be retrying internally. Retry amplification across two uncoordinated policies is a documented recipe for turning a partial degradation into a full one.
- **Bulkheads per vendor.** Separate connection pools, worker pools, or semaphore limits per dependency, so that one slow vendor consumes its own capacity rather than the shared pool. Circuit breaking (chapter on dependency state) then operates on a bounded resource instead of an unbounded one.
- **Cache aggressively where staleness is acceptable**, with negative caching for permission lookups and product metadata, and an explicit stale-if-error path so cache expiry does not synchronize into a thundering herd against the vendor.
- **Idempotency keys on every mutating call.** Vendor timeouts are not failed writes; without an operation identity you must either accept duplicates or build a reconciliation job, and the second option is what you will do anyway.
- **Webhooks treated as attacker-supplied input**: signature verification, timestamp tolerance with replay protection, idempotent handlers, and quarantine for events that fail schema validation rather than a dead letter that nobody reads.
- **Health defined by behaviour, not vendor status pages.** Track per-vendor success ratio, latency histogram, and rate-limit rejection as first-class signals; the vendor's status page is a communication channel, not an indicator.

For components on the critical path, "we have a second provider" is a hypothesis until it has been tested in production traffic. A real failover design needs routing that can be switched by configuration, per-vendor credentials and quotas provisioned in advance, feature parity verified by contract tests against both vendors, and a game day where the primary is actually cut off (chapter on resilience testing). Where failover is genuinely impossible because of cost or semantics, say so in writing and record the compensating control: degraded mode, queue-and-retry with a bounded backlog, or an explicit customer-visible error state.

## Upgrades as a Standing Program

Upgrade debt behaves like any other debt: it compounds, and the interest rate rises with the number of components that have moved on. The program is unglamorous and its metrics are honest:

| Practice | Mechanism | Failure mode it prevents |
|---|---|---|
| Automated update pull requests, grouped | A bot opens updates; grouping keeps review cost bounded; separate lanes for patch and major | Year-long freezes that turn a routine bump into a rewrite |
| Security-only fast lane | Patch upgrades reviewed within hours, with tests but no design review | Days of known exposure while a committee meets |
| Runtime and toolchain EOL calendar | Base images, language runtimes, and managed engine versions tracked with dates | Forced migration during the busiest week of the quarter |
| Dependency ownership register | Every direct dependency has an owning team and a named engineer | "Who approved this?" answered by silence at 3 a.m. |
| Removal as a first-class task | Deleting a dependency gets the same estimate as adding one | Zombie integrations that nobody dares to touch |
| Contract tests against vendors | Consumer-side assertions about response shape, error taxonomy, and rate-limit headers | Silent vendor behaviour changes discovered by customers |

**Documented**: the pattern behind most large-scale dependency failures is not a novel attack but an unmaintained component that nobody owned. This is why ownership registers outperform scanning tools in practice: a scanner produces findings, and findings without owners become background noise within a quarter.

For internal platforms the same logic applies with higher stakes, because your dependencies are other teams' code. Platform components need published deprecation policies with dates, upgrade windows, and tooling that does the migration automatically where possible. A platform team that ships breaking changes faster than its consumers can migrate has not built a platform; it has outsourced its work to every consuming team simultaneously (chapter on migrations).

## Failure Modes of Bought Components

| Failure mode | Symptom | Fix |
|---|---|---|
| Orphaned critical dependency | No owner, no changelog read, present in three services | Ownership register with a default: unowned dependencies get removed rather than adopted |
| Silent transitive adoption | A test-only tool pulls a runtime library into production images | Build-stage separation; scan the shipped artifact, not the source tree |
| Abstraction tax without an exit | Generic provider interface that models one provider badly | Delete the abstraction or add the second implementation; decide deliberately |
| Retry amplification across boundaries | Vendor degradation causes self-inflicted outage | One retry budget per request path, owned by the caller; verify with a load test |
| Status-page optimism | Vendor reports "degraded" while you are down | Own your vendor telemetry; alert on your own error budget burn |
| Pricing-model surprise | Per-request or per-egress billing grows with retries and polling | Model cost at 10x before adoption; add a cost dashboard for the integration |
| Data-flow drift | A vendor SDK starts collecting fields nobody reviewed | Classify egress at design time; test outbound calls in CI with an allow-list proxy |
| Build-time compromise | Unexpected install script, unreviewed lockfile change | Hermetic builds, hash verification, and lockfile diffs treated as security-relevant reviews |

## Key Takeaways

- Dependency decisions are architectural decisions: they import failure semantics, upgrade obligations, data flows, and legal terms at once, and only the first two show up in a pull request.
- Count the resolved graph, not the manifest. Direct dependencies are choices; transitive ones are accidents with the same blast radius.
- Capability-removing controls beat detection: allow-list proxies, digest pinning, hash verification, hermetic builds, and least-privilege pipelines close classes of attack that no scanner can reliably catch.
- Reachability analysis converts an unpayable vulnerability backlog into a patch queue someone can finish; severity-only triage produces noise and deferred accountability.
- Lock-in is exit cost across five dimensions (data format, API shape, operations, economics, skills), and the honest mitigation for most of it is a written exit note rather than an abstraction.
- Never build a provider-agnostic interface without either a real second implementation or a protected domain boundary; otherwise you pay a tax to model one vendor badly.
- Third parties are untrusted dependencies: timeouts, bounded retries with budgets, per-vendor bulkheads, idempotency keys, verified webhooks, and your own vendor telemetry are the minimum containment set.
- Vendor availability objectives do not compose with yours; put every synchronous third party on the SLO budget line or move it off the request path.
- Failover to a second provider is unverified until traffic has been switched in production under a game day; where it is impossible, record that fact and the compensating control.
- Upgrades are a standing program with owners, lanes, and an EOL calendar, because dependency debt compounds at the interest rate of everyone else's release cadence.

## References

- Hindle, A., Barr, E., Su, Z., Gabel, M., and Devanbu, P. "On the Naturalness of Software." ICSE 2012. Repetition in source corpora, the underlying evidence for preferring widely-used components whose behaviour is already known.
- Rauber, R., and Rüsch, A. "Choice of Third-Party Components and Its Impact on the Security of Open Source Software." Companion Proceedings of ICSE 2013. Empirical link between component selection criteria and downstream security exposure.
- Ralph, L. "A Leftpad Retrospective." npm blog, 2016. The availability incident that made transitive dependency risk a mainstream engineering conversation.
- Unit 42, Palo Alto Networks. "Log4Shell: The Story So Far." 2021. Anatomy of a transitive vulnerability reaching systems that never referenced the library directly.
- Mandiant. "Highly Evasive Attacker Leveled SolarWinds Customer Using Various Tactics, Techniques, and Procedures." 2020. Build-pipeline compromise as a systemic rather than local failure.
- Freund, A. Public disclosure of the xz/liblzma backdoor attempt. 2024. Multi-year social engineering aimed at an upstream compression library reachable from a system daemon.
- npm and GitHub security advisories for the event-stream and ua-parser-js incidents, 2018 and 2021. Compromised maintainer accounts producing targeted payloads inside widely-installed packages.
- OpenSSF. *Scorecard* project documentation. Heuristic signals for open source project security practices, and their documented limits as an adoption filter.
- OpenSSF. *SLSA: Supply-chain Levels for Software Artifacts* specification. Provenance requirements, build-integrity levels, and the graduated assurance model for artifacts.
- in-toto project documentation. Attestation format and verification model for delegating and recording supply chain steps.
- Cloud Native Computing Foundation Technical Advisory Committee. *Supply Chain Security Whitepaper*. 2021. Threat taxonomy for the software supply chain from source to runtime.
- SPDX specification and OWASP CycloneDX specification. Bill-of-materials exchange formats used as procurement and incident-response evidence.
- Executive Order 14028, "Improving the Nation's Cybersecurity." 2021, and CISA Secure Software Development Attestation guidance. The regulatory turn that made provenance and SBOMs contractual rather than optional.
- Evans, E. *Domain-Driven Design: Tackling Complexity in the Heart of Software*. Addison-Wesley, 2003. Anti-corruption layers as structural isolation of an external model.
- Cockburn, A. "Hexagonal Architecture." 2005. Ports and adapters as the seam that keeps a vendor's interface out of the domain.

---

## Related Chapters

- [Architecture Review and Technical Governance](./02-architecture-review-and-governance.md): adoption criteria only work if the decision lands before the dependency is in production
- [Threat Modeling and Abuse Case Engineering](../10-security/08-threat-modeling-and-abuse-cases.md): supply chain abuse cases belong to the design, not to the scanner report
- [Zero-Trust Service and Workload Architecture](../10-security/05-zero-trust-architecture.md): least-privilege identity is what limits a compromised build pipeline and a malicious package
- [Circuit Breakers](../06-scaling/06-circuit-breakers.md): the containment mechanism for a dependency whose state you cannot observe from its status page
- [Software Delivery Control Planes and GitOps Reconciliation](../15-deployment/04-cicd-gitops.md): where digest pinning, signature verification, and admission policy become enforceable
- [FinOps and Cost Engineering](../11-observability/06-finops-cost-engineering.md): vendor pricing metrics turn retries and polling into invoices
- [Service and Platform Migration](../15-deployment/06-migration-strategies.md): the exit note becomes a funded program once leaving is actually required
