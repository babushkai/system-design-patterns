# Architecture Review and Technical Governance

## TL;DR

An organization's architecture is the output of its decision rights, not of its diagrams. Whatever is not explicitly assigned gets decided by whoever speaks first, cares most, or is on call when it breaks, and those are exactly the decisions that later cost a quarter to reverse. **Documented**: DORA research has repeatedly found that requiring sign-off from an external change-approval body predicts worse delivery throughput and higher change failure rates, while small batches and informal design conversation predict the opposite. The finding is not an argument against review; it is an argument about placement. Reviewing *what* to build early is cheap and high-yield, approving *whether this particular deploy may proceed* late is a tax that buys little, because by then every cheaper option has already been spent.

Two distinctions carry most of the value here. The first is between review and decision rights: a review that cannot decide anything is education (valuable, and it should say so), while an authority that can never be overridden is a queue with opinions attached. Name which ceremony you are running, because the dominant failure in both directions comes from participants believing they are in the other one. The second is between standards and guardrails: a standard is text that decays at exactly the rate of your hiring, and a guardrail is code that fails a build, and only one of them requires anyone to remember it during an incident.

**Inference**: governance earns its cost when it moves constraints earlier, where they are paragraphs, and refuses to become a gate later, where it can only add latency and blame. The unit of governance is not the meeting; it is the default that a team hits at 2 a.m. with no one watching. **Reference design**: this chapter's default stack is deliberately small: written decisions in the design-record chapter, four named review ceremonies with published question lists, a golden path with a documented exit hatch, policy-as-code for every constraint that can be mechanically checked, and one standing rule that a veto must arrive with an alternative and a date.

## What Governance Actually Allocates

Governance is the allocation of three scarce things: the right to commit the organization to a cost, the information needed to notice a consequence, and the ability to enforce a constraint after the people who agreed to it have moved teams. Every governance artifact in the wild is one of those three wearing a costume.

Conway's law is usually quoted as an observation about system boundaries and communication structure. Its useful form for a working engineer is a governance statement: because the system reproduces the organization's decisions, changing the system without touching decision rights produces architecture that nobody chose. The teams that inherit an undocumented Kafka cluster inherit both its operational cost and its implicit constraints, and they will discover the second category through an outage.

Five mechanisms exist for getting the organization to do something, and they differ in cost, durability, and how they fail:

| Mechanism | What it actually is | Cost to change later | How it fails |
|---|---|---|---|
| Cultural norm | "Here we write a load test first" | Very high | Silent divergence as teams grow; invisible to newcomers |
| Written standard | A document with an owner | Medium | Decays at hiring rate; enforced by argument |
| Golden path | A template plus pipeline that makes the standard the easy choice | Medium-high | Abandoned when it lags the frontier; adoption theater |
| Policy-as-code | A check that fails a build, admission request, or deploy | Low | Overridden under deadline pressure unless overrides are logged |
| Org mandate | A decision made by an accountable leader | High | Correct-but-wrong: compliance without understanding, workarounds |

**Inference**: choose the cheapest mechanism that survives the failure you are guarding against. A constraint that has caused an incident twice deserves policy-as-code, not a reminder in a wiki. A preference about library style deserves a default, never a mandate, because mandates convert preferences into resentment and drag.

The reason governance is a staff-level concern rather than a management one is leverage. A senior engineer ships a system; a staff engineer changes what ten teams build by default. The only control that scales past one team's calendar is the default, and defaults are governance artifacts.

## Four Ceremonies That Constantly Get Confused

Naming these separately pays for itself the first time someone complains that "the architecture review is too slow" when the actual complaint is that a launch gate has been scheduled two days before the launch date.

| Ceremony | Question it answers | When it runs | Artifact | Decision right | Signature failure |
|---|---|---|---|---|---|
| Design review | Which option, under these constraints? | Before implementation | Design document | Advisory or deciding, must be stated | Polite discussion that produces no verdict |
| Portfolio or risk review | What have we collectively become? | Periodic, quarterly cadence | Service inventory and risk register | Sets standards and guardrails | Rubber-stamp queue of individual designs |
| Security and compliance review | How is this abused, and what must it satisfy? | Before build and before launch | Abuse-case register, control mapping | Veto on named controls | Arrives when the schema is already shipped |
| Production readiness review | May this carry customer load? | After build, before traffic | Readiness checklist with evidence | Scoped veto (what may not go live) | Checklist theater with no re-check |

A fifth ceremony, the incident review, is the only one that reliably produces new constraints, which is why a governance system that does not feed on postmortems eventually drifts into irrelevance.

The economic case for separating them is that each defect class has exactly one cheap catcher:

| Defect class | Cheapest catcher | Cost if caught at readiness review | Cost if caught in production |
|---|---|---|---|
| Wrong problem, or no problem | One-pager or design review | Rebuild of a sprint | Feature retired quietly after a support queue |
| Missing tenant isolation | Design and security reviews | Multi-tenant migration under duress | Cross-tenant data disclosure, contract loss |
| Dependency cannot meet the SLO | Design review with budget math | Vendor swap mid-project | Two vendors running in parallel forever |
| No rollback path for a schema change | Readiness review | Emergency change window at 3 a.m. | Extended write outage while a table is rebuilt |
| Retry amplification under failover | Resilience test before launch | Config change plus retest | Cascading self-inflicted outage |

**Documented**: Google publishes its production readiness review as a checklist owned by the reviewing team and explicitly scoped to what may block a launch, precisely so that the review cannot quietly expand into a design critique of choices already made. The accompanying guidance in the Site Reliability literature is that reviewers should be from outside the owning team, and that the outcome is a set of follow-up items with owners rather than a binary "approved" stamp.

## Why Review Boards Fail

A review board is a shared queue with calendar constraints, so it inherits the pathologies of every queue plus two of its own. Little's law applies without mercy: if the arrival rate of designs to review stays fixed and the reviewers are the same eight people, mean waiting time rises with the population being reviewed, which is exactly when an organization creates a board. **Inference**: a board created at 200 engineers that reviews everything will be bypassed by 400, and the bypass becomes the real architecture.

Concrete arithmetic makes this tangible. Eight senior engineers in an hour-long meeting have consumed eight person-hours of the scarcest capacity in the organization to discuss one design. Three such meetings a week is more than one full-time engineer, spent entirely on the middle of the decision curve, with nothing left for writing the guardrail that would make the discussion unnecessary.

| Failure mode | Symptom | First fix |
|---|---|---|
| Rubber stamp | Everything approves; incidents still trace to unreviewed choices | Sample five approved designs a quarter and audit whether the stated risk controls exist in code |
| Calendar governance | Reviews scheduled weeks out, so urgent work skips them | Split into async written review by default plus synchronous slots reserved for irreversible decisions only |
| Design by committee | Verdict is a blend of every reviewer's preference | One named decider; reviewers advise and are answered in writing |
| Decision by tenure | The most senior opinion wins without evidence | Require "here is what would change my mind" from every reviewer, including the senior one |
| Veto without alternative | A no that carries no cost for the person saying no | Rule: a veto must name an acceptable alternative and a re-review date |
| The ratchet | Standards written in 2019 now block correct designs | Every standard carries an owner and a review date; expired standards stop being enforceable |
| Invisible decisions | Architecture decided in chat, board ratifies afterwards | Decisions live in decision records; anything not recorded is explicitly not a decision |

What makes the small version work: a published service-level target for turnaround, triage tiers so that most designs never consume synchronous time, a stable core of three reviewers plus rotating members drawn from affected domains, a public log of prior verdicts so precedent exists, and an explicit rule that the board's real output is an updated constraint set. A board that finishes a quarter having changed no guardrail has been doing education, not governance.

## Decision Rights Without an Org Chart

Most staff-level architecture work happens with no formal authority at all, which means influence has to be earned in a specific order. Cheapest first:

1. **Prototype the smaller claim.** A working branch that shows option B is not slower beats four meetings about whether it might be.
2. **Write the comparison.** Two options, named constraints, numbers, and what was rejected (see the decision-record chapter). Writing is the only influence mechanism that scales past a conversation.
3. **Sponsor from inside.** Ask a respected engineer on the owning team to co-author. Adoption is a social process wearing a technical costume.
4. **Pilot with a metric attached.** One service, one quarter, before-and-after numbers chosen in advance so the pilot cannot be declared a success by vibes.
5. **Mandate last.** A mandate applied to a design that has not survived contact with a real team produces compliance plus a private workaround.

Two role distinctions prevent most escalation theater. The *accountable decider* is one person who can commit the organization and must publish the verdict; the *steward* is the person who maintains the constraint afterwards and whose consent matters for changes to it. Confusing these produces either unilateral decisions with no maintainer, or a maintainer with veto power over everyone else's roadmap.

Timeboxed escalation beats endless consensus: if two teams cannot converge within one review cycle, the disagreement gets written down as a comparison document and taken to the accountable decider, who decides within a stated window. The alternative is a stalemate that both sides resolve by building what they wanted anyway.

## Standards, Golden Paths, and Policy-as-Code

Three levels of standard exist and each is checked differently:

| Level | Statement form | Verification | Honest name for it |
|---|---|---|---|
| Aspirational | "Prefer idempotent consumers" | Nothing checks it | Advice |
| Accepted | "Consumers in this domain dedupe on operation identity" | Design review, code review | Norm |
| Enforced | Pipeline fails without a dedupe key | Policy-as-code, admission control | Constraint |

**Inference**: an organization that labels all three "standards" will eventually enforce none of them, because the presence of unenforced rules teaches reviewers that rules are optional. The honest move is to demote most standards to advice and promote a small number to constraints.

A golden path is the highest-leverage artifact a platform organization can produce, and its unit of adoption is not a document but a service template: repository skeleton, deploy pipeline, dashboards and SLOs wired, alert routing and on-call attached, structured logging configured, dependency allow-list applied, runbook stub committed, and load-test harness present. Adoption should be measured rather than mandated, with the number of services on the path published as the platform team's own scoreboard. Non-conformance then becomes interesting data instead of a sin: it tells you which part of the path is unusable.

The exit hatch matters as much as the path. A team that cannot get an exception without a written decision record will either route around the platform or stall, and both outcomes are worse than the deviation. Exceptions should be recorded as decision records with an expiry date, so that the fourth "we need something different" complaint becomes visible as a pattern in the standard rather than four separate secret decisions.

A technology-radar style inventory makes the portfolio legible: **adopt** (default choice), **trial** (approved for bounded use with a review date), **assess** (prototype only, not production), and **hold** (do not adopt; existing usage has a migration plan). The operational content lives entirely in what each ring *does*: does "hold" block a build, or is it a comment? Does "trial" expire automatically? Deprecating something already in production requires the boring machinery of the migration chapter: a named owner, an upgrade window, a support commitment, and a migration path with tooling.

Policy-as-code closes the loop, and placement determines whether it helps or hurts:

| Control intent | Enforcement point | Example | Failure if misplaced |
|---|---|---|---|
| Preventive | Pull request check, admission control | No public bucket without a policy; no unencrypted topic | If only in CI, the resource still gets created by hand |
| Detective | Continuous drift scan | Service without alert routing or owner label | Silent weeks of non-conformance |
| Corrective | Automated remediation with a kill switch | Re-apply retention policy, revert an override | Automated fixes that surprise on-call engineers |
| Advisory | Comment on the pull request | Prefer a bounded queue over an unbounded buffer | Ignored entirely once it fires on everything |

The GitOps reconciliation loop ([Software Delivery Control Planes and GitOps Reconciliation](../15-deployment/04-cicd-gitops.md)) is where governance finally becomes physical: a constraint expressed as declarative state is enforced continuously, while a constraint expressed in a wiki is enforced by whoever remembers.

The cheapest standard of all is structural. An anti-corruption layer or a port-and-adapter seam makes it impossible to spread a vendor's data model through the codebase, which means nobody has to police it in review. Governance that removes a whole category of decision is worth more than governance that reviews each instance of it.

## Sizing the Machinery

Thresholds here are heuristics, not measurements: **Inference**, treat the numbers as starting points to be corrected by your own queue latency.

| Organization size | Default mechanism | What to refuse to build yet |
|---|---|---|
| Under ~40 engineers | Written design docs for irreversible choices; review by whoever is affected | A standing board, a governance portal, a compliance matrix |
| 40 to 150 | Async request-for-comment process with named owners and a turnaround target | Approval requirements on routine deploys |
| 150 to 600 | Lightweight portfolio review (one hour, fortnightly) scoped to cross-cutting and irreversible changes; golden path as the main vehicle | Review of every design; a full-time architecture office |
| Above 600, or regulated | Domain-level review plus central guardrail ownership, with automated conformance evidence | Believing that any central body can keep up without policy-as-code |

Two scaling facts dominate regardless of size. First, cross-team decisions must be findable, which is why the decision-record corpus is the substrate and reviews are just its write path. Second, reviewer attention is the binding constraint, so triage is not optional: the practical split is roughly four fifths of designs handled asynchronously in writing, with synchronous time reserved for irreversible, cross-domain, or novel-infrastructure choices.

## Failure Modes of Governance

| Failure mode | How it announces itself | Where it actually gets fixed |
|---|---|---|
| Gate creep | Readiness review starts blocking deploys rather than launches | Split the checklist into launch criteria and steady-state follow-ups with dates |
| Compliance-shaped security | Controls documented, abuse cases never enumerated | Move the security review to design time and tie it to executable abuse tests |
| Governance debt | No one can say who owns a checklist item | Every control carries an owner and a review date or stops being enforced |
| Metric theater | Counts of documents, reviews held, tickets closed | Measure escapes and reversals instead; see the next section |
| Duplicate platforms | Four stream stores, three schedulers, all "temporary" | Portfolio review with a consolidation decision and a funded migration |
| Standard as religion | Arguments about tooling outlast arguments about requirements | Re-derive the constraint from an incident or a requirement, or retire it |
| Silent local optima | Every team is reasonable; the system is not | Cross-cutting reviews of end-to-end paths, not of components |

## Measuring Whether Governance Works

A governance system that measures activity optimizes for activity. Measure the escapes:

| Metric | What it proves | Useful target | Gaming risk |
|---|---|---|---|
| Share of high-blast-radius changes with an accepted decision record before implementation | Decisions precede code | Rising, and above half in year one | Retroactive records dated backwards |
| Decision lead time, p50 and p90 | The queue is not the product | p90 under one week for async review | Fast approvals of weak documents |
| Guardrail override rate with reasons | Whether constraints are wrong or culture is | Falling, with reasons triaged quarterly | Unlogged overrides in emergencies |
| Escape rate by ceremony | Which review should have caught it | Each incident mapped to a named control | Blame-driven underreporting |
| Golden-path adoption | Whether the default is actually good | Growth without mandates | Empty template services created to inflate it |
| Duplicate-platform count | Portfolio health | Flat or falling | Reclassifying a fork as a "variant" |
| Superseded-within-two-quarters rate | Whether decisions are premature | Low but not zero | Never superseding anything |

**Inference**: the single most diagnostic number is the escape rate, because it forces a conversation about which control failed rather than how many reviews happened. The second is override rate with reasons, since a guardrail that everyone overrides is a bug report written in the language of policy.

## Decision Framework

Choose the mechanism from the symptom, not from an org-chart template:

| Symptom | Mechanism to add first | Mechanism to refuse |
|---|---|---|
| Two teams independently built the same platform | Portfolio review plus one funded owning team | A committee to prevent the next duplicate |
| On-call inherits services with unknown dependencies | Readiness review with a dependency register and named owners | Deploy approvals for every release |
| Review queue longer than a week | Triage tiers; narrow what requires review at all | Adding reviewers while keeping scope fixed |
| Same argument about Kafka every quarter | Decision record corpus plus precedent log | A new board to settle it permanently |
| Incidents traced to schema and migration choices | Targeted readiness checks on migration plans | Full design review of every change |
| Regulated market with evidence demands | Control-to-test mapping with automated evidence | Manual screenshot archives |

For a first program, the sequence that survives contact: write decisions down for one quarter before adding any ceremony; then add production readiness review only for services that will carry customer load, because it produces visible value and no new gate for internal tools; then publish one golden path and measure adoption; then automate whichever constraint has caused the most incidents; and only then consider a portfolio review with decision rights. Build the substrate before the ceremony, and never create a body whose only output is opinions.

Two stop rules keep the machinery honest. Delete any control that has blocked nothing for four quarters, since it is now pure latency. And when a guardrail produces more overrides than violations, treat it as a defect in the guardrail rather than in the engineers.

## References

- Conway, M. "How Do Committees Invent?." *Datamation*, 1968. The original matching of system structure to communication structure, and the reason decision rights are an architectural input.
- Forsgren, N., Humble, J., and Kim, G. *Accelerate: The Science of Lean Software and DevOps*. IT Revolution Press, 2018. Survey evidence separating external change approval from design quality, and its correlation with delivery performance.
- DORA (Google Cloud). *Accelerate State of DevOps Reports*, 2019 onward. Repeated findings on change approval processes, deployment frequency, and where review helps or hurts.
- Beyer, B., Jones, C., Petoff, J., and Murphy, N. (eds). *Site Reliability Engineering: How Google Runs Production Systems*. O'Reilly Media, 2016. Error budgets as an explicit decision mechanism, and production readiness review guidance.
- Sharma, T., Bosworth, A., Zaharia, M., and Chakraborty, A. (eds). *Software Engineering at Google*. O'Reilly Media, 2020. Design document culture, readability as a governance mechanism, and review responsibilities.
- Ford, N., Richards, M., Tolbert, N., and West, M. *Building Evolutionary Architectures*. O'Reilly Media, 2017. Fitness functions and guided, testable evolution instead of periodic approval.
- Skelton, M., and Pais, S. *Team Topologies*. IT Revolution Press, 2019. Interaction modes, cognitive load, and loose coupling as the goal that governance either serves or undermines.
- Nygard, M. "Where Is the Architecture?." 2018. A working architect's critique of review boards that produce verdicts but no constraints.
- Bezos, J. Amazon shareholder letters, 2015 and 2016. Reversible versus irreversible decisions as the basis for deciding what deserves escalation.
- Thoughtworks. *Technology Radar*. An ongoing public example of ring-based technology inventories as a governance artifact.
- ISO/IEC/IEEE 42010:2022, "Systems and software engineering - Architecture description." Stakeholders, concerns, and viewpoints as the required content of an architecture description.

---

## Related Chapters

- [Design Documents and Architecture Decision Records](./01-design-documents-and-decision-records.md): governance without a written decision corpus has nothing to enforce or supersede
- [Verification Strategy for Distributed Systems](./04-verification-strategy-for-distributed-systems.md): constraints that cannot be tested will be violated, so review and verification are one system
- [Incident Command and Learning](../11-observability/07-incident-management.md): postmortems are the only reliable source of new constraints worth enforcing
- [Software Delivery Control Planes and GitOps Reconciliation](../15-deployment/04-cicd-gitops.md): declarative policy is where a standard stops needing an argument
- [Repository Architecture for Safe Agentic Change](../19-compound-engineering/04-ai-native-software-architecture.md): machine-readable constraints become the only enforceable review once agents write the code
- [Verification and Governance for Agentic Change](../19-compound-engineering/05-quality-engineering-with-ai-agents.md): the same review machinery re-derived for changes authored by coding agents
- [Progressive Delivery and Deployment Strategies](../15-deployment/01-deployment-strategies.md): staged rollout is governance applied to runtime instead of to documents
