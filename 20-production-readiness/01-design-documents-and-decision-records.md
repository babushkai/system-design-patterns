# Design Documents and Architecture Decision Records

## TL;DR

Code proves a design; only a document can decide one. The design document is the cheapest artifact that makes an architectural choice reviewable while changing it is still a paragraph rather than a migration, and the architecture decision record (ADR) is the cheapest artifact that tells the next engineer why the choice exists before they confidently reverse it. **Documented**: real designs are not produced in rational order. Parnas and Clements observed that the rational design process is faked after the fact, because discovery, dead ends, and constraints arrive interleaved. That is an argument for writing the rationalization deliberately, not for skipping it: whoever writes the document chooses which fiction becomes the team's shared model.

Two artifacts with different lifetimes do two different jobs, and most documentation failure comes from using one for the other's work. A design doc is persuasive, comparative, and disposable: its purpose ends when a decision is made, and a stack of un-filed design docs is a liability. An ADR is descriptive, single-topic, and effectively permanent: it records one constraint that is now true, with the forces that produced it, and it is superseded rather than edited. Teams that write only design docs lose the "why" within two quarters. Teams that write only ADRs never argue a decision properly, because an ADR has no room for the two rejected options that had real merit.

The quality bar for both artifacts is falsifiability: a reader who disagrees must be able to point at a sentence and argue with it. That means numbers instead of adjectives, named owners instead of "we decided", a stated scope boundary, an explicit list of what was considered and why it lost, and one line saying what evidence would change the answer. **Reference design**: this chapter's template puts a two-page verdict block first so a reviewer can stop early, forces at least two serious alternatives, and closes with open questions that each carry a name and a date. A document without a named question owner is a discussion; a document with dates is an agenda.

## The Document Is the Deliverable

A senior engineer's output is not code volume or diagram quality; it is a stream of decisions whose consequences are understood before they land. Diagrams, prototypes, spreadsheets, and whiteboards are all inputs to that stream, but only a written decision can be reviewed by someone who was not in the room, cited by a team that arrives later, or audited after an incident. The organization's architecture is, in practice, the set of decisions it can reconstruct. Whatever is not written down gets re-litigated or quietly violated, and both outcomes are expensive: re-litigation costs a quarter of arguing about Kafka again, violation costs a support queue that nobody owns.

The cost of changing a decision rises steeply and unevenly, which is why the artifact is worth its overhead:

| Decision state | Cost to reverse | Who has to be convinced |
|---|---|---|
| A paragraph comparing two options | Minutes; edit the file | The author and one reviewer |
| An accepted document with a review thread | Days; new doc plus migration note | Everyone who approved the original |
| Shipped behind a flag | One sprint; dual-write or adapter | Owning team, plus support training |
| Shipped and depended on by others | One to four quarters | Every consumer team, plus a migration plan |
| Shipped, and customers can see it | Effectively fixed | Product, legal, and existing integrations |

**Inference**: the ratio explains why experienced engineers spend disproportionate time on documents. A day spent making an option comparison rigorous is cheap against even the second row of that table, and the same day spent polishing a prototype usually buys less. This is also why "we'll figure it out in the pull request" fails as architecture: a pull request can review an implementation against its intent, but it cannot review an intent that was never written.

Writing forces enumeration, which is the actual mechanism at work. A sentence like "the cache should be reasonably fresh" survives conversation indefinitely and dies on paper, because paper makes the missing number visible. Every quantitative claim a document forces into the open (request rate, cardinality, retention window, tail latency target, tenant count, blast radius) is a decision that stops being implicit.

## What Each Artifact Owns

Name the artifact by the decision it enables and the person who has to agree. Six artifacts cover most of the surface, and each fails differently when used for the wrong job:

| Artifact | Question it answers | Lifetime | Decision it unblocks |
|---|---|---|---|
| One-pager or narrative memo | Is this worth building at all, and at what cost? | Until funded or rejected | Whether to staff the work |
| Design document | Which option, under which constraints, with which consequences? | Until implemented, then archived | What to build and how it is shaped |
| Architecture decision record | What is now true, and why did the alternative lose? | Years; superseded, not edited | Whether a later change is allowed |
| Architecture description or system overview | How does what exists actually fit together? | Living, versioned with the system | Onboarding, impact analysis, audit scope |
| Operations or launch review | Is this safe to run at expected load with current staffing? | Until launch, then re-run on change | Whether to put it in front of traffic |
| Runbook | What does the operator do at 3 a.m.? | Living; reviewed after every incident | Whether the team can carry a pager for it |

Three rules keep them separate:

- **Persuasion and record do not mix.** A design doc argues; an ADR states. If an ADR contains a comparison table with live options, it is a design doc that has lost its header.
- **A document that outlives its decision must be marked, not deleted.** Marking preserves the reasoning trail that stops the third re-design. Delete only when the artifact never carried a decision.
- **One owner per artifact, named in the first lines.** An unowned living document is a rumor with a URL; an owned obsolete document is still useful, because its expiry date is visible.

Larger systems formalize this as views over one model: ISO/IEC/IEEE 42010 requires that an architectural description name its stakeholders, their concerns, and the viewpoint that addresses each, precisely so that a document cannot hide an absence. The practical benefit for a working engineer is that "which view is this?" resolves reviewer complaints before they arrive: a capacity complaint against a deployment view is in scope, and the same complaint against a data-flow view is a misfiled review.

## Anatomy of a Design Document

```text
Title, author, status, reviewers, decision owner, date
1. Verdict (max 2 pages)
   - The recommendation in one paragraph
   - Constraints that are not negotiable (contracts, budgets, deadlines, compliance)
   - Requirements with numbers, and the non-goals that were explicitly dropped
   - What this decision does NOT cover
2. Background and forces
   - Current system as it behaves, not as it was designed
   - Measured baseline: p50/p99, error rate, cost per unit, headroom
   - The specific pain, expressed as a failure the team can reproduce
3. Options considered (at least two that could have won)
   Per option: mechanism, capacity ceiling, failure behavior, migration path,
   operational burden, cost, and the one condition under which it is the right answer
4. Decision and consequences
   - Why this option wins under these constraints specifically
   - What gets worse: latency, cost, complexity, on-call load, blast radius
   - Reversibility: what would it cost to change this in 12 months
5. Plan and seams
   - Interfaces that stay stable if the decision turns out wrong
   - Milestones with observable exit criteria
   - Rollout, dual-run, and rollback mechanics
6. Open questions
   - Each item carries a name and a date, or it is not an open question
7. Appendix
   - Data for the skeptic: benchmarks, arithmetic, vendor limits, prior incidents
```

Six writing rules do most of the quality work:

1. **Front-load the verdict.** Reviewers allocate attention by rank order; a document that reaches its recommendation on page nine gets reviewed as a tutorial and returns style comments.
2. **Quantify or delete the sentence.** "Reads dominate writes" becomes "8k reads/s against 60 writes/s, measured over thirty days at p99 of 41 ms." The second version can be wrong, which is what makes it reviewable.
3. **Name the non-goals.** Most scope creep in review comes from a reviewer solving a problem the author already decided to ignore; say so once and the thread disappears.
4. **Steelman every alternative.** If an option loses because it was described weakly, the decision is not actually made, and someone will re-open it after launch with a better-written document.
5. **Write the consequences section honestly.** A design whose downsides are "none" has not been thought about; naming the degradation buys credibility for the whole artifact and pre-registers the trade-off for the next planning cycle.
6. **State the reversibility, not just the choice.** Reversible decisions should be made fast by the person closest to them; irreversible ones deserve the document, the review, and the delay.

**Reference design**: keep one "what would change my mind" line per accepted option. It is the cheapest defense against sunk-cost drift, because it converts future contrary evidence from an argument into a scheduled re-decision.

## Decision Records That Survive Contact

An ADR records a decision that constrains future work. The format is deliberately boring, which is what makes it possible to keep hundreds of them:

```text
ADR-0142: Order events are published through the transactional outbox

Status: accepted (2026-03-11). Supersedes ADR-0088.
Context: order state lives in PostgreSQL and three consumers need change events.
         Dual writes were attempted in Q4 and lost 11k events during a failover.
Decision: publish inside the same transaction as the state change, then tail the
          outbox table with CDC. At-least-once delivery; consumers are idempotent.
Consequences: writers pay one extra row per event and outbox lag becomes a metric
              that must be alerted on. No exactly-once end to end; the inbox pattern
              covers effect boundaries where duplicates are unacceptable.
Alternatives rejected: CDC directly from WAL (no schema coupling but no transactional
                       guarantee for derived events), dual write (empirically broken).
```

Four rules keep the collection trustworthy:

- **Immutable once accepted.** Corrections arrive as a new record that supersedes the old one, and the old record's status changes to `superseded by ADR-nnn`. Editing history destroys the only thing an ADR is for: knowing what the team believed at the time.
- **Numbered and indexed, with the index generated.** Sequential IDs make "why did we do this?" answerable in one lookup; a directory listing plus a title table beats a taxonomy nobody maintains.
- **Stored next to the thing it constrains**, reviewed in the same pull request that changes it, so that CI can ask whether a change to a constrained path references an ADR. **Inference**: co-location is what makes drift visible; a record in a separate wiki decays at exactly the rate of whoever last remembered it.
- **Short enough to read in ninety seconds.** Context under ten lines. An ADR that needs background reading is a design doc appendix, and appendices are not read.

Decide whether a decision deserves a record by asking which of these is true: it is hard to reverse; it is expensive to change; it would surprise a competent new engineer; it constrains another team's options; or it was genuinely contested. Anything with three or more of those properties earns a record, and so does any decision that has already been argued twice, because the third argument is coming.

Recording what was *not* done is equally valuable. "We do not run a service mesh; sidecars were rejected at 40k rps per node" prevents an entire class of drift, and it answers the question an outsider will otherwise ask with an RFC of their own.

## Keeping Documents True

Documentation rot is not a discipline problem, it is an incentives problem: the person who writes the document is rarely the person who pays for its staleness. Three mechanisms do most of the work:

- **Expiry and owner in the header.** A living document carries `owner`, `last reviewed`, and `review cadence`. A document past its review date is treated as a claim of unknown validity rather than a spec, and readers can tell from the first screen.
- **Generate whatever can be generated.** Dependency graphs, API surface inventories, topic and consumer lists, capacity numbers, dashboards, and SLO tables come from code or telemetry. Handwritten copies of those facts are wrong within a quarter, and their being wrong poisons trust in the parts that matter.
- **Bind record maintenance to change.** A pull request that changes a constrained interface names the affected records; CI fails when it does not. This is the same mechanism that keeps a book's chapter list from drifting from its files: derive or check, never remember.

The failure mode this replaces is the invisible fork: three documents describing one system, each correct about a different quarter. When the fork is discovered, delete the losers explicitly and leave a pointer; silently keeping the newest is how the next engineer finds the wrong one first.

**Documented**: postmortems are the highest-yield source of missing records. Where an incident's root cause was an undocumented assumption, the corrective action is a record plus a test where possible, not a wiki page that restates the incident. Assumptions that cannot be written down are usually the ones that break.

## Review Mechanics

A review is a decision procedure with a deadline, and it fails in exactly two ways: too little scrutiny, or scrutiny that never terminates. Both are fixed by making the ask explicit rather than inviting opinion.

| Ask | What reviewers owe | Typical latency |
|---|---|---|
| "Is this the wrong problem?" | Scope and requirement objections only | 24 hours |
| "Pick A or B, I lean A" | A position, not a comment; silence means agreement | 48 hours, then the owner decides |
| "Find what breaks this" | Failure modes, capacity ceilings, migration gaps | 3 days with a named facilitator |
| "Sign off for launch" | Checklist evidence per domain: capacity, security, observability, cost | Blocking, per release policy |

Four mechanics keep reviews cheap:

1. **Async first, meeting last.** Written comments against a written document let three people think in parallel; a meeting lets thirteen people agree with whoever speaks first. Reserve the calendar for the one unresolved choice, and go in with exactly that choice on the invitation.
2. **Severity labels on every comment.** Blocker (this design causes an outage or data loss), major (a stated requirement is unmet), nit (style), curious (an unanswered question). Reviewers who must pick a label write fewer essays, and authors can triage without guessing intent.
3. **A decision log attached to the document.** Every resolved thread becomes one line: what was decided, by whom, on what evidence. This is where an ADR usually comes from, and it is the artifact that ends re-litigation.
4. **Named decider with an escalation path.** Absent consensus, the accountable owner decides and records the dissent. "Disagree and commit" only works when the disagreement is written down first, because otherwise the same argument returns with a new hire eighteen months later.

**Inference**: review latency is a design input, not just a schedule annoyance. If a decision costs five days of calendar to make and one day to reverse, the organization should push it down to the team; if it costs an hour to write down and four quarters to undo, it belongs in front of a reviewer with cross-system context.

## Failure Modes of Documentation Culture

| Anti-pattern | Symptom | Correction |
|---|---|---|
| Document theater | Long docs, no decisions, no dates, no owners | Reject any document without a verdict and an ask |
| ADR spam | Two hundred records restating "we used Postgres" | Only record decisions that are hard, costly, surprising, or contested |
| Rubber-stamp review | Approval in minutes with zero comments | Require one named risk per approval, or explicit "no risks seen in X, Y, Z" |
| CYA document | Written after the decision to justify it | Date and status honestly; mark retroactive records as such |
| Design by committee | Options merge into a hybrid nobody proposed | One owner decides; dissent recorded, not averaged |
| Template worship | All sections filled, no argument | Judge the verdict and the option comparison; delete unused sections |
| The forever draft | Status `draft` for eleven months while it is built from | Draft status blocks implementation, or it expires |
| Decision by prototype | The first demo wins the argument | Require the option table before the second prototype |
| Tribal decisions in pull requests | Rationale exists only in a closed thread | Promote any decision that constrains future code into a record |

The common root cause is treating documentation as reporting rather than as design work. Reporting is judged by completeness and pays out to the author's manager; design documents are judged by whether they changed a decision, and they pay out to whoever has to operate the system in a year. Teams that measure document count get volume; teams that read the consequences section get decisions.

## Measuring Whether It Works

Directional signals only, and none of them belongs on a dashboard for individual engineers:

- **Reversal rate**: what fraction of accepted decisions were superseded inside two quarters? A very low rate with high rework means nobody writes anything down; a high rate with stable systems means the team is exploring cheaply, which is good.
- **Time from question to decision**, tracked for the ten most expensive decisions rather than averaged across everything.
- **Predicted-incident share**: of the last ten Sev-1 incidents, how many had a constraint written down and ignored, versus a constraint nobody wrote? The first number says something about review quality, the second about documentation coverage, and they need different fixes.
- **Review latency and rework ratio**, watched together: fast reviews with rising rework mean approvals have become ceremonial.

**Inference**: expect these to be gamed if promoted. A team that optimizes reversal rate stops recording supersessions; a team that optimizes doc coverage produces document theater, which is the failure mode two rows above it in the previous table.

## Key Takeaways

1. Write the rationalization on purpose: real design discovery is messy, and whoever writes the document sets the model the team shares.
2. Separate persuasion from record: design docs argue between options and expire; decision records state what is now true and get superseded rather than edited.
3. Make every claim falsifiable, which in practice means attaching a number, a source, or an explicit assumption to each sentence that a reviewer could otherwise interpret two ways.
4. Include at least two alternatives that could have won, plus non-goals; a document with one option is a status report and readers will review it as one.
5. Record the consequences and the reversibility of the decision, because "which option wins under which constraint" is the sentence that saves the next design.
6. Keep records short, numbered, immutable, indexed, and next to the code they constrain; long records are not read and distant records are not found.
7. Generate what can be generated and bind the rest to change: an owner, a review date, and a CI check that asks which record this pull request honors.
8. Run reviews as decision procedures with an explicit ask, severity labels, a named decider, and a written decision log; meetings resolve one surviving disagreement, not thirteen opinions.
9. Treat postmortems as the primary source of missing records, since every undocumented assumption that broke is already paid for by an incident.
10. Judge the practice by reversal rate, decision latency, and predicted-incident share, and never by how much documentation exists.

## References

- Parnas, D., and Clements, P. "A Rational Design Process: How and Why to Fake It." IEEE Transactions on Software Engineering, 1986. The original argument that design proceeds by iterative guessing and that the documented rational structure is a deliberately constructed artifact.
- Parnas, D. "On the Criteria To Be Used in Decomposing Systems into Modules." Communications of the ACM, 1972. Information hiding as the reason module boundaries are decisions worth recording.
- Nygard, M. "Documenting Architecture Decisions." 2011. The decision-record format that most ADR tooling descends from: context, decision, consequences, and immutability by supersession.
- ISO/IEC/IEEE 42010:2022, "Systems and software engineering - Architecture description." Stakeholders, concerns, viewpoints, and the requirement that an architecture description state which of each it addresses.
- Brown, S. "The C4 Model for Visualising Software Architecture." 2011 onward. Altitudes for diagrams, and why one drawing cannot answer questions at three scales.
- Starke, G., and Reif, H. "arc42 - template for software architecture documentation." A fixed section set that keeps architecture descriptions comparable across teams.
- Bryar, W., and Carr, B. "Working Backwards: Insights, Stories, and Secrets from Inside Amazon." St. Martin's Press, 2021. Narrative memos instead of slide decks, and the six-page structure that forces the argument to be written in prose.
- Sharma, T., Bosworth, A., Zaharia, M., and Chakraborty, A. (eds). "Software Engineering at Google." O'Reilly Media, 2020. Design documents, code review norms, and documentation engineering as maintained artifacts with owners.
- Google Engineering Practices documentation. Public guidance on review responsibilities, comment severity, and resolving disagreement without escalation theater.
- Ousterhout, J. "A Philosophy of Software Design." 2nd ed. Yaknyam Press, 2021. Complexity as the thing design decisions either pay down or accumulate, and comments as documentation of intent rather than behavior.
- Brooks, F. "The Design of Design: Essays from a Computer Scientist." Addison-Wesley, 2010. Design processes, iterative sketches, and why the representation used for a design changes what can be noticed in it.
- Skelton, M., and Pais, S. "Team Topologies." IT Revolution Press, 2019. Interaction modes and cognitive load as reasons decisions need an owning team rather than a committee.

---

## Related Chapters

- [Threat Modeling and Abuse Case Engineering](../10-security/08-threat-modeling-and-abuse-cases.md): the abuse-case register attaches to a design while the design is still editable
- [Incident Command and Learning](../11-observability/07-incident-management.md): postmortems are the highest-yield source of decisions that were never written down
- [Capacity Planning and Back-of-the-Envelope Estimation](../01-foundations/10-capacity-planning.md): supplies the numbers that make a design document falsifiable instead of aspirational
- [ML Risk and Governance](../16-ml-systems/09-ml-risk-governance.md): auditable decision trails as a compliance requirement rather than an engineering preference
- [Repository Architecture for Safe Agentic Change](../19-compound-engineering/04-ai-native-software-architecture.md): written constraints become machine-readable guardrails once agents make the changes
- [Service and Platform Migration](../15-deployment/06-migration-strategies.md): migrations audit every decision that was never recorded, at full price
